"""One port for every model running on this machine (and, in a cluster, on
the other machines too: see ``cluster.py``).

Each model keeps its own port; the gateway (default :9000) routes to them the
way the OpenAI API does on a single base URL:

``GET /v1/models``
    The running models, read from ``running.json`` on every request (so a
    model started or stopped a moment ago is already reflected). ``id`` is
    the llmctl instance name; an ``llmctl`` block adds backend, port, API,
    context, vision, decision flag and live status.
``POST /v1/...`` with a JSON body
    Routed by its ``model`` field: an exact name, else a unique prefix, else
    (when ``model`` is absent) the only model serving that API.
    ``/v1/systemone`` goes to the decision endpoints, everything else to the
    llama-server / vLLM instances.
``/m/<name>/<path>``
    Any method and path, forwarded verbatim to that model (llama-server's
    ``/completion``, ``/tokenize``, ``/props``, a decision endpoint's
    ``/v1/spec``, ...).

With a registry configured, the list and the routing cover every node's
models; ``model@node`` pins a node, and an instance that can't take the
request (down, still loading, prompt too long) hands it to the next one.

Responses are streamed through as they arrive (SSE included); a client that
hangs up closes the upstream connection, so the model stops generating.
Standard library only, like systemone-proxy.
"""
from __future__ import annotations

import http.client
import ipaddress
import json
import signal
import socket
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlsplit

from llm_runner import __version__
from llm_runner import config as config_mod
from llm_runner import process as proc
from llm_runner.backends import DECIDE, DECIDER, GATEWAY, VLLM, is_decision_model
from llm_runner.cluster import (HEARTBEAT_EVERY, HOP_HEADER, OPENAI, PROTOCOL, SYSTEMONE,
                                TOKEN_HEADER, ClusterView, Instance, RegistryTable,
                                Unroutable, has_images, is_context_error, order, pick,
                                split_pin, token_ok)

DEFAULT_PORT = 9000
RETRY_BODY_MAX = 1 << 20     # an error reply this small is read to decide on failover
MAX_BODY = 256 * 1024 * 1024
HEALTH_TTL = 2.0
# Not forwarded either way: they describe one connection, not the message.
_HOP = {"connection", "keep-alive", "proxy-connection", "transfer-encoding", "te",
        "trailer", "upgrade", "host", "content-length"}


@dataclass(frozen=True)
class Model:
    name: str
    pid: int
    port: int
    backend: str
    started_at: float
    model_path: str
    cmd: tuple[str, ...]
    group: str = ""

    @property
    def api(self) -> str:
        return SYSTEMONE if self.backend in (DECIDE, DECIDER) else OPENAI


def _flag(cmd: tuple[str, ...], *names: str) -> str | None:
    for i, a in enumerate(cmd[:-1]):
        if a in names:
            return cmd[i + 1]
    return None


_decision_cache: dict[str, bool] = {}


def _is_decision(path: str) -> bool:
    if path not in _decision_cache:
        try:
            _decision_cache[path] = is_decision_model(Path(path))
        except OSError:
            _decision_cache[path] = False
    return _decision_cache[path]


def _group_of(rec: dict) -> str:
    if "group" in rec:
        return rec["group"]
    # Launched before groups existed: the default `run` would have given it.
    if rec.get("backend", "llama") != "llama" or not rec.get("model_path"):
        return ""
    from llm_runner.registry import default_group
    group = default_group(Path(rec["model_path"]))
    return "" if group == rec["name"] else group


class LocalModels:
    """The live models on this machine, from the state file (reloaded when it
    changes)."""

    def __init__(self, state_file: Path = proc.STATE_FILE):
        self.state_file = state_file
        self._mtime: float | None = None
        self._records: list[Model] = []
        self._health: dict[int, tuple[float, str]] = {}
        self._lock = threading.Lock()

    def models(self) -> list[Model]:
        with self._lock:
            try:
                mtime = self.state_file.stat().st_mtime
            except OSError:
                self._mtime, self._records = None, []
                return []
            if mtime != self._mtime:
                try:
                    raw = json.loads(self.state_file.read_text())
                except (OSError, ValueError):
                    raw = []
                self._mtime = mtime
                self._records = [
                    Model(r["name"], r["pid"], r["port"], r.get("backend", "llama"),
                          r.get("started_at", 0.0), r.get("model_path", ""),
                          tuple(r.get("cmd", ())), _group_of(r))
                    for r in raw if r.get("backend") != GATEWAY]
            records = list(self._records)
        # A model that died without `llmctl stop` is still in the file.
        return [m for m in records if proc._alive(m.pid)]

    def status(self, m: Model) -> str:
        now = time.monotonic()
        hit = self._health.get(m.port)
        if hit and now - hit[0] < HEALTH_TTL:
            return hit[1]
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{m.port}/health", timeout=1) as r:
                state = "ready" if r.status == 200 else "loading"
        except urllib.error.HTTPError:
            state = "loading"        # llama-server answers 503 while it loads
        except (OSError, ValueError):
            state = "down"
        self._health[m.port] = (now, state)
        return state

    def describe(self, m: Model, models: list[Model] | None = None) -> dict:
        info: dict = {"backend": m.backend, "api": m.api, "port": m.port,
                      "path_prefix": f"/m/{m.name}", "status": self.status(m),
                      "group": m.group or None, "model_path": m.model_path}
        if m.backend == DECIDE:
            # It reads images when the llama-server behind it has an mmproj.
            upstream = urlsplit(_flag(m.cmd, "--backend") or "").port
            behind = next((x for x in models or [] if x.port == upstream), None)
            info["vision"] = bool(behind and "--mmproj" in behind.cmd)
            info["decision"] = True
        elif m.backend == DECIDER:
            info["vision"] = False
            info["decision"] = True
        if m.api == OPENAI:
            ctx = _flag(m.cmd, "--max-model-len") if m.backend == VLLM else \
                _flag(m.cmd, "--ctx-size", "-c")
            info["context"] = int(ctx) if ctx and ctx.isdigit() else None
            info["vision"] = "--mmproj" in m.cmd
            info["embeddings"] = "--embeddings" in m.cmd or "pooling" in m.cmd
            info["decision"] = _is_decision(m.model_path)
            par = _flag(m.cmd, "--parallel", "-np", "--max-num-seqs")
            info["parallel"] = int(par) if par and par.isdigit() else None
        return {"id": m.name, "object": "model", "created": int(m.started_at),
                "owned_by": "llmctl", "llmctl": info}


def lan_ip() -> str:
    """This machine's address on the default route (no packet is sent)."""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("192.0.2.1", 9))
            return s.getsockname()[0]
    except OSError:
        return "127.0.0.1"


def _is_loopback(ip: str) -> bool:
    try:
        return ipaddress.ip_address(ip).is_loopback
    except ValueError:
        return False


class Gateway:
    """State shared by the request threads: local models, and in a cluster
    the node's view of the others (plus the registry table on the registry)."""

    def __init__(self, local: LocalModels | None = None,
                 cluster: config_mod.Cluster | None = None, port: int = DEFAULT_PORT):
        self.local = local or LocalModels()
        self.cluster = cluster if cluster and cluster.enabled else None
        self.node = cluster.node_name if cluster else socket.gethostname()
        self.priority = cluster.node_priority if cluster else 0
        self.port = port
        self.view = ClusterView()
        self.table = RegistryTable(__version__) if self.cluster and self.cluster.registry_serve \
            else None
        self._last_log: tuple | None = None

    @property
    def token(self) -> str | None:
        return self.cluster.token if self.cluster else None

    # -------------------------------------------------------------- models --
    def local_described(self) -> list[dict]:
        models = self.local.models()
        return [self.local.describe(m, models) for m in models]

    def own_registry(self) -> bool:
        """This gateway is the registry and reports to itself (no hop needed)."""
        return self.table is not None and not self.cluster.registry_url

    def sync_from_table(self) -> None:
        """On the registry, see node changes as they land, not a heartbeat later."""
        if self.own_registry() and self.view.age is not None:
            nodes = [n for n in self.table.snapshot() if n["node"] != self.node]
            self.view.update({"nodes": nodes, "warnings": self.view.warnings,
                              "node_url": self.view.node_url})

    def instances(self, local_only: bool = False) -> list[Instance]:
        out = [Instance(id=d["id"], node=self.node, priority=self.priority, local=True,
                        info=d["llmctl"], created=d["created"])
               for d in self.local_described()]
        return out if local_only else out + self.view.instances()

    @staticmethod
    def entry(cands: list[Instance]) -> dict:
        best = cands[0]
        return {"id": best.id, "object": "model", "created": best.created, "owned_by": "llmctl",
                "llmctl": {**best.info, "node": best.node,
                           "local": any(c.local for c in cands),
                           "nodes": [{"node": c.node, "priority": c.priority, "local": c.local,
                                      "status": c.status, "port": c.info.get("port"),
                                      "context": c.info.get("context"),
                                      "vision": c.info.get("vision")} for c in cands]}}

    def model_list(self, local_only: bool = False) -> list[dict]:
        insts = self.instances(local_only)
        by_id: dict[str, list[Instance]] = {}
        for i in insts:
            by_id.setdefault(i.id, []).append(i)
        out = [self.entry(order(c)) for c in by_id.values()]
        # A group shows as a model of its own once it has two members (one
        # member is reachable by its group name too, but needn't be listed twice).
        by_group: dict[str, list[Instance]] = {}
        for i in insts:
            if i.group:
                by_group.setdefault(i.group, []).append(i)
        for name, members in by_group.items():
            ids = sorted({i.id for i in members})
            if len(ids) > 1 and name not in by_id:
                e = self.entry(order(members))
                e["id"] = name
                e["llmctl"].update(group=name, members=ids, path_prefix=f"/m/{name}")
                for n, c in zip(e["llmctl"]["nodes"], order(members)):
                    n["model"] = c.id
                out.append(e)
        return out

    # ---------------------------------------------------------- heartbeats --
    def heartbeat_payload(self) -> dict:
        return {"protocol": PROTOCOL, "version": __version__, "node": self.node,
                "priority": self.priority, "port": self.port,
                "advertise": self.cluster.node_advertise if self.cluster else None,
                "models": self.local_described()}

    def _post_registry(self, path: str, payload: dict, timeout: float = 4) -> tuple[int, dict]:
        req = urllib.request.Request(
            self.cluster.registry_url.rstrip("/") + path, data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json", TOKEN_HEADER: self.token or ""})
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return r.status, json.loads(r.read() or b"{}")
        except urllib.error.HTTPError as e:
            try:
                return e.code, json.loads(e.read() or b"{}")
            except ValueError:
                return e.code, {"error": {"message": f"HTTP {e.code}"}}

    def heartbeat_once(self) -> None:
        payload = self.heartbeat_payload()
        try:
            if self.own_registry():
                if not payload["advertise"]:
                    payload["advertise"] = f"http://{lan_ip()}:{self.port}"
                status, reply = self.table.heartbeat(payload, "127.0.0.1")
            else:
                status, reply = self._post_registry("/registry/v1/heartbeat", payload)
        except (OSError, ValueError) as e:
            status, reply = 0, {"error": {"message": f"registry unreachable: {e}"}}
        if status == 200:
            self.view.update(reply)
        else:
            self.view.fail(reply.get("error", {}).get("message", f"HTTP {status}"))
        self._log_state()

    def _log_state(self) -> None:
        nodes = sorted(n["node"] for n in self.view.nodes())
        state = (self.view.error, tuple(self.view.warnings), tuple(nodes))
        if state == self._last_log:
            return
        self._last_log = state
        if self.view.error:
            print(f"cluster: {self.view.error}", flush=True)
        else:
            print(f"cluster: node {self.node} (priority {self.priority}) at "
                  f"{self.view.node_url}; other nodes: {', '.join(nodes) or '(none)'}", flush=True)
        for w in self.view.warnings:
            print(f"cluster warning: {w}", flush=True)

    def run_heartbeats(self, stop: threading.Event) -> None:
        last_sent, last_mtime = 0.0, None
        while not stop.is_set():
            try:
                mtime = self.local.state_file.stat().st_mtime
            except OSError:
                mtime = None
            if time.monotonic() - last_sent >= HEARTBEAT_EVERY or mtime != last_mtime:
                last_sent, last_mtime = time.monotonic(), mtime
                self.heartbeat_once()
            stop.wait(1.0)

    def goodbye(self) -> None:
        if self.cluster is None:
            return
        if self.own_registry():
            self.table.goodbye(self.node)
            return
        try:
            self._post_registry("/registry/v1/goodbye", {"node": self.node}, timeout=2)
        except (OSError, ValueError):
            pass

    def status(self) -> dict:
        age = self.view.age
        return {"node": self.node, "priority": self.priority, "version": __version__,
                "protocol": PROTOCOL, "cluster": self.cluster is not None,
                "registry": (self.cluster.registry_url or "self") if self.cluster else None,
                "registry_serve": self.table is not None,
                "registry_ok": self.cluster is not None and self.view.error is None
                and age is not None,
                "last_heartbeat_age": None if age is None else round(age, 1),
                "error": self.view.error, "warnings": self.view.warnings,
                "node_url": self.view.node_url,
                "local_models": [d["id"] for d in self.local_described()],
                "nodes": [{k: n[k] for k in ("node", "url", "priority", "version", "age")}
                          | {"models": [m["id"] for m in n.get("models", [])]}
                          for n in self.view.nodes()]}


def make_handler(gw: Gateway):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"
        server_version = "llmctl-gateway"

        def log_message(self, fmt, *args):  # one line per request, to the log file
            print(f"{self.address_string()} {fmt % args}", flush=True)

        # ---------------------------------------------------------- replies --
        def _json(self, status: int, payload: dict) -> None:
            body = json.dumps(payload).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            if self.command != "HEAD":
                self.wfile.write(body)

        def _error(self, status: int, code: str, message: str) -> None:
            kind = "invalid_request_error" if status < 500 else "server_error"
            self._json(status, {"error": {"message": message, "type": kind,
                                          "param": "model" if "model" in code else None,
                                          "code": code}})

        def _body(self) -> bytes | None:
            if self.headers.get("Transfer-Encoding", "").lower() == "chunked":
                out = bytearray()
                while True:
                    size = int(self.rfile.readline().split(b";")[0].strip() or b"0", 16)
                    if size == 0:
                        while self.rfile.readline() not in (b"\r\n", b"\n", b""):
                            pass
                        return bytes(out)
                    out += self.rfile.read(size)
                    self.rfile.readline()
                    if len(out) > MAX_BODY:
                        return None
            n = int(self.headers.get("Content-Length") or 0)
            if n > MAX_BODY:
                return None
            return self.rfile.read(n) if n else b""

        def _authorized(self) -> bool:
            if token_ok(gw.token, self.headers.get(TOKEN_HEADER)):
                return True
            self._error(401, "unauthorized", "Missing or wrong cluster token."
                        if gw.token else "This gateway is not in a cluster.")
            return False

        # ---------------------------------------------------------- routing --
        def _handle(self) -> None:
            path, _, query = self.path.partition("?")
            body = self._body()
            if body is None:
                self.close_connection = True
                return self._error(413, "body_too_large", f"Body over {MAX_BODY >> 20} MiB.")

            if path.startswith("/registry/v1/"):
                return self._registry(path, body)
            if path == "/cluster/v1/status":
                return self._json(200, gw.status())
            # A request from another gateway: authenticated, and never sent on.
            hop = HOP_HEADER in self.headers
            if hop and not self._authorized():
                return

            if path.startswith("/m/"):
                name, _, rest = path[3:].partition("/")
                name, node = split_pin(unquote(name))
                insts = gw.instances(local_only=hop)
                cands = order([i for i in insts if name in (i.id, i.group)
                               and (node is None or i.node == node)])
                if not cands:
                    where = f" on node \"{node}\"" if node else ""
                    return self._error(404, "model_not_found",
                                       f"No running model named \"{name}\"{where}. Running: "
                                       + (", ".join(sorted({i.id for i in insts})) or "(none)"))
                local_path = "/" + rest + ("?" + query if query else "")
                return self._forward(cands, lambda c: local_path if c.local
                                     else f"/m/{c.id}{local_path}", lambda c: body)

            if path in ("/health", "/"):
                return self._json(200, {"status": "ok", "node": gw.node,
                                        "models": len(gw.local.models())})
            if path == "/v1/models" and self.command in ("GET", "HEAD"):
                return self._json(200, {"object": "list", "data": gw.model_list(hop)})
            if path.startswith("/v1/models/") and self.command in ("GET", "HEAD"):
                name = unquote(path[len("/v1/models/"):])
                m = next((m for m in gw.model_list(hop) if m["id"] == name), None)
                if m is None:
                    return self._error(404, "model_not_found", f"No running model \"{name}\".")
                return self._json(200, m)

            api = SYSTEMONE if path.rstrip("/") in ("/v1/systemone", "/v1/decisions",
                                                    "/v1/spec") else OPENAI
            payload = None
            ctype = self.headers.get("Content-Type", "")
            if body and not ctype.startswith("multipart/"):
                # Parsed whatever the declared type: `curl -d` says form-urlencoded.
                try:
                    payload = json.loads(body)
                except ValueError:
                    if "json" in ctype:
                        return self._error(400, "invalid_json", "Body is not valid JSON.")
            by_header = not isinstance(payload, dict)
            requested = self.headers.get("X-Model") if by_header else payload.get("model")
            try:
                cands = pick(gw.instances(local_only=hop), requested, api,
                             needs_vision=has_images(payload))
            except Unroutable as e:
                return self._error(e.status, e.code, e.message)
            # A prefix, group, pin or nothing was sent: each instance gets its
            # own exact name (vLLM insists on it; a hop routes by it).
            bodies: dict[str, bytes] = {}

            def body_for(c: Instance) -> bytes:
                if by_header or requested == c.id:
                    return body
                if c.id not in bodies:
                    bodies[c.id] = json.dumps({**payload, "model": c.id}).encode()
                return bodies[c.id]
            if by_header and requested is not None:
                self.headers.replace_header("X-Model", cands[0].id)
            return self._forward(cands, lambda c: self.path, body_for)

        def _registry(self, path: str, body: bytes) -> None:
            if gw.table is None:
                return self._error(404, "not_a_registry",
                                   "This gateway is not the registry (`llmctl registry serve`).")
            if not self._authorized():
                return
            if path == "/registry/v1/nodes" and self.command == "GET":
                return self._json(200, {"registry_version": __version__, "protocol": PROTOCOL,
                                        "nodes": gw.table.snapshot()})
            try:
                payload = json.loads(body or b"{}")
            except ValueError:
                return self._error(400, "invalid_json", "Body is not valid JSON.")
            if path == "/registry/v1/heartbeat" and self.command == "POST":
                peer = self.client_address[0]
                if _is_loopback(peer):
                    peer = lan_ip()      # a node on this machine: others can't use 127.0.0.1
                status, reply = gw.table.heartbeat(payload, peer)
                gw.sync_from_table()
                return self._json(status, reply)
            if path == "/registry/v1/goodbye" and self.command == "POST":
                gw.table.goodbye(str(payload.get("node", "")))
                gw.sync_from_table()
                return self._json(200, {"ok": True})
            return self._error(404, "not_found", f"No registry endpoint {self.command} {path}.")

        # --------------------------------------------------------- forwarding --
        def _forward(self, cands: list[Instance], path_for, body_for) -> None:
            """Send to the best instance; on refusal before anything was relayed
            (unreachable, 502/503, gone, prompt too long) try the next one."""
            for n, c in enumerate(cands):
                if c.id != cands[0].id and "X-Model" in self.headers:
                    self.headers.replace_header("X-Model", c.id)
                if self._try(c, path_for(c), body_for(c), last=n == len(cands) - 1):
                    return

        def _try(self, c: Instance, path: str, body: bytes, last: bool) -> bool:
            if c.local:
                host, port, extra = "127.0.0.1", c.info["port"], {}
            else:
                u = urlsplit(c.url)
                host, port = u.hostname, u.port or 80
                extra = {HOP_HEADER: gw.node, TOKEN_HEADER: gw.token or ""}
            conn = http.client.HTTPConnection(host, port, timeout=10)
            try:
                headers = {k: v for k, v in self.headers.items()
                           if k.lower() not in _HOP and k not in (HOP_HEADER, TOKEN_HEADER)}
                headers.update(extra)
                if body or self.command in ("POST", "PUT", "PATCH"):
                    headers["Content-Length"] = str(len(body))
                try:
                    conn.connect()
                    conn.sock.settimeout(None)   # a long generation is not a hang
                    conn.request(self.command, path, body=body or None, headers=headers)
                    resp = conn.getresponse()
                except OSError as e:
                    where = f"port {port}" if c.local else f"node {c.node} ({c.url})"
                    if not last:
                        print(f"failover: {c.id} on {c.node} unreachable ({e})", flush=True)
                        return False
                    self._error(502, "upstream_unavailable",
                                f"{c.id} ({where}) is not answering: {e}")
                    return True
                length = resp.getheader("Content-Length")
                if not last and (resp.status in (502, 503) or (
                        resp.status in (400, 404, 413, 422) and length is not None
                        and int(length) <= RETRY_BODY_MAX)):
                    data = resp.read()
                    # 404 from another gateway: the model went away since its
                    # last heartbeat.
                    gone = resp.status == 404 and not c.local and b"model_not_found" in data
                    if resp.status in (502, 503) or gone or is_context_error(resp.status, data):
                        print(f"failover: {c.id} on {c.node} answered {resp.status}", flush=True)
                        return False
                    self._relay(resp, data)
                    return True
                self._relay(resp)
                return True
            finally:
                conn.close()

        def _relay(self, resp: http.client.HTTPResponse, data: bytes | None = None) -> None:
            self.send_response(resp.status, resp.reason)
            length = str(len(data)) if data is not None else resp.getheader("Content-Length")
            for k, v in resp.getheaders():
                if k.lower() not in _HOP:
                    self.send_header(k, v)
            chunked = length is None and self.command != "HEAD" and resp.status not in (204, 304)
            if chunked:
                self.send_header("Transfer-Encoding", "chunked")
            elif length is not None:
                self.send_header("Content-Length", length)
            self.end_headers()
            try:
                if data is not None:
                    self.wfile.write(data)
                    return
                while True:
                    chunk = resp.read1(65536)
                    if not chunk:
                        break
                    if chunked:
                        self.wfile.write(b"%x\r\n%s\r\n" % (len(chunk), chunk))
                    else:
                        self.wfile.write(chunk)
                if chunked:
                    self.wfile.write(b"0\r\n\r\n")
            except (BrokenPipeError, ConnectionResetError):
                self.close_connection = True   # client left; closing upstream stops it
            except (OSError, http.client.HTTPException):
                self.close_connection = True   # upstream died mid-reply

        do_GET = do_POST = do_PUT = do_PATCH = do_DELETE = do_HEAD = do_OPTIONS = _handle

    return Handler


def make_server(port: int, host: str = "0.0.0.0", gw: Gateway | None = None) -> ThreadingHTTPServer:
    gw = gw or Gateway(port=port)
    httpd = ThreadingHTTPServer((host, port), make_handler(gw))
    httpd.daemon_threads = True
    gw.port = httpd.server_address[1]
    return httpd


def serve(port: int = DEFAULT_PORT, host: str = "0.0.0.0") -> None:
    cluster = config_mod.get_cluster()
    gw = Gateway(LocalModels(), cluster, port)
    httpd = make_server(port, host, gw)
    print(f"llmctl gateway {__version__} on http://{host}:{port}/v1  "
          f"(node {gw.node}, state: {proc.STATE_FILE})", flush=True)
    stop = threading.Event()
    if gw.cluster:
        if not gw.token:
            print("cluster: no token set; registry and other nodes will refuse this one "
                  "(`llmctl registry set URL --token T`)", flush=True)
        role = "registry" if gw.table else "node"
        print(f"cluster: {role}, reporting to {gw.cluster.registry_url or 'itself'}", flush=True)
        threading.Thread(target=gw.run_heartbeats, args=(stop,), daemon=True).start()

    def _term(*_):
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM, _term)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        stop.set()
        gw.goodbye()
        httpd.server_close()

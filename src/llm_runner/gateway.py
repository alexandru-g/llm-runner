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
import os
import re
import uuid
import secrets
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
from llm_runner import auth
from llm_runner import config as config_mod
from llm_runner import logs
from llm_runner import metrics as metrics_mod
from llm_runner import process as proc
from llm_runner import supervisor as sup_mod
from llm_runner import tracing
from llm_runner.backends import (DECIDE, DECIDER, GATEWAY, SGLANG, VLLM, hf_has_vision,
                                 is_decision_model)
from llm_runner.cluster import (BUSY_HEADER, HEARTBEAT_EVERY, HOP_HEADER, OPENAI, PROTOCOL,
                                SPILL_HEADER, SYSTEMONE, TOKEN_HEADER, ClusterView,
                                Instance, RegistryTable,
                                Unroutable, has_images, is_context_error, order, pick,
                                split_pin, token_ok)

DEFAULT_PORT = 9000
RETRY_BODY_MAX = 1 << 20     # an error reply this small is read to decide on failover
MAX_BODY = 256 * 1024 * 1024
HEALTH_TTL = 2.0
# Answered without an API key: health checks come from proxies and Compose.
OPEN_PATHS = ("/", "/health", "/healthz", "/readyz")
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
    # (kind, image, digest, image_id); kind "host" for every record without
    # a runtime. image_id is what docker ran (an archive-loaded image runs by ID).
    runtime: tuple[str, str | None, str | None, str | None] = ("host", None, None, None)

    @property
    def api(self) -> str:
        return SYSTEMONE if self.backend in (DECIDE, DECIDER) else OPENAI


def _flag(cmd: tuple[str, ...], *names: str) -> str | None:
    for i, a in enumerate(cmd[:-1]):
        if a in names:
            return cmd[i + 1]
    return None


# Prometheus gauge of requests waiting for a free slot, per engine.
_QUEUE_METRIC = {VLLM: "vllm:num_requests_waiting", SGLANG: "sglang:num_queue_reqs"}
# Requests waiting for a slot, per engine, for llmctl_requests_queued.
_WAITING_METRIC = {**_QUEUE_METRIC, "llama": "llamacpp:requests_deferred"}
# Engines with a Prometheus /metrics on their own port.
_METRICS_BACKENDS = (VLLM, SGLANG, "llama")
# Metrics endpoints: the scrape-only token, else the API-key policy.
METRICS_PATHS = ("/metrics", "/metrics/sd")
_STATE_OF = {"ready": "ready", "loading": "starting", "down": "down"}
# Retry-After for a model held for a planned restart (`llmctl apply`), and
# for one whose engine says it is still loading.
HELD_RETRY_AFTER = 30
LOADING_RETRY_AFTER = 10
# A client's X-Request-Id is kept when it is up to 128 visible ASCII characters.
_REQUEST_ID = re.compile(r"[\x21-\x7e]{1,128}")
# The flag carrying the context size in each engine's recorded command.
_CTX_FLAGS = {VLLM: ("--max-model-len",), SGLANG: ("--context-length",)}


_decision_cache: dict[str, bool] = {}


def _is_decision(path: str) -> bool:
    if path not in _decision_cache:
        try:
            _decision_cache[path] = is_decision_model(Path(path))
        except OSError:
            _decision_cache[path] = False
    return _decision_cache[path]


_vision_cache: dict[str, bool] = {}
# Flags that keep vllm/sglang from loading a checkpoint's vision encoder, and
# the per-request multimodal limit that can turn images off (``{"image": 0}``).
_TEXT_ONLY_FLAGS = ("--language-model-only", "--language-only")
_MM_LIMIT_FLAGS = ("--limit-mm-per-prompt", "--limit-mm-data-per-request")


def _image_limit(cmd: tuple[str, ...]) -> int | None:
    """The image count a vllm/sglang command allows per request, or None when
    it sets no limit. Takes ``--flag JSON``, ``--flag=JSON`` and vLLM's
    ``--limit-mm-per-prompt.image N``."""
    for i, a in enumerate(cmd):
        name, eq, val = a.partition("=")
        if name in _MM_LIMIT_FLAGS:
            raw = val if eq else (cmd[i + 1] if i + 1 < len(cmd) else "")
            try:
                limits = json.loads(raw)
            except ValueError:
                continue
            if isinstance(limits, dict) and isinstance(limits.get("image"), int):
                return limits["image"]
        elif name in (f"{f}.image" for f in _MM_LIMIT_FLAGS):
            raw = val if eq else (cmd[i + 1] if i + 1 < len(cmd) else "")
            if raw.isdigit():
                return int(raw)
    return None


def _hf_vision(m: Model) -> bool:
    """A vllm/sglang instance reads images when its checkpoint has a vision
    tower (what ``llmctl list`` shows) and the command doesn't leave it out."""
    if any(a.partition("=")[0] in _TEXT_ONLY_FLAGS for a in m.cmd) or _image_limit(m.cmd) == 0:
        return False
    if m.model_path not in _vision_cache:
        try:
            _vision_cache[m.model_path] = hf_has_vision(Path(m.model_path))
        except OSError:
            _vision_cache[m.model_path] = False
    return _vision_cache[m.model_path]


def _runtime_of(rec: dict) -> tuple[str, str | None, str | None, str | None]:
    rt = rec.get("runtime") or {}
    return (rt.get("kind") or "host", rt.get("image"), rt.get("digest"), rt.get("image_id"))


def _group_of(rec: dict) -> str:
    from llm_runner.registry import group_of
    return group_of(rec["name"], rec.get("backend", "llama"), rec.get("model_path", ""),
                    rec.get("group"))


class LocalModels:
    """The live models on this machine, from the state file (reloaded when it
    changes)."""

    def __init__(self, state_file: Path = proc.STATE_FILE,
                 autostart_file: Path | None = None):
        self.state_file = state_file
        # Next to the state file, as proc.AUTOSTART_FILE is.
        self.autostart_file = autostart_file or state_file.with_name("autostart.json")
        self._mtime: float | None = None
        self._records: list[Model] = []
        self._health: dict[int, tuple[float, str]] = {}
        self._lock = threading.Lock()
        self._seen: set[tuple[str, int]] = set()

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
                          tuple(r.get("cmd", ())), _group_of(r), _runtime_of(r))
                    for r in raw if r.get("backend") != GATEWAY]
                self._log_new(self._records)
            records = list(self._records)
        # A model that died without `llmctl stop` is still in the file.
        return [m for m in records if proc._alive(m.pid)]

    def _log_new(self, records: list[Model]) -> None:
        """One log line per model the gateway starts routing to, with what
        runs it, so a benchmark log records the exact image."""
        now = {(m.name, m.pid) for m in records}
        for m in records:
            if (m.name, m.pid) not in self._seen:
                kind, image, digest, image_id = m.runtime
                where = (f"docker image={image} digest={digest} image_id={image_id}"
                         if kind == "docker" else kind)
                logs.emit(f"model {m.name}: backend={m.backend} port={m.port} runtime={where}",
                          event="model", model=m.name, backend=m.backend, port=m.port,
                          runtime=kind, image=image, image_digest=digest, image_id=image_id)
        self._seen = now

    def busy(self, name: str) -> bool:
        """Every slot of this local model is working, so a new request would
        queue. Unknown (no /slots, an error) counts as free."""
        models = self.models()
        m = next((x for x in models if x.name == name), None)
        if m is None:
            return False
        if m.backend == DECIDE:
            # A decision endpoint is as busy as the llama-server behind it.
            port = urlsplit(_flag(m.cmd, "--backend") or "").port
            m = next((x for x in models if x.port == port), None)
            if m is None:
                return False
        try:
            if m.backend in _QUEUE_METRIC:
                with urllib.request.urlopen(f"http://127.0.0.1:{m.port}/metrics",
                                            timeout=0.5) as r:
                    for line in r.read().decode().splitlines():
                        if line.startswith(_QUEUE_METRIC[m.backend]):
                            return float(line.rsplit(" ", 1)[1]) > 0
                return False
            if m.backend != "llama":
                return False
            with urllib.request.urlopen(f"http://127.0.0.1:{m.port}/slots", timeout=0.5) as r:
                slots = json.loads(r.read())
            if not isinstance(slots, list) or not slots:
                return False
            return all(isinstance(sl, dict) and sl.get("is_processing") for sl in slots)
        except (OSError, ValueError, KeyError, IndexError):
            return False

    def backend_metrics(self, m: Model, timeout: float = 2.0) -> str | None:
        """The engine's own /metrics text, or None (no endpoint, not answering)."""
        if m.backend not in _METRICS_BACKENDS:
            return None
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{m.port}/metrics",
                                        timeout=timeout) as r:
                return r.read().decode(errors="replace")
        except (OSError, ValueError):
            return None

    def waiting(self, m: Model) -> float | None:
        """Requests waiting for a slot on this model, from its engine."""
        name = _WAITING_METRIC.get(m.backend)
        text = self.backend_metrics(m, timeout=0.5) if name else None
        if text is None:
            return None
        total = None
        for line in text.splitlines():
            if line.startswith(name) and line[len(name):len(name) + 1] in ("{", " "):
                try:
                    total = (total or 0.0) + float(line.rsplit(" ", 1)[1])
                except ValueError:
                    pass
        return total

    def status(self, m: Model, fresh: bool = False) -> str:
        now = time.monotonic()
        hit = self._health.get(m.port)
        if hit and now - hit[0] < HEALTH_TTL and not fresh:
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

    def readiness(self) -> tuple[bool, dict[str, str]]:
        """Whether every model this machine should serve is ready, and each
        one's status. "Should serve" is the autostart manifest: `llmctl stop`
        takes a model out, a crash doesn't, so a dead backend reads
        ``missing``. With an empty manifest, the running models count. No
        model at all is not ready."""
        models = {m.name: m for m in self.models()}
        try:
            wanted = [s["name"] for s in json.loads(self.autostart_file.read_text())
                      if s.get("backend") != GATEWAY]
        except (OSError, ValueError, KeyError, TypeError):
            wanted = []
        states = {n: self.status(models[n]) if n in models else "missing"
                  for n in wanted or models}
        return bool(states) and all(s == "ready" for s in states.values()), states

    def describe(self, m: Model, models: list[Model] | None = None) -> dict:
        info: dict = {"backend": m.backend, "api": m.api, "port": m.port,
                      "path_prefix": f"/m/{m.name}", "status": self.status(m),
                      "group": m.group or None, "model_path": m.model_path,
                      "runtime": m.runtime[0], "image": m.runtime[1],
                      "image_digest": m.runtime[2], "image_id": m.runtime[3]}
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
            ctx = _flag(m.cmd, *_CTX_FLAGS.get(m.backend, ("--ctx-size", "-c")))
            info["context"] = int(ctx) if ctx and ctx.isdigit() else None
            # llama-server sees through its mmproj; vllm/sglang through the
            # checkpoint's own vision tower.
            info["vision"] = _hf_vision(m) if m.backend in (VLLM, SGLANG) else "--mmproj" in m.cmd
            info["embeddings"] = any(a in m.cmd for a in ("--embeddings", "pooling",
                                                          "--is-embedding"))
            info["decision"] = _is_decision(m.model_path)
            par = _flag(m.cmd, "--parallel", "-np", "--max-num-seqs",
                        "--max-running-requests")
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


def _answers_as(url: str, node: str, timeout: float = 1.0) -> bool:
    """The gateway at ``url`` is up and is ``node`` (not some other machine
    that has the same LAN address on another network)."""
    try:
        with urllib.request.urlopen(url.rstrip("/") + "/health", timeout=timeout) as r:
            return json.loads(r.read()).get("node") == node
    except (OSError, ValueError, AttributeError):
        return False


class GatewayMetrics:
    """What ``/metrics`` reports. Calls are counted by the gateway the client
    called (a call another gateway hands on is not counted again where it
    lands); ``node`` is where the model that answered runs."""

    def __init__(self, gw: "Gateway"):
        self.gw = gw
        r = self.registry = metrics_mod.Registry()
        lbl = ("model", "node", "backend")
        self.requests = r.counter("llmctl_requests_total",
                                  "Calls answered, by HTTP status.", lbl + ("status",))
        self.errors = r.counter("llmctl_request_errors_total",
                                "Calls that ended in an error, by error code.", lbl + ("code",))
        self.duration = r.histogram("llmctl_request_duration_seconds",
                                    "From the call to its last byte.", lbl)
        self.ttft = r.histogram("llmctl_time_to_first_token_seconds",
                                "Streamed calls: from the call to the first event.", lbl,
                                buckets=metrics_mod.TTFT_BUCKETS)
        self.tokens = r.counter("llmctl_tokens_total",
                                "Tokens from the replies' usage, in (prompt) and out.",
                                lbl + ("direction",))
        self.in_flight = r.gauge("llmctl_requests_in_flight",
                                 "Calls being relayed now.", lbl)
        r.gauge("llmctl_requests_queued", "Calls waiting for a slot in the engine.", lbl,
                collect=self._queued)
        self.spills = r.counter("llmctl_spills_total",
                                "Calls the best instance was too busy for, sent elsewhere.",
                                ("model", "from_node", "to_node"))
        self.failovers = r.counter("llmctl_failovers_total",
                                   "Instances that refused a call before answering.",
                                   ("model", "from_node", "reason"))
        r.gauge("llmctl_node_up", "Cluster nodes seen by this gateway: 1 up, 0 gone.",
                ("node",), collect=self._nodes)
        r.gauge("llmctl_instance_state", "1 for the current state of each local model.",
                ("model", "node", "backend", "state"), collect=self._states)
        self.restarts = r.counter("llmctl_instance_restarts_total",
                                  "Restarts by the gateway's supervisor.", ("model", "node"))
        r.gauge("llmctl_build_info", "The llmctl version.", ("version", "node"),
                collect=lambda: [({"version": __version__, "node": gw.node}, 1)])
        self._seen_nodes: set[str] = set()

    def _queued(self):
        for m in self.gw.local.models():
            n = self.gw.local.waiting(m)
            if n is not None:
                yield {"model": m.name, "node": self.gw.node, "backend": m.backend}, n

    def _nodes(self):
        up = {self.gw.node} | {n["node"] for n in self.gw.view.nodes()}
        self._seen_nodes |= up
        for node in sorted(self._seen_nodes):
            yield {"node": node}, 1 if node in up else 0

    def _states(self):
        for d in self.gw.local_described():
            info = d["llmctl"]
            yield {"model": d["id"], "node": self.gw.node, "backend": info["backend"],
                   "state": _STATE_OF.get(info["status"], info["status"])}, 1

    def render(self) -> str:
        return self.registry.render()


class Gateway:
    """State shared by the request threads: local models, and in a cluster
    the node's view of the others (plus the registry table on the registry)."""

    def __init__(self, local: LocalModels | None = None,
                 cluster: config_mod.Cluster | None = None, port: int = DEFAULT_PORT,
                 access=None):
        self.local = local or LocalModels()
        # Who needs an API key: a callable returning the current auth.Policy.
        self.access = access or auth.ConfigPolicy()
        self.cluster = cluster if cluster and cluster.enabled else None
        self.node = cluster.node_name if cluster else socket.gethostname()
        self.priority = cluster.node_priority if cluster else 0
        self.spill = cluster.spill if cluster else True
        self.port = port
        # Lets the registry tell this machine changing networks from another
        # machine using its name.
        self.instance = secrets.token_hex(8)
        self.view = ClusterView()
        # Node URL -> whether this machine reached it at the last probe.
        self._direct: dict[str, bool] = {}
        self.table = RegistryTable(__version__) if self.cluster and self.cluster.registry_serve \
            else None
        self._last_log: tuple | None = None
        self.metrics = GatewayMetrics(self)
        # OTLP span export, when configured (serve() sets it up).
        self.exporter: tracing.Exporter | None = None
        # Restarts crashed models; its loop runs in serve().
        self.supervisor = sup_mod.Supervisor(
            on_restart=lambda name: self.metrics.restarts.inc(model=name, node=self.node))

    @property
    def token(self) -> str | None:
        return self.cluster.token if self.cluster else None

    # -------------------------------------------------------------- models --
    def local_described(self) -> list[dict]:
        """The live local models, plus supervised ones that are down
        (``restarting`` or ``crashed``): routable, so a call gets a 503."""
        models = self.local.models()
        out = [self.local.describe(m, models) for m in models]
        live = {m.name for m in models}
        for d in out:
            t = self.supervisor.state(d["id"])
            d["llmctl"]["restarts"] = t.total_restarts if t else 0
        down = {t.name: t for t in self.supervisor.down() if t.name not in live}
        if down:
            for spec in proc.autostart_list():
                t = down.get(spec.name)
                if t is None:
                    continue
                m = Model(spec.name, 0, spec.port, spec.backend, 0.0, spec.model_path,
                          tuple(spec.cmd), _group_of(vars(spec)),
                          _runtime_of({"runtime": spec.runtime}))
                d = self.local.describe(m, models)
                d["llmctl"].update(status=t.state, restarts=t.total_restarts)
                out.append(d)
        return out

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
                                      "vision": c.info.get("vision"),
                                      "runtime": c.info.get("runtime") or "host",
                                      "image": c.info.get("image"),
                                      "image_digest": c.info.get("image_digest"),
                                      "image_id": c.info.get("image_id")}
                                     for c in cands]}}

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
                # For a registry behind a reverse proxy, which sees the proxy's address.
                "lan": f"http://{lan_ip()}:{self.port}", "instance": self.instance,
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

    def probe_nodes(self) -> None:
        """Check that each node answers at the address the registry gave. A
        laptop away from home can't reach LAN addresses: its calls for those
        nodes then go through the registry instead (``relay_url``)."""
        if self.cluster is None or self.own_registry() or not self.cluster.registry_url:
            return
        direct = {n["url"]: _answers_as(n["url"], n["node"]) for n in self.view.nodes()}
        for n in self.view.nodes():
            if direct[n["url"]] != self._direct.get(n["url"], True):
                logs.emit(f"cluster: {n['node']} at {n['url']} " +
                          ("answers again; calling it directly" if direct[n["url"]] else
                           f"does not answer; its calls go through the registry "
                           f"({self.cluster.registry_url})"), event="cluster",
                          level="info" if direct[n["url"]] else "warn", peer=n["node"])
        self._direct = direct

    def relay_url(self, c: Instance) -> str | None:
        """The registry's URL when ``c``'s node didn't answer the last probe,
        else None (call it directly)."""
        if c.local or self.cluster is None or not self.cluster.registry_url:
            return None
        if self._direct.get(c.url, True) or c.url == self.cluster.registry_url:
            return None
        return self.cluster.registry_url

    def _log_state(self) -> None:
        nodes = sorted(n["node"] for n in self.view.nodes())
        state = (self.view.error, tuple(self.view.warnings), tuple(nodes))
        if state == self._last_log:
            return
        self._last_log = state
        if self.view.error:
            logs.emit(f"cluster: {self.view.error}", event="cluster", level="error")
        else:
            logs.emit(f"cluster: node {self.node} (priority {self.priority}) at "
                      f"{self.view.node_url}; other nodes: {', '.join(nodes) or '(none)'}",
                      event="cluster", nodes=nodes)
        for w in self.view.warnings:
            logs.emit(f"cluster warning: {w}", event="cluster", level="warn")

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
                self.probe_nodes()
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


def _usage_and_code(body: bytes, stream: bool) -> tuple[dict | None, str | None]:
    """``usage`` and ``error.code`` from a reply body (JSON), or from the
    last SSE event that carries a usage."""
    if not body:
        return None, None
    docs: list = []
    if stream:
        for line in reversed(body.split(b"\n")):
            if line.startswith(b"data:") and b'"usage"' in line:
                try:
                    docs = [json.loads(line[5:])]
                except ValueError:
                    continue
                break
    else:
        try:
            docs = [json.loads(body)]
        except ValueError:
            return None, None
    doc = docs[0] if docs and isinstance(docs[0], dict) else {}
    usage = doc.get("usage") if isinstance(doc.get("usage"), dict) else None
    err = doc.get("error")
    code = err.get("code") if isinstance(err, dict) and isinstance(err.get("code"), str) \
        else None
    return usage, code


def make_handler(gw: Gateway):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"
        server_version = "llmctl-gateway"
        # Headers and body go out as separate small writes; with Nagle on, the
        # body waits for the client's delayed ACK (~40 ms on Linux) on every
        # keep-alive reply, and every SSE chunk after the first is held back too.
        disable_nagle_algorithm = True

        def log_message(self, fmt, *args):  # one line per request, to the log file
            if not logs.is_json():           # JSON mode writes its own (_record)
                print(f"{self.address_string()} {fmt % args}", flush=True)

        # ---------------------------------------------------------- replies --
        def _json(self, status: int, payload: dict, headers: dict | None = None) -> None:
            body = json.dumps(payload).encode()
            self._replied = True
            self._obs["status"] = status
            self.send_response(status)
            for k, v in (headers or {}).items():
                self.send_header(k, v)
            self.send_header("X-Request-Id", self._obs["rid"])
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            if self.command != "HEAD":
                self.wfile.write(body)

        def _error(self, status: int, code: str, message: str,
                   headers: dict | None = None) -> None:
            self._obs["code"] = code
            kind = "invalid_request_error" if status < 500 else "server_error"
            self._json(status, {"error": {"message": message, "type": kind,
                                          "param": "model" if "model" in code else None,
                                          "code": code}}, headers)

        def _unavailable(self, name: str, down: tuple[str, int | None]) -> None:
            """503 for a model that is configured here but not answering."""
            why, wait = down
            self._error(503, "model_unavailable", f"{name} is unavailable: {why}",
                        {"Retry-After": str(wait)} if wait is not None else None)

        def _configured_down(self, name: str | None) -> tuple[str, int | None] | None:
            """Why ``name`` (a model or group in this machine's autostart
            manifest, or one held for a planned restart) isn't running, and
            the Retry-After to send; None if it isn't configured here."""
            if not name:
                return None
            if held := self._held(name):
                return held
            for spec in proc.autostart_list():
                if spec.backend != GATEWAY and name in (spec.name, spec.group):
                    t = gw.supervisor.state(spec.name)
                    return (f"it is {t.state}" if t else "it is not running",
                            gw.supervisor.retry_after(spec.name))
            return None

        @staticmethod
        def _loading(name: str) -> bool:
            m = next((m for m in gw.local.models() if m.name == name), None)
            return m is not None and gw.local.status(m, fresh=True) == "loading"

        @staticmethod
        def _held(name: str) -> tuple[str, int] | None:
            for held, h in proc.holds().items():
                if name in (held, h.get("group")):
                    return f"it is being restarted ({h.get('reason')})", HELD_RETRY_AFTER
            return None

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

        def _key_ok(self) -> bool:
            """An API key, when this caller needs one (see auth.py)."""
            policy = gw.access()
            if policy.mode == auth.OFF or not policy.needs_key(self.client_address[0],
                                                              self.headers):
                return True
            if policy.key_name(self.headers):
                return True
            sent = bool(self.headers.get("Authorization") or self.headers.get("X-Api-Key"))
            self._error(401, "invalid_api_key",
                        "Wrong API key." if sent else
                        "This gateway needs an API key: Authorization: Bearer <key>.")
            return False

        def _authorized(self) -> bool:
            if token_ok(gw.token, self.headers.get(TOKEN_HEADER)):
                return True
            self._error(401, "unauthorized", "Missing or wrong cluster token."
                        if gw.token else "This gateway is not in a cluster.")
            return False

        # ---------------------------------------------------------- routing --
        def _handle(self) -> None:
            self._replied = False
            sent_id = self.headers.get("X-Request-Id") or ""
            # What /metrics and the request log learn about this call (_record).
            self._obs = {"t0": time.monotonic(), "count": False, "inst": None,
                         "status": None, "code": None, "ttft": None, "usage": None,
                         "rid": sent_id if _REQUEST_ID.fullmatch(sent_id) else uuid.uuid4().hex,
                         "failovers": 0, "bytes": 0, "trace": None}
            self._trace = tracing.Context.from_headers(self.headers)
            path = self.path.partition("?")[0]
            self._span = self._trace.span(
                f"{self.command} {path}", tracing.SERVER, **{
                    "http.request.method": self.command, "url.path": path,
                    "llmctl.request_id": self._obs["rid"], "llmctl.node": gw.node})
            self._obs["trace"] = {"trace_id": self._trace.trace_id}
            try:
                self._route()
            except Exception as e:  # never drop a connection without an answer
                import traceback
                traceback.print_exc()
                self.close_connection = True
                if not self._replied:
                    self._error(500, "gateway_error", f"llmctl gateway: {type(e).__name__}: {e}")
            finally:
                self._record()

        def _record(self) -> None:
            o = self._obs
            self._log_request()
            self._export_span()
            if not o["count"]:
                return
            c = o["inst"]
            lbl = {"model": c.id if c else "", "node": c.node if c else "",
                   "backend": (c.info.get("backend") or "") if c else ""}
            m = gw.metrics
            status = o["status"] or 0
            m.requests.inc(**lbl, status=str(status))
            if status >= 400 or o["code"]:
                m.errors.inc(**lbl, code=o["code"] or f"http_{status}")
            if c is not None:
                m.duration.observe(time.monotonic() - o["t0"], **lbl)
                if o["ttft"] is not None:
                    m.ttft.observe(o["ttft"], **lbl)
                usage = o["usage"] or {}
                for direction, key in (("in", "prompt_tokens"), ("out", "completion_tokens")):
                    if isinstance(usage.get(key), int):
                        m.tokens.inc(usage[key], **lbl, direction=direction)

        def _export_span(self) -> None:
            path = self.path.partition("?")[0]
            if gw.exporter is None or not self._trace.sampled or path in OPEN_PATHS \
                    or path.startswith("/metrics"):
                return
            o, c = self._obs, self._obs["inst"]
            usage = o["usage"] or {}
            self._span.error = (o["status"] or 0) >= 500 or o["code"] == "model_unavailable"
            self._span.end(**{
                "http.response.status_code": o["status"], "error.type": o["code"],
                "llmctl.model": c.id if c else None,
                "llmctl.backend": c.info.get("backend") if c else None,
                "llmctl.served_by": c.node if c else None,
                "llmctl.ttft_ms": None if o["ttft"] is None else round(o["ttft"] * 1000, 1),
                "gen_ai.usage.input_tokens": usage.get("prompt_tokens"),
                "gen_ai.usage.output_tokens": usage.get("completion_tokens"),
                "llmctl.failovers": o["failovers"] or None})
            gw.exporter.submit(self._span)

        def _log_request(self) -> None:
            """JSON mode: one line per call, without bodies. Probes and scrapes
            (health, /metrics) are left out."""
            if not logs.is_json():
                return
            path = self.path.partition("?")[0]
            if path in OPEN_PATHS or path.startswith("/metrics"):
                return
            o, c = self._obs, self._obs["inst"]
            usage = o["usage"] or {}
            logs.request({
                "request_id": o["rid"], "trace_id": (o["trace"] or {}).get("trace_id"),
                "method": self.command, "path": path,
                "model": c.id if c else None, "backend": c.info.get("backend") if c else None,
                "served_by": c.node if c else None, "status": o["status"], "code": o["code"],
                "duration_ms": round((time.monotonic() - o["t0"]) * 1000, 1),
                "ttft_ms": None if o["ttft"] is None else round(o["ttft"] * 1000, 1),
                "tokens_in": usage.get("prompt_tokens"),
                "tokens_out": usage.get("completion_tokens"),
                "bytes_out": o["bytes"], "failovers": o["failovers"] or None,
                "client": self.address_string()}, f"{self.command} {path} {o['status']}")

        def _metrics_ok(self) -> bool:
            """Scrapes: the scrape-only token when one is set, else the API-key
            policy (``/v1`` callers' keys)."""
            policy = gw.access()
            if policy.mode == auth.OFF or not policy.needs_key(self.client_address[0],
                                                              self.headers):
                return True
            if policy.metrics_token:
                if policy.is_metrics_token(self.headers):
                    return True
                self._error(401, "invalid_api_key", "Scrapes need the metrics token: "
                            "Authorization: Bearer <token> (`llmctl gateway metrics-token`).")
                return False
            return self._key_ok()

        def _metrics(self, path: str) -> None:
            if path == "/metrics":
                body = gw.metrics.render().encode()
                ctype = "text/plain; version=0.0.4; charset=utf-8"
            elif path == "/metrics/sd":
                host = self.headers.get("Host") or f"{lan_ip()}:{gw.port}"
                body = json.dumps([
                    {"targets": [host],
                     "labels": {"__metrics_path__": f"/metrics/backends/{m.name}",
                                "model": m.name, "node": gw.node, "backend": m.backend}}
                    for m in gw.local.models() if m.backend in _METRICS_BACKENDS]).encode()
                ctype = "application/json"
            else:
                name = unquote(path[len("/metrics/backends/"):])
                m = next((x for x in gw.local.models() if x.name == name), None)
                if m is None or m.backend not in _METRICS_BACKENDS:
                    return self._error(404, "model_not_found",
                                       f"No local model \"{name}\" with engine metrics.")
                text = gw.local.backend_metrics(m)
                if text is None:
                    return self._error(503, "model_unavailable",
                                       f"{name}'s engine is not answering /metrics.")
                body = metrics_mod.add_labels(text, {"model": m.name, "node": gw.node,
                                                     "backend": m.backend}).encode()
                ctype = "text/plain; version=0.0.4; charset=utf-8"
            self._replied = True
            self._obs["status"] = 200
            self.send_response(200)
            self.send_header("X-Request-Id", self._obs["rid"])
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            if self.command != "HEAD":
                self.wfile.write(body)

        def _route(self) -> None:
            path, _, query = self.path.partition("?")
            body = self._body()
            if body is None:
                self.close_connection = True
                return self._error(413, "body_too_large", f"Body over {MAX_BODY >> 20} MiB.")

            if path.startswith("/registry/v1/"):
                return self._registry(path, body)
            # A request from another gateway: authenticated, and never sent on.
            hop = HOP_HEADER in self.headers
            if hop and not self._authorized():
                return
            is_metrics = path in METRICS_PATHS or path.startswith("/metrics/backends/")
            if not hop and path not in OPEN_PATHS and not is_metrics and not self._key_ok():
                return
            if path == "/cluster/v1/status":
                return self._json(200, gw.status())
            if not hop and self.command in ("GET", "HEAD") and (
                    path in METRICS_PATHS or path.startswith("/metrics/backends/")):
                return self._metrics(path) if self._metrics_ok() else None

            if path.startswith("/m/"):
                self._obs["count"] = not hop
                name, _, rest = path[3:].partition("/")
                name, node = split_pin(unquote(name))
                insts = gw.instances(local_only=hop)
                cands = order([i for i in insts if name in (i.id, i.group)
                               and (node is None or i.node == node)])
                if not cands and node is None and (down := self._configured_down(name)):
                    return self._unavailable(name, down)
                if not cands:
                    where = f" on node \"{node}\"" if node else ""
                    return self._error(404, "model_not_found",
                                       f"No running model named \"{name}\"{where}. Running: "
                                       + (", ".join(sorted({i.id for i in insts})) or "(none)"))
                local_path = "/" + rest + ("?" + query if query else "")
                return self._forward(cands, lambda c: local_path if c.local
                                     else f"/m/{c.id}{local_path}", lambda c: body)

            if path in ("/health", "/healthz", "/"):
                return self._json(200, {"status": "ok", "node": gw.node,
                                        "models": len(gw.local.models())})
            if path == "/readyz":
                ready, models = gw.local.readiness()
                return self._json(200 if ready else 503,
                                  {"status": "ready" if ready else "not_ready",
                                   "models": models})
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
            self._obs["count"] = not hop
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
                name, pin = split_pin(requested) if isinstance(requested, str) else (None, None)
                if e.code == "model_not_found" and pin is None and \
                        (down := self._configured_down(name)):
                    return self._unavailable(name, down)
                return self._error(e.status, e.code, e.message)
            if hop and self.headers.get(SPILL_HEADER) == "1":
                cands = [c for c in cands if not gw.local.busy(c.id)]
                if not cands:
                    return self._busy()
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

        def _busy(self) -> None:
            self._obs.update(status=503, code="busy")
            body = json.dumps({"error": {"message": "all slots busy", "type": "server_error",
                                         "param": None, "code": "busy"}}).encode()
            self._replied = True
            self.send_response(503)
            self.send_header("Content-Type", "application/json")
            self.send_header("X-Request-Id", self._obs["rid"])
            self.send_header(BUSY_HEADER, "1")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

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
                status, reply = gw.table.heartbeat(
                    payload, peer, proxied=gw.access().via_proxy(peer, self.headers))
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
            (unreachable, 502/503, gone, prompt too long) try the next one.

            With spill on and a choice of instances, a first pass skips any
            whose slots are all busy, so a free one further down the list takes
            the call. If every one is busy, the call queues on the best one."""
            def attempt(c: Instance, last: bool, spill: bool) -> str:
                if "X-Model" in self.headers:
                    self.headers.replace_header("X-Model", c.id)
                return self._try(c, path_for(c), body_for(c), last=last, spill=spill)

            busy: list[Instance] = []
            if gw.spill and len(cands) > 1:
                for n, c in enumerate(cands):
                    # The last one may relay its error, unless busy ones wait.
                    last = n == len(cands) - 1 and not busy
                    if c.local and gw.local.busy(c.id):
                        result = "busy"
                    else:
                        result = attempt(c, last, spill=not c.local)
                    if result == "done":
                        if busy:
                            gw.metrics.spills.inc(model=c.id, from_node=busy[0].node,
                                                  to_node=c.node)
                        return
                    if result == "busy":
                        busy.append(c)
                if not busy:
                    return
                logs.emit(f"spill: every instance of {cands[0].id} is busy; queueing on "
                          f"{busy[0].node}", event="spill", model=cands[0].id,
                          request_id=self._obs["rid"])
                cands = busy
            for n, c in enumerate(cands):
                if attempt(c, n == len(cands) - 1, spill=False) == "done":
                    return

        def _try(self, c: Instance, path: str, body: bytes, last: bool,
                 spill: bool = False) -> str:
            """"done" (a reply was relayed), "failed" (try the next instance) or
            "busy" (a remote instance with every slot taken; only with spill)."""
            self._obs["inst"] = c
            lbl = {"model": c.id, "node": c.node, "backend": c.info.get("backend") or ""}
            tls, relay = False, None
            if c.local:
                host, port, extra = "127.0.0.1", c.info["port"], {}
            elif relay := gw.relay_url(c):
                # Through the registry's gateway as an ordinary client call (it
                # routes it on): with this machine's key, not the cluster token.
                u = urlsplit(relay)
                tls = u.scheme == "https"
                host, port = u.hostname, u.port or (443 if tls else 80)
                path = u.path.rstrip("/") + path
                key = config_mod.get_upstream_key()
                extra = {"Authorization": f"Bearer {key}"} if key else {}
                # This machine chose the node (and fails over itself): pin it,
                # or the registry would choose again.
                pinned = f"{c.id}@{c.node}"
                if path.startswith(f"{u.path.rstrip('/')}/m/{c.id}/"):
                    path = path.replace(f"/m/{c.id}/", f"/m/{pinned}/", 1)
                elif self.headers.get("X-Model") == c.id:
                    extra["X-Model"] = pinned
                elif body:
                    try:
                        payload = json.loads(body)
                    except ValueError:
                        payload = None
                    if isinstance(payload, dict) and payload.get("model") == c.id:
                        body = json.dumps({**payload, "model": pinned}).encode()
            else:
                u = urlsplit(c.url)
                host, port = u.hostname, u.port or 80
                extra = {HOP_HEADER: gw.node, TOKEN_HEADER: gw.token or ""}
                if spill:
                    extra[SPILL_HEADER] = "1"
            attempt = self._trace.span(f"upstream {c.id}", tracing.CLIENT,
                                       parent=self._span.span_id, **{
                                           "llmctl.model": c.id, "llmctl.node": c.node,
                                           "server.address": host, "server.port": port})
            extra = {**extra, "X-Request-Id": self._obs["rid"],
                     "traceparent": self._trace.traceparent(attempt)}
            conn = (http.client.HTTPSConnection if tls else http.client.HTTPConnection)(
                host, port, timeout=10)
            gw.metrics.in_flight.add(1, **lbl)
            outcome = "error"
            try:
                replaced = {k.lower() for k in extra}
                if "authorization" in replaced:
                    replaced.add("x-api-key")
                headers = {k: v for k, v in self.headers.items()
                           if k.lower() not in _HOP and k.lower() not in replaced
                           and k not in (HOP_HEADER, TOKEN_HEADER, SPILL_HEADER)}
                headers.update(extra)
                if body or self.command in ("POST", "PUT", "PATCH"):
                    headers["Content-Length"] = str(len(body))
                try:
                    conn.connect()
                    conn.sock.settimeout(None)   # a long generation is not a hang
                    conn.request(self.command, path, body=body or None, headers=headers)
                    resp = conn.getresponse()
                    attempt.attrs["http.response.status_code"] = resp.status
                except OSError as e:
                    attempt.attrs["error.type"] = type(e).__name__
                    where = f"port {port}" if c.local else \
                        f"node {c.node} (via {relay})" if relay else f"node {c.node} ({c.url})"
                    if not last:
                        logs.emit(f"failover: {c.id} on {c.node} unreachable ({e})",
                                  event="failover", level="warn", model=c.id, peer=c.node,
                                  reason="unreachable", request_id=self._obs["rid"])
                        self._obs["failovers"] += 1
                        gw.metrics.failovers.inc(model=c.id, from_node=c.node,
                                                 reason="unreachable")
                        return "failed"
                    if c.local:
                        self._unavailable(c.id, self._held(c.id) or (
                            f"its server ({where}) is not answering",
                            gw.supervisor.retry_after(c.id)))
                    else:
                        self._error(502, "upstream_unavailable",
                                    f"{c.id} ({where}) is not answering: {e}")
                    return "done"
                if spill and resp.status == 503 and resp.getheader(BUSY_HEADER):
                    resp.read()
                    return "busy"
                if last and c.local and resp.status == 503 and (down := self._held(c.id) or (
                        ("it is loading", LOADING_RETRY_AFTER) if self._loading(c.id) else None)):
                    # The engine's own "Loading model" (llama-server answers
                    # while it loads; vLLM doesn't listen yet): the same 503
                    # as for any model that is coming back.
                    resp.read()
                    attempt.attrs["error.type"] = "loading"
                    self._unavailable(c.id, down)
                    return "done"
                length = resp.getheader("Content-Length")
                if not last and (resp.status in (502, 503) or (
                        resp.status in (400, 404, 413, 422) and length is not None
                        and int(length) <= RETRY_BODY_MAX)):
                    data = resp.read()
                    # 404 from another gateway: the model went away since its
                    # last heartbeat.
                    gone = resp.status == 404 and not c.local and b"model_not_found" in data
                    if resp.status in (502, 503) or gone or is_context_error(resp.status, data):
                        logs.emit(f"failover: {c.id} on {c.node} answered {resp.status}",
                                  event="failover", level="warn", model=c.id, peer=c.node,
                                  reason=str(resp.status), request_id=self._obs["rid"])
                        self._obs["failovers"] += 1
                        gw.metrics.failovers.inc(
                            model=c.id, from_node=c.node,
                            reason="gone" if gone else str(resp.status)
                            if resp.status in (502, 503) else "context")
                        return "failed"
                    self._relay(resp, data)
                    return "done"
                self._relay(resp)
                outcome = "done"
                return "done"
            finally:
                gw.metrics.in_flight.add(-1, **lbl)
                conn.close()
                if gw.exporter is not None and self._trace.sampled:
                    status = attempt.attrs.get("http.response.status_code") or 0
                    attempt.error = outcome != "done" or status >= 500
                    attempt.end()
                    gw.exporter.submit(attempt)

        def _relay(self, resp: http.client.HTTPResponse, data: bytes | None = None) -> None:
            self._replied = True
            self._obs["status"] = resp.status
            ctype = (resp.getheader("Content-Type") or "").lower()
            stream = ctype.startswith("text/event-stream")
            # Kept to read `usage` (and an error code) from: a JSON reply
            # whole, up to 4 MiB; the tail of a stream, where the usage
            # chunk comes last.
            keep = bytearray()
            keep_json = "json" in ctype and int(resp.getheader("Content-Length") or 0) <= 4 << 20
            self.send_response(resp.status, resp.reason)
            length = str(len(data)) if data is not None else resp.getheader("Content-Length")
            for k, v in resp.getheaders():
                if k.lower() not in _HOP and k.lower() != "x-request-id":
                    self.send_header(k, v)
            self.send_header("X-Request-Id", self._obs["rid"])
            chunked = length is None and self.command != "HEAD" and resp.status not in (204, 304)
            if chunked:
                self.send_header("Transfer-Encoding", "chunked")
            elif length is not None:
                self.send_header("Content-Length", length)
            self.end_headers()
            sent, side = 0, "client"
            try:
                if data is not None:
                    keep += data
                    self.wfile.write(data)
                    self._obs["bytes"] = len(data)
                    return
                while True:
                    side = "upstream"
                    chunk = resp.read1(65536)
                    side = "client"
                    if not chunk:
                        break
                    if stream and self._obs["ttft"] is None:
                        self._obs["ttft"] = time.monotonic() - self._obs["t0"]
                    if stream:
                        keep += chunk
                        del keep[:-32768]
                    elif keep_json and len(keep) < 4 << 20:
                        keep += chunk
                    if chunked:
                        self.wfile.write(b"%x\r\n%s\r\n" % (len(chunk), chunk))
                    else:
                        self.wfile.write(chunk)
                    sent += len(chunk)
                    self._obs["bytes"] = sent
                if chunked:
                    self.wfile.write(b"0\r\n\r\n")
            except (OSError, http.client.HTTPException) as e:
                # Either way the connection is done: a client that left gets
                # nothing more, and one whose upstream broke sees the stream
                # cut short. Closing upstream (in _try) stops generation.
                who = "client left" if side == "client" else "upstream broke"
                logs.emit(f"relay: {who}: {self.command} {self.path} for "
                          f"{self.address_string()} after {sent} bytes ({e!r})",
                          event="relay", level="warn", request_id=self._obs["rid"],
                          side=side, bytes_out=sent)
                self._obs["code"] = "client_left" if side == "client" else "stream_aborted"
                if side == "upstream" and stream and chunked:
                    # Say so instead of cutting the stream: one last SSE
                    # event with an OpenAI-style error, then a clean end.
                    event = b"data: " + json.dumps({"error": {
                        "message": "The model stopped answering mid-stream.",
                        "type": "server_error", "param": None,
                        "code": "model_unavailable"}}).encode() + b"\n\n"
                    try:
                        self.wfile.write(b"%x\r\n%s\r\n0\r\n\r\n" % (len(event), event))
                        self._obs["code"] = "model_unavailable"
                    except OSError:
                        pass
                self.close_connection = True
            finally:
                self._obs["usage"], code = _usage_and_code(bytes(keep), stream)
                if resp.status >= 400 and code and not self._obs["code"]:
                    self._obs["code"] = code

        do_GET = do_POST = do_PUT = do_PATCH = do_DELETE = do_HEAD = do_OPTIONS = _handle

    return Handler


class _Server(ThreadingHTTPServer):
    # socketserver's default listen backlog is 5. A burst of new connections
    # (a load test opening 16 streams at once) overflowed it, and the kernel
    # dropped the extra SYNs, so clients saw stalls and retries.
    request_queue_size = 128


def make_server(port: int, host: str = "0.0.0.0", gw: Gateway | None = None) -> ThreadingHTTPServer:
    gw = gw or Gateway(port=port)
    httpd = _Server((host, port), make_handler(gw))
    httpd.daemon_threads = True
    gw.port = httpd.server_address[1]
    return httpd


def serve(port: int = DEFAULT_PORT, host: str = "0.0.0.0") -> None:
    cluster = config_mod.get_cluster()
    gw = Gateway(LocalModels(), cluster, port)
    httpd = make_server(port, host, gw)
    cfg = config_mod.load()
    logs.configure(os.environ.get("LLM_RUNNER_GATEWAY_LOG") or cfg.get("gateway_log_format"),
                   gw.node)
    endpoint = os.environ.get("LLM_RUNNER_GATEWAY_OTLP") or cfg.get("gateway_otlp_endpoint")
    if endpoint:
        gw.exporter = tracing.Exporter(endpoint, {"service.name": "llmctl-gateway",
                                                  "service.version": __version__,
                                                  "host.name": gw.node})
        logs.emit(f"otlp: exporting gateway spans to {gw.exporter.url}", event="otlp")
    logs.emit(f"llmctl gateway {__version__} on http://{host}:{port}/v1  "
              f"(node {gw.node}, state: {proc.STATE_FILE})", event="start",
              version=__version__, port=port)
    stop = threading.Event()
    if gw.cluster:
        if not gw.token:
            logs.emit("cluster: no token set; registry and other nodes will refuse this one "
                      "(`llmctl registry set URL --token T`)", event="cluster", level="warn")
        role = "registry" if gw.table else "node"
        logs.emit(f"cluster: {role}, reporting to {gw.cluster.registry_url or 'itself'}",
                  event="cluster")
        threading.Thread(target=gw.run_heartbeats, args=(stop,), daemon=True).start()
    threading.Thread(target=gw.supervisor.run, args=(stop,), daemon=True).start()

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
        if gw.exporter is not None:
            gw.exporter.close()
        httpd.server_close()

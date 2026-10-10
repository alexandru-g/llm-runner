"""One port for every model running on this machine.

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

Responses are streamed through as they arrive (SSE included); a client that
hangs up closes the upstream connection, so the model stops generating.
Standard library only, like systemone-proxy.
"""
from __future__ import annotations

import http.client
import json
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote

from llm_runner import process as proc
from llm_runner.backends import DECIDE, DECIDER, GATEWAY, VLLM, is_decision_model

DEFAULT_PORT = 9000
OPENAI = "openai"
SYSTEMONE = "systemone"
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


class Registry:
    """The live models, from the state file (reloaded when it changes)."""

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
                          tuple(r.get("cmd", ())))
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

    def describe(self, m: Model) -> dict:
        info: dict = {"backend": m.backend, "api": m.api, "port": m.port,
                      "path_prefix": f"/m/{m.name}", "status": self.status(m),
                      "model_path": m.model_path}
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


class Unroutable(Exception):
    def __init__(self, status: int, code: str, message: str):
        super().__init__(message)
        self.status, self.code, self.message = status, code, message


def resolve(models: list[Model], requested: object, api: str) -> Model:
    """Pick the model a request is for. Raises :class:`Unroutable`."""
    pool = [m for m in models if m.api == api]
    kind = "decision endpoint" if api == SYSTEMONE else "model"
    names = ", ".join(m.name for m in pool) or "(none running)"
    if requested is None or requested == "":
        if len(pool) == 1:
            return pool[0]
        if not pool:
            raise Unroutable(503, "no_models", f"No {kind} is running on this machine.")
        raise Unroutable(400, "model_required",
                         f"Several {kind}s are running; set \"model\" to one of: {names}")
    if not isinstance(requested, str):
        raise Unroutable(400, "invalid_model", "\"model\" must be a string.")
    exact = next((m for m in pool if m.name == requested), None)
    if exact:
        return exact
    prefixed = [m for m in pool if m.name.startswith(requested)]
    if len(prefixed) == 1:
        return prefixed[0]
    if prefixed:
        raise Unroutable(400, "model_ambiguous",
                         f"\"{requested}\" matches several {kind}s: "
                         + ", ".join(m.name for m in prefixed))
    raise Unroutable(404, "model_not_found",
                     f"No running {kind} named \"{requested}\". Running: {names}")


def make_handler(registry: Registry):
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

        # ---------------------------------------------------------- routing --
        def _handle(self) -> None:
            path, _, query = self.path.partition("?")
            body = self._body()
            if body is None:
                self.close_connection = True
                return self._error(413, "body_too_large", f"Body over {MAX_BODY >> 20} MiB.")
            models = registry.models()

            if path.startswith("/m/"):
                name, _, rest = path[3:].partition("/")
                name = unquote(name)
                target = next((m for m in models if m.name == name), None)
                if target is None:
                    return self._error(404, "model_not_found",
                                       f"No running model named \"{name}\". Running: "
                                       + (", ".join(m.name for m in models) or "(none)"))
                return self._forward(target, "/" + rest + ("?" + query if query else ""), body)

            if path in ("/health", "/"):
                return self._json(200, {"status": "ok", "models": len(models)})
            if path == "/v1/models" and self.command in ("GET", "HEAD"):
                return self._json(200, {"object": "list",
                                        "data": [registry.describe(m) for m in models]})
            if path.startswith("/v1/models/") and self.command in ("GET", "HEAD"):
                name = unquote(path[len("/v1/models/"):])
                m = next((m for m in models if m.name == name), None)
                if m is None:
                    return self._error(404, "model_not_found", f"No running model \"{name}\".")
                return self._json(200, registry.describe(m))

            api = SYSTEMONE if path.rstrip("/") in ("/v1/systemone", "/v1/spec") else OPENAI
            payload = None
            ctype = self.headers.get("Content-Type", "")
            if body and not ctype.startswith("multipart/"):
                # Parsed whatever the declared type: `curl -d` says form-urlencoded.
                try:
                    payload = json.loads(body)
                except ValueError:
                    if "json" in ctype:
                        return self._error(400, "invalid_json", "Body is not valid JSON.")
            requested = payload.get("model") if isinstance(payload, dict) else \
                self.headers.get("X-Model")
            try:
                target = resolve(models, requested, api)
            except Unroutable as e:
                return self._error(e.status, e.code, e.message)
            if isinstance(payload, dict) and requested != target.name:
                # A prefix (or nothing) was sent; vLLM insists on its own name.
                payload["model"] = target.name
                body = json.dumps(payload).encode()
            return self._forward(target, self.path, body)

        def _forward(self, target: Model, path: str, body: bytes) -> None:
            conn = http.client.HTTPConnection("127.0.0.1", target.port, timeout=10)
            try:
                headers = {k: v for k, v in self.headers.items() if k.lower() not in _HOP}
                if body or self.command in ("POST", "PUT", "PATCH"):
                    headers["Content-Length"] = str(len(body))
                try:
                    conn.connect()
                    conn.sock.settimeout(None)   # a long generation is not a hang
                    conn.request(self.command, path, body=body or None, headers=headers)
                    resp = conn.getresponse()
                except OSError as e:
                    return self._error(502, "upstream_unavailable",
                                       f"{target.name} (port {target.port}) is not answering: {e}")
                self.send_response(resp.status, resp.reason)
                length = resp.getheader("Content-Length")
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
            finally:
                conn.close()

        do_GET = do_POST = do_PUT = do_PATCH = do_DELETE = do_HEAD = do_OPTIONS = _handle

    return Handler


def serve(port: int = DEFAULT_PORT, host: str = "0.0.0.0",
          registry: Registry | None = None) -> None:
    httpd = make_server(port, host, registry)
    print(f"llmctl gateway on http://{host}:{port}/v1  (state: {proc.STATE_FILE})", flush=True)
    httpd.serve_forever()


def make_server(port: int, host: str = "0.0.0.0",
                registry: Registry | None = None) -> ThreadingHTTPServer:
    httpd = ThreadingHTTPServer((host, port), make_handler(registry or Registry()))
    httpd.daemon_threads = True
    return httpd


"""The all-models gateway, against fake upstream servers on ephemeral ports."""
from __future__ import annotations

import http.client
import json
import os
import re
import socket
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from llm_runner import gateway as gw
from llm_runner.cluster import OPENAI, SYSTEMONE, Instance, Unroutable, pick


class _Upstream(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    disable_nagle_algorithm = True   # so latency tests measure only the gateway

    def log_message(self, *a):
        pass

    def _reply(self, status: int, payload: dict) -> None:
        body = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/health":
            return self._reply(200, {"status": "ok"})
        self._reply(200, {"who": self.server.label, "path": self.path, "method": "GET"})

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        if body.get("stream"):
            # SSE, close-delimited like llama-server: no Content-Length.
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Connection", "close")
            self.end_headers()
            for i in range(3):
                self.wfile.write(f"data: {i}\n\n".encode())
                self.wfile.flush()
                time.sleep(0.05)
            self.wfile.write(b"data: [DONE]\n\n")
            self.close_connection = True
            return
        self._reply(200, {"who": self.server.label, "path": self.path, "body": body})


def _serve(handler, label=None):
    srv = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    srv.daemon_threads = True
    srv.label = label
    threading.Thread(target=srv.serve_forever, args=(0.05,), daemon=True).start()
    return srv


def _record(name, port, backend="llama", pid=None, cmd=()):
    return {"name": name, "pid": pid or os.getpid(), "port": port, "started_at": 1.7e9,
            "model_path": f"/models/{name}.gguf", "log_file": "", "cmd": list(cmd),
            "backend": backend, "env": {}}


@pytest.fixture
def env(tmp_path):
    a = _serve(_Upstream, "gemma")
    b = _serve(_Upstream, "glm")
    d = _serve(_Upstream, "decide")
    state = tmp_path / "running.json"
    records = [
        _record("gemma-4-26b-q4", a.server_address[1],
                cmd=("llama-server", "--ctx-size", "32768", "--mmproj", "x", "--parallel", "2")),
        _record("glm-ocr", b.server_address[1], cmd=("llama-server", "--ctx-size", "8192")),
        _record("quyet.decide", d.server_address[1], backend="decide"),
        _record("gateway", 1, backend="gateway"),
    ]
    state.write_text(json.dumps(records))
    httpd = gw.make_server(0, "127.0.0.1", gw.Gateway(gw.LocalModels(state)))
    threading.Thread(target=httpd.serve_forever, args=(0.05,), daemon=True).start()
    yield {"port": httpd.server_address[1], "state": state, "records": records}
    for s in (a, b, d, httpd):
        s.shutdown()


def _call(port, method, path, body=None, headers=None):
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
    data = json.dumps(body).encode() if body is not None else None
    conn.request(method, path, body=data,
                 headers={"Content-Type": "application/json", **(headers or {})})
    resp = conn.getresponse()
    raw = resp.read()
    conn.close()
    return resp.status, json.loads(raw) if raw else None


def test_models_lists_running_with_details(env):
    status, out = _call(env["port"], "GET", "/v1/models")
    assert status == 200 and out["object"] == "list"
    by_id = {m["id"]: m for m in out["data"]}
    assert set(by_id) == {"gemma-4-26b-q4", "glm-ocr", "quyet.decide"}    # no gateway
    g = by_id["gemma-4-26b-q4"]
    assert g["object"] == "model" and g["owned_by"] == "llmctl"
    assert g["llmctl"]["status"] == "ready"
    assert g["llmctl"]["context"] == 32768 and g["llmctl"]["vision"] is True
    assert g["llmctl"]["parallel"] == 2 and g["llmctl"]["api"] == "openai"
    assert by_id["quyet.decide"]["llmctl"]["api"] == "systemone"


def test_routes_by_model_and_rewrites_prefix(env):
    status, out = _call(env["port"], "POST", "/v1/chat/completions",
                        {"model": "glm", "messages": []})
    assert status == 200 and out["who"] == "glm"
    assert out["path"] == "/v1/chat/completions"
    assert out["body"]["model"] == "glm-ocr"           # vLLM needs its exact name
    status, out = _call(env["port"], "POST", "/v1/chat/completions",
                        {"model": "gemma-4-26b-q4", "messages": []})
    assert out["who"] == "gemma"


def test_errors_are_openai_shaped(env):
    status, out = _call(env["port"], "POST", "/v1/chat/completions", {"model": "llama-70b"})
    assert status == 404 and out["error"]["code"] == "model_not_found"
    assert "gemma-4-26b-q4" in out["error"]["message"]
    status, out = _call(env["port"], "POST", "/v1/chat/completions", {"messages": []})
    assert status == 400 and out["error"]["code"] == "model_required"
    status, out = _call(env["port"], "POST", "/v1/chat/completions", {"model": "g"})
    assert status == 400 and out["error"]["code"] == "model_ambiguous"


def test_systemone_goes_to_decision_endpoints(env):
    # The only decision endpoint: no model needed, and chat models aren't candidates.
    status, out = _call(env["port"], "POST", "/v1/systemone", {"questions": []})
    assert status == 200 and out["who"] == "decide"
    status, out = _call(env["port"], "GET", "/v1/spec")
    assert out["who"] == "decide"


def test_path_prefix_reaches_any_endpoint(env):
    status, out = _call(env["port"], "GET", "/m/glm-ocr/props?x=1")
    assert status == 200 and out == {"who": "glm", "path": "/props?x=1", "method": "GET"}
    status, out = _call(env["port"], "GET", "/m/nope/props")
    assert status == 404


def test_streaming_passes_chunks_through(env):
    conn = http.client.HTTPConnection("127.0.0.1", env["port"], timeout=5)
    conn.request("POST", "/v1/chat/completions",
                 body=json.dumps({"model": "gemma", "stream": True}),
                 headers={"Content-Type": "application/json"})
    resp = conn.getresponse()
    assert resp.getheader("Content-Type") == "text/event-stream"
    assert resp.getheader("Transfer-Encoding") == "chunked"
    first = resp.read1(1024)
    assert first.startswith(b"data: 0")             # arrives before the stream ends
    rest = resp.read()
    assert (first + rest).endswith(b"data: [DONE]\n\n")
    conn.close()


def test_healthz_and_readyz(env):
    assert _call(env["port"], "GET", "/healthz")[0] == 200
    # No autostart manifest: the running models are what must be ready.
    status, out = _call(env["port"], "GET", "/readyz")
    assert status == 200 and out["status"] == "ready"
    assert set(out["models"]) == {"gemma-4-26b-q4", "glm-ocr", "quyet.decide"}
    # A manifest model that isn't running (crashed) makes the node not ready.
    manifest = [{"name": "gemma-4-26b-q4", "backend": "llama"},
                {"name": "crashed", "backend": "vllm"},
                {"name": "gateway", "backend": "gateway"}]
    (env["state"].parent / "autostart.json").write_text(json.dumps(manifest))
    status, out = _call(env["port"], "GET", "/readyz")
    assert status == 503 and out["status"] == "not_ready"
    assert out["models"] == {"gemma-4-26b-q4": "ready", "crashed": "missing"}


def test_readyz_with_nothing_running(tmp_path):
    (tmp_path / "running.json").write_text("[]")
    httpd = gw.make_server(0, "127.0.0.1", gw.Gateway(gw.LocalModels(tmp_path / "running.json")))
    threading.Thread(target=httpd.serve_forever, args=(0.05,), daemon=True).start()
    try:
        assert _call(httpd.server_address[1], "GET", "/readyz")[0] == 503
    finally:
        httpd.shutdown()


class _BrokenStream(_Upstream):
    def do_POST(self):
        self.rfile.read(int(self.headers["Content-Length"]))
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Transfer-Encoding", "chunked")
        self.end_headers()
        self.wfile.write(b"9\r\ndata: 0\n\n\r\n")
        self.wfile.flush()
        self.close_connection = True       # dies before the last chunk


def test_upstream_dying_mid_stream_ends_with_an_error_event(tmp_path, capsys):
    up = _serve(_BrokenStream)
    state = tmp_path / "running.json"
    state.write_text(json.dumps([_record("gemma", up.server_address[1])]))
    httpd = gw.make_server(0, "127.0.0.1", gw.Gateway(gw.LocalModels(state)))
    threading.Thread(target=httpd.serve_forever, args=(0.05,), daemon=True).start()
    try:
        conn = http.client.HTTPConnection("127.0.0.1", httpd.server_address[1], timeout=5)
        conn.request("POST", "/v1/chat/completions",
                     body=json.dumps({"model": "gemma", "stream": True}),
                     headers={"Content-Type": "application/json"})
        resp = conn.getresponse()
        body = resp.read()                  # a clean end, not IncompleteRead
        conn.close()
        assert body.startswith(b"data: 0\n\n")
        last = json.loads(body.strip().split(b"\n\n")[-1][len(b"data: "):])
        assert last["error"]["code"] == "model_unavailable"
        assert last["error"]["type"] == "server_error"
        out = ""
        for _ in range(40):              # the gateway logs from its own thread
            out += capsys.readouterr().out
            if "relay:" in out:
                break
            time.sleep(0.05)
        assert "relay: upstream broke: POST /v1/chat/completions" in out
        assert "after 9 bytes" in out
    finally:
        up.shutdown(); httpd.shutdown()


def test_keep_alive_serves_several_requests(env):
    conn = http.client.HTTPConnection("127.0.0.1", env["port"], timeout=5)
    for name in ("glm", "gemma", "glm"):
        conn.request("POST", "/v1/completions", body=json.dumps({"model": name}),
                     headers={"Content-Type": "application/json"})
        assert json.loads(conn.getresponse().read())["who"] == name
    conn.close()


def test_keep_alive_replies_are_not_held_by_nagle(env):
    # With Nagle on, each reply's body waits ~40 ms for the client's delayed ACK.
    conn = http.client.HTTPConnection("127.0.0.1", env["port"], timeout=5)
    times = []
    for _ in range(6):
        t0 = time.perf_counter()
        conn.request("POST", "/v1/completions", body=json.dumps({"model": "glm"}),
                     headers={"Content-Type": "application/json"})
        conn.getresponse().read()
        times.append(time.perf_counter() - t0)
    conn.close()
    assert sorted(times[1:])[2] < 0.02, times


def test_dead_and_unreachable_models(env):
    records = env["records"]
    records[1]["pid"] = 2 ** 22 + 12345                 # glm's process is gone
    records[0]["port"] = 1                              # gemma's port refuses
    env["state"].write_text(json.dumps(records))
    os.utime(env["state"], (time.time() + 5, time.time() + 5))   # force a reload
    status, out = _call(env["port"], "GET", "/v1/models")
    assert {m["id"] for m in out["data"]} == {"gemma-4-26b-q4", "quyet.decide"}
    # A local model whose server refuses: 503, not 502 (since 1.2).
    status, out = _call(env["port"], "POST", "/v1/chat/completions", {"model": "gemma"})
    assert status == 503 and out["error"]["code"] == "model_unavailable"
    assert out["error"]["type"] == "server_error"


def test_one_model_needs_no_name():
    m = Instance("only", "pc", 0, True, {"api": "openai", "status": "ready"})
    assert pick([m], None, OPENAI) == [m]
    with pytest.raises(Unroutable) as e:
        pick([m], None, SYSTEMONE)
    assert e.value.status == 503


def test_model_read_whatever_the_declared_type(env):
    # `curl -d` declares form-urlencoded; the model must still be honoured.
    status, out = _call(env["port"], "POST", "/v1/chat/completions", {"model": "llama-70b"},
                        headers={"Content-Type": "application/x-www-form-urlencoded"})
    assert status == 404


# ------------------------------------------------- vision on vllm/sglang --

def _ckpt(tmp_path, name, vision):
    d = tmp_path / name
    d.mkdir()
    arch = "Qwen3_5ForConditionalGeneration" if vision else "Qwen3ForCausalLM"
    cfg = {"architectures": [arch], **({"vision_config": {}} if vision else {})}
    (d / "config.json").write_text(json.dumps(cfg))
    return str(d)


@pytest.mark.parametrize("cmd, limit", [
    ((), None),
    (("--limit-mm-per-prompt", '{"image": 0}'), 0),
    (('--limit-mm-per-prompt={"image":2,"video":0}',), 2),
    (("--limit-mm-per-prompt.image", "0"), 0),
    (("--limit-mm-per-prompt.image=1",), 1),
    (("--limit-mm-data-per-request", '{"image": 0}'), 0),
    (("--limit-mm-per-prompt", "not json"), None),
])
def test_image_limit(cmd, limit):
    assert gw._image_limit(("vllm", "serve", "m", *cmd)) == limit


@pytest.mark.parametrize("backend, extra, sees", [
    ("vllm", (), True),
    ("sglang", (), True),
    ("vllm", ("--limit-mm-per-prompt", '{"image":1}'), True),
    ("vllm", ("--language-model-only",), False),
    ("sglang", ("--language-only",), False),
    ("vllm", ("--limit-mm-per-prompt", '{"image": 0}'), False),
])
def test_vision_on_hf_engines(tmp_path, backend, extra, sees):
    vlm = _ckpt(tmp_path, f"vlm-{backend}-{len(extra)}-{sees}", vision=True)
    m = gw.Model("m", os.getpid(), 1, backend, 0, vlm, (backend, "serve", vlm, *extra))
    assert gw._hf_vision(m) is sees
    text = _ckpt(tmp_path, f"text-{backend}-{len(extra)}-{sees}", vision=False)
    assert gw._hf_vision(gw.Model("t", os.getpid(), 1, backend, 0, text, (backend,))) is False


def test_images_route_to_a_vllm_vision_model(tmp_path):
    up = _serve(_Upstream, "vlm")
    state = tmp_path / "running.json"
    vlm = _ckpt(tmp_path, "qwen3.5-4b", vision=True)
    rec = _record("t-vlm", up.server_address[1], backend="vllm",
                  cmd=("vllm", "serve", vlm, "--max-model-len", "4096",
                       "--limit-mm-per-prompt", '{"image":1}'))
    state.write_text(json.dumps([{**rec, "model_path": vlm}]))
    httpd = gw.make_server(0, "127.0.0.1", gw.Gateway(gw.LocalModels(state)))
    threading.Thread(target=httpd.serve_forever, args=(0.05,), daemon=True).start()
    try:
        port = httpd.server_address[1]
        image = {"model": "t-vlm", "messages": [{"role": "user", "content": [
            {"type": "text", "text": "what is this?"},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}}]}]}
        status, out = _call(port, "POST", "/v1/chat/completions", image)
        assert status == 200 and out["who"] == "vlm", out
        models = {m["id"]: m for m in _call(port, "GET", "/v1/models")[1]["data"]}
        assert models["t-vlm"]["llmctl"]["vision"] is True
    finally:
        for s in (up, httpd):
            s.shutdown()


def test_listen_backlog_takes_a_burst(tmp_path):
    httpd = gw.make_server(0, "127.0.0.1", gw.Gateway(gw.LocalModels(tmp_path / "running.json")))
    try:
        assert httpd.request_queue_size >= 128
    finally:
        httpd.server_close()


# ---------------------------------------------------- a busy model is not down --

def _probes(monkeypatch, *results):
    """urlopen in the gateway answers with these in turn: a status or an exception."""
    it = iter(results)

    class Reply:
        def __init__(self, status):
            self.status = status

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    def urlopen(url, timeout=None):
        got = next(it)
        if isinstance(got, Exception):
            raise got
        return Reply(got)

    monkeypatch.setattr(gw.urllib.request, "urlopen", urlopen)


def _local(tmp_path, pid=7):
    return gw.LocalModels(tmp_path / "running.json"), gw.Model("flux", pid, 8000, "omni", 0.0,
                                                               "/m", ())


def _clock(monkeypatch):
    now = [1000.0]
    monkeypatch.setattr(gw.time, "monotonic", lambda: now[0])
    return now


REFUSED = gw.urllib.error.URLError(ConnectionRefusedError(111, "refused"))


def test_a_model_too_busy_to_answer_stays_ready(tmp_path, monkeypatch):
    local, m = _local(tmp_path)
    _probes(monkeypatch, 200, TimeoutError(), TimeoutError(), 200)
    assert [local.status(m, fresh=True) for _ in range(4)] == ["ready"] * 4


def test_a_connect_timeout_counts_as_busy_too(tmp_path, monkeypatch):
    local, m = _local(tmp_path)
    _probes(monkeypatch, 200, gw.urllib.error.URLError(TimeoutError()))
    assert [local.status(m, fresh=True) for _ in range(2)] == ["ready", "ready"]


def test_silence_too_long_is_down_and_an_answer_starts_over(tmp_path, monkeypatch):
    local, m = _local(tmp_path)
    now = _clock(monkeypatch)
    _probes(monkeypatch, 200, TimeoutError(), TimeoutError(), TimeoutError(), 200, TimeoutError())
    assert local.status(m, fresh=True) == "ready"
    assert local.status(m, fresh=True) == "ready"
    now[0] += gw.sup_mod.BUSY_GRACE - 1
    assert local.status(m, fresh=True) == "ready"
    now[0] += 2
    assert local.status(m, fresh=True) == "down"
    assert local.status(m, fresh=True) == "ready"
    now[0] += gw.sup_mod.BUSY_GRACE + 1
    assert local.status(m, fresh=True) == "ready"       # silent for no time yet


def test_the_silence_is_counted_from_the_first_miss_not_the_last_answer(tmp_path, monkeypatch):
    """An idle gateway may not have probed for hours."""
    local, m = _local(tmp_path)
    now = _clock(monkeypatch)
    _probes(monkeypatch, 200, TimeoutError())
    assert local.status(m, fresh=True) == "ready"
    now[0] += 3600
    assert local.status(m, fresh=True) == "ready"


def test_a_refusal_after_answering_is_down_at_once(tmp_path, monkeypatch):
    local, m = _local(tmp_path)
    _probes(monkeypatch, 200, REFUSED)
    assert [local.status(m, fresh=True) for _ in range(2)] == ["ready", "down"]


def test_refused_before_it_ever_answered_is_loading(tmp_path, monkeypatch):
    """vLLM and Omni don't listen until they've loaded."""
    local, m = _local(tmp_path)
    _probes(monkeypatch, REFUSED, REFUSED, 200)
    assert [local.status(m, fresh=True) for _ in range(3)] == ["loading", "loading", "ready"]


def test_silent_before_it_ever_answered_is_loading(tmp_path, monkeypatch):
    local, m = _local(tmp_path)
    _probes(monkeypatch, TimeoutError())
    assert local.status(m, fresh=True) == "loading"


def test_another_process_on_the_port_has_not_answered(tmp_path, monkeypatch):
    """Also what a restart looks like: a new pid, loading again."""
    local, m = _local(tmp_path)
    _, other = _local(tmp_path, pid=8)
    _probes(monkeypatch, 200, TimeoutError(), REFUSED)
    assert local.status(m, fresh=True) == "ready"
    assert local.status(other, fresh=True) == "loading"
    assert local.status(other, fresh=True) == "loading"


def test_loading_is_not_touched(tmp_path, monkeypatch):
    local, m = _local(tmp_path)
    _probes(monkeypatch, 200,
            gw.urllib.error.HTTPError("u", 503, "loading", None, None))
    assert [local.status(m, fresh=True) for _ in range(2)] == ["ready", "loading"]


def test_a_real_socket_that_never_answers_reads_as_a_timeout(tmp_path, monkeypatch):
    local = gw.LocalModels(tmp_path / "running.json")
    with socket.socket() as listening:
        listening.bind(("127.0.0.1", 0))
        listening.listen()
        port = listening.getsockname()[1]
        monkeypatch.setattr(gw.config_mod, "local_url", lambda p, path="": f"http://127.0.0.1:{p}{path}")
        real = gw.urllib.request.urlopen
        monkeypatch.setattr(gw.urllib.request, "urlopen",
                            lambda url, timeout=None: real(url, timeout=0.05))
        m = gw.Model("flux", 7, port, "omni", 0.0, "/m", ())
        local._answering[port] = (7, None)
        assert local.status(m, fresh=True) == "ready"
    assert local.status(m, fresh=True) == "down"        # closed: refused


def test_real_sockets_read_loading_ready_then_down(tmp_path, monkeypatch):
    class Health(BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(200)
            self.end_headers()

        def log_message(self, *args):
            pass

    with socket.socket() as free:
        free.bind(("127.0.0.1", 0))
        port = free.getsockname()[1]
    monkeypatch.setattr(gw.config_mod, "local_url", lambda p, path="": f"http://127.0.0.1:{p}{path}")
    local = gw.LocalModels(tmp_path / "running.json")
    m = gw.Model("flux", 7, port, "omni", 0.0, "/m", ())
    assert local.status(m, fresh=True) == "loading"     # nothing listening yet
    server = ThreadingHTTPServer(("127.0.0.1", port), Health)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    assert local.status(m, fresh=True) == "ready"
    server.shutdown()
    server.server_close()
    assert local.status(m, fresh=True) == "down"


class _Silent:
    """A model that takes a request and says nothing, or one reply and then
    nothing; ``hung_up`` is set when the gateway closes its side."""

    def __init__(self, first: bytes = b"", then: bytes | None = None, after: float = 0.0):
        self.hung_up = threading.Event()
        self.asked = threading.Event()
        self._first, self._then, self._after = first, then, after
        self._listen = socket.create_server(("127.0.0.1", 0))
        self.port = self._listen.getsockname()[1]
        threading.Thread(target=self._run, daemon=True).start()

    def _run(self):
        while True:
            conn, _ = self._listen.accept()
            threading.Thread(target=self._one, args=(conn,), daemon=True).start()

    def _one(self, conn):
        data = b""
        while b"\r\n\r\n" not in data:
            chunk = conn.recv(65536)
            if not chunk:
                conn.close()
                return
            data += chunk
        head, _, rest = data.partition(b"\r\n\r\n")
        if head.startswith(b"GET "):        # the gateway's health probe
            conn.sendall(b"HTTP/1.1 200 OK\r\nContent-Length: 0\r\nConnection: close\r\n\r\n")
            conn.close()
            return
        length = int(re.search(rb"(?i)content-length: (\d+)", head).group(1))
        while len(rest) < length:
            rest += conn.recv(65536)
        self.asked.set()
        if self._first:
            conn.sendall(self._first)
        if self._then is not None:
            time.sleep(self._after)
            conn.sendall(self._then)
            conn.close()
            return
        conn.settimeout(10)
        try:
            while conn.recv(65536):
                pass
        except OSError:
            pass
        self.hung_up.set()
        conn.close()


SSE_HEAD = (b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\n"
            b"Transfer-Encoding: chunked\r\n\r\n")
PIECE = b"d\r\ndata: piece\n\n\r\n"
END = b"0\r\n\r\n"
ASK = {"model": "silent", "messages": [], "stream": True}


@pytest.fixture
def silent_on(tmp_path, monkeypatch):
    """A gateway in front of a _Silent model, and the relay log lines."""
    lines = []
    real = gw.logs.emit
    monkeypatch.setattr(gw.logs, "emit",
                        lambda msg, **kw: (lines.append(msg), real(msg, **kw))[1])
    servers = []

    def start(model):
        state = tmp_path / "running.json"
        state.write_text(json.dumps([_record("silent", model.port)]))
        httpd = gw.make_server(0, "127.0.0.1", gw.Gateway(gw.LocalModels(state)))
        servers.append(httpd)
        threading.Thread(target=httpd.serve_forever, args=(0.05,), daemon=True).start()
        return httpd.server_address[1], lines

    yield start
    for s in servers:
        s.shutdown()


def _ask(port, extra=b""):
    body = json.dumps(ASK).encode()
    client = socket.create_connection(("127.0.0.1", port), timeout=5)
    client.sendall(b"POST /v1/chat/completions HTTP/1.1\r\nHost: x\r\nContent-Type: application/json\r\n"
                   b"Content-Length: %d\r\n\r\n%s%s" % (len(body), body, extra))
    return client


def _until_gone(client):
    data = b""
    while chunk := client.recv(65536):
        data += chunk
        if data.endswith(END):
            break
    return data


def test_a_client_leaving_while_queued_hangs_up_on_the_model(silent_on):
    model = _Silent()
    port, lines = silent_on(model)
    client = _ask(port)
    assert model.asked.wait(3)
    client.close()
    assert model.hung_up.wait(3)
    deadline = time.time() + 3
    while not any("while waiting for the model" in x for x in lines) and time.time() < deadline:
        time.sleep(0.02)
    assert any("client left" in x and "while waiting for the model" in x for x in lines)


def test_a_client_leaving_a_silent_reply_hangs_up_on_the_model(silent_on):
    model = _Silent(SSE_HEAD + PIECE)
    port, lines = silent_on(model)
    client = _ask(port)
    data = b""
    while b"data: piece" not in data:
        data += client.recv(65536)
    client.close()
    assert model.hung_up.wait(3)
    deadline = time.time() + 3
    while not any("client left" in x for x in lines) and time.time() < deadline:
        time.sleep(0.02)
    assert any("client left" in x for x in lines)
    assert not any("upstream broke" in x for x in lines)


def test_a_client_still_there_gets_the_reply_after_a_long_silence(silent_on):
    wait = 3 * gw.WATCH_EVERY_MS / 1000
    port, lines = silent_on(_Silent(SSE_HEAD + PIECE, PIECE + END, after=wait))
    client = _ask(port)
    data = _until_gone(client)
    client.close()
    assert data.count(b"data: piece") == 2
    assert not any("client left" in x for x in lines)


def test_bytes_from_the_client_are_not_a_hang_up(silent_on):
    wait = 3 * gw.WATCH_EVERY_MS / 1000
    port, lines = silent_on(_Silent(SSE_HEAD + PIECE, PIECE + END, after=wait))
    client = _ask(port, extra=b"GET /health HTTP/1.1\r\nHost: x\r\n\r\n")
    data = _until_gone(client)
    assert data.count(b"data: piece") == 2
    assert not any("client left" in x for x in lines)
    client.settimeout(3)
    assert b"200" in client.recv(65536)     # the next request wasn't eaten by the watcher
    client.close()


def test_a_finished_reply_is_not_called_a_cancel(silent_on):
    port, lines = silent_on(_Silent(SSE_HEAD + PIECE + END, b""))
    client = _ask(port)
    assert _until_gone(client).endswith(END)
    client.close()
    time.sleep(2 * gw.WATCH_EVERY_MS / 1000)
    assert not any("client left" in x for x in lines)

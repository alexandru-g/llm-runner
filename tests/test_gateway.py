"""The all-models gateway, against fake upstream servers on ephemeral ports."""
from __future__ import annotations

import http.client
import json
import os
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from llm_runner import gateway as gw
from llm_runner.cluster import OPENAI, SYSTEMONE, Instance, Unroutable, pick


class _Upstream(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

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


def test_keep_alive_serves_several_requests(env):
    conn = http.client.HTTPConnection("127.0.0.1", env["port"], timeout=5)
    for name in ("glm", "gemma", "glm"):
        conn.request("POST", "/v1/completions", body=json.dumps({"model": name}),
                     headers={"Content-Type": "application/json"})
        assert json.loads(conn.getresponse().read())["who"] == name
    conn.close()


def test_dead_and_unreachable_models(env):
    records = env["records"]
    records[1]["pid"] = 2 ** 22 + 12345                 # glm's process is gone
    records[0]["port"] = 1                              # gemma's port refuses
    env["state"].write_text(json.dumps(records))
    os.utime(env["state"], (time.time() + 5, time.time() + 5))   # force a reload
    status, out = _call(env["port"], "GET", "/v1/models")
    assert {m["id"] for m in out["data"]} == {"gemma-4-26b-q4", "quyet.decide"}
    status, out = _call(env["port"], "POST", "/v1/chat/completions", {"model": "gemma"})
    assert status == 502 and out["error"]["code"] == "upstream_unavailable"


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

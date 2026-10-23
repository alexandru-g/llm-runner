"""Request ids, JSON request logs, trace propagation and OTLP export."""
from __future__ import annotations

import http.client
import json
import threading
import time
from http.server import BaseHTTPRequestHandler
from pathlib import Path

import pytest

from llm_runner import backends
from llm_runner import gateway as gw
from llm_runner import logs
from llm_runner import tracing

from test_gateway import _record, _serve

SECRET = "the secret prompt text"


class _Echo(BaseHTTPRequestHandler):
    """Records the headers it got; replies with usage."""
    protocol_version = "HTTP/1.1"
    seen: list[dict] = []

    def log_message(self, *a):
        pass

    def do_GET(self):
        self._send(200, b"{}")

    def do_POST(self):
        self.rfile.read(int(self.headers["Content-Length"]))
        _Echo.seen.append(dict(self.headers))
        self._send(200, json.dumps({"usage": {"prompt_tokens": 3, "completion_tokens": 2},
                                    "choices": [{"message": {"content": "secret answer"}}]
                                    }).encode())

    def _send(self, status, body):
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("X-Request-Id", "upstream-own-id")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


class _Collector(BaseHTTPRequestHandler):
    got: list[dict] = []

    def log_message(self, *a):
        pass

    def do_POST(self):
        _Collector.got.append(json.loads(self.rfile.read(int(self.headers["Content-Length"]))))
        self.send_response(200)
        self.send_header("Content-Length", "0")
        self.end_headers()


@pytest.fixture
def env(tmp_path):
    _Echo.seen.clear()
    up = _serve(_Echo)
    state = tmp_path / "running.json"
    state.write_text(json.dumps([_record("qwen", up.server_address[1], backend="vllm")]))
    g = gw.Gateway(gw.LocalModels(state))
    httpd = gw.make_server(0, "127.0.0.1", g)
    threading.Thread(target=httpd.serve_forever, args=(0.05,), daemon=True).start()
    yield {"port": httpd.server_address[1], "gw": g}
    logs.configure("text", "")
    up.shutdown()
    httpd.shutdown()


def _call(port, body, headers=None):
    c = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
    c.request("POST", "/v1/chat/completions", body=json.dumps(body),
              headers={"Content-Type": "application/json", **(headers or {})})
    r = c.getresponse()
    r.read()
    c.close()
    return r


def test_request_ids_are_kept_or_made_and_forwarded(env):
    r = _call(env["port"], {"model": "qwen"}, {"X-Request-Id": "client-42"})
    assert r.getheader("X-Request-Id") == "client-42"          # not the upstream's own
    assert _Echo.seen[-1]["X-Request-Id"] == "client-42"
    r = _call(env["port"], {"model": "qwen"}, {"X-Request-Id": "bad id with spaces"})
    rid = r.getheader("X-Request-Id")
    assert len(rid) == 32 and _Echo.seen[-1]["X-Request-Id"] == rid
    r = _call(env["port"], {"model": "nope"})
    assert r.status == 404 and len(r.getheader("X-Request-Id")) == 32


def test_traceparent_continues_the_clients_trace(env):
    tp = "00-0af7651916cd43dd8448eb211c80319c-b7ad6b7169203331-01"
    _call(env["port"], {"model": "qwen"}, {"traceparent": tp, "tracestate": "k=v"})
    sent = _Echo.seen[-1]
    trace, parent = sent["traceparent"].split("-")[1:3]
    assert trace == "0af7651916cd43dd8448eb211c80319c" and parent != "b7ad6b7169203331"
    assert sent["tracestate"] == "k=v"
    _call(env["port"], {"model": "qwen"})
    assert tracing._TRACEPARENT.fullmatch(_Echo.seen[-1]["traceparent"])


def test_json_request_lines_have_ids_and_no_content(env, capsys):
    logs.configure("json", env["gw"].node)
    _call(env["port"], {"model": "qwen", "messages": [{"role": "user", "content": SECRET}]},
          {"X-Request-Id": "rid-1", "traceparent":
           "00-0af7651916cd43dd8448eb211c80319c-b7ad6b7169203331-01"})
    lines = []
    for _ in range(40):                 # the handler thread may still be writing
        lines += [json.loads(x) for x in capsys.readouterr().out.splitlines() if x.startswith("{")]
        if any(x.get("event") == "request" for x in lines):
            break
        time.sleep(0.05)
    [req] = [x for x in lines if x.get("event") == "request"]
    assert req["request_id"] == "rid-1" and req["trace_id"] == "0af7651916cd43dd8448eb211c80319c"
    assert req["model"] == "qwen" and req["status"] == 200 and req["path"] == "/v1/chat/completions"
    assert req["tokens_in"] == 3 and req["tokens_out"] == 2 and req["duration_ms"] >= 0
    assert {"ts", "level", "node", "msg"} <= set(req)
    assert SECRET not in json.dumps(lines) and "secret answer" not in json.dumps(lines)


def test_spans_go_to_the_collector(env):
    _Collector.got.clear()
    col = _serve(_Collector)
    try:
        env["gw"].exporter = tracing.Exporter(f"http://127.0.0.1:{col.server_address[1]}",
                                              {"service.name": "llmctl-gateway"}, every=0.05)
        tp = "00-0af7651916cd43dd8448eb211c80319c-b7ad6b7169203331-01"
        _call(env["port"], {"model": "qwen"}, {"traceparent": tp})
        _call(env["port"], {"model": "qwen"},
              {"traceparent": "00-1af7651916cd43dd8448eb211c80319c-b7ad6b7169203331-00"})
        time.sleep(0.3)
        env["gw"].exporter.flush()
        spans = [s for b in _Collector.got for rs in b["resourceSpans"]
                 for ss in rs["scopeSpans"] for s in ss["spans"]]
        assert {s["traceId"] for s in spans} == {"0af7651916cd43dd8448eb211c80319c"}  # unsampled out
        server = next(s for s in spans if s["kind"] == tracing.SERVER)
        client = next(s for s in spans if s["kind"] == tracing.CLIENT)
        assert server["parentSpanId"] == "b7ad6b7169203331"
        assert client["parentSpanId"] == server["spanId"]
        assert _Echo.seen[-2]["traceparent"].split("-")[2] == client["spanId"]
        attrs = {a["key"]: a["value"] for a in server["attributes"]}
        assert attrs["llmctl.model"] == {"stringValue": "qwen"}
        assert attrs["http.response.status_code"] == {"intValue": "200"}
        assert attrs["gen_ai.usage.input_tokens"] == {"intValue": "3"}
    finally:
        env["gw"].exporter.close()
        env["gw"].exporter = None
        col.shutdown()


def test_a_dead_collector_drops_spans_without_blocking(env):
    env["gw"].exporter = tracing.Exporter("http://127.0.0.1:1", {}, every=0.05, timeout=0.2)
    try:
        t0 = time.monotonic()
        assert _call(env["port"], {"model": "qwen"}).status == 200
        assert time.monotonic() - t0 < 1
        time.sleep(0.5)
        assert env["gw"].exporter.dropped >= 2 and env["gw"].exporter.sent == 0
    finally:
        env["gw"].exporter.close()
        env["gw"].exporter = None


def test_engines_are_never_told_to_log_prompts(tmp_path):
    from llm_runner import cli
    noisy = {"--enable-log-requests", "--log-requests", "--verbose", "-v", "--log-verbose"}
    model = tmp_path / "m"
    model.mkdir()
    cmds = [backends.build_vllm_cmd("vllm", model, name="m", port=1, ctx_size=4096),
            backends.build_sglang_cmd("sglang", model, name="m", port=1, ctx_size=4096),
            cli._build_llama_cmd(Path("/m/x.gguf"), None, 1, 4096, True)]
    for cmd in cmds:
        assert not noisy & set(cmd), cmd

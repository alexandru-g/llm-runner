"""Gateway /metrics, re-exported engine metrics, HTTP SD and the scrape token."""
from __future__ import annotations

import http.client
import json
import threading
from http.server import BaseHTTPRequestHandler

import pytest

from llm_runner import auth
from llm_runner import gateway as gw
from llm_runner import metrics as m

from test_gateway import _record, _serve

ENGINE_METRICS = """\
# HELP vllm:num_requests_waiting Requests waiting.
# TYPE vllm:num_requests_waiting gauge
vllm:num_requests_waiting{engine="0",model_name="qwen"} 2.0
# TYPE vllm:prompt_tokens_total counter
vllm:prompt_tokens_total{model_name="qwen"} 1234.0
"""


class _Engine(BaseHTTPRequestHandler):
    """An engine with /metrics, JSON replies with usage, and SSE with usage last."""
    protocol_version = "HTTP/1.1"

    def log_message(self, *a):
        pass

    def _send(self, status, ctype, body: bytes):
        self.send_response(status)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/health":
            return self._send(200, "application/json", b"{}")
        if self.path == "/metrics":
            return self._send(200, "text/plain", ENGINE_METRICS.encode())
        self._send(404, "application/json", b"{}")

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        usage = {"prompt_tokens": 11, "completion_tokens": 7, "total_tokens": 18}
        if body.get("model") == "bad":
            return self._send(400, "application/json",
                              b'{"error": {"code": "context_length_exceeded"}}')
        if body.get("stream"):
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Connection", "close")
            self.end_headers()
            self.wfile.write(b'data: {"choices": [{"delta": {"content": "hi"}}]}\n\n')
            self.wfile.write(b'data: {"choices": [], "usage": %s}\n\n' % json.dumps(usage).encode())
            self.wfile.write(b"data: [DONE]\n\n")
            self.close_connection = True
            return
        self._send(200, "application/json", json.dumps({"usage": usage}).encode())


@pytest.fixture
def env(tmp_path):
    eng = _serve(_Engine)
    port = eng.server_address[1]
    state = tmp_path / "running.json"
    state.write_text(json.dumps([_record("qwen", port, backend="vllm"),
                                 _record("bad", port, backend="vllm"),
                                 _record("quyet.decide", port, backend="decide")]))
    gateway = gw.Gateway(gw.LocalModels(state))
    httpd = gw.make_server(0, "127.0.0.1", gateway)
    threading.Thread(target=httpd.serve_forever, args=(0.05,), daemon=True).start()
    yield {"port": httpd.server_address[1], "gw": gateway}
    eng.shutdown()
    httpd.shutdown()


def _get(port, path, headers=None):
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
    conn.request("GET", path, headers=headers or {})
    r = conn.getresponse()
    out = r.status, r.getheader("Content-Type"), r.read().decode()
    conn.close()
    return out


def _post(port, body):
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
    conn.request("POST", "/v1/chat/completions", body=json.dumps(body),
                 headers={"Content-Type": "application/json"})
    r = conn.getresponse()
    r.read()
    conn.close()
    return r.status


def _value(text: str, sample: str) -> float | None:
    for line in text.splitlines():
        if line.startswith(sample + " "):
            return float(line.rsplit(" ", 1)[1])
    return None


# ------------------------------------------------------------------- unit --

def test_exposition_format():
    r = m.Registry()
    c = r.counter("x_total", "Calls.", ("model",))
    h = r.histogram("x_seconds", "Time.", ("model",), buckets=(0.1, 1))
    c.inc(model='a"b')
    h.observe(0.5, model="a")
    h.observe(5, model="a")
    text = r.render()
    assert '# TYPE x_total counter\nx_total{model="a\\"b"} 1' in text
    assert 'x_seconds_bucket{model="a",le="0.1"} 0' in text
    assert 'x_seconds_bucket{model="a",le="1"} 1' in text
    assert 'x_seconds_bucket{model="a",le="+Inf"} 2' in text
    assert 'x_seconds_sum{model="a"} 5.5' in text and 'x_seconds_count{model="a"} 2' in text


def test_a_failing_collector_does_not_break_the_scrape():
    r = m.Registry()
    r.gauge("bad", "x", collect=lambda: 1 / 0)
    r.counter("ok_total", "y").inc()
    text = r.render()
    assert "# bad: collect failed: ZeroDivisionError" in text and "ok_total 1" in text


def test_added_labels_keep_the_engines_own():
    out = m.add_labels(ENGINE_METRICS + "plain 3\n", {"model": "q", "model_name": "x"})
    assert 'vllm:num_requests_waiting{engine="0",model_name="qwen",model="q"} 2.0' in out
    assert 'plain{model="q",model_name="x"} 3' in out
    assert "# TYPE vllm:prompt_tokens_total counter" in out


# ---------------------------------------------------------------- gateway --

def test_calls_are_counted_with_latency_ttft_and_tokens(env):
    port = env["port"]
    assert _post(port, {"model": "qwen", "messages": []}) == 200
    assert _post(port, {"model": "qwen", "stream": True}) == 200
    assert _post(port, {"model": "bad"}) == 400
    assert _post(port, {"model": "nope"}) == 404
    status, ctype, text = _get(port, "/metrics")
    assert status == 200 and ctype.startswith("text/plain; version=0.0.4")
    node = env["gw"].node
    q = f'model="qwen",node="{node}",backend="vllm"'
    assert _value(text, f'llmctl_requests_total{{{q},status="200"}}') == 2
    assert _value(text, f'llmctl_request_duration_seconds_count{{{q}}}') == 2
    assert _value(text, f'llmctl_time_to_first_token_seconds_count{{{q}}}') == 1
    assert _value(text, f'llmctl_tokens_total{{{q},direction="in"}}') == 22
    assert _value(text, f'llmctl_tokens_total{{{q},direction="out"}}') == 14
    b = f'model="bad",node="{node}",backend="vllm"'
    assert _value(text, f'llmctl_request_errors_total{{{b},code="context_length_exceeded"}}') == 1
    assert _value(text, 'llmctl_request_errors_total{model="",node="",backend="",'
                        'code="model_not_found"}') == 1
    assert _value(text, f'llmctl_requests_queued{{{q}}}') == 2
    assert _value(text, f'llmctl_requests_in_flight{{{q}}}') == 0
    assert _value(text, f'llmctl_instance_state{{{q},state="ready"}}') == 1
    assert _value(text, f'llmctl_node_up{{node="{node}"}}') == 1
    assert "llmctl_build_info{" in text
    # Scrapes and health checks are not calls.
    _get(port, "/healthz")
    assert _value(_get(port, "/metrics")[2], f'llmctl_requests_total{{{q},status="200"}}') == 2


def test_sd_lists_engines_and_backends_re_export_them(env):
    port = env["port"]
    status, ctype, text = _get(port, "/metrics/sd", {"Host": "ws:9000"})
    assert status == 200 and ctype == "application/json"
    sd = json.loads(text)
    assert [t["labels"]["model"] for t in sd] == ["qwen", "bad"]        # no decide
    assert sd[0]["targets"] == ["ws:9000"]
    assert sd[0]["labels"]["__metrics_path__"] == "/metrics/backends/qwen"
    status, _, text = _get(port, "/metrics/backends/qwen")
    node = env["gw"].node
    assert status == 200
    assert (f'vllm:num_requests_waiting{{engine="0",model_name="qwen",model="qwen",'
            f'node="{node}",backend="vllm"}} 2.0') in text
    assert _get(port, "/metrics/backends/quyet.decide")[0] == 404
    assert _get(port, "/metrics/backends/nope")[0] == 404


class _Outside:
    """An access policy for which every caller is outside the local networks."""

    def __init__(self, metrics_token=None):
        self.mode = auth.ON
        self._p = auth.Policy(mode=auth.ON, keys={"app": auth.hash_key("app-key")},
                              metrics_token=metrics_token and auth.hash_key(metrics_token))
        self.metrics_token = self._p.metrics_token

    def needs_key(self, peer, headers):
        return True

    def is_metrics_token(self, headers):
        return self._p.is_metrics_token(headers)

    def key_name(self, headers):
        return self._p.key_name(headers)

    def via_proxy(self, *a):
        return False


def test_scrapes_take_the_metrics_token_and_it_runs_nothing(env):
    port, gateway = env["port"], env["gw"]
    gateway.access = lambda: _Outside("scrape-tok")
    bearer = lambda k: {"Authorization": f"Bearer {k}"}    # noqa: E731
    assert _get(port, "/metrics")[0] == 401
    assert _get(port, "/metrics", bearer("scrape-tok"))[0] == 200
    assert _get(port, "/metrics/sd", bearer("scrape-tok"))[0] == 200
    assert _get(port, "/metrics", bearer("app-key"))[0] == 401   # the token replaces keys here
    assert _get(port, "/v1/models", bearer("scrape-tok"))[0] == 401
    assert _get(port, "/v1/models", bearer("app-key"))[0] == 200


def test_without_a_metrics_token_scrapes_need_an_api_key(env):
    port, gateway = env["port"], env["gw"]
    gateway.access = lambda: _Outside()
    assert _get(port, "/metrics")[0] == 401
    assert _get(port, "/metrics", {"Authorization": "Bearer app-key"})[0] == 200


def test_llama_server_gets_metrics():
    from pathlib import Path
    from llm_runner import cli
    for emb in (False, True):
        cmd = cli._build_llama_cmd(Path("/m/x.gguf"), None, 1, 4096, True, embeddings=emb)
        assert "--metrics" in cmd

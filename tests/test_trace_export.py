"""Span export for Phoenix: protobuf, a project per API key, llama-server
timings, and never any content (prompts, replies, tools, decision state)."""
from __future__ import annotations

import http.client
import json
import struct
import threading
import time
from http.server import BaseHTTPRequestHandler

import pytest
from click.testing import CliRunner

from llm_runner import auth
from llm_runner import cli
from llm_runner import config as config_mod
from llm_runner import gateway as gw
from llm_runner import tracing

from test_gateway import _record, _serve

SECRET = "zebra-secret-7731"          # in every request, and echoed in every reply
TP = "00-0af7651916cd43dd8448eb211c80319c-b7ad6b7169203331-01"
TRACE = "0af7651916cd43dd8448eb211c80319c"
TIMINGS = {"prompt_n": 120, "prompt_ms": 35.25, "cache_n": 64, "predicted_n": 9,
           "predicted_ms": 80.5, "predicted_per_second": 111.8}


class _Model(BaseHTTPRequestHandler):
    """Echoes the whole request back (the worst case for a leak), with
    llama-server's usage and timings; label "down" answers 503 to POSTs."""
    protocol_version = "HTTP/1.1"

    def log_message(self, *a):
        pass

    def do_GET(self):
        self._send(200, b"{}")

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        if self.server.label == "down":
            return self._send(503, b'{"error": {"message": "busy", "code": "unavailable"}}')
        echo = {"echo": body, "usage": {"prompt_tokens": 120, "completion_tokens": 9},
                "timings": TIMINGS, "choices": [{"message": {"content": f"re: {SECRET}"}}]}
        if body.get("stream"):
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Connection", "close")
            self.end_headers()
            self.wfile.write(f'data: {{"choices": [{{"delta": {{"content": "{SECRET}"}}}}]}}\n\n'
                             .encode())
            self.wfile.write(f"data: {json.dumps(echo)}\n\ndata: [DONE]\n\n".encode())
            self.close_connection = True
            return
        self._send(200, json.dumps(echo).encode())

    def _send(self, status, body):
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


class _Collector(BaseHTTPRequestHandler):
    got: list[tuple[str, bytes, dict]] = []

    def log_message(self, *a):
        pass

    def do_POST(self):
        _Collector.got.append((self.headers["Content-Type"],
                               self.rfile.read(int(self.headers["Content-Length"])),
                               dict(self.headers)))
        self.send_response(200)
        self.send_header("Content-Length", "0")
        self.end_headers()


class _Keys:
    """An auth policy that lets everyone in and names two keys."""
    mode = auth.OFF
    names = {"Bearer k-pa": "pa", "Bearer k-bench": "bench"}

    def key_name(self, headers):
        return self.names.get(headers.get("Authorization") or "")


@pytest.fixture
def env(tmp_path):
    _Collector.got.clear()
    down, up = _serve(_Model, "down"), _serve(_Model, "up")
    col = _serve(_Collector)
    state = tmp_path / "running.json"
    recs = [{**_record("chat-a", down.server_address[1]), "group": "chat"},
            {**_record("chat-b", up.server_address[1]), "group": "chat"},
            _record("emb", up.server_address[1], cmd=("llama-server", "--embeddings")),
            _record("d.decide", up.server_address[1], backend="decide")]
    state.write_text(json.dumps(recs))
    g = gw.Gateway(gw.LocalModels(state), access=_Keys)
    httpd = gw.make_server(0, "127.0.0.1", g)
    threading.Thread(target=httpd.serve_forever, args=(0.05,), daemon=True).start()
    col_url = f"http://127.0.0.1:{col.server_address[1]}"
    yield {"port": httpd.server_address[1], "gw": g, "col": col_url}
    if g.exporter:
        g.exporter.close()
    for s in (down, up, col, httpd):
        s.shutdown()


def _post(port, path, body, headers=None):
    c = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
    c.request("POST", path, body=json.dumps(body),
              headers={"Content-Type": "application/json", "traceparent": TP, **(headers or {})})
    r = c.getresponse()
    data = r.read()
    c.close()
    return r.status, data


def _export(env, protocol):
    env["gw"].exporter = tracing.Exporter(env["col"], {"service.name": "llmctl-gateway"},
                                          protocol=protocol, every=0.05,
                                          headers={"Authorization": "Bearer phoenix"})


def _drain(env):
    time.sleep(0.3)
    env["gw"].exporter.flush()


# Everything a client might send that must not leave in a span.
CALLS = [
    ("/v1/chat/completions", {"model": "chat", "messages": [{"role": "user", "content": SECRET}],
                              "tools": [{"type": "function", "function": {
                                  "name": "f", "description": SECRET, "parameters": {}}}]}),
    ("/v1/chat/completions", {"model": "chat", "stream": True,
                              "messages": [{"role": "tool", "content": SECRET,
                                            "tool_call_id": SECRET}]}),
    ("/v1/embeddings", {"model": "emb", "input": [SECRET]}),
    ("/v1/systemone", {"model": "d.decide", "state": {"inbox": SECRET},
                       "questions": [{"id": "q1", "text": SECRET}]}),
]


@pytest.mark.parametrize("protocol", [tracing.JSON, tracing.PROTOBUF])
def test_no_content_ever_leaves_in_a_span(env, protocol):
    _export(env, protocol)
    for path, body in CALLS:
        status, reply = _post(env["port"], path, body, {"Authorization": "Bearer k-pa"})
        assert status == 200 and SECRET.encode() in reply, path      # the echo did come back
    _drain(env)
    assert _Collector.got
    for ctype, raw, headers in _Collector.got:
        assert SECRET.encode() not in raw
        assert ctype == ("application/x-protobuf" if protocol == tracing.PROTOBUF
                         else "application/json")
        assert headers["Authorization"] == "Bearer phoenix"


def _spans_json(got):
    return [s for _, raw, _ in got for rs in json.loads(raw)["resourceSpans"]
            for ss in rs["scopeSpans"] for s in ss["spans"]]


def test_every_call_and_attempt_nests_under_the_clients_span(env):
    _export(env, tracing.JSON)
    for path, body in CALLS:
        _post(env["port"], path, body)
    _drain(env)
    spans = _spans_json(_Collector.got)
    assert {s["traceId"] for s in spans} == {TRACE}
    servers = [s for s in spans if s["kind"] == tracing.SERVER]
    assert {s["name"] for s in servers} == {"POST /v1/chat/completions", "POST /v1/embeddings",
                                            "POST /v1/systemone"}
    assert all(s["parentSpanId"] == TP.split("-")[2] for s in servers)
    by_id = {s["spanId"]: s for s in servers}
    attempts = [s for s in spans if s["kind"] == tracing.CLIENT]
    assert all(a["parentSpanId"] in by_id for a in attempts)
    # Each chat call failed over: chat-a's 503 then chat-b, both under one server span.
    chats = [s for s in servers if s["name"] == "POST /v1/chat/completions"]
    for chat in chats:
        tried = [a["name"] for a in attempts if a["parentSpanId"] == chat["spanId"]]
        assert tried == ["upstream chat-a", "upstream chat-b"]
        attrs = {a["key"]: a["value"] for a in chat["attributes"]}
        assert attrs["llmctl.failovers"] == {"intValue": "1"}


def test_timings_from_llama_server(env):
    _export(env, tracing.JSON)
    for path, body in CALLS[:2]:                       # JSON reply, then a stream
        _post(env["port"], path, body)
    _drain(env)
    servers = [s for s in _spans_json(_Collector.got) if s["kind"] == tracing.SERVER]
    assert len(servers) == 2
    for s in servers:
        attrs = {a["key"]: a["value"] for a in s["attributes"]}
        assert attrs["llmctl.prefill_ms"] == {"doubleValue": round(35.25, 1)}
        assert attrs["llmctl.prefill_tokens"] == {"intValue": "120"}
        assert attrs["llmctl.cached_tokens"] == {"intValue": "64"}
        assert attrs["llmctl.decode_ms"] == {"doubleValue": 80.5}
        assert attrs["llmctl.decode_tokens_per_s"] == {"doubleValue": 111.8}
        assert attrs["gen_ai.usage.output_tokens"] == {"intValue": "9"}


def test_a_project_per_api_key(env):
    _export(env, tracing.JSON)
    env["gw"].trace_projects = {"pa": "personal-assistant", "*": "llmctl"}
    body = CALLS[2][1]
    _post(env["port"], "/v1/embeddings", body, {"Authorization": "Bearer k-pa"})
    _post(env["port"], "/v1/embeddings", body, {"Authorization": "Bearer k-bench"})
    _post(env["port"], "/v1/embeddings", body)
    _drain(env)
    projects: dict[str, list[str]] = {}
    for _, raw, _ in _Collector.got:
        for rs in json.loads(raw)["resourceSpans"]:
            res = {a["key"]: a["value"]["stringValue"] for a in rs["resource"]["attributes"]}
            for ss in rs["scopeSpans"]:
                for s in ss["spans"]:
                    key = next((a["value"]["stringValue"] for a in s["attributes"]
                                if a["key"] == "llmctl.key_name"), None)
                    projects.setdefault(res[tracing.PROJECT_ATTR], []).append(
                        f"{s['kind']}:{key}")
    assert projects == {"personal-assistant": ["3:None", "2:pa"],
                        "llmctl": ["3:None", "2:bench", "3:None", "2:None"]}


# --------------------------------------------------- protobuf by hand --

def _fields(buf: bytes) -> list[tuple[int, object]]:
    """A protobuf message as (field number, raw value) pairs."""
    out, i = [], 0
    while i < len(buf):
        key, i = _uvarint(buf, i)
        num, wire = key >> 3, key & 7
        if wire == 0:
            val, i = _uvarint(buf, i)
        elif wire == 1:
            val, i = buf[i:i + 8], i + 8
        elif wire == 2:
            n, i = _uvarint(buf, i)
            val, i = buf[i:i + n], i + n
        else:
            raise AssertionError(f"wire type {wire}")
        out.append((num, val))
    return out


def _uvarint(buf: bytes, i: int) -> tuple[int, int]:
    n = shift = 0
    while True:
        b = buf[i]
        i += 1
        n |= (b & 0x7F) << shift
        shift += 7
        if not b & 0x80:
            return n, i


def _one(fields, num):
    [v] = [v for n, v in fields if n == num]
    return v


def test_protobuf_matches_the_json_encoding():
    ctx = tracing.Context(TRACE, "b7ad6b7169203331", True, project="pa")
    s = ctx.span("POST /v1/x", tracing.SERVER, **{"a.int": -3, "a.float": 1.5, "a.bool": True,
                                                  "a.str": "x"})
    s.error = True
    s.end()
    other = tracing.Context(TRACE, None, True).span("GET /v1/models", tracing.SERVER)
    other.end()
    raw = tracing.to_otlp_proto([s, other], {"service.name": "gw"})
    rss = [v for n, v in _fields(raw) if n == 1]                  # resource_spans
    assert len(rss) == 2                                         # one per project
    rs = _fields(rss[0])
    res = [_fields(kv) for n, kv in _fields(_one(rs, 1)) if n == 1]
    assert {_one(kv, 1).decode(): _fields(_one(kv, 2)) for kv in res} == {
        "service.name": [(1, b"gw")], tracing.PROJECT_ATTR: [(1, b"pa")]}
    scope = _fields(_one(rs, 2))
    assert _fields(_one(scope, 1)) == [(1, tracing.SCOPE.encode())]
    sp = _fields(_one(scope, 2))
    assert _one(sp, 1).hex() == TRACE and _one(sp, 2).hex() == s.span_id
    assert _one(sp, 4).hex() == "b7ad6b7169203331" and _one(sp, 5) == b"POST /v1/x"
    assert _one(sp, 6) == tracing.SERVER
    assert struct.unpack("<Q", _one(sp, 7))[0] == s.start_ns
    assert struct.unpack("<Q", _one(sp, 8))[0] == s.end_ns
    attrs = {_one(_fields(kv), 1).decode(): _fields(_one(_fields(kv), 2))
             for n, kv in sp if n == 9}
    assert attrs["a.int"] == [(3, (1 << 64) - 3)]                # int64 two's complement
    assert struct.unpack("<d", attrs["a.float"][0][1])[0] == 1.5
    assert attrs["a.bool"] == [(2, 1)] and attrs["a.str"] == [(1, b"x")]
    assert _fields(_one(sp, 15)) == [(3, tracing.STATUS_ERROR)]
    other_rs = _fields(rss[1])
    assert [_one(_fields(kv), 1) for n, kv in _fields(_one(other_rs, 1))] == [b"service.name"]


# ------------------------------------------------------------ the CLI --

def _cli(*args):
    return CliRunner().invoke(cli.cli, ["config", *args])


def test_config_commands():  # conftest: own config
    assert _cli("gateway-otlp", "--protocol", "protobuf").exit_code != 0   # no endpoint yet
    r = _cli("gateway-otlp", "http://lab:6006", "--protocol", "protobuf",
             "--header", "Authorization=Bearer s3cr3t")
    assert r.exit_code == 0, r.output
    assert "http://lab:6006/v1/traces as protobuf" in r.output
    assert _cli("gateway-otlp-project", "pa", "personal-assistant").exit_code == 0
    assert _cli("gateway-otlp-project", "*", "llmctl").exit_code == 0
    shown = _cli("gateway-otlp").output
    assert "protobuf" in shown and "Authorization: (set)" in shown and "s3cr3t" not in shown
    assert "project personal-assistant: key pa" in shown
    c = config_mod.load()
    assert c["gateway_otlp_headers"] == {"Authorization": "Bearer s3cr3t"}
    assert c["gateway_otlp_projects"] == {"pa": "personal-assistant", "*": "llmctl"}
    assert _cli("gateway-otlp-project", "pa", "--remove").exit_code == 0
    assert _cli("gateway-otlp", "--clear").exit_code == 0
    c = config_mod.load()
    assert "gateway_otlp_endpoint" not in c and "gateway_otlp_headers" not in c
    assert _cli("gateway-otlp", "http://x", "--header", "bad").exit_code != 0


def test_headers_from_the_environment():
    assert gw.otlp_headers("Authorization=Bearer a=b, x-project-name = pa,junk") == {
        "Authorization": "Bearer a=b", "x-project-name": "pa"}


def test_calls_without_a_project_can_stay_out(env):
    _export(env, tracing.JSON)
    env["gw"].trace_projects = {"pa": "personal-assistant", "*": ""}
    body = CALLS[2][1]
    _post(env["port"], "/v1/embeddings", body, {"Authorization": "Bearer k-pa"})
    _post(env["port"], "/v1/embeddings", body, {"Authorization": "Bearer k-bench"})
    _post(env["port"], "/v1/embeddings", body)
    _drain(env)
    spans = _spans_json(_Collector.got)
    assert len(spans) == 2 and all(s["traceId"] == TRACE for s in spans)
    keys = [a["value"]["stringValue"] for s in spans for a in s["attributes"]
            if a["key"] == "llmctl.key_name"]
    assert keys == ["pa"]
    assert _cli("gateway-otlp-project", "*", "none").exit_code == 0
    assert config_mod.load()["gateway_otlp_projects"] == {"*": ""}
    assert "(not exported)" in _cli("gateway-otlp-project").output

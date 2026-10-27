"""Native mode: llama-server's own /v1/systemone behind the unified contract."""
import json
import struct
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from systemone_proxy import gguf, native
from systemone_proxy.backend import LlamaServer
from systemone_proxy.errors import ApiError
from systemone_proxy.spec import build_spec


def _gguf(path, kvs):
    """A header-only GGUF: kvs is [(key, type, value)]."""
    def s(x):
        b = x.encode()
        return struct.pack("<Q", len(b)) + b
    out = b"GGUF" + struct.pack("<IQQ", 3, 0, len(kvs))
    for key, t, v in kvs:
        out += s(key) + struct.pack("<I", t)
        if t == 8:
            out += s(v)
        elif t == 9:
            et, items = v
            out += struct.pack("<IQ", et, len(items))
            out += b"".join(s(i) if et == 8 else struct.pack("<i", i) for i in items)
        elif t == 7:
            out += struct.pack("<?", v)
        elif t == 6:
            out += struct.pack("<f", v)
    path.write_bytes(out)
    return path


@pytest.fixture
def d3_gguf(tmp_path):
    return _gguf(tmp_path / "vllm-d3-mini-Q8_0.gguf", [
        ("general.architecture", 8, "qwen35"), ("general.name", 8, "vllm-d3-mini"),
        ("tokenizer.ggml.tokens", 9, (8, ["a"] * 5000)), ("tokenizer.ggml.token_type", 9, (5, [1] * 5000)),
        ("qwen35.attention.causal", 7, False),
        ("qwen35.decision.type", 8, "vllm-d3"),
        ("qwen35.decision.temperature.choice", 6, 1.0), ("qwen35.decision.temperature.noul", 6, 1.0),
    ])


def test_the_whole_header_is_read_past_big_arrays(d3_gguf):
    meta = gguf.decision_metadata(d3_gguf)
    assert meta["qwen35.decision.type"] == "vllm-d3" and meta["qwen35.attention.causal"] is False
    assert "tokenizer.ggml.tokens" not in meta
    assert gguf.general_metadata(d3_gguf) == {"general.architecture": "qwen35",
                                              "general.name": "vllm-d3-mini"}
    m = native.NativeModel.from_gguf(str(d3_gguf), "vllm-d3-mini")
    assert m.decision_type == "vllm-d3" and not m.causal
    assert m.temperatures == {"choice": 1.0, "noul": 1.0} and m.limits.max_choice == 255
    assert native.NativeModel.from_gguf(str(_gguf(d3_gguf.parent / "plain.gguf", [
        ("general.architecture", 8, "qwen35")])), "x") is None


class _Llama(BaseHTTPRequestHandler):
    seen: list = []
    reply: tuple = (200, {})

    def log_message(self, *a):
        pass

    def _send(self, status, body):
        data = json.dumps(body).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        self._send(200, {"data": [{"id": "d3", "architecture": {"output_modalities": ["decisions"]}}]})

    def do_POST(self):
        _Llama.seen.append(json.loads(self.rfile.read(int(self.headers["Content-Length"]))))
        self._send(*_Llama.reply)


@pytest.fixture
def svc(d3_gguf):
    srv = ThreadingHTTPServer(("127.0.0.1", 0), _Llama)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    backend = LlamaServer(f"http://127.0.0.1:{srv.server_address[1]}")
    yield native.NativeService(native.NativeModel.from_gguf(str(d3_gguf), "vllm-d3-mini"), backend)
    srv.shutdown()


REQ = {"state": {"ticket": "charged twice"}, "questions": {
    "team": {"type": "choice", "instructions": "Which team?", "criteria": ["billing", "tech"]},
    "twice": {"type": "noul", "instructions": "Charged twice?"}}}


def test_answers_are_adapted_to_the_contract(svc):
    _Llama.reply = (200, {"model": "x", "answers": {
        "team": {"type": "choice", "choice": "billing", "probabilities": {"billing": 0.9, "tech": 0.1},
                 "confidence": 0.53},
        "twice": {"type": "noul", "noul": 0.3}}, "usage": {"input_tokens": 211, "output_tokens": 0}})
    out = svc.decide({**REQ, "debug": True})
    sent = _Llama.seen[-1]
    assert sent["questions"]["team"]["criteria"] == {"billing": None, "tech": None}   # list → object
    assert "criteria" not in sent["questions"]["twice"] and sent["state"] == REQ["state"]
    assert out["model"] == "vllm-d3-mini" and out["warnings"] == []
    assert out["usage"] == {"input_tokens": 211, "output_tokens": 0, "backend_calls": 1}
    assert out["answers"]["twice"]["confidence"] == 0.7 and out["answers"]["team"]["confidence"] == 0.53
    assert out["answers"]["team"]["debug"] == {"native": True}
    assert native.detect(svc.backend)


@pytest.mark.parametrize("status,body,code", [
    (400, {"error": {"code": 400, "type": "exceed_context_size_error", "message": "too big"}}, "input_too_long"),
    (500, {"error": {"code": 500, "type": "server_error",
                     "message": "input (9000 tokens) is too large to process. increase the physical batch size"}},
     "input_too_long"),
    (400, {"error": {"code": 400, "type": "invalid_request_error", "message": "bad"}}, "invalid_request"),
    (503, {"error": {"code": 503, "type": "unavailable_error", "message": "Loading model"}}, "not_ready"),
    (500, {"error": {"code": 500, "type": "server_error", "message": "boom"}}, "backend_error"),
])
def test_errors_are_mapped(svc, status, body, code):
    _Llama.reply = (status, body)
    with pytest.raises(ApiError) as e:
        svc.decide(REQ)
    assert e.value.code == code


def test_requests_are_validated_before_forwarding(svc):
    n = len(_Llama.seen)
    with pytest.raises(ApiError) as e:
        svc.decide({"state": "s", "questions": {"q": {"type": "score", "instructions": "x",
                                                      "criteria": [str(i) for i in range(11)]}}})
    assert e.value.code == "invalid_request" and len(_Llama.seen) == n


def test_spec(svc):
    spec = build_spec(svc.profile, svc.profile)
    assert spec["model"]["id"] == "vllm-d3-mini"
    assert spec["model"]["limits"]["max_score_levels"] == 10
    assert "normalized entropy" in spec["model"]["confidence"]["choice"]
    assert any("non-causal" in n for n in spec["model"]["notes"])


def test_detect_and_spec_cli(d3_gguf, capsys):
    from systemone_proxy import cli
    cli.main(["detect", str(d3_gguf)])
    assert capsys.readouterr().out.strip() == "native"
    cli.main(["spec", "--gguf", str(d3_gguf), "--json"])
    spec = json.loads(capsys.readouterr().out)
    assert spec["model"]["id"] == "vllm-d3-mini" and "notes" in spec["model"]

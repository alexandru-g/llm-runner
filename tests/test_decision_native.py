"""Natively served decision GGUFs (llama.cpp decision metadata): flags and batch sizes."""
import struct
from pathlib import Path

from click.testing import CliRunner

from llm_runner import backends, cli


def _gguf(path, kvs):
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
            out += struct.pack("<IQ", et, len(items)) + b"".join(
                s(i) if et == 8 else struct.pack("<i", i) for i in items)
        elif t == 7:
            out += struct.pack("<?", v)
    path.write_bytes(out)
    return path


def _d3(tmp_path, causal=False, name="vllm-d3-mini-Q8_0.gguf"):
    return _gguf(tmp_path / name, [
        ("general.architecture", 8, "qwen35"), ("general.tags", 9, (8, ["decision-model"])),
        ("tokenizer.ggml.tokens", 9, (8, ["t"] * 3000)), ("tokenizer.ggml.token_type", 9, (5, [1] * 3000)),
        ("qwen35.attention.causal", 7, causal), ("qwen35.decision.type", 8, "vllm-d3")])


def test_noncausal_is_found_past_the_tokenizer(tmp_path):
    d3 = _d3(tmp_path)
    assert backends.gguf_is_noncausal(d3) and backends.is_decision_model(d3)
    assert not backends.gguf_is_noncausal(_d3(tmp_path, causal=True, name="causal.gguf"))
    meta = backends.gguf_metadata(d3, lambda k: ".decision." in k)
    assert meta == {"qwen35.decision.type": "vllm-d3"}


def test_batch_sizes_reach_llama_server():
    cmd = cli._build_llama_cmd(Path("/m/x.gguf"), None, 8080, 8192, False,
                               batch_size=8192, ubatch_size=8192)
    assert cmd[cmd.index("--batch-size") + 1] == "8192" and cmd[cmd.index("--ubatch-size") + 1] == "8192"
    assert "--batch-size" not in cli._build_llama_cmd(Path("/m/x.gguf"), None, 8080, 8192, False)


def test_run_defaults_batches_to_the_context_for_noncausal(tmp_path, monkeypatch):
    d3 = _d3(tmp_path)
    started = {}
    monkeypatch.setattr(cli.proc, "start", lambda **kw: started.update(kw) or cli.proc.RunningProcess(
        name=kw["name"], pid=1, port=kw["port"], started_at=0, model_path=kw["model_path"],
        log_file="l", cmd=kw["cmd"], backend=kw["backend"]))
    monkeypatch.setattr(cli, "_ensure_gateway", lambda: None)
    monkeypatch.setattr(cli, "_port_in_use", lambda port: False)
    monkeypatch.setattr(backends, "find", lambda backend: "/usr/bin/llama-server")
    r = CliRunner().invoke(cli.cli, ["run", "--ft", str(d3), "-c", "8192", "--port", "8199"])
    assert r.exit_code == 0, r.output
    cmd = started["cmd"]
    assert cmd[cmd.index("--batch-size") + 1] == "8192" == cmd[cmd.index("--ubatch-size") + 1]
    assert "Non-causal model" in r.output

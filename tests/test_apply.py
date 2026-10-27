"""`llmctl apply`: serving files, validation with lines, idempotent plans."""
import json
import os
import subprocess
from pathlib import Path

import pytest
from click.testing import CliRunner

from llm_runner import apply as apply_mod
from llm_runner import backends
from llm_runner import cli
from llm_runner import process as proc
from llm_runner import registry


def _write(tmp_path: Path, name: str, text: str) -> Path:
    p = tmp_path / name
    p.write_text(text)
    return p


QWEN = """\
models:
  qwen:
    model: qwen3-v1
    group: qwen3
    port: 8101
    ctx: 8192
    n_parallel: 4
    kv_cache_mem: 6G
    args: ["--max-num-batched-tokens", 8192]
    env: {VLLM_LOGGING_LEVEL: INFO}
"""


def test_an_entry_becomes_its_run_command(tmp_path):
    [e] = apply_mod.load([_write(tmp_path, "a.yaml", QWEN)])
    assert e.run_argv("qwen3-v1") == [
        "qwen3-v1", "--name", "qwen", "--port", "8101", "--group", "qwen3",
        "--ctx-size", "8192", "--n-parallel", "4", "--kv-cache-mem", "6G",
        "--extra", "--max-num-batched-tokens 8192", "--env", "VLLM_LOGGING_LEVEL=INFO"]


def test_fragments_merge_and_collisions_name_both_files(tmp_path):
    a = _write(tmp_path, "a.yaml", QWEN)
    b = _write(tmp_path, "b.yaml", "models:\n  embed:\n    model: bge\n    port: 8102\n")
    assert [e.name for e in apply_mod.load([a, b])] == ["qwen", "embed"]
    c = _write(tmp_path, "c.yaml", "models:\n  qwen:\n    model: x\n    port: 8103\n")
    d = _write(tmp_path, "d.yaml", "models:\n  other:\n    model: x\n    port: 8101\n")
    with pytest.raises(apply_mod.ApplyError) as e:
        apply_mod.load([a, c, d])
    text = str(e.value)
    assert f"{c}:2: model 'qwen' is also defined at {a}:2" in text
    assert f"{d}:2: port 8101 is also used by 'qwen' ({a}:2)" in text


def test_errors_carry_file_and_line(tmp_path):
    f = _write(tmp_path, "bad.yaml", """\
models:
  qwen:
    model: qwen3-v1
    port: eighty
    ctx: 8192
    ctx: 4096
    colour: blue
    trust_remote_code: "yes"
  nameless:
    port: 9000
extra: 1
""")
    with pytest.raises(apply_mod.ApplyError) as e:
        apply_mod.load([f])
    p = e.value.problems
    assert f"{f}:4: models.qwen.port must be an integer, not 'eighty'" in p
    assert any(x.startswith(f"{f}:6: models.qwen.ctx is set twice (first at line 5)") for x in p)
    assert any(x.startswith(f"{f}:7: models.qwen.colour is not a known setting") for x in p)
    assert f"{f}:8: models.qwen.trust_remote_code must be true or false, not 'yes'" in p
    assert f"{f}:9: models.nameless.model is required" in p
    assert any(x.startswith(f"{f}:11: unknown top-level key 'extra'") for x in p)


def test_yaml_syntax_errors_give_the_line(tmp_path):
    f = _write(tmp_path, "bad.yaml", "models:\n  qwen:\n    model: [x\n    port: 1\n")
    with pytest.raises(apply_mod.ApplyError) as e:
        apply_mod.load([f])
    assert e.value.problems[0].startswith(f"{f}:")
    assert "not valid YAML" in e.value.problems[0]


def test_group_null_means_no_group(tmp_path):
    [e] = apply_mod.load([_write(tmp_path, "a.yaml",
                                 "models:\n  q:\n    model: m\n    port: 1\n    group: null\n")])
    assert e.run_argv("m")[-2:] == ["--group", "none"]


# ------------------------------------------------------------- end to end --

@pytest.fixture
def host(tmp_path, monkeypatch):
    """A fake host: real state files, servers are `sleep` processes."""
    state = tmp_path / "state"
    monkeypatch.setattr(proc, "STATE_DIR", state)
    monkeypatch.setattr(proc, "STATE_FILE", state / "running.json")
    monkeypatch.setattr(proc, "AUTOSTART_FILE", state / "autostart.json")
    monkeypatch.setattr(proc, "LOG_DIR", state / "logs")
    models = tmp_path / "models"
    for name in ("qwen3-v1", "qwen3-v2"):
        d = models / name
        d.mkdir(parents=True)
        (d / "config.json").write_text('{"architectures": ["Qwen3ForCausalLM"]}')
        (d / "model.safetensors").write_bytes(b"")
    monkeypatch.setattr(registry, "MODELS_DIR", models)
    monkeypatch.setattr(cli, "MODELS_DIR", models)
    monkeypatch.setattr(cli, "discover_models", lambda: registry.discover_models(models))
    monkeypatch.setattr(backends, "find_engine", lambda backend: "/usr/bin/vllm")
    monkeypatch.setattr(backends, "default_gpu_mem", lambda backend=None: None)
    monkeypatch.setattr(cli, "_ensure_gateway", lambda: None)
    monkeypatch.setattr(cli, "_port_in_use", lambda port: False)
    started = []
    real = subprocess.Popen

    def popen(argv, **kw):
        started.append(argv)
        return real(["sleep", "60"], start_new_session=True)
    monkeypatch.setattr(proc.subprocess, "Popen", popen)
    yield started
    for p in proc._load():
        try:
            os.killpg(p.pid, 9)
        except ProcessLookupError:
            pass


def _apply(*args):
    r = CliRunner().invoke(cli.cli, ["apply", *map(str, args), "--no-wait", "--json"])
    return r, (json.loads(r.stdout) if r.stdout.strip().startswith("{") else None)


def test_apply_starts_then_is_idempotent_then_rolls_back(tmp_path, host):
    f = _write(tmp_path, "serve.yaml", QWEN)
    r, out = _apply(f)
    assert r.exit_code == 0, r.output
    assert [a["action"] for a in out["actions"]] == ["start"]
    assert len(host) == 1 and "/fast" not in host[0][0]
    [rec] = proc.list_running()
    assert rec.name == "qwen" and rec.port == 8101 and rec.group == "qwen3"

    r, out = _apply(f)                                  # nothing changed
    assert [a["action"] for a in out["actions"]] == ["unchanged"] and len(host) == 1

    # Rollback/upgrade: point the same name at another version folder.
    f.write_text(f.read_text().replace("qwen3-v1", "qwen3-v2"))
    r, out = _apply(f)
    assert r.exit_code == 0, r.output
    [a] = out["actions"]
    assert a["action"] == "restart" and "+qwen3-v2" in a["reason"]
    [rec] = proc.list_running()
    assert rec.model_path.endswith("qwen3-v2") and len(host) == 2
    assert [s.name for s in proc.autostart_list()] == ["qwen"]


def test_dry_run_changes_nothing_and_prune_stops_the_rest(tmp_path, host):
    f = _write(tmp_path, "serve.yaml", QWEN)
    _apply(f)
    other = _write(tmp_path, "other.yaml", "models:\n  q2:\n    model: qwen3-v2\n    port: 8102\n")
    r, out = _apply(other, "--dry-run", "--prune")
    assert {a["name"]: a["action"] for a in out["actions"]} == {"q2": "start", "qwen": "stop"}
    assert out["dry_run"] and [p.name for p in proc.list_running()] == ["qwen"]
    r, out = _apply(other, "--prune")
    assert r.exit_code == 0, r.output
    assert [p.name for p in proc.list_running()] == ["q2"]
    assert [s.name for s in proc.autostart_list()] == ["q2"]


def test_apply_reports_unknown_models_and_bad_values(tmp_path, host):
    f = _write(tmp_path, "serve.yaml",
               "models:\n  a:\n    model: nope\n    port: 1\n"
               "  b:\n    model: qwen3-v1\n    port: 2\n    kv_quant: q2\n")
    r, out = _apply(f)
    assert r.exit_code == 3 and out["error"] == "invalid"
    assert any(p.startswith(f"{f}:2: models.a.model 'nope'") for p in out["problems"])
    assert any(p.startswith(f"{f}:5: models.b:") and "q2" in p for p in out["problems"])
    assert host == []


def test_dns_takes_a_list_or_a_string_and_checks_addresses(tmp_path):
    f = _write(tmp_path, "a.yaml", "models:\n  q:\n    model: m\n    port: 1\n"
               "    dns: [127.0.0.1, '::1']\n"
               "  r:\n    model: m\n    port: 2\n    dns: 127.0.0.1\n")
    q, r = apply_mod.load([f])
    assert q.run_argv("m")[-2:] == ["--dns", "127.0.0.1,::1"]
    assert r.options["dns"] == "127.0.0.1"
    f = _write(tmp_path, "b.yaml", "models:\n  q:\n    model: m\n    port: 1\n"
               "    dns: [dns.example]\n")
    with pytest.raises(apply_mod.ApplyError) as e:
        apply_mod.load([f])
    assert "b.yaml:5: models.q.dns: 'dns.example' is not an IP address" in e.value.problems[0]

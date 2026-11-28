"""What the Stream Deck integration reads and does (1.18): list --json, VRAM
in ps --json, run's fit check, --profile last without a last run, and --node
in --help."""
from __future__ import annotations

import json

import pytest
from click.testing import CliRunner

from llm_runner import backends, cli, gpu, launch, registry
from llm_runner import process as proc

from test_apply import host  # noqa: F401  (fixture)
from test_launch import _ctx, _spec, state  # noqa: F401  (fixture)


@pytest.fixture
def gguf(host, monkeypatch):  # noqa: F811
    """A 4 GiB model with a 1 GiB projector and a 0.5 GiB drafter (sparse files)."""
    folder = registry.MODELS_DIR / "big"
    folder.mkdir()
    for name, gib in (("big-Q4.gguf", 4), ("mmproj-F16.gguf", 1), ("mtp-big.gguf", 0.5)):
        with open(folder / name, "wb") as f:
            f.truncate(int(gib * 1024 ** 3))
    monkeypatch.setattr(backends, "find_llama_server", lambda: "/usr/bin/llama-server")
    monkeypatch.setattr(cli, "_require_backend", lambda backend: "/usr/bin/llama-server")
    return host


def _free(monkeypatch, mib, unified=False):
    monkeypatch.setattr(gpu, "gpus", lambda: [gpu.GPU(0, "Test GPU", 24576, mib, unified)])


def _run(*args):
    return CliRunner().invoke(cli.cli, ["run", "big-q4", "--no-menu", "--port", "8130", *args])


def test_run_refuses_what_cannot_fit(gguf, monkeypatch):
    _free(monkeypatch, 5000)                      # 5.5 GiB to load, ~4.9 free
    r = _run()
    assert r.exit_code == 1 and "won't fit: 5.5 GiB to load" in r.output
    assert "weights 4.0 + projector 1.0 + drafter 0.5" in r.output
    assert proc.find("big-q4") is None
    assert _run("--no-vision", "--no-draft").exit_code == 0      # 4 GiB: tight but fits
    assert "--force" not in proc.autostart_find("big-q4").run_args


def test_force_and_partial_offload_skip_the_check(gguf, monkeypatch):
    _free(monkeypatch, 1000)
    r = _run("--force")
    assert r.exit_code == 0, r.output
    assert "--force" not in proc.autostart_find("big-q4").run_args    # not remembered


def test_a_tight_fit_warns_and_starts(gguf, monkeypatch):
    _free(monkeypatch, 6000)
    r = _run()
    assert r.exit_code == 0 and "tight fit" in r.output


def test_no_gpu_information_means_no_check(gguf, monkeypatch):
    monkeypatch.setattr(gpu, "gpus", lambda: [])
    assert _run().exit_code == 0


def test_layers_on_the_cpu_skip_the_check(gguf, monkeypatch):
    _free(monkeypatch, 1000)
    assert _run("--n-gpu-layers", "20").exit_code == 0


def test_ps_json_has_vram_per_model_and_the_gpus(gguf, monkeypatch):
    _free(monkeypatch, 20000)
    assert _run().exit_code == 0
    pid = proc.find("big-q4").pid
    monkeypatch.setattr(gpu, "process_mib", lambda: {pid: 6100})
    out = json.loads(CliRunner().invoke(cli.cli, ["ps", "--json"]).stdout)
    [m] = [m for m in out["models"] if m["name"] == "big-q4"]
    assert m["vram_mib"] == 6100
    assert out["gpus"] == [{"index": 0, "name": "Test GPU", "total_mib": 24576,
                            "free_mib": 20000, "unified": False}]
    monkeypatch.setattr(gpu, "process_mib", lambda: None)      # a Jetson: no per process
    out = json.loads(CliRunner().invoke(cli.cli, ["ps", "--json"]).stdout)
    assert [m["vram_mib"] for m in out["models"] if m["name"] == "big-q4"] == [None]


def test_tree_mib_adds_up_child_processes(monkeypatch):
    monkeypatch.setattr(gpu, "_children", lambda: {10: [11, 12], 12: [13]})
    assert gpu.tree_mib([10, 20], {11: 100, 13: 50, 20: 7}) == {10: 150, 20: 7}
    assert gpu.tree_mib([10], None) == {10: None}


def test_jetson_reports_system_memory(monkeypatch, tmp_path):
    monkeypatch.setattr(gpu, "is_jetson", lambda: True)
    monkeypatch.setattr(gpu, "_meminfo", lambda: {"MemTotal": 62828, "MemAvailable": 23820})
    [g] = gpu.read_gpus()
    assert (g.total_mib, g.free_mib, g.unified) == (62828, 23820, True)
    assert gpu.process_mib() is None


def test_list_json(gguf):
    from llm_runner import usage
    usage.note_start(str(registry.MODELS_DIR / "big" / "big-Q4.gguf"), "big-q4", when=1.8e9)
    out = json.loads(CliRunner().invoke(cli.cli, ["list", "--json", "-t"]).stdout)
    assert out["models_dir"] == str(registry.MODELS_DIR)
    first = out["models"][0]
    assert first == {"key": "big-q4", "backend": "llama", "size": 4 * 1024 ** 3,
                     "last_used": 1.8e9, "vision": True, "draft": True, "decision": False,
                     "labels": ["chat", "vision"],
                     "path": str(registry.MODELS_DIR / "big" / "big-Q4.gguf")}
    assert {m["key"] for m in out["models"]} >= {"big-q4", "qwen3-v1", "qwen3-v2"}


def test_profile_last_without_a_last_run_uses_the_defaults(state, capsys):  # noqa: F811
    target = launch.Target(_spec(state), "llama")
    m = launch._Menu(_ctx(), target)
    assert m.find("last").name == "default"
    assert "no last run remembered" in capsys.readouterr().err
    launch._save_choices(target.spec.key, {"ctx_size": 8192})
    assert launch._Menu(_ctx(), target).find("last").name == "last"


def test_node_shows_in_help_of_remote_commands_only():
    r = CliRunner()
    for cmd in (["run"], ["ps"], ["list"], ["stop"], ["decide", "start"], ["--help"]):
        out = r.invoke(cli.cli, [*cmd, "--help"] if cmd != ["--help"] else cmd).output
        assert "--node NAME" in out, cmd
    for cmd in (["config", "show"], ["gateway", "control"], ["setup"]):
        assert "--node NAME" not in r.invoke(cli.cli, [*cmd, "--help"]).output, cmd

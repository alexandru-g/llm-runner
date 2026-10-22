"""Telemetry off for every backend; offline mode keeps llmctl off the hub."""
import pytest
from click.testing import CliRunner

from llm_runner import cli, config as config_mod, process as proc


@pytest.fixture
def cfg(tmp_path, monkeypatch):
    monkeypatch.setattr(config_mod, "CONFIG_DIR", tmp_path)
    monkeypatch.setattr(config_mod, "CONFIG_FILE", tmp_path / "config.json")
    monkeypatch.setattr(proc, "STATE_FILE", tmp_path / "running.json")
    monkeypatch.setattr(proc, "AUTOSTART_FILE", tmp_path / "autostart.json")
    monkeypatch.setattr(proc, "LOG_DIR", tmp_path / "logs")
    monkeypatch.delenv("LLM_RUNNER_OFFLINE", raising=False)
    return tmp_path


@pytest.fixture
def launched(cfg, monkeypatch):
    """The environment of the next process.start()."""
    seen = {}

    class FakePopen:
        pid = 4242

        def __init__(self, cmd, env=None, **kw):
            seen.update(env)

    monkeypatch.setattr(proc.subprocess, "Popen", FakePopen)
    return seen


def _start(**kw):
    proc.start(name="m", cmd=["vllm", "serve", "m"], port=8000, model_path="m",
               backend="vllm", **kw)


def test_telemetry_is_always_off(launched):
    _start()
    assert launched["VLLM_NO_USAGE_STATS"] == "1" and launched["DO_NOT_TRACK"] == "1"
    assert launched["HF_HUB_DISABLE_TELEMETRY"] == "1"
    assert "HF_HUB_OFFLINE" not in launched


def test_offline_mode_reaches_backends_and_e_still_wins(launched):
    config_mod.set_offline(True)
    _start(env={"DO_NOT_TRACK": "0"})
    assert launched["HF_HUB_OFFLINE"] == "1" and launched["TRANSFORMERS_OFFLINE"] == "1"
    assert launched["DO_NOT_TRACK"] == "0"
    # Only what -E set is remembered; the rest is applied fresh on each start.
    assert proc.autostart_list()[0].env == {"DO_NOT_TRACK": "0"}


def test_env_var_overrides_the_setting(cfg, monkeypatch):
    config_mod.set_offline(True)
    monkeypatch.setenv("LLM_RUNNER_OFFLINE", "0")
    assert not config_mod.is_offline()
    config_mod.set_offline(False)
    monkeypatch.setenv("LLM_RUNNER_OFFLINE", "1")
    assert config_mod.is_offline()


def test_offline_refuses_download_and_update(cfg):
    runner = CliRunner()
    assert runner.invoke(cli.cli, ["config", "offline", "on"]).exit_code == 0
    assert runner.invoke(cli.cli, ["config", "offline"]).output.strip() == "on"
    for args in (["download", "Qwen/Qwen3-8B"], ["update", "Qwen/Qwen3-8B"]):
        out = runner.invoke(cli.cli, args)
        assert out.exit_code != 0 and "Offline mode is on" in out.output
    assert runner.invoke(cli.cli, ["config", "offline", "off"]).exit_code == 0
    assert "offline" not in config_mod.load()


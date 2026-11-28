import pytest

from llm_runner import backends
from llm_runner import config as config_mod
from llm_runner import process as proc


@pytest.fixture(autouse=True)
def _own_config(tmp_path_factory, monkeypatch):
    """Never read or write the real config (the gateway reads its API key
    policy from it on every request)."""
    d = tmp_path_factory.mktemp("config")
    monkeypatch.setattr(config_mod, "CONFIG_DIR", d)
    monkeypatch.setattr(config_mod, "CONFIG_FILE", d / "config.json")


@pytest.fixture(autouse=True)
def _own_state(tmp_path_factory, monkeypatch):
    """Never touch the real state dir: `proc.stop` writes holds.json there,
    which the machine's live gateway reads."""
    d = tmp_path_factory.mktemp("state")
    monkeypatch.setattr(proc, "STATE_DIR", d)
    monkeypatch.setattr(proc, "STATE_FILE", d / "running.json")
    monkeypatch.setattr(proc, "AUTOSTART_FILE", d / "autostart.json")
    monkeypatch.setattr(proc, "LOG_DIR", d / "logs")


@pytest.fixture(autouse=True)
def _no_toolchain_probe(monkeypatch):
    """Launches never probe this machine's nvcc and GCC (the probe runs
    subprocesses, which tests that fake Popen would intercept)."""
    monkeypatch.setattr(backends, "toolchain_env", dict)


@pytest.fixture(autouse=True)
def _no_gpu_probe(monkeypatch):
    """No nvidia-smi: the fit check and `ps --json` see no GPU unless a test
    says what it has (results don't depend on this machine's free VRAM)."""
    from llm_runner import gpu
    monkeypatch.setattr(gpu, "gpus", lambda: [])
    monkeypatch.setattr(gpu, "process_mib", lambda: None)

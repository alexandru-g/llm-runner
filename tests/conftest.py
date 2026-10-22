import pytest

from llm_runner import backends
from llm_runner import config as config_mod


@pytest.fixture(autouse=True)
def _own_config(tmp_path_factory, monkeypatch):
    """Never read or write the real config (the gateway reads its API key
    policy from it on every request)."""
    d = tmp_path_factory.mktemp("config")
    monkeypatch.setattr(config_mod, "CONFIG_DIR", d)
    monkeypatch.setattr(config_mod, "CONFIG_FILE", d / "config.json")


@pytest.fixture(autouse=True)
def _no_toolchain_probe(monkeypatch):
    """Launches never probe this machine's nvcc and GCC (the probe runs
    subprocesses, which tests that fake Popen would intercept)."""
    monkeypatch.setattr(backends, "toolchain_env", dict)

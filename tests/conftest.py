"""Test isolation. Two guards are set before llm_runner is imported, for
the whole session, because a function-scoped patch can be undone mid-test
(on 2026-11-28 a test's ``monkeypatch.undo()`` let a fixture read the real
running.json and SIGKILL the live gateway and a served model):

* the state and config dirs point at a throwaway directory, so even paths
  fixed at import time (launch.json, speedtest.json) never reach the real ones;
* ``os.kill``/``os.killpg`` refuse any process this pytest didn't start.
"""
import os
import tempfile
from pathlib import Path

_SANDBOX = Path(tempfile.mkdtemp(prefix="llmctl-tests-"))
os.environ["LLM_RUNNER_STATE_DIR"] = str(_SANDBOX / "state")
os.environ["LLM_RUNNER_CONFIG_DIR"] = str(_SANDBOX / "config")

_PYTEST = os.getpid()
_real_kill, _real_killpg = os.kill, os.killpg


def _parent(pid: int) -> int | None:
    try:
        return int(Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[1])
    except (OSError, ValueError, IndexError):
        return None


def _ours(pid: int) -> bool | None:
    """Whether ``pid`` descends from this pytest; None when it's gone."""
    if _parent(pid) is None:
        return None
    for _ in range(64):
        if pid == _PYTEST:
            return True
        pid = _parent(pid)
        if pid is None or pid <= 1:
            return False
    return False


def _group(pgid: int) -> list[int]:
    out = []
    for d in os.listdir("/proc"):
        if d.isdigit():
            try:
                fields = Path(f"/proc/{d}/stat").read_text().rsplit(")", 1)[1].split()
            except (OSError, IndexError):
                continue
            if int(fields[2]) == pgid:
                out.append(int(d))
    return out


def _guarded_kill(pid, sig):
    if sig and pid > 0 and _ours(pid) is False:
        raise PermissionError(f"tests may only signal their own processes, not pid {pid}")
    return _real_kill(pid, sig)


def _guarded_killpg(pgid, sig):
    if sig and pgid > 0 and any(_ours(p) is False for p in _group(pgid)):
        raise PermissionError(f"tests may only signal their own processes, not group {pgid}")
    return _real_killpg(pgid, sig)


os.kill, os.killpg = _guarded_kill, _guarded_killpg

import pytest  # noqa: E402

from llm_runner import backends  # noqa: E402
from llm_runner import config as config_mod  # noqa: E402
from llm_runner import process as proc  # noqa: E402


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

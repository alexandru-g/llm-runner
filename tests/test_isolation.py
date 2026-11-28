"""The test session can't reach this machine's real llmctl state or its
processes (see conftest.py)."""
import os
import signal
import subprocess

import pytest

import conftest
from llm_runner import config as config_mod
from llm_runner import launch
from llm_runner import process as proc


def test_state_and_config_live_in_the_sandbox(monkeypatch):
    monkeypatch.undo()                  # even with every patch undone
    for path in (proc.STATE_DIR, proc.STATE_FILE, launch.CHOICES_FILE, config_mod.CONFIG_FILE):
        assert str(path).startswith(str(conftest._SANDBOX)), path


def test_signals_reach_only_our_own_processes():
    with pytest.raises(PermissionError, match="tests may only"):
        os.kill(os.getppid(), signal.SIGCONT)
    with pytest.raises(PermissionError, match="tests may only"):
        os.killpg(os.getpgid(os.getppid()), signal.SIGCONT)
    child = subprocess.Popen(["sleep", "30"], start_new_session=True)
    os.killpg(child.pid, signal.SIGKILL)
    assert child.wait() == -signal.SIGKILL
    with pytest.raises(ProcessLookupError):
        os.kill(child.pid, signal.SIGKILL)
    os.kill(os.getppid(), 0)            # probing is fine

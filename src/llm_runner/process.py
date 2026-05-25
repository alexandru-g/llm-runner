"""Lifecycle for detached llama-server processes.

State is persisted to ``.state/running.json`` under a user-level state dir so
the CLI is stateful across invocations regardless of cwd. Each running entry
tracks its PID, listening port, log path, and the command that started it.

Override location with ``LLM_RUNNER_STATE_DIR`` (otherwise falls back to a
repo-local ``.state/`` if one already exists, else ``$XDG_STATE_HOME/llm-runner``
or ``~/.local/state/llm-runner``).
"""
from __future__ import annotations

import json
import os
import signal
import subprocess
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path


def _resolve_state_dir() -> Path:
    env = os.environ.get("LLM_RUNNER_STATE_DIR")
    if env:
        return Path(env).expanduser().resolve()
    repo_local = Path(__file__).resolve().parents[2] / ".state"
    if repo_local.exists():
        return repo_local
    xdg = os.environ.get("XDG_STATE_HOME")
    base = Path(xdg).expanduser() if xdg else Path.home() / ".local" / "state"
    return base / "llm-runner"


STATE_DIR = _resolve_state_dir()
STATE_FILE = STATE_DIR / "running.json"
LOG_DIR = STATE_DIR / "logs"


@dataclass
class RunningProcess:
    name: str
    pid: int
    port: int
    started_at: float
    model_path: str
    log_file: str
    cmd: list[str] = field(default_factory=list)


def _load() -> list[RunningProcess]:
    if not STATE_FILE.exists():
        return []
    try:
        raw = json.loads(STATE_FILE.read_text())
    except json.JSONDecodeError:
        return []
    return [RunningProcess(**r) for r in raw]


def _save(procs: list[RunningProcess]) -> None:
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    STATE_FILE.write_text(json.dumps([asdict(p) for p in procs], indent=2))


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def reconcile() -> list[RunningProcess]:
    """Drop dead PIDs from the state file and return the live ones."""
    procs = _load()
    alive = [p for p in procs if _alive(p.pid)]
    if len(alive) != len(procs):
        _save(alive)
    return alive


def list_running() -> list[RunningProcess]:
    return reconcile()


def find(identifier: str) -> RunningProcess | None:
    procs = reconcile()
    if identifier.isdigit():
        pid = int(identifier)
        match = next((p for p in procs if p.pid == pid), None)
        if match:
            return match
    return next((p for p in procs if p.name == identifier), None)


def start(*, name: str, cmd: list[str], port: int, model_path: str) -> RunningProcess:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_file = LOG_DIR / f"{name}-{port}-{int(time.time())}.log"
    fh = open(log_file, "ab", buffering=0)
    proc = subprocess.Popen(
        cmd,
        stdout=fh,
        stderr=subprocess.STDOUT,
        stdin=subprocess.DEVNULL,
        start_new_session=True,
        close_fds=True,
    )
    rec = RunningProcess(
        name=name,
        pid=proc.pid,
        port=port,
        started_at=time.time(),
        model_path=model_path,
        log_file=str(log_file),
        cmd=list(cmd),
    )
    procs = reconcile()
    procs.append(rec)
    _save(procs)
    return rec


def stop(identifier: str, *, timeout: float = 10.0) -> RunningProcess | None:
    target = find(identifier)
    if target is None:
        return None
    _signal_group(target.pid, signal.SIGTERM)
    deadline = time.time() + timeout
    while time.time() < deadline and _alive(target.pid):
        time.sleep(0.25)
    if _alive(target.pid):
        _signal_group(target.pid, signal.SIGKILL)
    procs = [p for p in reconcile() if p.pid != target.pid]
    _save(procs)
    return target


def _signal_group(pid: int, sig: int) -> None:
    try:
        os.killpg(os.getpgid(pid), sig)
    except ProcessLookupError:
        pass

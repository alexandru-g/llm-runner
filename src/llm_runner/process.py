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
# Durable "what the user wants running" manifest. Unlike running.json (which is
# reconciled — dead PIDs are pruned), this survives a reboot so `restore` can
# relaunch the last-running set. Kept in sync: `start` upserts, `stop` removes.
AUTOSTART_FILE = STATE_DIR / "autostart.json"
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


@dataclass
class LaunchSpec:
    """The minimal recipe needed to relaunch a model after a reboot."""
    name: str
    cmd: list[str]
    port: int
    model_path: str


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


def _load_autostart() -> list[LaunchSpec]:
    if not AUTOSTART_FILE.exists():
        return []
    try:
        raw = json.loads(AUTOSTART_FILE.read_text())
    except json.JSONDecodeError:
        return []
    return [LaunchSpec(**r) for r in raw]


def _save_autostart(specs: list[LaunchSpec]) -> None:
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    AUTOSTART_FILE.write_text(json.dumps([asdict(s) for s in specs], indent=2))


def autostart_list() -> list[LaunchSpec]:
    return _load_autostart()


def autostart_add(*, name: str, cmd: list[str], port: int, model_path: str) -> None:
    """Upsert a launch spec into the autostart manifest (keyed by name)."""
    specs = [s for s in _load_autostart() if s.name != name]
    specs.append(LaunchSpec(name=name, cmd=list(cmd), port=port,
                            model_path=model_path))
    _save_autostart(specs)


def autostart_remove(name: str) -> None:
    specs = _load_autostart()
    kept = [s for s in specs if s.name != name]
    if len(kept) != len(specs):
        _save_autostart(kept)


def restore() -> list[tuple[LaunchSpec, str]]:
    """Relaunch every manifest model that isn't already running.

    Idempotent: a model whose name is already live is left alone. Returns
    ``(spec, status)`` pairs for reporting, where status is ``"started …"``,
    ``"already running"``, or ``"error: …"``.
    """
    live = {p.name for p in list_running()}
    results: list[tuple[LaunchSpec, str]] = []
    for spec in _load_autostart():
        if spec.name in live:
            results.append((spec, "already running"))
            continue
        try:
            rec = start(name=spec.name, cmd=spec.cmd, port=spec.port,
                        model_path=spec.model_path)
            results.append((spec, f"started pid={rec.pid} port={rec.port}"))
        except Exception as e:  # never let one bad spec abort the rest
            results.append((spec, f"error: {e}"))
    return results


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
    # Remember this launch so `restore` can bring it back after a reboot.
    autostart_add(name=name, cmd=cmd, port=port, model_path=model_path)
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
    # An explicit stop means "don't autostart this next boot".
    autostart_remove(target.name)
    return target


def _signal_group(pid: int, sig: int) -> None:
    try:
        os.killpg(os.getpgid(pid), sig)
    except ProcessLookupError:
        pass

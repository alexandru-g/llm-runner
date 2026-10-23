"""Restarts models that crash, from inside the gateway.

A model is supervised when it is in the autostart manifest (every ``run``
puts it there; ``llmctl stop`` takes it out first, so a stop is never
mistaken for a crash) and this gateway has seen it ready. One that dies
before it was ever ready is a failed start, not a crash: it is left alone,
as are manifest entries at boot, which ``llmctl restore`` starts.

Policy per model (``run --restart``, ``--restart-max``, the same keys in
serving files): ``on-failure`` (the default) restarts with exponential
backoff, 2, 4, 8 … s capped at 60 s; after ``restart_max`` restarts within
:data:`WINDOW` the model is ``crashed`` and left down until someone starts
it. ``never`` marks it ``crashed`` at once. Ten minutes ready resets the
count.

States: ``starting`` (alive, not ready yet), ``ready``, ``restarting``
(down, a restart is due), ``crashed`` (down, given up). They go to
``<state>/supervisor.json`` for ``llmctl ps`` and to ``/v1/models``; each
change is one JSON line in ``<state>/events.jsonl`` (``llmctl events``)
and the gateway log.
"""
from __future__ import annotations

import json
import os
import threading
import time
import urllib.error
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from llm_runner import process as proc
from llm_runner.backends import GATEWAY

ON_FAILURE, NEVER = "on-failure", "never"
POLICIES = (ON_FAILURE, NEVER)
DEFAULT_MAX = 5
WINDOW = 600.0          # restarts counted over this long; this long ready resets them
BACKOFF_CAP = 60.0
TICK = 2.0
STARTING, READY, RESTARTING, CRASHED = "starting", "ready", "restarting", "crashed"
DOWN_STATES = (RESTARTING, CRASHED)


def backoff(n: int) -> float:
    """Wait before restart number ``n`` (0-based): 2, 4, 8 … s, capped."""
    return min(2.0 * (2 ** n), BACKOFF_CAP)


def healthy(port: int, timeout: float = 1.0) -> bool:
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=timeout) as r:
            return r.status == 200
    except (urllib.error.HTTPError, OSError, ValueError):
        return False


@dataclass
class Tracked:
    name: str
    state: str = STARTING
    pid: int | None = None
    ready_since: float | None = None
    seen_ready: bool = False
    restarts: list[float] = field(default_factory=list)     # monotonic times, in WINDOW
    total_restarts: int = 0
    next_at: float | None = None
    last_error: str | None = None


class Supervisor:
    def __init__(self, state_dir: Path | None = None, *,
                 health: Callable[[int], bool] = healthy,
                 restore: Callable[[list[str]], list] = lambda names: proc.restore(names),
                 on_restart: Callable[[str], None] | None = None,
                 clock: Callable[[], float] = time.monotonic):
        self.state_dir = state_dir or proc.STATE_DIR
        self.health, self.restore, self.on_restart = health, restore, on_restart
        self.clock = clock
        self.tracked: dict[str, Tracked] = {}
        # Servers this process started: reaped here, or each would stay a
        # zombie after it exits (proc._alive treats zombies as dead).
        self.children: set[int] = set()
        self._lock = threading.Lock()
        self._written: str | None = None

    # ------------------------------------------------------------ queries --
    def state(self, name: str) -> Tracked | None:
        with self._lock:
            t = self.tracked.get(name)
            return None if t is None else Tracked(**vars(t))

    def down(self) -> list[Tracked]:
        """Supervised models that are down now (restarting or crashed)."""
        with self._lock:
            return [Tracked(**vars(t)) for t in self.tracked.values() if t.state in DOWN_STATES]

    def retry_after(self, name: str) -> int | None:
        """Seconds a client should wait for ``name``: the next restart plus a
        typical start. None when it isn't coming back by itself."""
        t = self.state(name)
        if t is None or t.state == CRASHED:
            return None
        wait = max(0.0, (t.next_at or self.clock()) - self.clock())
        return int(wait) + 30

    # --------------------------------------------------------------- loop --
    def run(self, stop: threading.Event) -> None:
        while not stop.wait(TICK):
            try:
                self.tick()
            except Exception as e:  # a bad pass mustn't end supervision
                print(f"supervisor: {type(e).__name__}: {e}", flush=True)

    def _reap(self) -> None:
        for pid in list(self.children):
            try:
                done, _ = os.waitpid(pid, os.WNOHANG)
            except ChildProcessError:
                done = pid
            if done:
                self.children.discard(pid)

    def tick(self) -> None:
        self._reap()
        now = self.clock()
        live = {p.name: p for p in proc.list_running() if p.backend != GATEWAY}
        specs = {s.name: s for s in proc.autostart_list() if s.backend != GATEWAY}
        due: list[str] = []
        with self._lock:
            for name in list(self.tracked):
                if name not in specs and name not in live:
                    t = self.tracked.pop(name)
                    if t.state in DOWN_STATES:
                        self._event(name, "forgotten", "no longer in autostart (stopped)")
            for name, p in live.items():
                t = self.tracked.setdefault(name, Tracked(name))
                if t.pid != p.pid:
                    t.pid, t.ready_since = p.pid, None
                    if t.state in DOWN_STATES:
                        t.state = STARTING
                        self._event(name, STARTING, f"pid {p.pid}")
                if self.health(p.port):
                    if t.state != READY:
                        t.state, t.ready_since, t.seen_ready = READY, now, True
                        self._event(name, READY, f"pid {p.pid} port {p.port}")
                    elif t.ready_since is not None and now - t.ready_since >= WINDOW:
                        t.restarts.clear()
                elif t.state == READY:
                    t.state = STARTING          # alive, not answering (reloading, stuck)
            for name, spec in specs.items():
                if name in live:
                    continue
                t = self.tracked.get(name)
                if t is None or not t.seen_ready:
                    continue                     # never ready here: boot, or a failed start
                t.restarts = [r for r in t.restarts if now - r < WINDOW]
                if t.state not in DOWN_STATES:
                    t.pid, t.ready_since = None, None
                    if getattr(spec, "restart", ON_FAILURE) == NEVER:
                        t.state = CRASHED
                        self._event(name, CRASHED, "died; restart policy is never")
                        continue
                    t.state, t.next_at = RESTARTING, now + backoff(len(t.restarts))
                    self._event(name, RESTARTING,
                                f"died; restart in {backoff(len(t.restarts)):.0f} s")
                if t.state == RESTARTING:
                    limit = getattr(spec, "restart_max", DEFAULT_MAX) or DEFAULT_MAX
                    if len(t.restarts) >= limit:
                        t.state, t.next_at = CRASHED, None
                        self._event(name, CRASHED, f"gave up after {len(t.restarts)} restarts "
                                                   f"in {WINDOW / 60:.0f} min")
                    elif now >= (t.next_at or now):
                        due.append(name)
        for name in due:
            self._restart(name, specs[name].port)
        self._write()

    def _restart(self, name: str, port: int) -> None:
        if proc.port_in_use(port):
            with self._lock:
                t = self.tracked[name]
                t.next_at, t.last_error = self.clock() + 5, f"port {port} is in use"
            return
        results = self.restore([name])
        status = results[0][1] if results else "error: nothing restored"
        with self._lock:
            t = self.tracked[name]
            now = self.clock()
            t.restarts.append(now)
            t.total_restarts += 1
            if status.startswith("error"):
                t.last_error = status
                t.next_at = now + backoff(len(t.restarts))
                self._event(name, "restart_failed", status)
            else:
                t.state, t.last_error = STARTING, None
                self._event(name, "restarted", status)
                pid = status.partition("pid=")[2].split(" ")[0]
                if pid.isdigit():
                    self.children.add(int(pid))
        if self.on_restart:
            self.on_restart(name)

    # ------------------------------------------------------------- output --
    def _event(self, name: str, event: str, detail: str) -> None:
        line = {"ts": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "model": name,
                "event": event, "detail": detail}
        print(f"supervisor: {name}: {event} ({detail})", flush=True)
        try:
            self.state_dir.mkdir(parents=True, exist_ok=True)
            with open(self.state_dir / "events.jsonl", "a") as f:
                f.write(json.dumps(line) + "\n")
        except OSError:
            pass

    def snapshot(self) -> dict:
        now = self.clock()
        with self._lock:
            return {t.name: {"state": t.state, "pid": t.pid,
                             "restarts": t.total_restarts,
                             "next_restart_in": None if t.next_at is None or t.state != RESTARTING
                             else max(0, round(t.next_at - now, 1)),
                             "last_error": t.last_error}
                    for t in self.tracked.values()}

    def _write(self) -> None:
        text = json.dumps({"updated": time.time(), "models": self.snapshot()}, indent=2)
        if text.split("\n", 2)[-1] == (self._written or "").split("\n", 2)[-1]:
            return                       # only the timestamp changed
        try:
            self.state_dir.mkdir(parents=True, exist_ok=True)
            tmp = self.state_dir / "supervisor.json.tmp"
            tmp.write_text(text)
            os.replace(tmp, self.state_dir / "supervisor.json")
            self._written = text
        except OSError:
            pass


def read_states(state_dir: Path | None = None) -> dict[str, dict]:
    """What the gateway's supervisor last wrote, for ``llmctl ps``."""
    try:
        data = json.loads(((state_dir or proc.STATE_DIR) / "supervisor.json").read_text())
    except (OSError, ValueError):
        return {}
    return data.get("models") or {}

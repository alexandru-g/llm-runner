"""When each on-disk model was last used, for ``llmctl list --sort used``.

``<state>/usage.json`` maps a model's path to when it last served a request
through this machine's gateway (``last_request``, with a running ``requests``
count) and when ``llmctl run`` last started it (``last_started``). Its "last
used" is the later of the two. A request for a model on another node is
recorded there, by the gateway whose model served it. Requests straight to a
model's port (``speedtest``, a client that skips the gateway) aren't seen.

The gateway counts requests in memory and writes them at most every
:data:`FLUSH_EVERY` seconds, and when it exits; ``run`` writes its start at
once. Both merge into the file under a lock, so neither loses the other's.

For a model with no entry yet (used before llmctl 1.17), the latest start in
``events.jsonl`` under its key stands in.
"""
from __future__ import annotations

import contextlib
import fcntl
import json
import os
import threading
import time
from pathlib import Path

from llm_runner import process as proc

FILE = "usage.json"
FLUSH_EVERY = 30.0
_STARTS = ("started", "restarted")


def _path(state_dir: Path | None = None) -> Path:
    return (state_dir or proc.STATE_DIR) / FILE


def load(state_dir: Path | None = None) -> dict[str, dict]:
    try:
        data = json.loads(_path(state_dir).read_text())
    except (OSError, ValueError):
        return {}
    return {k: v for k, v in data.items() if isinstance(v, dict)} if isinstance(data, dict) else {}


@contextlib.contextmanager
def _locked(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path.with_suffix(".lock"), "a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


def merge(updates: dict[str, dict], state_dir: Path | None = None) -> None:
    """Fold ``{model_path: {last_request, requests, last_started, name}}`` into
    the file: times keep the later value, ``requests`` adds up."""
    if not updates:
        return
    path = _path(state_dir)
    try:
        with _locked(path):
            data = load(state_dir)
            for model_path, new in updates.items():
                old = data.setdefault(model_path, {})
                for key in ("last_request", "last_started"):
                    if new.get(key) is not None:
                        old[key] = max(old.get(key) or 0.0, new[key])
                if new.get("requests"):
                    old["requests"] = int(old.get("requests") or 0) + new["requests"]
                if new.get("name"):
                    old["name"] = new["name"]
            tmp = path.with_name(f".{path.name}.{os.getpid()}")
            tmp.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
            tmp.replace(path)
    except OSError:
        pass                    # remembering use is a convenience, never a failure


def note_start(model_path: str, name: str, when: float | None = None,
               state_dir: Path | None = None) -> None:
    merge({model_path: {"last_started": when or time.time(), "name": name}}, state_dir)


class Tracker:
    """The gateway's side: count requests per model path, write them now and
    then (and on :meth:`flush` at exit)."""

    def __init__(self, state_dir: Path | None = None, every: float = FLUSH_EVERY):
        self.state_dir, self.every = state_dir, every
        self._pending: dict[str, dict] = {}
        self._lock = threading.Lock()
        self._flushed = time.monotonic()

    def note(self, model_path: str, name: str, when: float | None = None) -> None:
        if not model_path:
            return
        with self._lock:
            e = self._pending.setdefault(model_path, {"requests": 0})
            e["requests"] += 1
            e["last_request"] = max(e.get("last_request") or 0.0, when or time.time())
            e["name"] = name
            due = time.monotonic() - self._flushed >= self.every
        if due:
            self.flush()

    def flush(self) -> None:
        with self._lock:
            pending, self._pending = self._pending, {}
            self._flushed = time.monotonic()
        merge(pending, self.state_dir)


def last_used(entry: dict | None) -> float | None:
    times = [t for t in ((entry or {}).get("last_request"), (entry or {}).get("last_started"))
             if t]
    return max(times) if times else None


def starts_from_events(state_dir: Path | None = None) -> dict[str, float]:
    """``{model name: its latest start}`` from events.jsonl, for models used
    before usage.json existed."""
    out: dict[str, float] = {}
    try:
        f = open((state_dir or proc.STATE_DIR) / "events.jsonl")
    except OSError:
        return out
    with f:
        for line in f:
            try:
                ev = json.loads(line)
                if ev.get("event") not in _STARTS:
                    continue
                ts = time.mktime(time.strptime(ev["ts"][:19], "%Y-%m-%dT%H:%M:%S"))
            except (ValueError, KeyError, TypeError):
                continue
            name = ev.get("model")
            if isinstance(name, str) and ts > out.get(name, 0.0):
                out[name] = ts
    return out


def ago(ts: float | None, now: float | None = None) -> str:
    """``5m ago``, ``3h ago``, ``12d ago``; a date past 60 days; ``never``."""
    if not ts:
        return "never"
    s = max(0.0, (now or time.time()) - ts)
    if s < 60:
        return "just now"
    for unit, size, upto in (("m", 60, 3600), ("h", 3600, 86400 * 2), ("d", 86400, 86400 * 60)):
        if s < upto:
            return f"{int(s // size)}{unit} ago"
    return time.strftime("%Y-%m-%d", time.localtime(ts))

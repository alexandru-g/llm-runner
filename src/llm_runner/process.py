"""Lifecycle for detached model-server processes (llama-server or vllm).

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
import re
import signal
import socket
import subprocess
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

from llm_runner import config as config_mod
from llm_runner.backends import launch_env, with_bind_host


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
    backend: str = "llama"  # default covers records written before vllm support
    env: dict[str, str] = field(default_factory=dict)  # extra vars set at launch
    # Second name the gateway routes by (e.g. every quant of one model); "" =
    # none, None = not recorded (launched before groups: the default applies).
    group: str | None = None
    # How it runs: None = a host process (also every record written before
    # containers). Else {"kind": "host"|"docker", "image", "digest", "gpus",
    # "network", "container", "cache_dir", "cache_rule"} (see containers.py).
    runtime: dict | None = None


@dataclass
class LaunchSpec:
    """The minimal recipe needed to relaunch a model after a reboot.

    ``cmd`` is the fully-resolved server argv (what actually runs);
    ``run_args`` is the original ``llmctl run`` argv (minus ``--port``, which
    is pinned separately) so the launch can be shown and re-issued with
    overrides. Empty for manifests written before ``run_args`` existed.
    """
    name: str
    cmd: list[str]
    port: int
    model_path: str
    run_args: list[str] = field(default_factory=list)
    backend: str = "llama"
    env: dict[str, str] = field(default_factory=dict)
    group: str | None = None
    # The runtime as chosen (kind, image, digest, gpus, network); the
    # container name and cache dir are worked out again at each start.
    runtime: dict | None = None
    # What the gateway's supervisor does when it crashes (supervisor.py).
    restart: str = "on-failure"
    restart_max: int = 5

    def run_cmd(self) -> list[str]:
        """The ``llmctl run …`` argv that reproduces this launch, port pinned."""
        return ["run", *self.run_args, "--port", str(self.port)]


def port_in_use(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.2)
        return s.connect_ex((config_mod.local_host(), port)) == 0


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
    # A zombie still answers kill(0) until its parent reaps it, and `run
    # --wait` is the parent of the server it waits on, so a crash looked
    # like a slow start.
    try:
        state = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[0]
    except (OSError, IndexError):
        return True
    return state != "Z"


def _load_autostart() -> list[LaunchSpec]:
    if not AUTOSTART_FILE.exists():
        return []
    try:
        raw = json.loads(AUTOSTART_FILE.read_text())
    except json.JSONDecodeError:
        return []
    known = set(LaunchSpec.__dataclass_fields__)
    return [LaunchSpec(**{k: v for k, v in r.items() if k in known}) for r in raw]


def _save_autostart(specs: list[LaunchSpec]) -> None:
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    AUTOSTART_FILE.write_text(json.dumps([asdict(s) for s in specs], indent=2))


def autostart_list() -> list[LaunchSpec]:
    return _load_autostart()


def autostart_add(*, name: str, cmd: list[str], port: int, model_path: str,
                  run_args: list[str] | None = None,
                  backend: str = "llama",
                  env: dict[str, str] | None = None, group: str | None = None,
                  runtime: dict | None = None, restart: str | None = None,
                  restart_max: int | None = None) -> None:
    """Upsert a launch spec into the autostart manifest (keyed by name).

    Ports are pinned, so a port can only be restored to one model: entries
    for other models on the same port are dropped. They are leftovers from
    launches that died without an explicit ``stop`` (crash, failed load,
    external kill) and would otherwise fight over the port at boot."""
    specs = [s for s in _load_autostart() if s.name != name and s.port != port]
    specs.append(LaunchSpec(name=name, cmd=list(cmd), port=port,
                            model_path=model_path, run_args=list(run_args or []),
                            backend=backend, env=dict(env or {}), group=group,
                            runtime=chosen_runtime(runtime),
                            restart=restart or "on-failure", restart_max=restart_max or 5))
    _save_autostart(specs)


# The runtime fields that describe the choice, not one start of it.
_RUNTIME_CHOICE = ("kind", "image", "digest", "gpus", "network", "dns",
                   "engine_version")


def chosen_runtime(runtime: dict | None) -> dict | None:
    if not runtime:
        return None
    return {k: runtime[k] for k in _RUNTIME_CHOICE if runtime.get(k) is not None}


def autostart_find(name: str) -> LaunchSpec | None:
    return next((s for s in _load_autostart() if s.name == name), None)


def autostart_remove(name: str) -> None:
    specs = _load_autostart()
    kept = [s for s in specs if s.name != name]
    if len(kept) != len(specs):
        _save_autostart(kept)


def autostart_sync() -> tuple[list[LaunchSpec], list[LaunchSpec]]:
    """Rewrite the manifest to exactly the live models. Returns ``(kept,
    dropped)``. A live model's existing entry keeps its ``run_args`` (port
    and launch details refreshed from the live record); one without an entry
    gets a legacy-style spec from its resolved argv."""
    old = {s.name: s for s in _load_autostart()}
    kept = [
        LaunchSpec(name=p.name, cmd=list(p.cmd), port=p.port,
                   model_path=p.model_path,
                   run_args=old[p.name].run_args if p.name in old else [],
                   backend=p.backend, env=dict(p.env), group=p.group,
                   runtime=chosen_runtime(p.runtime),
                   restart=old[p.name].restart if p.name in old else "on-failure",
                   restart_max=old[p.name].restart_max if p.name in old else 5)
        for p in list_running()
    ]
    live = {s.name for s in kept}
    dropped = [s for s in old.values() if s.name not in live]
    _save_autostart(kept)
    return kept, dropped


def restore(names: list[str] | None = None,
            host: str | None = None) -> list[tuple[LaunchSpec, str]]:
    """Relaunch manifest models that aren't already running.

    ``names`` restricts the set (default: everything in the manifest).
    ``host``, when given, replaces the listen address each entry recorded.
    Idempotent: a model whose name is already live is left alone. Ports are
    pinned — an entry whose port is taken is reported, not moved. Returns
    ``(spec, status)`` pairs for reporting, where status is ``"started …"``,
    ``"already running"``, or ``"error: …"``.
    """
    live = {p.name for p in list_running()}
    specs = _load_autostart()
    if names:
        by_name = {s.name: s for s in specs}
        specs = []
        for n in names:
            if n in by_name:
                specs.append(by_name[n])
            else:
                specs.append(LaunchSpec(name=n, cmd=[], port=0, model_path=""))
    results: list[tuple[LaunchSpec, str]] = []
    # A server launched earlier in this loop may not have bound its port yet,
    # so port_in_use() alone can't stop two entries racing for the same port.
    claimed: set[int] = set()
    for spec in specs:
        if not spec.cmd:
            results.append((spec, "error: not in autostart manifest"))
            continue
        if spec.name in live:
            results.append((spec, "already running"))
            continue
        if spec.port in claimed or port_in_use(spec.port):
            results.append((spec, f"error: port {spec.port} is in use"))
            continue
        claimed.add(spec.port)
        try:
            cmd = spec.cmd if host is None else with_bind_host(spec.cmd, spec.backend, host)
            rec = start(name=spec.name, cmd=cmd, port=spec.port,
                        model_path=spec.model_path, run_args=spec.run_args,
                        backend=spec.backend, env=spec.env, group=spec.group,
                        runtime=spec.runtime, restart=spec.restart,
                        restart_max=spec.restart_max)
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


def start(*, name: str, cmd: list[str], port: int, model_path: str,
          run_args: list[str] | None = None,
          backend: str = "llama",
          env: dict[str, str] | None = None, group: str | None = None,
          runtime: dict | None = None, restart: str | None = None,
          restart_max: int | None = None) -> RunningProcess:
    """Launch ``cmd`` detached. ``env`` is layered over the current environment
    (e.g. VLLM_* tuning knobs) and remembered so restore/restart reproduce it.
    Telemetry is always off, offline mode keeps backends off the hub, and
    vllm/sglang get a CUDA toolchain their JIT can use (``backends.launch_env``);
    those are applied fresh, not remembered.

    ``runtime`` ``{"kind": "docker", "image": …}`` runs ``cmd`` in a container
    (``containers.docker_argv``); ``cmd`` stays the host form either way."""
    from llm_runner import containers
    from llm_runner.registry import MODELS_DIR

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_file = LOG_DIR / f"{name}-{port}-{int(time.time())}.log"
    runtime = dict(runtime) if runtime else None
    kind = (runtime or {}).get("kind", containers.HOST)
    header: list[str] = []
    if runtime is not None:
        cache_dir, rule, line = containers.prepare_cache(
            state_dir=STATE_DIR, name=name, model_path=model_path, backend=backend,
            runtime=runtime)
        runtime["cache_dir"] = str(cache_dir) if cache_dir else None
        runtime["cache_rule"] = rule
        if line:
            header.append(line)
    else:
        cache_dir = None
    offline = config_mod.is_offline()
    if kind == containers.DOCKER:
        from llm_runner import images
        runtime["container"] = containers.container_name(name, port)
        # The ref when it resolves, else the image ID it was loaded as from
        # an archive (looked up on every start, since a load can come later).
        found = images.local_image(runtime["image"], STATE_DIR)
        runtime["run_image"], runtime["image_id"] = found or (runtime["image"], None)
        containers.stop_container(runtime["container"], timeout=0)   # a leftover
        argv = containers.docker_argv(
            name=name, cmd=cmd, port=port, backend=backend, model_path=model_path,
            env=containers.container_env(offline, backend, env or {},
                                         with_cache=cache_dir is not None),
            runtime=runtime, models_dir=MODELS_DIR, cache_dir=cache_dir)
        popen_env = dict(os.environ)
        header.insert(0, f"llmctl: runtime=docker image={runtime['image']} "
                         f"digest={runtime.get('digest')} container={runtime['container']}"
                         + (f" image_id={runtime['image_id']}" if runtime.get("image_id") else "")
                         + (" (loaded from an archive, run by ID)"
                            if runtime["run_image"] != runtime["image"] else ""))
    else:
        argv = cmd
        extra: dict[str, str] = {}
        gpus = (runtime or {}).get("gpus")
        if gpus and gpus != "all":
            extra["CUDA_VISIBLE_DEVICES"] = gpus
        if cache_dir is not None:
            extra["VLLM_CACHE_ROOT"] = str(cache_dir / "vllm")
        popen_env = {**os.environ, **launch_env(offline, backend), **extra, **(env or {})}
        if runtime is not None:
            header.insert(0, "llmctl: runtime=host")
    fh = open(log_file, "ab", buffering=0)
    for line in header:
        fh.write((line + "\n").encode())
    if kind == containers.DOCKER:
        fh.write(("llmctl: " + " ".join(argv) + "\n").encode())
    proc = subprocess.Popen(
        argv,
        env=popen_env,
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
        backend=backend,
        env=dict(env or {}),
        group=group,
        runtime=runtime,
    )
    procs = reconcile()
    procs.append(rec)
    _save(procs)
    # Remember this launch so `restore` can bring it back after a reboot.
    autostart_add(name=name, cmd=cmd, port=port, model_path=model_path,
                  run_args=run_args, backend=backend, env=env, group=group,
                  runtime=runtime, restart=restart, restart_max=restart_max)
    return rec


def stop(identifier: str, *, timeout: float = 10.0,
         keep_autostart: bool = False) -> RunningProcess | None:
    """Stop a model. ``keep_autostart`` is for a planned restart, which
    holds the model first (:func:`hold`) so nothing takes it for a crash."""
    target = find(identifier)
    if target is None:
        return None
    # An explicit stop means "don't autostart this next boot". Done first, so
    # the gateway's supervisor never takes the dying process for a crash.
    if not keep_autostart:
        autostart_remove(target.name)
    container = (target.runtime or {}).get("container")
    if container:
        # SIGTERM to `docker run` reaches the engine too, but a SIGKILL'd
        # client would leave the container running and holding the GPU.
        from llm_runner import containers
        containers.stop_container(container, timeout=int(timeout))
    _signal_group(target.pid, signal.SIGTERM)
    deadline = time.time() + timeout
    while time.time() < deadline and _alive(target.pid):
        time.sleep(0.25)
    if _alive(target.pid):
        _signal_group(target.pid, signal.SIGKILL)
    procs = [p for p in reconcile() if p.pid != target.pid]
    _save(procs)
    return target


# ----------------------------------------------------- planned restarts --
# A model being restarted on purpose (`llmctl apply` swapping its files or
# flags) is "held": the gateway answers 503 model_unavailable for it, not
# 404, and the supervisor leaves it alone. A hold lapses by itself, so one
# left by a killed `apply` can't pin a model down for ever.
HOLD_TTL = 900.0


def _holds_file(state_dir: Path | None) -> Path:
    return (state_dir or STATE_DIR) / "holds.json"


def holds(state_dir: Path | None = None) -> dict[str, dict]:
    """Held models still in their time: ``{name: {reason, group, until}}``."""
    try:
        raw = json.loads(_holds_file(state_dir).read_text())
    except (OSError, ValueError):
        return {}
    now = time.time()
    return {n: h for n, h in raw.items() if isinstance(h, dict) and h.get("until", 0) > now}


def hold(name: str, reason: str, *, group: str | None = None,
         ttl: float = HOLD_TTL) -> None:
    held = holds()
    held[name] = {"reason": reason, "group": group or None, "until": time.time() + ttl}
    _write_holds(held)


def release(name: str) -> None:
    held = holds()
    if held.pop(name, None) is not None:
        _write_holds(held)


def _write_holds(held: dict) -> None:
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    tmp = _holds_file(None).with_suffix(".tmp")
    tmp.write_text(json.dumps(held, indent=2))
    os.replace(tmp, _holds_file(None))


# --------------------------------------------------------------- events --
def record_event(name: str, event: str, detail: str = "", *,
                 state_dir: Path | None = None, **fields) -> bool:
    """Append one line to ``<state>/events.jsonl`` (``llmctl events``).

    ``fields`` (pid, port, by, reason, log …) go in as keys; None is left
    out. The same event for the same pid twice in a row is written once:
    ``run --wait`` and the gateway's supervisor both see a model get
    ready. Returns whether a line was written."""
    path = (state_dir or STATE_DIR) / "events.jsonl"
    pid = fields.get("pid")
    if pid is not None and _last_event(path, name) == (event, pid):
        return False
    line = {"ts": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "model": name, "event": event,
            "detail": detail, **{k: v for k, v in fields.items() if v is not None}}
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a") as f:
            f.write(json.dumps(line) + "\n")
    except OSError:
        return False
    return True


def _last_event(path: Path, name: str) -> tuple[str, int] | None:
    try:
        with open(path, "rb") as f:
            f.seek(0, os.SEEK_END)
            f.seek(max(0, f.tell() - 16384))
            tail = f.read().decode(errors="replace").splitlines()
    except OSError:
        return None
    for raw in reversed(tail):
        try:
            e = json.loads(raw)
        except ValueError:
            continue
        if e.get("model") == name:
            return e.get("event"), e.get("pid")
    return None


def last_log(name: str) -> Path | None:
    """The newest log of a model called ``name``, running or not."""
    pattern = re.compile(re.escape(name) + r"-\d+-\d+\.log")
    try:
        logs = [p for p in LOG_DIR.iterdir() if pattern.fullmatch(p.name)]
    except OSError:
        return None
    return max(logs, key=lambda p: p.stat().st_mtime, default=None)


def _signal_group(pid: int, sig: int) -> None:
    try:
        os.killpg(os.getpgid(pid), sig)
    except ProcessLookupError:
        pass

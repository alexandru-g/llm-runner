"""Persisted user configuration for llm-runner.

Settings live in a JSON file under a user-level config dir so they survive
across invocations regardless of cwd. Override the location with
``LLM_RUNNER_CONFIG_DIR`` (otherwise ``$XDG_CONFIG_HOME/llm-runner`` or
``~/.config/llm-runner``).

Settings: ``models_dir`` (where GGUF / HF checkpoints live), ``vllm_bin``
(path to the ``vllm`` CLI when it isn't on PATH) and ``decider_dir`` (the
Mapika/decider clone whose venv serves decider checkpoints) and
``gateway_port`` (the all-models port started with the first model, or
``"off"``), ``bind_host`` (the address the gateway and every backend listen
on, default ``0.0.0.0``), ``offline`` (never reach Hugging Face; also
``LLM_RUNNER_OFFLINE=1``), the gateway's API key settings (``gateway_auth``,
``gateway_keys``, ``gateway_local_networks``, ``gateway_proxies`` and
``gateway_upstream_key``, see ``auth.py``), plus the cluster settings: ``node_name``, ``node_priority``,
``node_advertise``, ``registry_url``, ``cluster_token`` and
``registry_serve`` (see ``cluster.py``). Environment variables still
take precedence over persisted config (see ``registry._resolve_models_dir``
and ``backends.find_vllm``).
"""
from __future__ import annotations

import json
import os
import socket
from dataclasses import dataclass
from pathlib import Path


def _resolve_config_dir() -> Path:
    env = os.environ.get("LLM_RUNNER_CONFIG_DIR")
    if env:
        return Path(env).expanduser().resolve()
    xdg = os.environ.get("XDG_CONFIG_HOME")
    base = Path(xdg).expanduser() if xdg else Path.home() / ".config"
    return base / "llm-runner"


CONFIG_DIR = _resolve_config_dir()
CONFIG_FILE = CONFIG_DIR / "config.json"


def load() -> dict:
    """Return the persisted config dict (empty if missing or unreadable)."""
    if not CONFIG_FILE.exists():
        return {}
    try:
        data = json.loads(CONFIG_FILE.read_text())
    except (json.JSONDecodeError, OSError):
        return {}
    return data if isinstance(data, dict) else {}


def save(cfg: dict) -> None:
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    CONFIG_FILE.write_text(json.dumps(cfg, indent=2) + "\n")
    CONFIG_FILE.chmod(0o600)   # it can hold the cluster token


def get_models_dir() -> Path | None:
    """Return the persisted models dir, or None if unset."""
    val = load().get("models_dir")
    return Path(val).expanduser().resolve() if val else None


def set_models_dir(path: Path) -> Path:
    """Persist ``path`` as the models dir and return its resolved value."""
    resolved = Path(path).expanduser().resolve()
    cfg = load()
    cfg["models_dir"] = str(resolved)
    save(cfg)
    return resolved


def get_vllm_bin() -> Path | None:
    """Return the persisted path to the ``vllm`` CLI, or None if unset."""
    val = load().get("vllm_bin")
    return Path(val).expanduser() if val else None


def set_vllm_bin(path: Path | None) -> Path | None:
    """Persist ``path`` as the vllm CLI (None clears it)."""
    cfg = load()
    if path is None:
        cfg.pop("vllm_bin", None)
        save(cfg)
        return None
    resolved = Path(path).expanduser().resolve()
    cfg["vllm_bin"] = str(resolved)
    save(cfg)
    return resolved


def get_decider_dir() -> Path | None:
    """Return the persisted decider clone dir, or None if unset."""
    val = load().get("decider_dir")
    return Path(val).expanduser() if val else None


def set_decider_dir(path: Path | None) -> Path | None:
    """Persist ``path`` as the decider clone dir (None clears it)."""
    cfg = load()
    if path is None:
        cfg.pop("decider_dir", None)
        save(cfg)
        return None
    resolved = Path(path).expanduser().resolve()
    cfg["decider_dir"] = str(resolved)
    save(cfg)
    return resolved


DEFAULT_GATEWAY_PORT = 9000


def get_gateway_port() -> int | None:
    """The gateway port, or None when the gateway is turned off."""
    val = load().get("gateway_port", DEFAULT_GATEWAY_PORT)
    return None if val == "off" else int(val)


def set_gateway_port(port: int | None) -> None:
    """Persist the gateway port (None turns the gateway off)."""
    cfg = load()
    cfg["gateway_port"] = "off" if port is None else int(port)
    save(cfg)


DEFAULT_BIND_HOST = "0.0.0.0"


def get_bind_host() -> str:
    """The address the gateway and the backends listen on."""
    return load().get("bind_host") or DEFAULT_BIND_HOST


def set_bind_host(host: str | None) -> None:
    """Persist the listen address (None restores the default)."""
    update(bind_host=None if host in (None, DEFAULT_BIND_HOST) else host)


def get_upstream_key() -> str | None:
    """The API key this gateway sends when it relays a call through the
    registry's gateway (`llmctl gateway key set`)."""
    return load().get("gateway_upstream_key") or None


def is_offline() -> bool:
    """Offline mode: ``LLM_RUNNER_OFFLINE`` (1/0) if set, else the setting."""
    env = os.environ.get("LLM_RUNNER_OFFLINE", "").strip().lower()
    if env:
        return env not in ("0", "false", "no", "off")
    return load().get("offline") is True


def set_offline(on: bool) -> None:
    update(offline=True if on else None)


# ----------------------------------------------------------------- cluster --

@dataclass(frozen=True)
class Cluster:
    node_name: str
    node_priority: int
    node_advertise: str | None     # URL other nodes use to reach this gateway
    registry_url: str | None       # the registry this node reports to
    token: str | None
    registry_serve: bool           # this gateway is the registry
    spill: bool = True             # move calls off a busy instance to a free one

    @property
    def enabled(self) -> bool:
        return bool(self.registry_url or self.registry_serve)


def get_cluster() -> Cluster:
    cfg = load()
    return Cluster(
        node_name=cfg.get("node_name") or socket.gethostname(),
        node_priority=int(cfg.get("node_priority", 0)),
        node_advertise=cfg.get("node_advertise") or None,
        registry_url=cfg.get("registry_url") or None,
        token=cfg.get("cluster_token") or None,
        registry_serve=bool(cfg.get("registry_serve", False)),
        spill=cfg.get("node_spill", True) is not False,
    )


def update(**values) -> None:
    """Set (or, with None, remove) top-level settings."""
    cfg = load()
    for key, val in values.items():
        if val is None:
            cfg.pop(key, None)
        else:
            cfg[key] = val
    save(cfg)

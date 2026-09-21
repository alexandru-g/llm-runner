"""Persisted user configuration for llm-runner.

Settings live in a JSON file under a user-level config dir so they survive
across invocations regardless of cwd. Override the location with
``LLM_RUNNER_CONFIG_DIR`` (otherwise ``$XDG_CONFIG_HOME/llm-runner`` or
``~/.config/llm-runner``).

Settings: ``models_dir`` (where GGUF / HF checkpoints live), ``vllm_bin``
(path to the ``vllm`` CLI when it isn't on PATH) and ``decider_dir`` (the
Mapika/decider clone whose venv serves decider checkpoints). Environment variables still
take precedence over persisted config (see ``registry._resolve_models_dir``
and ``backends.find_vllm``).
"""
from __future__ import annotations

import json
import os
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

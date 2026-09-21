"""Disk-discovered model specs.

Models live under ``MODELS_DIR``. Override with the ``LLM_RUNNER_MODELS_DIR``
env var to point at an existing collection (e.g. /fast/ml/models), or persist
a choice with ``llmctl config models-dir <path>`` (env var still wins).

A "model" is either

* any ``*.gguf`` file under ``MODELS_DIR`` that is not an ``mmproj-*`` vision
  projector — served by ``llama-server`` (``backend="llama"``), or
* any directory under ``MODELS_DIR`` holding a Hugging Face checkpoint
  (``config.json`` + safetensors/bin weights) — served by ``vllm``
  (``backend="vllm"``), or by ``decider.serve`` when the directory also
  carries ``decider_config.json`` (``backend="decider"``).

The friendly key for a GGUF is its filename stem, lowercased; for an HF dir
it is the directory name, lowercased. If two models share a key (across
subfolders) the key is prefixed with the parent directory name to
disambiguate.

If the model's folder also contains a file matching ``*mmproj*.gguf``, it is
attached as the vision projector automatically.

Likewise, a sibling GGUF whose name marks it as a speculative-decoding drafter
(an MTP / "assistant" / draft head — see ``_DRAFT_NAME_HINTS``) is attached as
the base model's draft model instead of being listed as a model in its own
right. ``run`` then wires it up as ``--model-draft`` for faster generation.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from llm_runner import backends


def _resolve_models_dir() -> Path:
    env = os.environ.get("LLM_RUNNER_MODELS_DIR")
    if env:
        return Path(env).expanduser().resolve()
    from llm_runner import config
    configured = config.get_models_dir()
    if configured:
        return configured
    repo_local = Path(__file__).resolve().parents[2] / "models"
    if repo_local.exists():
        return repo_local
    xdg = os.environ.get("XDG_DATA_HOME")
    base = Path(xdg).expanduser() if xdg else Path.home() / ".local" / "share"
    return base / "llm-runner" / "models"


MODELS_DIR = _resolve_models_dir()


# Substrings (matched against the lowercased filename stem) that mark a GGUF as
# a speculative-decoding drafter — an MTP / "assistant" / draft head that pairs
# with a base model rather than serving on its own. Such files are attached to
# their sibling base model (like an mmproj) instead of listed as standalone
# models. Canonical Gemma-4 MTP names match (`*-MTP.gguf`, `mtp-*.gguf`); the
# `-assistant` / `-draft` / `-eagle` variants cover other drafter conventions.
_DRAFT_NAME_HINTS = ("-mtp", "mtp-", "_mtp", "-assistant", "-draft", "-eagle")


def _looks_like_draft(stem: str) -> bool:
    low = stem.lower()
    return any(hint in low for hint in _DRAFT_NAME_HINTS)


@dataclass(frozen=True)
class ModelSpec:
    key: str
    model_path: Path  # GGUF file (llama) or checkpoint directory (vllm)
    mmproj_path: Path | None
    size: int
    draft_path: Path | None = None
    backend: str = backends.LLAMA
    # HF checkpoints declare vision in config.json rather than via an mmproj.
    hf_vision: bool = False

    @property
    def has_vision(self) -> bool:
        return self.mmproj_path is not None or self.hf_vision

    @property
    def has_draft(self) -> bool:
        return self.draft_path is not None


def _find_mmproj(folder: Path) -> Path | None:
    matches = sorted(f for f in folder.glob("*.gguf") if "mmproj" in f.name.lower())
    return matches[0] if matches else None


def _find_draft(folder: Path) -> Path | None:
    matches = sorted(
        f for f in folder.glob("*.gguf")
        if "mmproj" not in f.name.lower() and _looks_like_draft(f.stem)
    )
    return matches[0] if matches else None


def _find_hf_dirs(base: Path) -> list[Path]:
    """Checkpoint directories under ``base`` (outermost only — a nested
    ``vision_tower/config.json`` inside a checkpoint is not its own model)."""
    found: list[Path] = []
    for cfg in sorted(base.rglob("config.json")):
        d = cfg.parent
        if any(part.startswith(".") for part in d.relative_to(base).parts):
            continue  # .cache/ etc.
        if any(d.is_relative_to(f) for f in found):
            continue
        if backends.is_hf_model_dir(d):
            found.append(d)
    return found


def discover_models(models_dir: Path | None = None) -> dict[str, ModelSpec]:
    """Walk MODELS_DIR and return {key: ModelSpec} for every GGUF (minus
    companions) and every HF checkpoint directory."""
    base = models_dir or MODELS_DIR
    if not base.exists():
        return {}

    candidates: list[Path] = [
        p for p in sorted(base.rglob("*.gguf"))
        if "mmproj" not in p.name.lower() and not _looks_like_draft(p.stem)
    ]
    hf_dirs = _find_hf_dirs(base)

    # Group by preferred key so collisions can be disambiguated by parent.
    by_key: dict[str, list[Path]] = {}
    for p in candidates:
        by_key.setdefault(p.stem.lower(), []).append(p)
    for d in hf_dirs:
        by_key.setdefault(d.name.lower(), []).append(d)

    specs: dict[str, ModelSpec] = {}
    for stem, paths in by_key.items():
        for p in paths:
            key = stem if len(paths) == 1 else f"{p.parent.name.lower()}/{stem}"
            if p.is_dir():
                specs[key] = ModelSpec(
                    key=key,
                    model_path=p,
                    mmproj_path=None,
                    size=backends.hf_dir_size(p),
                    backend=backends.infer_backend(p),
                    hf_vision=backends.hf_has_vision(p),
                )
            else:
                specs[key] = ModelSpec(
                    key=key,
                    model_path=p,
                    mmproj_path=_find_mmproj(p.parent),
                    size=p.stat().st_size,
                    draft_path=_find_draft(p.parent),
                )
    return dict(sorted(specs.items()))

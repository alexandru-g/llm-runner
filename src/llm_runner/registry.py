"""Disk-discovered model specs.

Models live under ``MODELS_DIR``. Override with the ``LLM_RUNNER_MODELS_DIR``
env var to point at an existing collection (e.g. /fast/ml/models), or persist
a choice with ``llmctl config models-dir <path>`` (env var still wins).

A "model" is any ``*.gguf`` file under ``MODELS_DIR`` that is not an
``mmproj-*`` vision projector. The friendly key for each model is its
filename stem, lowercased. If two files share a stem (across subfolders)
the key is prefixed with the parent directory name to disambiguate.

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
    model_path: Path
    mmproj_path: Path | None
    size: int
    draft_path: Path | None = None

    @property
    def has_vision(self) -> bool:
        return self.mmproj_path is not None

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


def discover_models(models_dir: Path | None = None) -> dict[str, ModelSpec]:
    """Walk MODELS_DIR and return {key: ModelSpec} for every non-mmproj GGUF."""
    base = models_dir or MODELS_DIR
    if not base.exists():
        return {}

    candidates: list[Path] = [
        p for p in sorted(base.rglob("*.gguf"))
        if "mmproj" not in p.name.lower() and not _looks_like_draft(p.stem)
    ]

    by_stem: dict[str, list[Path]] = {}
    for p in candidates:
        by_stem.setdefault(p.stem.lower(), []).append(p)

    specs: dict[str, ModelSpec] = {}
    for stem, paths in by_stem.items():
        for p in paths:
            key = stem if len(paths) == 1 else f"{p.parent.name.lower()}/{stem}"
            specs[key] = ModelSpec(
                key=key,
                model_path=p,
                mmproj_path=_find_mmproj(p.parent),
                size=p.stat().st_size,
                draft_path=_find_draft(p.parent),
            )
    return dict(sorted(specs.items()))

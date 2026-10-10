"""Profile registry. Every module in this package that defines a Profile
subclass with an ``id`` is picked up automatically."""
from __future__ import annotations

import importlib
import pkgutil
import sys
from pathlib import Path

from systemone_proxy.gguf import general_metadata
from systemone_proxy.profiles.base import Profile


def all_profiles() -> dict[str, type[Profile]]:
    found: dict[str, type[Profile]] = {}
    for mod in pkgutil.iter_modules(__path__):
        if mod.name == "base":
            continue
        try:
            m = importlib.import_module(f"{__name__}.{mod.name}")
        except Exception as e:  # one broken profile must not take the others down
            print(f"systemone-proxy: skipping profile module {mod.name}: {e}", file=sys.stderr)
            continue
        for obj in vars(m).values():
            if (isinstance(obj, type) and issubclass(obj, Profile) and obj is not Profile
                    and "id" in vars(obj)):
                found[obj.id] = obj
    return dict(sorted(found.items()))


def get(profile_id: str) -> type[Profile]:
    profiles = all_profiles()
    if profile_id not in profiles:
        raise KeyError(f"unknown profile {profile_id!r}; known: {', '.join(profiles)}")
    return profiles[profile_id]


def detect(gguf_path: str | Path) -> type[Profile] | None:
    """The profile whose model this GGUF is, or None."""
    meta = general_metadata(gguf_path)
    name = Path(gguf_path).name
    hits = [p for p in all_profiles().values() if p.matches(meta, name)]
    if len(hits) > 1:
        raise RuntimeError(f"{name} matches several profiles: {[p.id for p in hits]}")
    return hits[0] if hits else None

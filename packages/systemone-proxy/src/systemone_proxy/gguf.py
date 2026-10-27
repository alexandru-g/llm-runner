"""Read the leading ``general.*`` metadata block of a GGUF file.

Only the header is read (a few KB), never the tensors. Used to recognise which
decision model a llama-server is serving.
"""
from __future__ import annotations

import struct
from pathlib import Path

_SCALAR = {0: "B", 1: "b", 2: "H", 3: "h", 4: "I", 5: "i", 6: "f", 7: "?",
           10: "Q", 11: "q", 12: "d"}


def general_metadata(path: str | Path) -> dict:
    """``{"general.name": ..., "general.tags": [...], ...}``; empty when the
    file is missing or not a GGUF."""
    out: dict = {}
    try:
        with open(path, "rb") as f:
            def read(fmt: str):
                return struct.unpack(fmt, f.read(struct.calcsize(fmt)))

            def string() -> str:
                (n,) = read("<Q")
                return f.read(n).decode("utf-8", "replace")

            def value(t: int):
                if t == 8:
                    return string()
                if t == 9:
                    et, n = read("<IQ")
                    return [value(et) for _ in range(n)]
                return read("<" + _SCALAR[t])[0]

            if f.read(4) != b"GGUF":
                return {}
            (version,) = read("<I")
            if version < 2:
                return {}
            _, n_kv = read("<QQ")
            for _ in range(n_kv):
                key = string()
                (t,) = read("<I")
                if not key.startswith("general."):
                    break  # writers emit general.* first
                out[key] = value(t)
    except (OSError, struct.error, KeyError, ValueError):
        pass
    return out


_SIZE = {k: struct.calcsize("<" + v) for k, v in _SCALAR.items()}


def metadata(path: str | Path, want=lambda key: True) -> dict:
    """Every header key ``want(key)`` accepts, reading the whole key-value
    block. Arrays of keys not wanted are skipped, not decoded (the
    tokenizer's vocabulary alone is ~10 MB on Qwen3.5)."""
    out: dict = {}
    try:
        with open(path, "rb") as f:
            def read(fmt: str):
                return struct.unpack(fmt, f.read(struct.calcsize(fmt)))

            def string() -> str:
                (n,) = read("<Q")
                return f.read(n).decode("utf-8", "replace")

            def value(t: int, keep: bool):
                if t == 8:
                    if keep:
                        return string()
                    (n,) = read("<Q")
                    f.seek(n, 1)
                    return None
                if t == 9:
                    et, n = read("<IQ")
                    if not keep and et in _SIZE:
                        f.seek(_SIZE[et] * n, 1)
                        return None
                    items = [value(et, keep) for _ in range(n)]
                    return items if keep else None
                return read("<" + _SCALAR[t])[0]

            if f.read(4) != b"GGUF":
                return {}
            (version,) = read("<I")
            if version < 2:
                return {}
            _, n_kv = read("<QQ")
            for _ in range(n_kv):
                key = string()
                (t,) = read("<I")
                keep = want(key)
                v = value(t, keep)
                if keep:
                    out[key] = v
    except (OSError, struct.error, KeyError, ValueError):
        pass
    return out


def decision_metadata(path: str | Path) -> dict:
    """``general.*`` plus the llama.cpp decision keys (``<arch>.decision.*``,
    ``<arch>.attention.causal``); ``decision.type`` is absent for a GGUF that
    llama-server can't serve natively."""
    return metadata(path, lambda k: k.startswith("general.") or ".decision." in k
                    or k.endswith(".attention.causal"))

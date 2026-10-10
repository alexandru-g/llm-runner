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

"""The vendor's own tokenizer, loaded from its ``tokenizer.json``.

Prompts are tokenized here, not by llama-server, and sent as token-id arrays:
several vendors tokenize their prompt piecewise, which gives different ids
from tokenizing the finished string. The profile's self-test checks at startup
that the GGUF behind llama-server agrees with this tokenizer.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class TokenizerSource:
    repo: str
    revision: str               # pinned commit
    sha256: str                 # of tokenizer.json, checked after download
    filename: str = "tokenizer.json"


# One byte-identical tokenizer.json is shipped by every Gemma-4-based decision
# model we support (Quyet-1.0-Large, Rune v3, Xor, Bobcat) and by Google's
# gemma-4-26B-A4B-it. Juspay's copy is public (Google's and Rune's are gated).
GEMMA4 = TokenizerSource(
    repo="juspay/Xor-26B-A4B",
    revision="8259b45a58286fd58ed3b9dacc462360eba66d5b",
    sha256="cc8d3a0ce36466ccc1278bf987df5f71db1719b9ca6b4118264f45cb627bfe0f",
)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch(source: TokenizerSource) -> Path:
    """Download (or reuse from the HF cache) the pinned tokenizer.json and
    verify its hash."""
    from huggingface_hub import hf_hub_download
    path = Path(hf_hub_download(source.repo, source.filename, revision=source.revision))
    got = _sha256(path)
    if got != source.sha256:
        raise RuntimeError(f"{source.repo}@{source.revision[:8]}/{source.filename}: sha256 {got} "
                           f"does not match the pinned {source.sha256}")
    return path


class Tok:
    """``encode(text, special=...)`` with special-token parsing on or off.

    ``special=True`` matches the vendors' default (``tokenizer(text,
    add_special_tokens=False)``: special-token text like ``<|turn>`` becomes the
    special id). ``special=False`` treats such text as plain characters.
    Two instances keep this thread-safe."""

    def __init__(self, path: str | Path):
        from tokenizers import Tokenizer
        self.path = Path(path)
        self._special = Tokenizer.from_file(str(path))
        self._plain = Tokenizer.from_file(str(path))
        self._plain.encode_special_tokens = True

    def encode(self, text: str, *, special: bool = True) -> list[int]:
        t = self._special if special else self._plain
        return t.encode(text, add_special_tokens=False).ids

    def decode(self, ids: list[int]) -> str:
        return self._special.decode(ids, skip_special_tokens=False)

    def token_to_id(self, token: str) -> int | None:
        return self._special.token_to_id(token)

    def id_to_token(self, i: int) -> str | None:
        return self._special.id_to_token(i)


def load(source: TokenizerSource, override: str | Path | None = None) -> Tok:
    return Tok(override if override else fetch(source))

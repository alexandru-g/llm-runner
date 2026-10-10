"""The profile interface: everything model-specific lives in a Profile.

A profile turns a parsed :class:`~systemone_proxy.questions.Request` into
backend rows (token-id prompts plus the label token ids to read after them),
and turns each question's label logprobs back into a unified answer.

Unified answer shapes (every profile returns exactly these keys, plus the
optional extras listed in its ``spec()``):

* choice: ``{"type": "choice", "choice": label, "confidence": float,
  "probabilities": {label: p}}``
* score: ``{"type": "score", "score": float, "confidence": float,
  "legend": {"0": text, ...}, "probabilities": {"0": p, ...}}``
* noul: ``{"type": "noul", "noul": P(true), "confidence": float}``

Probabilities are the vendor's own (same prompt, temperature and arithmetic);
nothing is rounded.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, ClassVar

from systemone_proxy.questions import Limits, Question, Request
from systemone_proxy.tokenizer import Tok, TokenizerSource


@dataclass
class Row:
    """One backend call: read ``label_ids`` right after the prompt ``ids``.
    ``tag`` tells rows of the same question apart (e.g. "forward"/"reverse")."""
    qid: str
    ids: list[int]
    label_ids: list[int]
    tag: str = "main"


@dataclass
class Built:
    rows: list[Row]
    warnings: list[dict] = field(default_factory=list)
    truncated: set[str] = field(default_factory=set)
    # None → the service reports the sum of all row prompt lengths.
    input_tokens: int | None = None


class Profile:
    id: ClassVar[str]                  # stable id, e.g. "rune-26b-a4b-v3"
    name: ClassVar[str]                # human name
    upstream: ClassVar[str]            # where the protocol comes from (repo/code)
    gguf_repos: ClassVar[tuple[str, ...]] = ()   # known GGUF repos, for docs
    tokenizer_source: ClassVar[TokenizerSource]
    temperatures: ClassVar[dict[str, float]]     # by question type
    limits: ClassVar[Limits] = Limits()

    def __init__(self, tok: Tok):
        self.tok = tok
        # llama-server's context size, set by the service before the first
        # build(); profiles whose vendor degrades gracefully near the limit
        # (e.g. drops an optional row) can use it. None in offline tests.
        self.n_ctx: int | None = None

    # -- detection -------------------------------------------------------
    @classmethod
    def matches(cls, meta: dict, filename: str) -> bool:
        """``meta`` is the GGUF ``general.*`` metadata, ``filename`` its base
        name."""
        raise NotImplementedError

    # -- request -> rows -------------------------------------------------
    def build(self, req: Request) -> Built:
        raise NotImplementedError

    # -- rows -> answer --------------------------------------------------
    def answer(self, q: Question, logprobs: dict[str, list[float]]) -> dict:
        """``logprobs[tag]`` is aligned with that row's ``label_ids``."""
        raise NotImplementedError

    # -- startup ---------------------------------------------------------
    def selftest_texts(self) -> list[str]:
        """Strings whose tokenization must be identical in the vendor
        tokenizer and the GGUF (special tokens parsed). Typically a full
        golden prompt."""
        return []

    def selftest_labels(self) -> dict[int, str]:
        """``{token_id: text}`` the GGUF must detokenize each label id to."""
        return {}

    def selftest(self, backend) -> list[str]:
        problems = []
        for text in self.selftest_texts():
            mine = self.tok.encode(text, special=True)
            theirs = backend.tokenize(text, parse_special=True)
            if mine != theirs:
                at = next((i for i, (a, b) in enumerate(zip(mine, theirs)) if a != b),
                          min(len(mine), len(theirs)))
                problems.append(f"tokenizer mismatch at token {at}: vendor "
                                f"{mine[at:at + 5]} vs llama-server {theirs[at:at + 5]} "
                                f"({len(mine)} vs {len(theirs)} tokens)")
        labels = self.selftest_labels()
        if labels:
            ids = list(labels)
            got = [backend.detokenize([i]) for i in ids]
            bad = [(i, labels[i], g) for i, g in zip(ids, got) if g != labels[i]]
            if bad:
                problems.append("label ids decode differently in the GGUF: "
                                + ", ".join(f"{i}: want {w!r} got {g!r}" for i, w, g in bad[:5]))
        return problems

    # -- documentation ---------------------------------------------------
    def spec(self) -> dict:
        """Model-specific part of ``systemone-proxy spec`` / ``GET /v1/spec``.
        Keys: confidence (by type), labels, noul_order, state, limits, calls,
        extras, notes."""
        raise NotImplementedError


# ------------------------------------------------------------- helpers --

def py_json(value: Any, *, compact: bool = False, ensure_ascii: bool = False,
            sort_keys: bool = False, indent: int | None = None) -> str:
    """Python's json.dumps, the serialiser every vendor uses (float repr,
    escapes and key order matter for the prompt bytes)."""
    seps = (",", ":") if compact else None
    return json.dumps(value, ensure_ascii=ensure_ascii, separators=seps,
                      sort_keys=sort_keys, indent=indent)


def gemma4_chat(system: str | None, user: str) -> str:
    """Gemma 4's chat template (shared byte-for-byte by Quyet-1.0-Large, Rune
    v3, Xor and Bobcat) rendered for ``[system?, user]`` with
    ``add_generation_prompt=True, enable_thinking=False``. The answer slot is
    the token right after the returned text. Verified against the jinja
    template in tests/test_gemma4_template.py."""
    out = "<bos>"
    if system is not None:
        out += "<|turn>system\n" + system.strip() + "<turn|>\n"
    out += "<|turn>user\n" + user.strip() + "<turn|>\n"
    return out + "<|turn>model\n<|channel>thought\n<channel|>"

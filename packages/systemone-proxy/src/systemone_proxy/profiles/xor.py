"""Xor 26B-A4B v1.1 (juspay), a post-trained Gemma 4 26B-A4B.

Source: ``serving/xn_base_server.py`` in juspay/Xor-26B-A4B v1.1 (sha256
2eed5693…, the frozen wrapper the release was benchmarked with) plus the
production settings in ``serving/docker-compose.yml`` / ``RELEASE_MANIFEST.json``:
temperatures choice 2.0, noul 1.7, score 1.0 and two-order scoring on.

One row per question, the state first so rows share a prefix::

    <bos><|turn>user\\nContext:\\n<state>\\n\\nQuestion: <instructions>\\nOptions:
    (A) <opt>\\n(B) <opt>…<turn|>\\n<|turn>model\\n<|channel>thought\\n<channel|>Answer: (

The answer is the next-token distribution over the label tokens after
``Answer: (``. The wrapper tokenizes piece by piece (head, ``Context:\\n``,
state, question, each option line, tail, ``Answer: (``) and concatenates the
ids. That is not always the same as tokenizing the whole string (a state that
ends in ``\\n`` merges differently), so the pieces are copied exactly.
"""
from __future__ import annotations

import hashlib
import json
import math
import string

from systemone_proxy.errors import too_long
from systemone_proxy.profiles.base import Built, Profile, Row
from systemone_proxy.questions import Limits, Question, Request
from systemone_proxy.readout import argmax, binary_confidence, expected_index
from systemone_proxy.tokenizer import GEMMA4

# xn_base_server.py: the chat template rendered around a sentinel user turn
# (add_generation_prompt=True, enable_thinking=False), split at the sentinel.
HEAD_TEXT = "<bos><|turn>user\n"
TAIL_TEXT = "<turn|>\n<|turn>model\n<|channel>thought\n<channel|>"
ANSWER_TEXT = "Answer: ("
HEAD_IDS = [2, 105, 2364, 107]
TAIL_IDS = [106, 107, 105, 4368, 107, 100, 45518, 107, 101]
ANSWER_IDS = [7925, 236787, 568]

MAX_OPTIONS = 255
MAX_CONTEXT = 131072          # XN_MAX_CONTEXT: a row needs len + 1 <= this
NOUL_INSTRUCTIONS = "Which answer fits the context?"   # only used for empty instructions
# RELEASE_MANIFEST.json "label_ids_sha256": sha256(json.dumps(LABEL_IDS))
LABEL_IDS_SHA256 = "239b0c213883a3ccf3c0effc7b398e66cd755d5765233128b1ed15a2ec5f7c1b"


def txt(value) -> str:
    """xn_base_server._txt: strings as-is, anything else as JSON with spaced
    separators."""
    return value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)


def render_state(state) -> str:
    """xn_base_server.render_state: strings as-is, anything else compact JSON."""
    return state if isinstance(state, str) else json.dumps(state, ensure_ascii=False,
                                                           separators=(",", ":"))


def _empty(value) -> bool:
    # the wrapper's ``value in (None, "")``
    return value is None or (isinstance(value, str) and value == "")


def softmax_T(lp: list[float], T: float) -> list[float]:
    """xn_base_server.softmax_T, operation for operation."""
    m = max(lp)
    e = [math.exp((x - m) / T) for x in lp]
    s = sum(e)
    return [x / s for x in e]


def option_texts(q: Question) -> list[str]:
    """Option lines in the wrapper's label order (noul: A = no, B = yes)."""
    if q.type == "choice":
        return [o.key if _empty(o.desc) else f"{o.key}: {txt(o.desc)}" for o in q.options]
    if q.type == "score":
        return [f"{i}: {txt(o.desc)}" for i, o in enumerate(q.options)]
    d_true, d_false = q.options[0].desc, q.options[1].desc
    return ["no" if _empty(d_false) else f"no: {txt(d_false)}",
            "yes" if _empty(d_true) else f"yes: {txt(d_true)}"]


def instructions(q: Question) -> str:
    ins = q.instructions
    if q.type == "noul" and _empty(ins):
        ins = NOUL_INSTRUCTIONS
    return txt(ins)


class Xor(Profile):
    id = "xor-26b-a4b"
    name = "Xor 26B-A4B v1.1"
    upstream = ("https://huggingface.co/juspay/Xor-26B-A4B (tag v1.1, serving/xn_base_server.py, "
                "docker-compose.yml)")
    gguf_repos = ("mradermacher/Xor-26B-A4B-GGUF", "mradermacher/Xor-26B-A4B-i1-GGUF")
    tokenizer_source = GEMMA4
    temperatures = {"choice": 2.0, "noul": 1.7, "score": 1.0}
    limits = Limits(max_choice=MAX_OPTIONS, max_score=10)

    @classmethod
    def matches(cls, meta: dict, filename: str) -> bool:
        base = str(meta.get("general.basename", "")).lower()
        size = str(meta.get("general.size_label", "")).lower()
        arch = str(meta.get("general.architecture", "gemma4")).lower()
        return base == "xor" and size == "26b-a4b" and arch == "gemma4"

    def __init__(self, tok):
        super().__init__(tok)
        E = self.E
        for text, want in ((HEAD_TEXT, HEAD_IDS), (TAIL_TEXT, TAIL_IDS), (ANSWER_TEXT, ANSWER_IDS)):
            if E(text) != want:
                raise RuntimeError(f"{text!r} tokenizes to {E(text)}, expected {want}")
        self.label_names, self.label_ids = self._labels()
        digest = hashlib.sha256(json.dumps(self.label_ids).encode()).hexdigest()
        if digest != LABEL_IDS_SHA256:
            raise RuntimeError(f"label ids hash {digest} != release manifest {LABEL_IDS_SHA256}")

    def E(self, text: str) -> list[int]:
        """``tok.encode(text, add_special_tokens=False)``: special-token text is
        parsed, as in the wrapper."""
        return self.tok.encode(text, special=True)

    def _labels(self) -> tuple[list[str], list[int]]:
        """xn_base_server._labels: A..Z then upper-case pairs; the label id is
        the actual next token of ``Answer: (<name>``; names that merge into
        the prefix or repeat an id are skipped (GZ). 255 labels."""
        names = list(string.ascii_uppercase) + [a + b for a in string.ascii_uppercase
                                                for b in string.ascii_uppercase]
        out, seen = [], set()
        for n in names:
            t = self.E(ANSWER_TEXT + n)
            if len(t) <= len(ANSWER_IDS) or t[-1] in seen:
                continue
            seen.add(t[-1])
            out.append((n, t[-1]))
            if len(out) == MAX_OPTIONS:
                break
        if len(out) != MAX_OPTIONS:
            raise RuntimeError(f"only {len(out)} usable labels")
        return [n for n, _ in out], [i for _, i in out]

    # xn_base_server.build_row
    def row_ids(self, state_ids: list[int], ins: str, opts: list[str]) -> list[int]:
        E = self.E
        ids = HEAD_IDS + E("Context:\n") + state_ids + E(f"\n\nQuestion: {ins}\nOptions:")
        for j, o in enumerate(opts):
            ids += E("\n(" + self.label_names[j] + ") " + o)
        return ids + TAIL_IDS + ANSWER_IDS

    def row_text(self, state_text: str, ins: str, opts: list[str]) -> str:
        """The same row as one string (for docs and the tokenizer self-test)."""
        lines = "".join(f"\n({self.label_names[j]}) {o}" for j, o in enumerate(opts))
        return (HEAD_TEXT + "Context:\n" + state_text + f"\n\nQuestion: {ins}\nOptions:" + lines
                + TAIL_TEXT + ANSWER_TEXT)

    def build(self, req: Request) -> Built:
        state_ids = self.E(render_state(req.state))
        # The vendor's own limit, or the server's context when that is smaller.
        limit = min(MAX_CONTEXT, self.n_ctx) if self.n_ctx else MAX_CONTEXT
        built = Built(rows=[])
        forward_tokens = 0
        for q in req.questions:
            ins, opts = instructions(q), option_texts(q)
            labels = self.label_ids[:len(opts)]
            ids = self.row_ids(state_ids, ins, opts)
            if len(ids) + 1 > limit:
                raise too_long(f"question {q.qid!r}: prompt of {len(ids)} tokens exceeds the "
                               f"context length of {limit}")
            forward_tokens += len(ids)
            if q.type != "choice":
                built.rows.append(Row(q.qid, ids, labels, "main"))
                continue
            built.rows.append(Row(q.qid, ids, labels, "forward"))
            rev = self.row_ids(state_ids, ins, opts[::-1])
            if len(rev) + 1 > limit:
                built.warnings.append({"code": "reverse_order_skipped", "question": q.qid,
                                       "prompt_tokens": len(rev), "max_context": limit})
                continue
            built.rows.append(Row(q.qid, rev, labels, "reverse"))
        built.input_tokens = forward_tokens
        return built

    def answer(self, q: Question, logprobs: dict[str, list[float]]) -> dict:
        keys = q.keys
        if q.type == "choice":
            lp = list(logprobs["forward"])
            rev = logprobs.get("reverse")
            K = len(lp)
            if rev is not None and K >= 2 and len(rev) == K:
                # two-order averaging in the original index space, then the
                # merged probabilities re-enter as pseudo-logprobs
                p_f = softmax_T(lp, 1.0)
                p_r = softmax_T(rev, 1.0)
                pa = [(p_f[j] + p_r[K - 1 - j]) / 2 for j in range(K)]
                lp = [math.log(max(p, 1e-12)) for p in pa]
            p = softmax_T(lp, self.temperatures["choice"])
            i = argmax(p)
            return {"type": "choice", "choice": keys[i], "confidence": p[i],
                    "probabilities": dict(zip(keys, p))}
        p = softmax_T(logprobs["main"], self.temperatures[q.type])
        if q.type == "score":
            return {"type": "score", "score": expected_index(p), "confidence": max(p),
                    "legend": {k: txt(o.desc) for k, o in zip(keys, q.options)},
                    "probabilities": dict(zip(keys, p))}
        p_true = p[1]                                   # A = no, B = yes
        return {"type": "noul", "noul": p_true, "confidence": binary_confidence(p_true)}

    def selftest_texts(self) -> list[str]:
        # README example (noul) and a JSON-state choice; for both, the whole
        # string tokenizes exactly like the wrapper's pieces.
        return [
            self.row_text("The customer was charged twice.", "Is this a billing issue?",
                          ["no", "yes"]),
            self.row_text('{"message":"Karte verloren – bitte sperren.","amount":12.5}',
                          "What does the customer want?",
                          ["cancel: close the card", "limit: change the limit", "other"]),
        ]

    def selftest_labels(self) -> dict[int, str]:
        idx = list(range(6)) + [26, 27, 254]           # A..F, AA, AB, IV
        return {self.label_ids[i]: self.label_names[i] for i in idx}

    def spec(self) -> dict:
        return {
            "confidence": {"choice": "max probability (after two-order averaging and T=2.0)",
                           "score": "max probability", "noul": "max(p, 1 - p)"},
            "labels": ("A..Z, then upper-case pairs AA, AB, … (GZ skipped), one per option in the "
                       "order given; up to 255. Score levels are also lettered A.. and the option "
                       "line reads '(A) 0: <level>'."),
            "noul_order": "A = no (criteria.false), B = yes (criteria.true)",
            "state": ("strings verbatim, objects/arrays as compact JSON (no spaces, non-ASCII kept). "
                      "Never truncated: a prompt over 131072 tokens (or over the server's context) "
                      "is rejected with input_too_long."),
            "limits": {"options": "choice 2..255, score 2..10",
                       "prompt_tokens": MAX_CONTEXT - 1},
            "calls": ("one llama-server call per noul/score question and two per choice question "
                      "(options in request order and reversed)"),
            "extras": {},
            "notes": [
                "Prompt, labels, temperatures (choice 2.0, noul 1.7, score 1.0) and two-order "
                "choice scoring are the released v1.1 serving configuration.",
                "Choice: the forward and reversed-order distributions (at T=1) are averaged per "
                "option, and the average is sharpened/flattened with T=2.0 (p ∝ avg^0.5). If only "
                "the reversed prompt is too long, the choice is scored forward-only and a "
                "reverse_order_skipped warning is added.",
                "Option descriptions and score levels that are not strings are rendered as JSON "
                "text with spaced separators; the score legend holds that text.",
                "usage.input_tokens counts only the forward prompts, as the vendor does.",
                "Text only: the vendor's image and video inputs are not supported.",
            ],
        }

"""Quyet-1.0-Large (chinhnc), Gemma-4-31B-it fine-tune, prompt version 2.

Source: the ``quyet`` 1.0.2 package (``quyet/llm/prompt.py``, ``runtime.py``,
``questions.py``, ``answers.py``, ``calibration.py``) and the model's
``quyet_config.json``. One prompt per question; the answer is the next-token
distribution over the letters A.. after the generation prompt.
"""
from __future__ import annotations

import json

from systemone_proxy.errors import invalid, too_long
from systemone_proxy.profiles.base import Built, Profile, Row, gemma4_chat
from systemone_proxy.questions import Limits, Option, Question, Request
from systemone_proxy.readout import argmax, binary_confidence, expected_index, softmax_t
from systemone_proxy.tokenizer import GEMMA4

LETTERS = "ABCDEFGHIJ"
KIND = {
    "choice": "Choose the option that fits best.",
    "score": "Choose the level that fits best (levels are ordered from lowest to highest).",
    "noul": "Choose A if the statement is true for this state, B if it is not.",
}
# quyet_config.json of chinhnc/Quyet-1.0-Large
MAX_STATE_TOKENS = 6000
MAX_PROMPT_TOKENS = 8000
MIN_STATE_TOKENS = 256


def describe(value) -> str | None:
    """quyet.questions.describe: falsy → no description; strings as-is; other
    JSON as text with spaced separators."""
    if not value:
        return None
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, separators=(", ", ": "), default=str)


class QuyetLarge(Profile):
    id = "quyet-1.0-large"
    name = "Quyet-1.0-Large"
    upstream = "https://huggingface.co/chinhnc/Quyet-1.0-Large (pip package quyet 1.0.2)"
    gguf_repos = ("mradermacher/Quyet-1.0-Large-GGUF",)
    tokenizer_source = GEMMA4
    temperatures = {"choice": 1.3007, "score": 1.3159, "noul": 1.4957}
    limits = Limits(max_choice=10, max_score=10)

    @classmethod
    def matches(cls, meta: dict, filename: str) -> bool:
        base = str(meta.get("general.basename", "")).lower()
        size = str(meta.get("general.size_label", "")).lower()
        return base == "quyet-1.0" and size == "large"

    def __init__(self, tok):
        super().__init__(tok)
        self.letter_ids = []
        for letter in LETTERS:
            ids = tok.encode(letter, special=True)
            if len(ids) != 1:
                raise RuntimeError(f"letter {letter} is not a single token: {ids}")
            self.letter_ids.append(ids[0])

    # quyet.llm.prompt.option_texts
    @staticmethod
    def option_texts(q: Question) -> list[str]:
        texts = [describe(o.desc) for o in q.options]
        if q.type == "noul":
            return [texts[0] or "true", texts[1] or "false"]
        if q.type == "score":
            return [t if t is not None else str(i) for i, t in enumerate(texts)]
        return [t if t is not None else o.key for o, t in zip(q.options, texts)]

    @staticmethod
    def user_text(state_text: str, q: Question) -> str:
        opts = "\n".join(f"{LETTERS[i]}. {t}" for i, t in enumerate(QuyetLarge.option_texts(q)))
        return f"State:\n{state_text}\n\nQuestion: {q.instructions}\n{KIND[q.type]}\n\nOptions:\n{opts}"

    def prompt_ids(self, state_text: str, q: Question) -> list[int]:
        return self.tok.encode(gemma4_chat(None, self.user_text(state_text, q)), special=True)

    def _truncate(self, state, cap: int) -> tuple[str, int, bool]:
        """quyet.llm.prompt.truncate_state with compact JSON (prompt v2)."""
        s = state if isinstance(state, str) else json.dumps(state, ensure_ascii=False,
                                                            separators=(",", ":"))
        ids = self.tok.encode(s, special=True)
        if len(ids) <= cap:
            return s, len(ids), False
        if isinstance(state, list):
            return "… " + self.tok.decode(ids[-cap:]), len(ids), True
        return self.tok.decode(ids[:cap]) + " …", len(ids), True

    def build(self, req: Request) -> Built:
        # quyet.llm.runtime.LLMModel._prompt_ids: shrink the state budget until
        # the longest question prompt fits max_prompt_tokens.
        cap = MAX_STATE_TOKENS
        for _ in range(4):
            state_text, n_state, truncated = self._truncate(req.state, cap)
            prompts = [self.prompt_ids(state_text, q) for q in req.questions]
            over = max(len(p) for p in prompts) - MAX_PROMPT_TOKENS
            if over <= 0:
                break
            cap = min(cap, n_state) - over - 16
            if cap < MIN_STATE_TOKENS:
                raise too_long("question and option text leave no room for the state: a prompt "
                               f"would exceed {MAX_PROMPT_TOKENS} tokens")
        else:
            raise too_long(f"prompt exceeds {MAX_PROMPT_TOKENS} tokens")
        built = Built(rows=[Row(q.qid, p, self.letter_ids[:len(q.options)])
                            for q, p in zip(req.questions, prompts)])
        if truncated:
            built.truncated = {q.qid for q in req.questions}
            built.warnings.append({"code": "state_truncated", "state_tokens": n_state,
                                   "kept_tokens": cap,
                                   "kept": "tail" if isinstance(req.state, list) else "head"})
        return built

    def answer(self, q: Question, logprobs: dict[str, list[float]]) -> dict:
        p = softmax_t(logprobs["main"], self.temperatures[q.type])
        keys = q.keys
        if q.type == "choice":
            i = argmax(p)
            return {"type": "choice", "choice": keys[i], "confidence": p[i],
                    "probabilities": dict(zip(keys, p))}
        if q.type == "score":
            texts = [describe(o.desc) for o in q.options]
            return {"type": "score", "score": expected_index(p), "confidence": max(p),
                    "legend": {k: (t if t is not None else k) for k, t in zip(keys, texts)},
                    "probabilities": dict(zip(keys, p))}
        return {"type": "noul", "noul": p[0], "confidence": binary_confidence(p[0])}

    def selftest_texts(self) -> list[str]:
        q = Question("q", "noul", "The request is urgent.",
                     (Option("true", None), Option("false", None)))
        return [gemma4_chat(None, self.user_text('{"message":"Thẻ bị mất, khóa giúp tôi."}', q))]

    def selftest_labels(self) -> dict[int, str]:
        return dict(zip(self.letter_ids, LETTERS))

    def spec(self) -> dict:
        return {
            "confidence": {"choice": "max probability", "score": "max probability",
                           "noul": "max(p, 1 - p)"},
            "labels": "letters A..J, one per option, in the order given",
            "noul_order": "A = true, B = false",
            "state": (f"strings as-is, other JSON as compact JSON. Over {MAX_STATE_TOKENS} tokens "
                      "(or when a prompt would exceed 8000) the state is TRUNCATED, not rejected: "
                      "arrays keep their tail, everything else its head; answers then carry "
                      "\"truncated\": true and a state_truncated warning is added."),
            "limits": {"options": "2..10 for choice and score", "prompt_tokens": MAX_PROMPT_TOKENS},
            "calls": "one llama-server call per question",
            "extras": {},
            "notes": [
                "Temperatures are per question type, from quyet_config.json (prompt version 2).",
                "Option descriptions that are not strings are rendered as JSON text; empty "
                "descriptions fall back to the label (choice), the level index (score) or "
                "true/false (noul).",
            ],
        }

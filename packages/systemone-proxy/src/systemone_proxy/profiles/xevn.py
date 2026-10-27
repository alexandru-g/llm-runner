"""Xevn-4B (quevirus), a Qwen3.5-4B full fine-tune for Korean decisions.

Source: ``xevn_infer.py`` and ``temps.json`` of quevirus/Xevn-4B at
ba11f0c2d8100a1e8b7789a1174ddee7c24dcab0 (identical in the GGUF and MLX
repos; ``xevn.json`` says ``prompt_version: 2``). There is no server, only a
library; this profile ports its ``build``, ``prompt_parts`` and ``readout``.

No chat template, no system prompt, no BOS. Each question is two pieces,
tokenized separately and concatenated, as in training::

    "{state}\\n\\n"
    "{instructions}{hint}\\n{A. key: desc ...}\\nAnswer:"

The answer slot is the token after ``Answer:``, read as ``" A"`` .. ``" Z"``.
The vendor's Transformers helper parses special-token text in the pieces
(``tokenizer.encode``, ``split_special_tokens: false``); its GGUF helper
doesn't. We follow the Transformers helper, the 16-bit reference.
"""
from __future__ import annotations

import math

from systemone_proxy.profiles.base import Built, Profile, Row, py_json
from systemone_proxy.questions import Limits, Option, Question, Request
from systemone_proxy.readout import argmax, binary_confidence, expected_index, normalized_confidence
from systemone_proxy.tokenizer import TokenizerSource

LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
SCORE_HINT = " (levels in ascending order)"
CLIP = 1e-12
NOUL_LOW, NOUL_HIGH = 0.2, 0.8

XEVN_TOKENIZER = TokenizerSource(
    repo="quevirus/Xevn-4B",
    revision="ba11f0c2d8100a1e8b7789a1174ddee7c24dcab0",
    sha256="83e6fcc1562987d9c67e21a2692af3b7d56cf9c8546d22315eb9de9b8c823110",
)


def to_text(value) -> str:
    """xevn_infer.to_text: strings as is, anything else indent-1 JSON with
    non-ASCII kept."""
    return value if isinstance(value, str) else py_json(value, indent=1)


def options_of(q: Question) -> list[tuple[str, str]]:
    """xevn_infer.build: ``(key, description)`` per option, in option order."""
    if q.type == "noul":
        t, f = q.options[0].desc, q.options[1].desc
        if t is not None and f is not None:      # both given, or neither is used
            return [("yes", "Yes: " + to_text(t)), ("no", "No: " + to_text(f))]
        return [("yes", "Yes"), ("no", "No")]
    if q.type == "score":
        return [(str(i), to_text(o.desc)) for i, o in enumerate(q.options)]
    return [(o.key, to_text(o.desc) if o.desc not in (None, "") else o.key) for o in q.options]


def prompt_parts(state_text: str, q: Question) -> tuple[str, str]:
    """xevn_infer.prompt_parts."""
    opts = "\n".join(f"{LETTERS[i]}. {d}" if d == k or q.type == "noul" else f"{LETTERS[i]}. {k}: {d}"
                     for i, (k, d) in enumerate(options_of(q)))
    hint = SCORE_HINT if q.type == "score" else ""
    return f"{state_text}\n\n", f"{to_text(q.instructions)}{hint}\n{opts}\nAnswer:"


def temper(logits: list[float], t: float) -> list[float]:
    """xevn_infer.readout: softmax at T=1, clip to [1e-12, 1], raise to 1/T,
    renormalise. Equals softmax(logit / T) unless a label is below 1e-12."""
    peak = max(logits)
    p = [math.exp(float(v) - peak) for v in logits]
    total = sum(p)
    p = [max(CLIP, min(1.0, v / total)) ** (1 / t) for v in p]
    total = sum(p)
    return [v / total for v in p]


def clamp_noul(p_yes: float) -> float:
    """xevn_infer.readout: a P(yes) strictly between 0.2 and 0.8 goes to 0.2
    (below 0.5) or 0.8 (0.5 and above)."""
    if NOUL_LOW < p_yes < NOUL_HIGH:
        return NOUL_LOW if p_yes < 0.5 else NOUL_HIGH
    return p_yes


class Xevn4B(Profile):
    id = "xevn-4b"
    name = "Xevn-4B"
    upstream = "https://huggingface.co/quevirus/Xevn-4B (xevn_infer.py, temps.json)"
    gguf_repos = ("quevirus/Xevn-4B-GGUF",)
    tokenizer_source = XEVN_TOKENIZER
    temperatures = {"choice": 1.1, "noul": 1.2, "score": 1.2}
    limits = Limits(max_choice=len(LETTERS), max_score=10)

    @classmethod
    def matches(cls, meta: dict, filename: str) -> bool:
        arch = str(meta.get("general.architecture", "")).lower()
        base = str(meta.get("general.basename", "")).lower()
        size = str(meta.get("general.size_label", "")).lower()
        if arch != "qwen35":
            return False
        if base == "xevn" and size == "4b":
            return True
        return not base and filename.lower().startswith("xevn-4b")

    def __init__(self, tok):
        super().__init__(tok)
        # xevn_infer.letter_ids: " A".."Z" encoded alone, one token each
        self.label_ids: list[int] = []
        for letter in LETTERS:
            ids = tok.encode(" " + letter, special=True)
            if len(ids) != 1:
                raise RuntimeError(f"label ' {letter}' is not one token: {ids}")
            self.label_ids.append(ids[0])
        if len(set(self.label_ids)) != len(LETTERS):
            raise RuntimeError("label tokens are not distinct")

    # -- prompt --------------------------------------------------------------
    def question_ids(self, state_text: str, q: Question) -> list[int]:
        a, b = prompt_parts(state_text, q)
        return self.tok.encode(a, special=True) + self.tok.encode(b, special=True)

    def build(self, req: Request) -> Built:
        # The vendor never truncates: the service refuses a prompt the
        # context can't hold (input_too_long).
        state_text = to_text(req.state)
        return Built(rows=[Row(q.qid, self.question_ids(state_text, q),
                               self.label_ids[:len(q.options)]) for q in req.questions])

    # -- answer ----------------------------------------------------------------
    def answer(self, q: Question, logprobs: dict[str, list[float]]) -> dict:
        p = temper(logprobs["main"], self.temperatures[q.type])
        keys = q.keys
        if q.type == "choice":
            return {"type": "choice", "choice": keys[argmax(p)],
                    "confidence": normalized_confidence(p), "probabilities": dict(zip(keys, p))}
        if q.type == "score":
            return {"type": "score", "score": expected_index(p),
                    "confidence": normalized_confidence(p),
                    "legend": {k: to_text(o.desc) for k, o in zip(keys, q.options)},
                    "probabilities": dict(zip(keys, p))}
        py = clamp_noul(p[0])
        return {"type": "noul", "noul": py, "confidence": binary_confidence(py)}

    # -- self-test -------------------------------------------------------------
    def selftest_texts(self) -> list[str]:
        """The README's prompt pieces (Korean state) and a 26-option piece with
        non-ASCII and JSON descriptions, ending at the slot of " Z"."""
        wide = Question("w", "choice", "어느 쪽? (naïve café, ½ €)",
                        tuple(Option(f"o{i}", {"d": i, "ü": [1e-05]}) for i in range(26)))
        a, b = prompt_parts(README_STATE, README_TEAM)
        return [a, b, prompt_parts("s", wide)[1] + " Z"]

    def selftest_labels(self) -> dict[int, str]:
        return {self.label_ids[LETTERS.index(c)]: " " + c for c in "ABCZ"}

    # -- documentation ---------------------------------------------------------
    def spec(self) -> dict:
        return {
            "confidence": {"choice": "(max p - 1/n) / (1 - 1/n)",
                           "score": "(max p - 1/n) / (1 - 1/n)",
                           "noul": "max(p, 1 - p) of the reported (clamped) P(true)"},
            "labels": "A..Z, one per option in the order given; read as the token \" <letter>\" "
                      "after \"Answer:\"",
            "noul_order": "A = Yes = true, B = No = false",
            "state": ("strings as is, other JSON as indented JSON (indent 1, non-ASCII kept). "
                      "Never truncated: a prompt longer than the server's context is refused "
                      "with input_too_long."),
            "limits": {"options": f"2..{len(LETTERS)} for choice, 2..10 levels for score"},
            "calls": "one llama-server call per question; the state is repeated in each",
            "extras": {},
            "notes": [
                "Temperatures from temps.json: choice 1.1, noul 1.2, score 1.2, applied as the "
                "vendor does: softmax, each probability clipped to at least 1e-12, raised to "
                "1/T and renormalised.",
                "noul answers are clamped as the vendor ships them: a P(true) strictly between "
                "0.2 and 0.8 is reported as 0.2 (below 0.5) or 0.8 (0.5 and above).",
                "The vendor defines no confidence; these are the same formulas as for "
                "H2O-Lightning.",
                "noul descriptions are used only when both true and false are given (\"Yes: "
                "…\" / \"No: …\"); with one or none the options are plain \"Yes\" / \"No\".",
                "A choice option without a description, or with an empty one, is shown as its "
                "label. Non-string descriptions, score levels and states are indented JSON.",
                "Special-token text inside the state or questions (e.g. <|im_end|>) is parsed "
                "as special tokens, like the vendor's Transformers helper (its GGUF helper "
                "doesn't).",
                "Built and trained for Korean; English works but isn't its focus. The vendor's "
                "published numbers are rounded to 3 decimals.",
            ],
        }


README_STATE = "고객: 어제 주문한 이어폰 값이 카드에서 두 번 결제됐어요. 오늘 안에 한 건 취소해 주세요."
README_TEAM = Question("team", "choice", "Which team should handle this ticket?",
                       (Option("billing", "Payments and refunds"), Option("shipping", "Delivery"),
                        Option("account", "Login and account")))

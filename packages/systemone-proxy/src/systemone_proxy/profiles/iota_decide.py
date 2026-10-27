"""iota-decide-4b v0.3 (macrocosm-os), Qwen3.5-4B with a rank-16 LoRA merged.

Source: ``letter_decide.py`` and ``joint_head.safetensors`` of
macrocosm-os/iota-decide-4b-v0.3 at 1da87f01c949e2f8a65e5958ef7cc43ec18da999
(the repo's only commit). There is no server; this profile ports its
``options``, ``prompt``, ``fit_state`` and ``systemone_answer``. One prompt
per question, built by hand (not through the chat template)::

    <|im_start|>system\\n{SYSTEM}<|im_end|>\\n<|im_start|>user\\nSTATE:\\n{state}\\n\\n
    QUESTION: {instructions}\\n\\nOPTIONS:\\n{A) text ...}\\n\\n
    Answer with the letter of the one best option.<|im_end|>\\n
    <|im_start|>assistant\\n<think>\\n\\n</think>\\n\\nAnswer:

(one string; the line breaks above are for reading). It is tokenized whole
with special-token text parsed and no BOS. The answer slot is the token after
``Answer:``, read as ``" A"`` .. ``" Z"``. Despite the file names, there is no
extra head at inference: ``joint_head.safetensors`` holds only the log of the
one temperature used for every question type.

No GGUF of v0.3 is published (the iota GGUFs on the Hub are v0.2, architecture
``clef``, a different model). Convert it with llama.cpp's convert_hf_to_gguf.py
after moving ``joint_head_config.json`` aside (its presence makes the converter
take the repo for Clef), with ``--no-mtp`` (the config declares an MTP layer the
weights don't have), and set ``general.basename: iota-decide``,
``general.version: v0.3`` and the ``decision-model`` tag with ``--metadata``.
"""
from __future__ import annotations

import math

from systemone_proxy.profiles.base import Built, Profile, Row, py_json
from systemone_proxy.questions import Limits, Option, Question, Request
from systemone_proxy.readout import (argmax, binary_confidence, expected_index, softmax_t,
                                     top_confidence)
from systemone_proxy.tokenizer import TokenizerSource

LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
# letter_decide.py, verbatim
SYSTEM = ("You are a careful decision engine. Read the state, then answer the question by choosing "
          "exactly one of the lettered options. Apply any stated rules literally.")
TAIL = "Answer with the letter of the one best option."
PREFILL = "<|im_start|>assistant\n<think>\n\n</think>\n\nAnswer:"
NOUL_FLOOR = 0.801
MAX_LENGTH = 16384                     # the whole prompt, per question
# joint_head.safetensors: log_temperature (F32) = -0.04101192206144333
TEMPERATURE = math.exp(-0.04101192206144333)

IOTA_TOKENIZER = TokenizerSource(
    repo="macrocosm-os/iota-decide-4b-v0.3",
    revision="1da87f01c949e2f8a65e5958ef7cc43ec18da999",
    sha256="a5cd9732badce41de57e6efce8302930ded1c1188c5f81feb2bd6c24c4a1941f",
)


def render_state(state) -> str:
    """letter_decide.prompt: strings as is, else JSON with default separators
    and non-ASCII kept."""
    return state if isinstance(state, str) else py_json(state)


def option_texts(q: Question) -> list[str]:
    """letter_decide.options. Descriptions go through Python's str(), not JSON
    (a null score level is "None"); a falsy choice or noul description is
    left out."""
    if q.type == "noul":
        t, f = q.options[0].desc, q.options[1].desc
        return ["Yes" + (f": {t}" if t else ""), "No" + (f": {f}" if f else "")]
    if q.type == "score":
        return [str(o.desc) for o in q.options]
    return [str(o.key) + (f": {o.desc}" if o.desc else "") for o in q.options]


def prompt(state_text: str, q: Question) -> str:
    """letter_decide.prompt, with the state already rendered."""
    lines = "\n".join(f"{LETTERS[i]}) {t}" for i, t in enumerate(option_texts(q)))
    user = (f"STATE:\n{state_text}\n\nQUESTION: {q.instructions}\n\nOPTIONS:\n{lines}\n\n{TAIL}")
    return (f"<|im_start|>system\n{SYSTEM}<|im_end|>\n<|im_start|>user\n{user}<|im_end|>\n"
            + PREFILL)


def commit_noul(p_yes: float) -> float:
    """letter_decide.systemone_answer: under the floor, P(true) moves to the
    floor on its own side (a tie goes to yes)."""
    if max(p_yes, 1.0 - p_yes) < NOUL_FLOOR:
        return NOUL_FLOOR if p_yes >= 0.5 else 1.0 - NOUL_FLOOR
    return p_yes


class IotaDecide4B(Profile):
    id = "iota-decide-4b-v0.3"
    name = "iota-decide-4b v0.3"
    upstream = ("https://huggingface.co/macrocosm-os/iota-decide-4b-v0.3 (letter_decide.py, "
                "joint_head.safetensors)")
    gguf_repos = ()                    # none published; converted locally
    tokenizer_source = IOTA_TOKENIZER
    temperatures = {"choice": TEMPERATURE, "noul": TEMPERATURE, "score": TEMPERATURE}
    limits = Limits(max_choice=len(LETTERS), max_score=len(LETTERS))

    @classmethod
    def matches(cls, meta: dict, filename: str) -> bool:
        arch = str(meta.get("general.architecture", "")).lower()
        base = str(meta.get("general.basename", "")).lower()
        version = str(meta.get("general.version", "")).lower().lstrip("v")
        if arch != "qwen35":           # v0.2 is "clef"
            return False
        if base == "iota-decide":
            return version == "0.3"
        return not base and filename.lower().startswith("iota-decide-4b-v0.3")

    def __init__(self, tok):
        super().__init__(tok)
        # letter_decide.letter_ids; also checked to be the token after "Answer:"
        head = tok.encode(PREFILL, special=True)
        self.label_ids: list[int] = []
        for letter in LETTERS:
            ids = tok.encode(" " + letter, special=True)
            full = tok.encode(PREFILL + " " + letter, special=True)
            if len(ids) != 1 or full != head + ids:
                raise RuntimeError(f"label ' {letter}' is not one token at the answer slot")
            self.label_ids.append(ids[0])
        if len(set(self.label_ids)) != len(LETTERS):
            raise RuntimeError("label tokens are not distinct")

    # -- prompt --------------------------------------------------------------
    def fit_state(self, state_text: str, q: Question) -> tuple[list[int], bool]:
        """letter_decide.fit_state: over 16384 tokens, the state keeps its first
        and last characters around "\\n[...]\\n", sized by the overshoot, and the
        ids are then cut to their last 16384 (which can cut the system turn)."""
        ids = self.tok.encode(prompt(state_text, q), special=True)
        if len(ids) <= MAX_LENGTH:
            return ids, False
        text = state_text
        keep = max(0, len(text) - int(len(text) * (len(ids) - MAX_LENGTH) / len(ids)) - 200)
        cut = text[: keep // 2] + "\n[...]\n" + text[len(text) - keep // 2:]
        return self.tok.encode(prompt(cut, q), special=True)[-MAX_LENGTH:], True

    def build(self, req: Request) -> Built:
        state_text = render_state(req.state)
        rows, truncated = [], set()
        for q in req.questions:
            ids, cut = self.fit_state(state_text, q)
            if cut:
                truncated.add(q.qid)
            rows.append(Row(q.qid, ids, self.label_ids[:len(q.options)]))
        built = Built(rows=rows, truncated=truncated)
        if truncated:
            built.warnings.append({"code": "state_truncated", "questions": sorted(truncated),
                                   "kept_tokens": MAX_LENGTH, "kept": "head and tail (characters)"})
        return built

    # -- answer ----------------------------------------------------------------
    def answer(self, q: Question, logprobs: dict[str, list[float]]) -> dict:
        p = softmax_t(logprobs["main"], TEMPERATURE)
        keys = q.keys
        if q.type == "choice":
            return {"type": "choice", "choice": keys[argmax(p)], "confidence": top_confidence(p),
                    "probabilities": dict(zip(keys, p))}
        if q.type == "score":
            return {"type": "score", "score": expected_index(p), "confidence": top_confidence(p),
                    "legend": {k: o.desc for k, o in zip(keys, q.options)},
                    "probabilities": dict(zip(keys, p))}
        py = commit_noul(p[0])
        return {"type": "noul", "noul": py, "confidence": binary_confidence(py)}

    # -- self-test -------------------------------------------------------------
    def selftest_texts(self) -> list[str]:
        """The card's two prompts, and a 26-option prompt with non-ASCII text and
        one of the special tokens v0.3 added (<|audio_start|>), ending at " Z"."""
        wide = Question("w", "choice", "Welche Option? (naïve café, 日本語, ½ €)",
                        tuple(Option(f"o{i}", {"d": i}) for i in range(26)))
        texts = [prompt(render_state(README_STATE), q) for q in README_QUESTIONS]
        return texts + [prompt("x <|audio_start|> y", wide) + " Z"]

    def selftest_labels(self) -> dict[int, str]:
        return {self.label_ids[LETTERS.index(c)]: " " + c for c in "ABCZ"}

    # -- documentation ---------------------------------------------------------
    def spec(self) -> dict:
        return {
            "confidence": {"choice": "max p", "score": "max p",
                           "noul": "max(p, 1 - p) of the reported (floored) P(true)"},
            "labels": "A..Z, one per option in the order given; read as the token \" <letter>\" "
                      "after the \"Answer:\" prefill",
            "noul_order": "A = Yes = true, B = No = false",
            "state": (f"strings as is, other JSON as json.dumps with default separators and "
                      f"non-ASCII kept. A prompt over {MAX_LENGTH} tokens is TRUNCATED, not "
                      "rejected: the state keeps its first and last characters around a "
                      "\"[...]\" line, then the prompt keeps its last "
                      f"{MAX_LENGTH} tokens. Answers then carry \"truncated\": true and a "
                      "state_truncated warning is added. Run llama-server with a context "
                      f"above {MAX_LENGTH} (e.g. -c 16896)."),
            "limits": {"options": f"2..{len(LETTERS)} for choice and score"},
            "calls": "one llama-server call per question; the state is repeated in each",
            "extras": {},
            "notes": [
                f"One temperature for every question type: T = exp(-0.0410119) = "
                f"{TEMPERATURE:.6f}, from joint_head.safetensors.",
                f"noul answers are floored as the vendor ships them: when max(p, 1 - p) < "
                f"{NOUL_FLOOR}, P(true) is reported as {NOUL_FLOOR} (or "
                f"{1 - NOUL_FLOOR:.3f}) on its own side.",
                "The vendor gives no noul confidence; this is max(p, 1 - p). Choice and score "
                "confidence is the top probability, as the vendor computes it. The vendor "
                "rounds to 4 decimals; nothing is rounded here.",
                "Option descriptions are rendered with Python's str(), not JSON: a null score "
                "level is \"None\", true is \"True\", an object is a Python dict repr. A "
                "falsy choice or noul description (null, \"\", 0, false) is left out.",
                "The score legend echoes the levels as given.",
                "Special-token text inside the state or questions (e.g. <|im_end|>) is parsed "
                "as special tokens, as the vendor's tokenizer call does.",
                "No GGUF is published for v0.3; the GGUF is converted locally from the "
                "pinned checkpoint (see the profile's docstring).",
            ],
        }


README_STATE = {"ticket": "Customer asks for a 40 EUR refund",
                "policy": "Refunds up to 50 EUR need no approval"}
README_QUESTIONS = (
    Question("approve", "noul", "Can the refund be issued without approval?",
             (Option("true", None), Option("false", None))),
    Question("route", "choice", "Which team handles it?",
             (Option("billing", "payments and refunds"), Option("tech", "technical faults"))),
)

"""The unified request: parsing and validation shared by every model.

A request is::

    {"state": <string | object | array>,
     "questions": {"<id>": <question>, ...},
     "images": [<data URI | base64 | image part>, ...],   (image models only)
     "model": <ignored>, "debug": <bool, optional>}

A question is ``{"type", "instructions", "criteria", "threshold"?}``:

* ``choice``: criteria ``{label: description}`` (description may be any JSON
  value, or null for none) or a list of labels. Answer keys are the labels.
* ``score``: criteria is a list of level descriptions, lowest first. Answer
  keys are ``"0"`` .. ``"n-1"``.
* ``noul`` (yes/no): criteria is null or ``{"true"?: description,
  "false"?: description}``. The answer is P(true).

Options keep their *raw* description values here; how they are turned into
prompt text is model-specific (see the profiles).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

from systemone_proxy import images as imagelib
from systemone_proxy.errors import ApiError, invalid

TYPES = ("choice", "score", "noul")


@dataclass(frozen=True)
class Option:
    key: str          # answer key: the choice label, the score level index, or "true"/"false"
    desc: Any         # raw JSON description, None when not given


@dataclass(frozen=True)
class Question:
    qid: str
    type: str
    instructions: str
    options: tuple[Option, ...]   # noul is always (true, false) in this order
    threshold: float | None = None
    raw: dict = field(default_factory=dict, compare=False)

    @property
    def keys(self) -> list[str]:
        return [o.key for o in self.options]


@dataclass(frozen=True)
class Request:
    state: Any
    questions: tuple[Question, ...]
    debug: bool = False
    # fitted images, in [image N] order; the state then holds the placeholders
    images: tuple[imagelib.Image, ...] = ()


@dataclass(frozen=True)
class Limits:
    max_questions: int = 64
    max_choice: int = 255
    max_score: int = 10


def _is_number(v: Any) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)


def parse_question(qid: str, q: Any, limits: Limits) -> Question:
    where = f"question {qid!r}"
    if not isinstance(q, dict):
        raise invalid(f"{where}: must be an object with type, instructions and criteria")
    t = q.get("type")
    if t not in TYPES:
        raise invalid(f"{where}: unknown type {t!r}; expected one of {list(TYPES)}")
    instr = q.get("instructions")
    if not isinstance(instr, str) or not instr.strip():
        raise invalid(f"{where}: instructions must be a non-empty string")
    crit = q.get("criteria")

    if t == "choice":
        if isinstance(crit, dict):
            opts = [Option(str(k), v) for k, v in crit.items()]
        elif isinstance(crit, list):
            if not all(isinstance(k, str) or _is_number(k) for k in crit):
                raise invalid(f"{where}: a choice criteria list must hold labels (strings)")
            opts = [Option(str(k), None) for k in crit]
        else:
            raise invalid(f"{where}: choice criteria must be an object {{label: description}} "
                          "or a list of labels")
        if len({o.key for o in opts}) != len(opts):
            raise invalid(f"{where}: choice labels must be unique")
        if not 2 <= len(opts) <= limits.max_choice:
            raise invalid(f"{where}: choice needs 2..{limits.max_choice} options, got {len(opts)}")
    elif t == "score":
        if not isinstance(crit, list):
            raise invalid(f"{where}: score criteria must be a list of level descriptions, lowest first")
        opts = [Option(str(i), v) for i, v in enumerate(crit)]
        if not 2 <= len(opts) <= limits.max_score:
            raise invalid(f"{where}: score needs 2..{limits.max_score} levels, got {len(opts)}")
    else:
        crit = {} if crit is None else crit
        if not isinstance(crit, dict) or not set(crit) <= {"true", "false"}:
            raise invalid(f"{where}: noul criteria must be null or an object with optional "
                          "keys \"true\" and \"false\"")
        opts = [Option("true", crit.get("true")), Option("false", crit.get("false"))]

    threshold = q.get("threshold")
    if threshold is not None:
        if not _is_number(threshold) or not 0.0 <= threshold <= 1.0:
            raise invalid(f"{where}: threshold must be a number in [0, 1]")
        threshold = float(threshold)
    return Question(str(qid), t, instr, tuple(opts), threshold, q)


def parse_request(body: Any, limits: Limits,
                  image_support: imagelib.ImageSupport | None = None) -> Request:
    if not isinstance(body, dict):
        raise ApiError(400, "bad_request", "the body must be a JSON object")
    if "state" not in body:
        raise invalid("state is required")
    state = body["state"]
    if not isinstance(state, (str, dict, list)):
        raise invalid(f"state must be a string, object or array, got {type(state).__name__}")
    questions = body.get("questions")
    if not isinstance(questions, dict) or not questions:
        raise invalid("questions must be a non-empty object {id: question}")
    if len(questions) > limits.max_questions:
        raise invalid(f"at most {limits.max_questions} questions per request, got {len(questions)}")
    debug = body.get("debug", False)
    if not isinstance(debug, bool):
        raise invalid("debug must be a boolean")
    parsed = tuple(parse_question(str(k), v, limits) for k, v in questions.items())
    if image_support is None:
        if body.get("images") is not None:
            raise invalid("this model does not read images")
        return Request(state, parsed, debug)
    state, raws = imagelib.extract(body)
    if len(raws) > image_support.max_images:
        raise invalid(f"{len(raws)} images; this model reads at most "
                      f"{image_support.max_images} per request")
    return Request(state, parsed, debug, tuple(imagelib.fit(r, image_support) for r in raws))

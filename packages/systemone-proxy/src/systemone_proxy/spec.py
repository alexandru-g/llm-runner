"""The client contract: one common part plus each model's specifics.

Printed by ``systemone-proxy spec`` (and ``llmctl decide spec``), served at
``GET /v1/spec``.
"""
from __future__ import annotations

import json

from systemone_proxy.profiles.base import Profile

EXAMPLE_REQUEST = {
    "state": {"message": "Please close my card, I lost it yesterday."},
    "questions": {
        "intent": {"type": "choice", "instructions": "What does the customer want?",
                   "criteria": {"cancel": "close the card", "limit": "change the limit",
                                "other": None}},
        "urgent": {"type": "noul", "instructions": "The request is urgent."},
        "mood": {"type": "score", "instructions": "How upset is the customer?",
                 "criteria": ["calm", "annoyed", "angry"]},
    },
}

EXAMPLE_RESPONSE = {
    "model": "<profile id>",
    "answers": {
        "intent": {"type": "choice", "choice": "cancel", "confidence": 0.93,
                   "probabilities": {"cancel": 0.95, "limit": 0.01, "other": 0.04}},
        "urgent": {"type": "noul", "noul": 0.83, "confidence": 0.83},
        "mood": {"type": "score", "score": 0.71, "confidence": 0.52,
                 "legend": {"0": "calm", "1": "annoyed", "2": "angry"},
                 "probabilities": {"0": 0.38, "1": 0.53, "2": 0.09}},
    },
    "usage": {"input_tokens": 412, "output_tokens": 0, "backend_calls": 3},
    "warnings": [],
    "latency_ms": 240,
}

COMMON = {
    "endpoints": {
        "POST /v1/systemone": "make decisions (alias: POST /v1/decisions)",
        "GET /v1/spec": "this document as JSON",
        "GET /health": "200 {\"status\": \"ok\", \"model\": id} once ready",
        "GET /v1/models": "OpenAI-style model list with the profile id",
    },
    "request": {
        "state": "required; string, object or array — the record the questions are about",
        "questions": "required; non-empty object {question_id: question}, answered independently, "
                     "returned in the same order",
        "question.type": "\"choice\" | \"score\" | \"noul\"",
        "question.instructions": "required non-empty string: the question",
        "question.criteria": {
            "choice": "{label: description} (description: any JSON, null for none) or [label, ...]",
            "score": "[level description, ...] lowest first; answer keys are \"0\"..\"n-1\"",
            "noul": "null or {\"true\"?: description, \"false\"?: description}",
        },
        "question.threshold": "optional number in [0, 1] (used by models that list it under extras)",
        "images": "optional, image models only (see model.images): list of data URIs, base64 "
                  "strings or image parts ({\"type\": \"image_url\", \"image_url\": {\"url\": "
                  "data URI}}). Images may also sit inline in state. URLs are not fetched. "
                  "Multipart/form-data works too: the JSON in a \"request\" field, each image a "
                  "file part. A text-only model rejects the images field (422).",
        "debug": "optional bool: adds answers[id].debug with the raw label logprobs and prompt sizes",
        "model": "optional, ignored",
    },
    "response": {
        "model": "profile id of the model that answered",
        "answers.<id> (choice)": "{type, choice: label, confidence, probabilities: {label: p}}",
        "answers.<id> (score)": "{type, score: expected level index (float, 0-based), confidence, "
                                "legend: {\"0\": description, ...}, probabilities: {\"0\": p, ...}}",
        "answers.<id> (noul)": "{type, noul: P(true), confidence}",
        "answers.<id>.truncated": "present (true) only when the model's protocol truncated the state",
        "usage": "{input_tokens, output_tokens: 0, backend_calls, images (only when present)}; "
                 "input_tokens includes image tokens",
        "warnings": "list of {code, ...}; codes: state_truncated, label_outside_top_n, "
                    "image_read_as_text (a text-only model got image data in state)",
        "latency_ms": "wall time spent in the proxy",
    },
    "semantics": [
        "Probabilities are the model vendor's own: same prompt bytes, same temperature, same "
        "arithmetic. They sum to 1 over the offered options. Nothing is rounded.",
        "confidence is computed with the vendor's formula (see model.confidence); compare it "
        "within one model, not across models.",
        "noul is always P(true), whatever label order the model uses internally.",
        "label_outside_top_n means an option's token was not in llama-server's top-n list; "
        "its logprob was bounded by the smallest returned one (the probability is tiny).",
    ],
    "errors": {
        "shape": "{\"error\": {\"code\": str, \"message\": str}}",
        "400 bad_request": "body is not a JSON object",
        "404 not_found": "unknown path",
        "413 body_too_large": "body over 128 MiB",
        "422 invalid_request": "a field breaks the contract (type, criteria, option count, …)",
        "422 input_too_long": "a question's prompt does not fit the model's context "
                              "(never truncated unless the model's protocol truncates)",
        "502 backend_error": "llama-server failed or is unreachable",
        "503 not_ready": "the proxy is still starting",
        "503 vision_unavailable": "a request with images, but llama-server has no vision "
                                  "projector loaded",
    },
}


def build_spec(profile_cls: type[Profile] | None, profile: Profile | None = None) -> dict:
    out: dict = {"api": COMMON, "example_request": EXAMPLE_REQUEST,
                 "example_response": EXAMPLE_RESPONSE}
    if profile_cls is not None:
        details = (profile or profile_cls.__new__(profile_cls)).spec()
        out["model"] = {
            "id": profile_cls.id,
            "name": profile_cls.name,
            "upstream": profile_cls.upstream,
            "gguf_repos": list(profile_cls.gguf_repos),
            "temperatures": dict(profile_cls.temperatures),
            "limits": {"max_questions": profile_cls.limits.max_questions,
                       "max_choice_options": profile_cls.limits.max_choice,
                       "max_score_levels": profile_cls.limits.max_score,
                       **details.get("limits", {})},
            **{k: v for k, v in details.items() if k != "limits"},
        }
        out["example_response"] = {**EXAMPLE_RESPONSE, "model": profile_cls.id}
    return out


def _lines(value, indent: str = "  ") -> list[str]:
    if isinstance(value, dict):
        out = []
        for k, v in value.items():
            if isinstance(v, (dict, list)) and v:
                out.append(f"{indent}{k}:")
                out += _lines(v, indent + "  ")
            else:
                out.append(f"{indent}{k}: {v if v != {} else '(none)'}")
        return out
    if isinstance(value, list):
        return [f"{indent}- {v}" for v in value]
    return [f"{indent}{value}"]


def render_text(spec: dict) -> str:
    lines: list[str] = []
    model = spec.get("model")
    if model:
        lines += [f"# {model['name']}  (profile: {model['id']})", ""]
        lines += ["## Model specifics"] + _lines({k: v for k, v in model.items()
                                                  if k not in ("id", "name")}) + [""]
    else:
        lines += ["# systemone-proxy API", ""]
    api = spec["api"]
    for section in ("endpoints", "request", "response", "semantics", "errors"):
        lines += [f"## {section.capitalize()}"] + _lines(api[section]) + [""]
    lines += ["## Example request", json.dumps(spec["example_request"], indent=2), "",
              "## Example response (illustrative numbers)",
              json.dumps(spec["example_response"], indent=2)]
    return "\n".join(lines)

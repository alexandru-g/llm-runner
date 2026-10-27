"""Native mode: llama-server answers /v1/systemone itself.

Since build b11361, llama-server serves ``POST /v1/systemone`` for GGUFs that
carry ``<arch>.decision.*`` metadata and a ``systemone`` chat template, and
then refuses plain completions (501), so no profile can read label logprobs
from it. ``vllm-d3`` (b11572+) exists only in this form.

The proxy then forwards each request and turns the answer into the unified
contract: the same request validation and error codes, a noul confidence,
``warnings``, ``latency_ms`` and ``usage.backend_calls``, and ``/v1/spec``.
The numbers are llama-server's own.
"""
from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field

from systemone_proxy.errors import ApiError, backend_error, invalid, too_long
from systemone_proxy.gguf import decision_metadata
from systemone_proxy.questions import Limits, parse_request
from systemone_proxy.readout import binary_confidence

# Per decision type: (max choice options, max score levels), from llama.cpp's
# server-decision.cpp. Types not listed get the strictest common limits.
_LIMITS = {"vllm-d3": (255, 10), "vllm-decision2": (255, 10), "h2o-lightning": (255, 255)}
_CONFIDENCE = {
    "vllm-d3": {"choice": "1 - H(p) / ln n (normalized entropy)",
                "score": "1 - H(p) / ln n (normalized entropy)"},
    "vllm-decision2": {"choice": "1 - H(p) / ln n (normalized entropy)",
                       "score": "1 - H(p) / ln n (normalized entropy)"},
}
_DEFAULT_CONFIDENCE = {"choice": "(max p - 1/n) / (1 - 1/n)",
                       "score": "1 - mean |i - mode| / the same for a uniform p"}


@dataclass
class NativeModel:
    """Stands in for a profile class and instance (``id``, ``limits``,
    ``image_support``, ``spec()``)."""
    id: str
    name: str
    decision_type: str
    temperatures: dict = field(default_factory=dict)
    causal: bool = True
    limits: Limits = field(default_factory=Limits)
    image_support = None
    upstream = "llama-server's native /v1/systemone (GGUF decision metadata)"
    gguf_repos: tuple = ()

    @classmethod
    def from_gguf(cls, path: str, model_id: str) -> "NativeModel | None":
        meta = decision_metadata(path)
        arch = meta.get("general.architecture", "")
        dtype = meta.get(f"{arch}.decision.type")
        if not dtype:
            return None
        prefix = f"{arch}.decision.temperature."
        temps = {k[len(prefix):]: v for k, v in meta.items() if k.startswith(prefix)}
        max_choice, max_score = _LIMITS.get(dtype, (26, 10))
        return cls(id=model_id, name=str(meta.get("general.name") or model_id), decision_type=dtype,
                   temperatures=temps, causal=meta.get(f"{arch}.attention.causal", True) is not False,
                   limits=Limits(max_choice=max_choice, max_score=max_score))

    def spec(self) -> dict:
        conf = {**_DEFAULT_CONFIDENCE, **_CONFIDENCE.get(self.decision_type, {}),
                "noul": "max(p, 1 - p), added by the proxy (llama-server reports none)"}
        notes = [f"Served natively by llama-server (decision type {self.decision_type}): the "
                 "prompt template, labels and temperatures come from the GGUF's metadata.",
                 "The proxy validates requests and adds noul confidence, warnings, latency_ms "
                 "and usage.backend_calls; probabilities are llama-server's.",
                 "Text only through the proxy: requests with images are refused."]
        if not self.causal:
            notes.append("The model attends to the whole prompt at once (non-causal): each "
                         "prompt must fit in llama-server's batch (-b / -ub, which llmctl sets "
                         "to the context size), or the request is refused as too long.")
        return {"confidence": conf,
                "labels": "set by llama-server from the GGUF (A, B, … in option order)",
                "noul_order": "true first",
                "state": "strings as is, other JSON as JSON text; no truncation by the proxy",
                "limits": {"options": f"2..{self.limits.max_choice} for choice, "
                                      f"2..{self.limits.max_score} levels for score"},
                "calls": "one llama-server /v1/systemone call per request",
                "extras": {}, "notes": notes}


class NativeService:
    """The DecisionService interface (``profile``, ``backend``, ``decide``)."""

    def __init__(self, model: NativeModel, backend, *, timeout: float = 600.0):
        self.profile = model
        self.backend = backend
        self.timeout = timeout

    def _post(self, body: dict) -> dict:
        req = urllib.request.Request(self.backend.url + "/v1/systemone",
                                     data=json.dumps(body, ensure_ascii=False).encode(),
                                     method="POST", headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as r:
                return json.loads(r.read())
        except urllib.error.HTTPError as e:
            raise _map_error(e.code, e.read().decode("utf-8", "replace")) from None
        except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
            raise backend_error(f"llama-server /v1/systemone unreachable: {e}") from None

    def decide(self, body) -> dict:
        t0 = time.monotonic()
        req = parse_request(body, self.profile.limits)
        if req.images:
            raise ApiError(422, "invalid_request",
                           "images aren't supported for natively served models")
        questions = {}
        for q in req.questions:
            raw = q.raw.get("criteria")
            if q.type == "choice" and isinstance(raw, list):      # llama-server wants an object
                raw = {str(k): None for k in raw}
            questions[q.qid] = {"type": q.type, "instructions": q.instructions,
                                **({"criteria": raw} if raw is not None else {})}
        out = self._post({"state": req.state, "questions": questions})
        got = out.get("answers") or {}
        answers = {}
        for q in req.questions:
            a = dict(got.get(q.qid) or {})
            if not a:
                raise backend_error(f"llama-server returned no answer for {q.qid!r}")
            if q.type == "noul" and "confidence" not in a:
                a["confidence"] = binary_confidence(float(a["noul"]))
            if req.debug:
                a["debug"] = {"native": True}
            answers[q.qid] = a
        usage = out.get("usage") or {}
        return {"model": self.profile.id, "answers": answers,
                "usage": {"input_tokens": usage.get("input_tokens", 0), "output_tokens": 0,
                          "backend_calls": 1},
                "warnings": [], "latency_ms": round((time.monotonic() - t0) * 1000)}


def _map_error(status: int, text: str) -> ApiError:
    """llama-server's ``{"error": {"code": <int>, "message", "type"}}`` as the
    contract's errors."""
    try:
        err = json.loads(text).get("error") or {}
    except (ValueError, AttributeError):
        err = {}
    message, kind = str(err.get("message") or text[:500]), str(err.get("type") or "")
    if kind == "exceed_context_size_error" or "too large to process" in message:
        return too_long(f"llama-server: {message}")
    if status == 400:
        return invalid(f"llama-server: {message}")
    if status == 503:
        return ApiError(503, "not_ready", f"llama-server: {message}")
    return backend_error(f"llama-server /v1/systemone returned {status}: {message}")


def detect(backend) -> bool:
    """Whether llama-server serves /v1/systemone itself (``/v1/models`` lists
    the "decisions" output modality)."""
    try:
        with urllib.request.urlopen(backend.url + "/v1/models", timeout=10) as r:
            data = json.loads(r.read()).get("data") or []
    except (urllib.error.URLError, TimeoutError, ConnectionError, ValueError):
        return False
    return any("decisions" in ((m.get("architecture") or {}).get("output_modalities") or [])
               for m in data)

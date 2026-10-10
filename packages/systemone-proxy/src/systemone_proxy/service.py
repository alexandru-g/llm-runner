"""Request → profile rows → llama-server → unified response."""
from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor

from systemone_proxy.backend import LlamaServer
from systemone_proxy.errors import too_long
from systemone_proxy.profiles.base import Profile, Row
from systemone_proxy.questions import parse_request

# Extra top-n entries requested beyond the number of labels, so every label
# of a well-behaved model comes back.
N_PROBS_MARGIN = 64


class DecisionService:
    def __init__(self, profile: Profile, backend: LlamaServer, *, n_ctx: int,
                 n_probs: int = 128, cache: str = "off", concurrency: int = 1):
        self.profile = profile
        self.backend = backend
        self.n_ctx = n_ctx
        profile.n_ctx = n_ctx
        self.n_probs = n_probs
        self.cache = cache            # "off" | "request"
        self.concurrency = max(1, concurrency)

    def _run_row(self, row: Row, first: bool) -> tuple[list[float], list[str]]:
        n = max(self.n_probs, len(row.label_ids) + N_PROBS_MARGIN)
        # Never reuse a prompt cache across requests: on Gemma 4 (sliding
        # window) that changed answers depending on what the slot held before.
        cache_prompt = self.cache == "request" and not first
        top = self.backend.next_token_logprobs(row.ids, n_probs=n, cache_prompt=cache_prompt)
        floor = min(top.values())
        missing = [i for i in row.label_ids if i not in top]
        # A label outside the top n is at most as likely as the last one
        # returned; use that bound (it can only overstate a tiny probability).
        return [top.get(i, floor) for i in row.label_ids], [str(i) for i in missing]

    def decide(self, body) -> dict:
        t0 = time.monotonic()
        req = parse_request(body, self.profile.limits)
        built = self.profile.build(req)
        for row in built.rows:
            if len(row.ids) + 1 > self.n_ctx:
                raise too_long(f"question {row.qid!r}: the prompt is {len(row.ids)} tokens; "
                               f"the model's context is {self.n_ctx}")

        rows = built.rows
        if self.concurrency == 1 or len(rows) == 1:
            results = [self._run_row(r, i == 0) for i, r in enumerate(rows)]
        else:
            first = self._run_row(rows[0], True)
            with ThreadPoolExecutor(self.concurrency) as pool:
                rest = list(pool.map(lambda r: self._run_row(r, False), rows[1:]))
            results = [first, *rest]

        warnings = list(built.warnings)
        by_q: dict[str, dict[str, list[float]]] = {}
        for row, (lps, missing) in zip(rows, results):
            by_q.setdefault(row.qid, {})[row.tag] = lps
            if missing:
                warnings.append({"code": "label_outside_top_n", "question": row.qid,
                                 "row": row.tag, "token_ids": missing})

        answers = {}
        for q in req.questions:
            a = self.profile.answer(q, by_q[q.qid])
            if q.qid in built.truncated:
                a["truncated"] = True
            if req.debug:
                a["debug"] = {"label_logprobs": by_q[q.qid],
                              "prompt_tokens": {r.tag: len(r.ids) for r in rows if r.qid == q.qid}}
            answers[q.qid] = a

        input_tokens = (built.input_tokens if built.input_tokens is not None
                        else sum(len(r.ids) for r in rows))
        return {
            "model": self.profile.id,
            "answers": answers,
            "usage": {"input_tokens": input_tokens, "output_tokens": 0,
                      "backend_calls": len(rows)},
            "warnings": warnings,
            "latency_ms": round((time.monotonic() - t0) * 1000),
        }

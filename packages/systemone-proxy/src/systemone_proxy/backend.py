"""llama-server client: tokenizer parity checks and next-token label logprobs."""
from __future__ import annotations

import json
import time
import urllib.error
import urllib.request

from systemone_proxy.errors import ApiError, backend_error


class LlamaServer:
    def __init__(self, url: str, timeout: float = 600.0):
        self.url = url.rstrip("/")
        self.timeout = timeout

    def _call(self, method: str, path: str, body: dict | None = None,
              timeout: float | None = None) -> dict:
        data = None if body is None else json.dumps(body).encode()
        req = urllib.request.Request(self.url + path, data=data, method=method,
                                     headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=timeout or self.timeout) as r:
                return json.loads(r.read())
        except urllib.error.HTTPError as e:
            detail = e.read().decode("utf-8", "replace")[:500]
            raise backend_error(f"llama-server {path} returned {e.code}: {detail}") from None
        except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
            raise backend_error(f"llama-server {path} unreachable: {e}") from None

    def props(self) -> dict:
        return self._call("GET", "/props", timeout=10)

    def wait_ready(self, timeout: float, log=print) -> dict:
        """Poll /health until the model is loaded; returns /props."""
        deadline = time.monotonic() + timeout
        last = ""
        while True:
            try:
                with urllib.request.urlopen(self.url + "/health", timeout=5) as r:
                    if r.status == 200:
                        return self.props()
            except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
                msg = str(getattr(e, "reason", e))
                if msg != last:
                    log(f"waiting for llama-server at {self.url} ({msg})")
                    last = msg
            if time.monotonic() > deadline:
                raise RuntimeError(f"llama-server at {self.url} not ready after {timeout:.0f}s")
            time.sleep(2)

    def tokenize(self, text: str, *, parse_special: bool = True) -> list[int]:
        out = self._call("POST", "/tokenize", {"content": text, "add_special": False,
                                               "parse_special": parse_special})
        return [t if isinstance(t, int) else t["id"] for t in out["tokens"]]

    def detokenize(self, ids: list[int]) -> str:
        return self._call("POST", "/detokenize", {"tokens": ids})["content"]

    def next_token_logprobs(self, ids: list[int], *, n_probs: int,
                            cache_prompt: bool) -> dict[int, float]:
        """Top ``n_probs`` next-token log-probabilities after the prompt
        ``ids``, from llama-server's pre-sampling full-vocabulary softmax of the
        raw logits (no temperature, no samplers)."""
        out = self._call("POST", "/completion", {
            "prompt": ids, "n_predict": 1, "n_probs": n_probs, "temperature": 0.0,
            "post_sampling_probs": False, "cache_prompt": cache_prompt,
        })
        try:
            top = out["completion_probabilities"][0]["top_logprobs"]
        except (KeyError, IndexError, TypeError):
            raise backend_error("llama-server returned no completion_probabilities "
                                "(is this a llama-server build with n_probs support?)") from None
        # A second <bos> or a server-side template would change the count.
        if not cache_prompt and out.get("tokens_evaluated") not in (None, len(ids)):
            raise ApiError(500, "prompt_mismatch",
                           f"llama-server evaluated {out['tokens_evaluated']} tokens, sent {len(ids)}")
        return {int(t["id"]): float(t["logprob"]) for t in top}

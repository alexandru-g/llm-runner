"""llama-server client: tokenizer parity checks and next-token label logprobs."""
from __future__ import annotations

import base64
import json
import re
import time
import urllib.error
import urllib.request

from systemone_proxy.errors import ApiError, backend_error, too_long
from systemone_proxy.images import MEDIA

_CONTEXT_ERROR = re.compile(r"context|too long|exceed", re.I)


class LlamaServer:
    def __init__(self, url: str, timeout: float = 600.0):
        self.url = url.rstrip("/")
        self.timeout = timeout
        self._marker: str | None = None

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
            if e.code == 400 and path == "/completion" and _CONTEXT_ERROR.search(detail):
                raise too_long(f"llama-server: {detail}") from None
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

    def vision(self, props: dict | None = None) -> bool:
        """Whether llama-server was started with a vision projector (mmproj)."""
        props = props if props is not None else self.props()
        return bool((props.get("modalities") or {}).get("vision"))

    def media_marker(self) -> str:
        """The server's image placeholder (random per process unless pinned)."""
        if self._marker is None:
            self._marker = self.props().get("media_marker") or "<__media__>"
        return self._marker

    def tokenize(self, text: str, *, parse_special: bool = True) -> list[int]:
        out = self._call("POST", "/tokenize", {"content": text, "add_special": False,
                                               "parse_special": parse_special})
        return [t if isinstance(t, int) else t["id"] for t in out["tokens"]]

    def detokenize(self, ids: list[int]) -> str:
        return self._call("POST", "/detokenize", {"tokens": ids})["content"]

    def next_token_logprobs(self, ids: list[int], *, n_probs: int,
                            cache_prompt: bool, text: str | None = None,
                            images: tuple[bytes, ...] = (),
                            extra_tokens: int = 0) -> dict[int, float]:
        """Top ``n_probs`` next-token log-probabilities after the prompt
        ``ids``, from llama-server's pre-sampling full-vocabulary softmax of the
        raw logits (no temperature, no samplers).

        With ``images`` the prompt goes as ``text`` (one ``images.MEDIA`` per
        image) plus the image data; ``ids`` then only sets the expected token
        count, ``len(ids) + extra_tokens``."""
        if images:
            prompt = {"prompt_string": text.replace(MEDIA, self.media_marker()),
                      "multimodal_data": [base64.b64encode(b).decode("ascii") for b in images]}
        else:
            prompt = ids
        out = self._call("POST", "/completion", {
            "prompt": prompt, "n_predict": 1, "n_probs": n_probs, "temperature": 0.0,
            "post_sampling_probs": False, "cache_prompt": cache_prompt,
        })
        try:
            top = out["completion_probabilities"][0]["top_logprobs"]
        except (KeyError, IndexError, TypeError):
            raise backend_error("llama-server returned no completion_probabilities "
                                "(is this a llama-server build with n_probs support?)") from None
        # A second <bos>, a server-side template or (for images) a different
        # resize in llama.cpp would change the count.
        want = len(ids) + extra_tokens
        if not cache_prompt and out.get("tokens_evaluated") not in (None, want):
            raise ApiError(500, "prompt_mismatch",
                           f"llama-server evaluated {out['tokens_evaluated']} tokens, expected {want}"
                           + (" (were llama-server's image token limits changed, e.g. "
                              "--image-max-tokens?)" if images else ""))
        return {int(t["id"]): float(t["logprob"]) for t in top}

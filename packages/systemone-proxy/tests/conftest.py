"""Shared fixtures. Tokenizer-dependent tests need the pinned tokenizer.json
files; they are fetched once into the HF cache (or skipped when offline)."""
from __future__ import annotations

from pathlib import Path

import pytest

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture(scope="session")
def gemma_tok():
    from systemone_proxy import tokenizer
    try:
        return tokenizer.load(tokenizer.GEMMA4)
    except Exception as e:  # offline, no HF access
        pytest.skip(f"Gemma 4 tokenizer unavailable: {e}")


@pytest.fixture(scope="session")
def gemma_template():
    jinja2 = pytest.importorskip("jinja2")
    env = jinja2.Environment(loader=jinja2.FileSystemLoader(str(FIXTURES)),
                             extensions=["jinja2.ext.loopcontrols"])

    def raise_exception(msg):
        raise RuntimeError(msg)

    env.globals["raise_exception"] = raise_exception
    tpl = env.get_template("gemma4_chat_template.jinja")

    def render(messages):
        return tpl.render(messages=messages, bos_token="<bos>", add_generation_prompt=True,
                          enable_thinking=False)
    return render


class FakeBackend:
    """Stands in for llama-server: returns fixed logprobs per label position."""

    url = "fake://"

    def __init__(self, logprobs_by_call):
        self.calls = []
        self._lps = list(logprobs_by_call)

    def next_token_logprobs(self, ids, *, n_probs, cache_prompt):
        self.calls.append({"ids": ids, "n_probs": n_probs, "cache_prompt": cache_prompt})
        return self._lps.pop(0)

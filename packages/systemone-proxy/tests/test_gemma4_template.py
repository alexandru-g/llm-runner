"""gemma4_chat must render exactly what Gemma 4's chat template renders."""
import pytest

from systemone_proxy.profiles.base import gemma4_chat

CASES = [
    (None, "State:\nhello\n\nQuestion: q?"),
    ("Make one decision.", "SHARED STATE (JSON string):\n{\"a\": 1}\n\nQUESTION:\nq"),
    ("  padded system  ", "  padded user\n\n"),
    (None, "Thẻ bị mất — <not special> {\"k\":[1,2]}"),
]


@pytest.mark.parametrize("system,user", CASES)
def test_matches_jinja(gemma_template, system, user):
    msgs = ([{"role": "system", "content": system}] if system is not None else [])
    msgs.append({"role": "user", "content": user})
    assert gemma4_chat(system, user) == gemma_template(msgs)


def test_tail_is_empty_thought_channel():
    assert gemma4_chat(None, "x").endswith("<|turn>model\n<|channel>thought\n<channel|>")

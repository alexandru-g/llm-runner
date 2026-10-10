"""Quyet-1.0-Large: prompt text from quyet 1.0.2 (prompt version 2) and answer maths."""
import math

import pytest

from systemone_proxy.profiles.quyet import QuyetLarge
from systemone_proxy.questions import Limits, parse_request

from conftest import FakeBackend

REQ = {
    "state": {"message": "Please close my card, I lost it yesterday."},
    "questions": {
        "intent": {"type": "choice", "instructions": "What does the customer want?",
                   "criteria": {"cancel": "close the card", "limit": "change the limit", "other": None}},
        "urgent": {"type": "noul", "instructions": "The request is urgent."},
        "mood": {"type": "score", "instructions": "How upset is the customer?",
                 "criteria": ["calm", "annoyed", "angry"]},
    },
}


def test_user_text_v2():
    req = parse_request(REQ, QuyetLarge.limits)
    state = '{"message":"Please close my card, I lost it yesterday."}'
    intent, urgent, mood = req.questions
    assert QuyetLarge.user_text(state, intent) == (
        "State:\n" + state + "\n\nQuestion: What does the customer want?\n"
        "Choose the option that fits best.\n\nOptions:\nA. close the card\nB. change the limit\nC. other")
    assert QuyetLarge.user_text(state, urgent).endswith(
        "Choose A if the statement is true for this state, B if it is not.\n\nOptions:\nA. true\nB. false")
    assert QuyetLarge.user_text(state, mood).endswith(
        "(levels are ordered from lowest to highest).\n\nOptions:\nA. calm\nB. annoyed\nC. angry")


def test_build_rows(gemma_tok):
    prof = QuyetLarge(gemma_tok)
    built = prof.build(parse_request(REQ, QuyetLarge.limits))
    assert [r.qid for r in built.rows] == ["intent", "urgent", "mood"]
    row = built.rows[0]
    assert row.ids[0] == 2 and row.ids.count(2) == 1          # one <bos>
    assert row.ids[-5:] == [107, 100, 45518, 107, 101]          # …model\n<|channel>thought\n<channel|>
    assert len(row.label_ids) == 3 and gemma_tok.decode(row.label_ids) == "ABC"
    assert not built.truncated


def test_answers_use_vendor_temperatures(gemma_tok):
    prof = QuyetLarge(gemma_tok)
    req = parse_request(REQ, QuyetLarge.limits)
    noul = prof.answer(req.questions[1], {"main": [-0.1, -2.5]})
    t = QuyetLarge.temperatures["noul"]
    want = 1 / (1 + math.exp(-(2.4) / t))
    assert noul["noul"] == pytest.approx(want)
    assert noul["confidence"] == pytest.approx(max(want, 1 - want))
    score = prof.answer(req.questions[2], {"main": [-3.0, -0.2, -1.0]})
    assert score["legend"] == {"0": "calm", "1": "annoyed", "2": "angry"}
    assert sum(score["probabilities"].values()) == pytest.approx(1.0)


def test_truncates_long_state(gemma_tok):
    prof = QuyetLarge(gemma_tok)
    req = parse_request({"state": ["turn " + str(i) for i in range(4000)],
                         "questions": {"q": REQ["questions"]["urgent"]}}, QuyetLarge.limits)
    built = prof.build(req)
    assert built.truncated == {"q"}
    assert built.warnings[0]["code"] == "state_truncated" and built.warnings[0]["kept"] == "tail"
    assert len(built.rows[0].ids) <= 8000


def test_service_end_to_end(gemma_tok):
    from systemone_proxy.service import DecisionService
    prof = QuyetLarge(gemma_tok)
    a, b, c = prof.letter_ids[:3]
    fake = FakeBackend([{a: -0.05, b: -3.0, c: -5.0, 9: -9.0},
                        {a: -0.3, b: -1.4},
                        {a: -2.0, b: -0.4}])           # C missing → bounded by the min returned
    svc = DecisionService(prof, fake, n_ctx=16384)
    out = svc.decide(REQ)
    assert out["model"] == "quyet-1.0-large"
    assert list(out["answers"]) == ["intent", "urgent", "mood"]
    assert out["answers"]["intent"]["choice"] == "cancel"
    assert out["usage"]["backend_calls"] == 3 and out["usage"]["output_tokens"] == 0
    assert [w["code"] for w in out["warnings"]] == ["label_outside_top_n"]
    assert all(call["cache_prompt"] is False for call in fake.calls)


@pytest.mark.parametrize("body,code", [
    ([], 400),
    ({"questions": {"q": {"type": "noul", "instructions": "x"}}}, 422),
    ({"state": 5, "questions": {"q": {"type": "noul", "instructions": "x"}}}, 422),
    ({"state": "s", "questions": {}}, 422),
    ({"state": "s", "questions": {"q": {"type": "pick", "instructions": "x"}}}, 422),
    ({"state": "s", "questions": {"q": {"type": "choice", "instructions": "x", "criteria": ["a"]}}}, 422),
    ({"state": "s", "questions": {"q": {"type": "noul", "instructions": "x", "criteria": {"yes": 1}}}}, 422),
    ({"state": "s", "questions": {"q": {"type": "noul", "instructions": "x", "threshold": True}}}, 422),
])
def test_validation(body, code):
    from systemone_proxy.errors import ApiError
    with pytest.raises(ApiError) as e:
        parse_request(body, Limits())
    assert e.value.status == code


def test_eleven_options_rejected():
    from systemone_proxy.errors import ApiError
    body = {"state": "s", "questions": {"q": {"type": "choice", "instructions": "x",
                                              "criteria": [str(i) for i in range(11)]}}}
    with pytest.raises(ApiError):
        parse_request(body, QuyetLarge.limits)

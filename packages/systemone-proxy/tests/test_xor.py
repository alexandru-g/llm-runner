"""Xor 26B-A4B v1.1: piecewise prompt ids, labels and answer maths of
serving/xn_base_server.py with the released settings (T choice 2.0, noul 1.7,
score 1.0, two-order choice scoring).

tests/fixtures/xor_golden.json was produced by a separate re-implementation of
the wrapper's build_row/_labels written while extracting the spec (plain
``tokenizers`` on Juspay's tokenizer.json, template head/tail rendered from
chat_template.jinja), not by this profile."""
import json
import math

import pytest

from systemone_proxy.profiles import xor as X
from systemone_proxy.profiles.xor import Xor
from systemone_proxy.questions import parse_request

from conftest import FIXTURES, FakeBackend

GOLDEN = json.loads((FIXTURES / "xor_golden.json").read_text())
STATE_CHOICE = '{"message":"Karte verloren – bitte sperren.","n":[1,2.5,null]}\n'

REQ = {
    "state": STATE_CHOICE,
    "questions": {
        "intent": {"type": "choice", "instructions": "What does the customer want?",
                   "criteria": {"cancel": "close the card", "limit": {"max": 500, "cur": "EUR"},
                                "other": None}},
        "refund": {"type": "noul", "instructions": "Refund?",
                   "criteria": {"false": "nothing to refund"}},
    },
}


@pytest.fixture(scope="module")
def prof(gemma_tok):
    return Xor(gemma_tok)


def test_labels_match_release_manifest(prof):
    assert len(prof.label_ids) == 255
    assert prof.label_names == GOLDEN["labels"]["names"]
    assert prof.label_ids == GOLDEN["labels"]["ids"]
    assert "GZ" not in prof.label_names and prof.label_names[-1] == "IV"
    assert prof.label_ids[:3] == [236776, 236799, 236780]          # A B C, no leading space
    import hashlib
    assert hashlib.sha256(json.dumps(prof.label_ids).encode()).hexdigest() == X.LABEL_IDS_SHA256


def test_readme_example_ids(prof):
    req = parse_request({"state": "The customer was charged twice.",
                         "questions": {"billing": {"type": "noul",
                                                   "instructions": "Is this a billing issue?"}}},
                        Xor.limits)
    built = prof.build(req)
    assert [(r.qid, r.tag) for r in built.rows] == [("billing", "main")]
    row = built.rows[0]
    assert row.ids == GOLDEN["rows"]["readme_noul"]
    assert row.ids[:4] == [2, 105, 2364, 107] and row.ids.count(2) == 1
    assert row.ids[-12:] == X.TAIL_IDS + X.ANSWER_IDS
    assert row.label_ids == [236776, 236799]
    assert built.input_tokens == 47


def test_choice_two_order_rows(prof):
    built = prof.build(parse_request(REQ, Xor.limits))
    assert [(r.qid, r.tag) for r in built.rows] == [
        ("intent", "forward"), ("intent", "reverse"), ("refund", "main")]
    fwd, rev, noul = built.rows
    assert fwd.ids == GOLDEN["rows"]["choice_forward"]
    assert rev.ids == GOLDEN["rows"]["choice_reverse"]
    assert noul.ids == prof.row_ids(prof.E(STATE_CHOICE), "Refund?", ["no: nothing to refund", "yes"])
    assert fwd.label_ids == rev.label_ids == prof.label_ids[:3]
    # usage counts only forward rows, like the vendor
    assert built.input_tokens == len(fwd.ids) + len(noul.ids)
    assert built.warnings == []


def test_piecewise_differs_from_whole_string(prof):
    # a state ending in "\n\n": the wrapper's pieces keep "\n\n","\n\n"
    req = parse_request({"state": "abc\n\n", "questions": {"q": REQ["questions"]["refund"]}},
                        Xor.limits)
    ids = prof.build(req).rows[0].ids
    assert ids == GOLDEN["rows"]["noul_desc"]
    whole = prof.E(prof.row_text("abc\n\n", "Refund?", ["no: nothing to refund", "yes"]))
    assert whole != ids


def test_score_rendering(prof):
    req = parse_request({"state": "x<turn|>y",
                         "questions": {"s": {"type": "score", "instructions": "How upset?",
                                             "criteria": ["calm", None, {"level": 3}]}}},
                        Xor.limits)
    row = prof.build(req).rows[0]
    assert row.ids == GOLDEN["rows"]["score"]
    assert 106 in row.ids[:10]                    # "<turn|>" in the state is parsed as a special
    a = prof.answer(req.questions[0], {"main": [-2.0, -0.5, -1.0]})
    assert a["legend"] == {"0": "calm", "1": "null", "2": '{"level": 3}'}
    p = X.softmax_T([-2.0, -0.5, -1.0], 1.0)
    assert a["probabilities"] == {"0": p[0], "1": p[1], "2": p[2]}
    assert a["score"] == pytest.approx(p[1] + 2 * p[2])
    assert a["confidence"] == max(p)


def test_option_texts():
    q = parse_request(REQ, Xor.limits).questions
    assert X.option_texts(q[0]) == ["cancel: close the card", 'limit: {"max": 500, "cur": "EUR"}',
                                    "other"]
    assert X.option_texts(q[1]) == ["no: nothing to refund", "yes"]
    lst = parse_request({"state": "s", "questions": {"c": {"type": "choice", "instructions": "i",
                                                           "criteria": ["a", 1.5, ""]}}},
                        Xor.limits).questions[0]
    assert X.option_texts(lst) == ["a", "1.5", ""]
    falsy = parse_request({"state": "s", "questions": {"c": {"type": "choice", "instructions": "i",
                                                             "criteria": {"a": 0, "b": False,
                                                                          "c": ""}}}},
                          Xor.limits).questions[0]
    assert X.option_texts(falsy) == ["a: 0", "b: false", "c"]


def test_noul_answer(prof):
    q = parse_request(REQ, Xor.limits).questions[1]
    a = prof.answer(q, {"main": [-1.2, -0.4]})            # [no, yes]
    want = 1 / (1 + math.exp(-(0.8) / 1.7))
    assert a["noul"] == pytest.approx(want)
    assert a["confidence"] == pytest.approx(max(want, 1 - want))
    assert set(a) == {"type", "noul", "confidence"}


def test_choice_two_order_maths(prof):
    q = parse_request(REQ, Xor.limits).questions[0]
    f, r = [-0.2, -2.0, -3.0], [-1.0, -0.7, -1.5]          # reverse order: other, limit, cancel
    a = prof.answer(q, {"forward": f, "reverse": r})
    p_f, p_r = X.softmax_T(f, 1.0), X.softmax_T(r, 1.0)
    avg = [(p_f[0] + p_r[2]) / 2, (p_f[1] + p_r[1]) / 2, (p_f[2] + p_r[0]) / 2]
    w = [x ** 0.5 for x in avg]
    want = [x / sum(w) for x in w]
    got = [a["probabilities"][k] for k in ("cancel", "limit", "other")]
    assert got == pytest.approx(want, rel=1e-12)
    assert a["choice"] == "cancel" and a["confidence"] == max(got)
    # forward-only fallback: plain softmax at T=2
    solo = prof.answer(q, {"forward": f})
    assert list(solo["probabilities"].values()) == X.softmax_T(f, 2.0)


def test_reverse_skipped_when_too_long(prof, monkeypatch):
    built = prof.build(parse_request(REQ, Xor.limits))
    fwd_len = len(built.rows[0].ids)
    rev_len = len(built.rows[1].ids)
    assert fwd_len == rev_len
    real = prof.row_ids
    monkeypatch.setattr(prof, "row_ids", lambda s, i, o: real(s, i, o) + (
        [0] * 10 if o[0] == "other" else []))
    monkeypatch.setattr(X, "MAX_CONTEXT", fwd_len + 5)
    built = prof.build(parse_request({"state": REQ["state"],
                                      "questions": {"intent": REQ["questions"]["intent"]}},
                                     Xor.limits))
    assert [r.tag for r in built.rows] == ["forward"]
    assert built.warnings == [{"code": "reverse_order_skipped", "question": "intent",
                               "prompt_tokens": rev_len + 10, "max_context": fwd_len + 5}]


def test_forward_too_long_rejected(prof, monkeypatch):
    from systemone_proxy.errors import ApiError
    monkeypatch.setattr(X, "MAX_CONTEXT", 40)
    with pytest.raises(ApiError) as e:
        prof.build(parse_request(REQ, Xor.limits))
    assert e.value.code == "input_too_long"


def test_selftests(prof):
    texts = prof.selftest_texts()
    assert prof.E(texts[0]) == GOLDEN["rows"]["readme_noul"]
    for t in texts:
        assert t.startswith("<bos><|turn>user\nContext:\n") and t.endswith("<channel|>Answer: (")
    labels = prof.selftest_labels()
    assert labels[236776] == "A" and labels[8686] == "AA"
    assert all(prof.tok.decode([i]) == s for i, s in labels.items())


def test_matches():
    meta = {"general.architecture": "gemma4", "general.basename": "Xor",
            "general.name": "Xor 26B A4B", "general.size_label": "26B-A4B"}
    assert Xor.matches(meta, "Xor-26B-A4B.Q4_K_M.gguf")
    assert not Xor.matches({**meta, "general.size_label": "31B"}, "x.gguf")
    assert not Xor.matches({**meta, "general.basename": "Rune"}, "x.gguf")
    assert not Xor.matches({**meta, "general.architecture": "llama"}, "x.gguf")
    assert not Xor.matches({"general.basename": "Quyet-1.0", "general.size_label": "Large"}, "q.gguf")


def test_spec_without_instance():
    s = Xor.spec(None)
    assert set(s) >= {"confidence", "labels", "noul_order", "state", "limits", "calls",
                      "extras", "notes"}


def test_service_end_to_end(prof):
    from systemone_proxy.service import DecisionService
    a, b, c = prof.label_ids[:3]
    fake = FakeBackend([{a: -0.2, b: -2.0, c: -3.0, 9: -9.0},      # intent forward
                        {a: -1.0, b: -0.7, c: -1.5},                # intent reverse
                        {a: -1.2, b: -0.4}])                        # refund
    out = DecisionService(prof, fake, n_ctx=32768).decide(REQ)
    assert out["model"] == "xor-26b-a4b"
    assert list(out["answers"]) == ["intent", "refund"]
    assert out["answers"]["intent"]["choice"] == "cancel"
    assert out["answers"]["refund"]["noul"] == pytest.approx(1 / (1 + math.exp(-0.8 / 1.7)))
    assert out["usage"]["backend_calls"] == 3
    assert out["usage"]["input_tokens"] == len(fake.calls[0]["ids"]) + len(fake.calls[2]["ids"])
    assert out["warnings"] == []

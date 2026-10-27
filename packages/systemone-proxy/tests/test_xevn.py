"""Xevn-4B: piecewise prompt ids from xevn_infer.py, option rendering, readout maths."""
import json
import math

import pytest

from systemone_proxy.errors import ApiError
from systemone_proxy.profiles import xevn as X
from systemone_proxy.profiles.xevn import Xevn4B
from systemone_proxy.questions import parse_request
from systemone_proxy.service import DecisionService

from conftest import FIXTURES, FakeBackend

GOLDEN = json.loads((FIXTURES / "xevn_readme.json").read_text())

# The model card's example, verbatim
README_REQ = {
    "state": X.README_STATE,
    "questions": {
        "team": {"type": "choice", "instructions": "Which team should handle this ticket?",
                 "criteria": {"billing": "Payments and refunds", "shipping": "Delivery",
                              "account": "Login and account"}},
        "charged_twice": {"type": "noul", "instructions": "Was the customer charged more than once?"},
        "urgency": {"type": "score", "instructions": "How urgent is this ticket?",
                    "criteria": ["Low", "Medium", "High"]},
    },
}
LABEL_IDS = [357, 417, 351, 414, 458, 426, 469, 462, 353, 604, 710, 436, 380, 443, 496, 387,
             1167, 423, 326, 345, 533, 629, 457, 1543, 783, 1799]


@pytest.fixture(scope="session")
def xevn_tok():
    from systemone_proxy import tokenizer
    try:
        return tokenizer.load(Xevn4B.tokenizer_source)
    except Exception as e:  # offline, no HF access
        pytest.skip(f"Xevn tokenizer unavailable: {e}")


@pytest.fixture(scope="session")
def prof(xevn_tok):
    return Xevn4B(xevn_tok)


def req(body):
    return parse_request(body, Xevn4B.limits)


# ------------------------------------------------------------------ prompt --

def test_readme_pieces_and_golden_ids(prof):
    r = req(README_REQ)
    for q in r.questions:
        assert X.prompt_parts(README_REQ["state"], q) == (GOLDEN[q.qid]["a"], GOLDEN[q.qid]["b"])
    built = prof.build(r)
    assert [row.ids for row in built.rows] == [GOLDEN[q]["ids"] for q in ("team", "charged_twice", "urgency")]
    assert [len(row.ids) for row in built.rows] == [58, 45, 62]
    assert built.input_tokens is None                    # the service sums the rows: 165
    assert built.rows[0].ids[25] == 271 and built.rows[0].ids[-2:] == [15666, 25]   # "\n\n", "Answer:"
    assert built.rows[0].label_ids == [357, 417, 351] and built.rows[1].label_ids == [357, 417]


def test_pieces_are_tokenized_apart(prof, xevn_tok):
    """Instructions starting with a newline tokenize differently as one string."""
    q = req({"state": "s", "questions": {"q": {"type": "noul", "instructions": "\nok?"}}}).questions[0]
    a, b = X.prompt_parts("s", q)
    ids = prof.question_ids("s", q)
    assert ids == xevn_tok.encode(a, special=True) + xevn_tok.encode(b, special=True)
    assert ids != xevn_tok.encode(a + b, special=True)


def test_option_lines_follow_xevn_infer():
    r = req({"state": "s", "questions": {
        "c": {"type": "choice", "instructions": "pick",
              "criteria": {"x": None, "y": "", "z": "z", "w": 3, "v": {"k": "ü", "n": [1.0]}}},
        "l": {"type": "choice", "instructions": "pick", "criteria": ["p", "q"]},
        "s": {"type": "score", "instructions": "rate", "criteria": ["lo", None, "", "3"]},
        "n2": {"type": "noul", "instructions": "ok?", "criteria": {"true": "yes it is", "false": ""}},
        "n1": {"type": "noul", "instructions": "ok?", "criteria": {"true": "only this"}},
    }})
    c, lst, s, n2, n1 = r.questions
    assert X.prompt_parts("S", c)[1] == ('pick\nA. x\nB. y\nC. z\nD. w: 3\n'
                                         'E. v: {\n "k": "ü",\n "n": [\n  1.0\n ]\n}\nAnswer:')
    assert X.prompt_parts("S", lst)[1] == "pick\nA. p\nB. q\nAnswer:"
    assert X.prompt_parts("S", s)[1] == ("rate (levels in ascending order)\nA. 0: lo\nB. 1: null\n"
                                         "C. 2: \nD. 3\nAnswer:")   # level text == index
    assert X.prompt_parts("S", n2)[1] == "ok?\nA. Yes: yes it is\nB. No: \nAnswer:"
    assert X.prompt_parts("S", n1)[1] == "ok?\nA. Yes\nB. No\nAnswer:"


def test_state_is_indent_1_json_with_raw_unicode():
    assert X.to_text({"ü": [1, 1e-05]}) == '{\n "ü": [\n  1,\n  1e-05\n ]\n}'
    assert X.to_text("  raw  ") == "  raw  "


def test_label_ids(prof, xevn_tok):
    assert prof.label_ids == LABEL_IDS
    assert [xevn_tok.decode([i]) for i in prof.label_ids] == [" " + c for c in X.LETTERS]


def test_26_options_and_no_more(prof):
    body = {"state": "s", "questions": {"q": {"type": "choice", "instructions": "x",
                                              "criteria": {f"o{i}": f"d{i}" for i in range(26)}}}}
    assert prof.build(req(body)).rows[0].label_ids == LABEL_IDS
    body["questions"]["q"]["criteria"]["o26"] = "d"
    with pytest.raises(ApiError):
        req(body)
    with pytest.raises(ApiError):
        req({"state": "s", "questions": {"q": {"type": "score", "instructions": "x",
                                               "criteria": [str(i) for i in range(11)]}}})


def test_a_prompt_over_the_context_is_refused(prof):
    body = {"state": "word " * 100, "questions": {"q": {"type": "noul", "instructions": "ok?"}}}
    with pytest.raises(ApiError) as e:
        DecisionService(prof, FakeBackend([]), n_ctx=64).decide(body)
    assert e.value.code == "input_too_long"


# ------------------------------------------------------------------ answers --

def vendor(lp, t):
    """xevn_infer.readout, re-typed."""
    peak = max(lp)
    p = [math.exp(v - peak) for v in lp]
    total = sum(p)
    p = [max(1e-12, min(1.0, v / total)) ** (1 / t) for v in p]
    return [v / sum(p) for v in p]


def test_temperature_matches_the_vendor_even_past_the_clip():
    lp = [0.0, -30.0]
    assert X.temper(lp, 1.1) == pytest.approx(vendor(lp, 1.1), rel=1e-12)
    assert X.temper(lp, 1.1)[1] == pytest.approx(1e-12 ** (1 / 1.1), rel=1e-6)   # not softmax(lp/T)
    lp = [-0.2, -1.9, -3.0]
    e = [math.exp(v / 1.1) for v in lp]
    assert X.temper(lp, 1.1) == pytest.approx([v / sum(e) for v in e], rel=1e-12)


def test_choice_score_and_the_noul_clamp(prof):
    team, charged, urgency = req(README_REQ).questions
    lp = [-0.01, -6.0, -7.0]
    a = prof.answer(team, {"main": lp})
    p = vendor(lp, 1.1)
    assert a["choice"] == "billing" and list(a["probabilities"].values()) == pytest.approx(p)
    assert a["confidence"] == pytest.approx((max(p) - 1 / 3) / (2 / 3))
    s = prof.answer(urgency, {"main": [-3.0, -1.7, -0.2]})
    ps = vendor([-3.0, -1.7, -0.2], 1.2)
    assert s["score"] == pytest.approx(ps[1] + 2 * ps[2])
    assert s["legend"] == {"0": "Low", "1": "Medium", "2": "High"}
    assert set(s) == {"type", "score", "confidence", "legend", "probabilities"}
    assert prof.answer(charged, {"main": [math.log(0.6), math.log(0.4)]})["noul"] == 0.8
    assert prof.answer(charged, {"main": [math.log(0.5), math.log(0.5)]})["noul"] == 0.8
    low = prof.answer(charged, {"main": [math.log(0.3), math.log(0.7)]})
    assert low == {"type": "noul", "noul": 0.2, "confidence": 0.8}
    hi = prof.answer(charged, {"main": [0.0, -8.0]})
    assert hi["noul"] == pytest.approx(vendor([0.0, -8.0], 1.2)[0]) and hi["noul"] > 0.8


def test_service_reads_one_row_per_question(prof):
    A, B, C = LABEL_IDS[:3]
    fake = FakeBackend([{A: -0.01, B: -6.0, C: -7.0}, {A: -0.01, B: -5.0},
                        {A: -3.0, B: -1.7, C: -0.2}])
    out = DecisionService(prof, fake, n_ctx=8192).decide(README_REQ)
    assert [len(c["ids"]) for c in fake.calls] == [58, 45, 62]
    assert out["usage"]["input_tokens"] == 165
    assert out["answers"]["team"]["choice"] == "billing"
    assert out["answers"]["charged_twice"]["noul"] > 0.98


# -------------------------------------------------------------- detection --

def test_matches():
    meta = {"general.architecture": "qwen35", "general.basename": "Xevn", "general.size_label": "4B"}
    assert Xevn4B.matches(meta, "Xevn-4B-Q8_0.gguf")
    assert Xevn4B.matches({"general.architecture": "qwen35"}, "xevn-4b.Q4_K_M.gguf")
    assert not Xevn4B.matches({**meta, "general.basename": "Qwen3.5"}, "Qwen3.5-4B-Q8_0.gguf")
    assert not Xevn4B.matches({**meta, "general.basename": "h2o-lightning"}, "h2o.gguf")
    assert not Xevn4B.matches({**meta, "general.architecture": "gemma4"}, "Xevn-4B.gguf")


def test_spec_needs_no_instance():
    s = Xevn4B.spec(object.__new__(Xevn4B))
    assert {"confidence", "labels", "noul_order", "state", "limits", "calls", "notes"} <= set(s)

"""H2O-Lightning-4B: prompt bytes and ids from h2o_lightning_shim.py, readout maths, truncation."""
import json
import math

import pytest

from systemone_proxy.errors import ApiError
from systemone_proxy.profiles import h2o_lightning as H
from systemone_proxy.profiles.h2o_lightning import H2OLightning4B
from systemone_proxy.questions import parse_request

from conftest import FIXTURES, FakeBackend

GOLDEN = json.loads((FIXTURES / "h2o_lightning_readme.json").read_text())

# README.md L257-264, verbatim
README_REQ = {
    "state": "Ticket 4471: since 09:10 the checkout page returns HTTP 500 for every customer; no orders have gone through in 40 minutes. Severity guide: P1 = a revenue path fully down; P2 = degraded but working; P3 = cosmetic.",
    "questions": {
        "priority": {"type": "choice", "instructions": "Which priority does the severity guide assign?",
                     "criteria": {"p1": "Revenue path fully down", "p2": "Degraded but working", "p3": "Cosmetic"}},
        "all_users": {"type": "noul", "instructions": "The problem affects every customer.",
                      "criteria": {"true": "Affects everyone", "false": "Affects only some"}},
        "urgency": {"type": "score", "instructions": "How urgent is this ticket?", "criteria": ["Low", "Medium", "High"]},
    },
}

# vocab ids of "Ġ" + label (checked against tokenizer.json while writing the spec)
FIRST_LABEL_IDS = {"A": 357, "B": 417, "C": 351, "D": 414, "E": 458, "Z": 1799, "a": 264, "z": 1110,
                   "α": 18470, "ω": 162391, "in": 303, "the": 279, "The": 561, "ell": 24792}


@pytest.fixture(scope="session")
def h2o_tok():
    from systemone_proxy import tokenizer
    try:
        return tokenizer.load(H2OLightning4B.tokenizer_source)
    except Exception as e:  # offline, no HF access
        pytest.skip(f"H2O-Lightning tokenizer unavailable: {e}")


@pytest.fixture(scope="session")
def prof(h2o_tok):
    return H2OLightning4B(h2o_tok)


@pytest.fixture(scope="session")
def qwen_template():
    jinja2 = pytest.importorskip("jinja2")
    env = jinja2.Environment(loader=jinja2.FileSystemLoader(str(FIXTURES)),
                             extensions=["jinja2.ext.loopcontrols"])

    def raise_exception(msg):
        raise RuntimeError(msg)

    env.globals["raise_exception"] = raise_exception
    return env.get_template("h2o_lightning_chat_template.jinja")


def req(body):
    return parse_request(body, H2OLightning4B.limits)


# ------------------------------------------------------------------ prompt --

def test_readme_prompt_text():
    r = req(README_REQ)
    for q in r.questions:
        assert H.render_chat(H.user_turn(README_REQ["state"], q)) == GOLDEN["prompts"][q.qid]


def test_readme_golden_ids_and_usage(prof):
    built = prof.build(req(README_REQ))
    assert [r.qid for r in built.rows] == ["priority", "all_users", "urgency"]
    for row in built.rows:
        assert row.ids == GOLDEN["ids"][row.qid]
    assert [len(r.ids) for r in built.rows] == [153, 140, 144]
    assert built.input_tokens == GOLDEN["readme_input_tokens"] == 231
    assert built.rows[0].ids[0] == 248045                          # <|im_start|>, no BOS
    assert built.rows[0].ids[-11:] == [248046, 198, 248045, 74455, 198, 248068, 271, 248069, 271, 15666, 25]
    assert built.rows[0].ids[-1] == 25                             # ':' of "Answer:"
    assert built.rows[0].label_ids == [357, 417, 351]
    assert built.rows[1].label_ids == [357, 417]                    # A = true, B = false
    assert not built.truncated and not built.warnings


@pytest.mark.parametrize("system_user", [
    "record: x\nquestion: q\noptions:\nA) a: b",
    "record: {\"k\":\"ü — <not special>\"}\nquestion: q?\noptions:\nA) true: t\nB) false: f  \n",
])
def test_render_chat_matches_jinja(qwen_template, system_user):
    """The shim's string equals the model's template for [system, user] with the
    generation prompt and thinking off, followed by the prefill; it also equals
    the continued assistant turn holding the prefill (the image path)."""
    msgs = [{"role": "system", "content": H.SYSTEM}, {"role": "user", "content": system_user}]
    gen = qwen_template.render(messages=msgs, add_generation_prompt=True, enable_thinking=False)
    assert H.render_chat(system_user) == gen + "Answer:"
    cont = qwen_template.render(messages=msgs + [{"role": "assistant", "content": "Answer:"}],
                                add_generation_prompt=False, enable_thinking=False)
    assert cont == gen + "Answer:<|im_end|>\n"   # vLLM's continue_final_message drops the end


def test_option_texts_follow_the_shim():
    r = req({"state": "s", "questions": {
        "c": {"type": "choice", "instructions": "  pick", "criteria": {"x": None, "y": 3, "z": {"k": "ü"}, "w": ""}},
        "l": {"type": "choice", "instructions": "pick", "criteria": ["p", "q"]},
        "s": {"type": "score", "instructions": "rate", "criteria": ["lo", None, 2]},
        "n": {"type": "noul", "instructions": "ok?"},
        "m": {"type": "noul", "instructions": "ok?", "criteria": {"true": "", "false": False}},
    }})
    c, l, s, n, m = r.questions
    assert H.user_turn("S", c) == ('record: S\nquestion: pick\noptions:\nA) x: x\nB) y: 3\n'
                                   'C) z: {"k": "ü"}\nD) w: ')
    assert H.option_texts(l) == ["p", "q"]
    assert H.option_texts(s) == ["lo", "null", "2"]
    assert H.user_turn("S", s).endswith("A) 0: lo\nB) 1: null\nC) 2: 2")
    assert H.option_texts(n) == ["the statement holds", "it does not"]
    assert H.option_texts(m) == ["the statement holds", "false"]


def test_state_is_compact_python_json():
    assert H.render_state({"a": [1, 1.0, 1e-05], "ü": "x/y"}) == '{"a":[1,1.0,1e-05],"ü":"x/y"}'
    assert H.render_state("  raw  ") == "  raw  "


def test_label_ids(prof, h2o_tok):
    assert len(prof.label_ids) == 255 and len(set(prof.label_ids)) == 255
    for label, i in FIRST_LABEL_IDS.items():
        assert prof.label_ids[H.LABELS.index(label)] == i
        assert h2o_tok.decode([i]) == " " + label
    assert {k: prof.label_ids[H.LABELS.index(k)] for k in "ABC"} == GOLDEN["label_ids_first"]


def test_255_options_and_more(prof):
    body = {"state": "s", "questions": {"q": {"type": "choice", "instructions": "x",
                                              "criteria": {f"o{i}": f"d{i}" for i in range(255)}}}}
    built = prof.build(req(body))
    assert built.rows[0].label_ids == prof.label_ids
    body["questions"]["q"]["criteria"]["o255"] = "d"
    with pytest.raises(ApiError):
        req(body)


# --------------------------------------------------------------- truncation --

def test_short_state_is_untouched(prof):
    text = "w " * 15000                                   # 30000 bytes: below the byte shortcut
    assert prof.truncate_state(text) == (text, None)


def test_long_state_keeps_head_and_tail(prof, h2o_tok):
    state = " ".join(f"w{i}" for i in range(20000))       # > 32000 tokens
    n = len(h2o_tok.encode(state))
    assert n > 32000
    built = prof.build(req({"state": state, "questions": {
        "a": {"type": "noul", "instructions": "ok?"},
        "b": {"type": "score", "instructions": "rate", "criteria": ["lo", "hi"]}}}))
    assert built.truncated == {"a", "b"}
    assert built.warnings == [{"code": "state_truncated", "state_tokens": n, "kept_tokens": 32000,
                               "kept": "head and tail"}]
    ids = h2o_tok.encode(state)
    want = h2o_tok.decode(ids[:16000]) + "\n...\n" + h2o_tok.decode(ids[-16000:])
    text = h2o_tok.decode(built.rows[0].ids)
    assert "record: " + want + "\nquestion: ok?" in text
    assert text.startswith("<|im_start|>system\n") and " w19999\nquestion" in text


# ------------------------------------------------------------------ answers --

def sm(lp, t):
    e = [math.exp(x / t) for x in lp]
    return [x / sum(e) for x in e]


def test_choice_and_score(prof):
    r = req(README_REQ)
    priority, _, urgency = r.questions
    lp = [-0.2, -1.9, -3.0]
    a = prof.answer(priority, {"main": lp})
    p = sm(lp, 0.75)
    assert a["choice"] == "p1"
    assert list(a["probabilities"]) == ["p1", "p2", "p3"]
    for k, x in zip(["p1", "p2", "p3"], p):
        assert a["probabilities"][k] == pytest.approx(x, abs=1e-12)
    assert a["confidence"] == pytest.approx((max(p) - 1 / 3) / (1 - 1 / 3))
    s = prof.answer(urgency, {"main": [-1.0, -0.5, -2.0]})
    ps = sm([-1.0, -0.5, -2.0], 0.65)
    assert s["score"] == pytest.approx(ps[1] + 2 * ps[2], abs=1e-12)
    assert s["legend"] == {"0": "Low", "1": "Medium", "2": "High"}
    assert s["confidence"] == pytest.approx(max(ps))           # v1.4.1
    assert set(s) == {"type", "score", "confidence", "legend", "probabilities"}


def test_noul_floor_and_threshold(prof):
    q = req({"state": "s", "questions": {"d": {"type": "noul", "instructions": "Holds?",
                                               "threshold": 0.9}}}).questions[0]
    a = prof.answer(q, {"main": [math.log(0.55), math.log(0.45)]})
    assert a == {"type": "noul", "noul": 0.801, "confidence": 0.801, "threshold": 0.9, "decision": False}
    a = prof.answer(q, {"main": [math.log(0.45), math.log(0.55)]})
    assert a["noul"] == pytest.approx(0.199) and a["confidence"] == pytest.approx(0.801)
    lp = [math.log(0.02), math.log(0.98)]                          # committed: left alone
    a = prof.answer(q, {"main": lp})
    assert a["noul"] == pytest.approx(sm(lp, 0.8)[0], abs=1e-12)
    tie = prof.answer(q, {"main": [-1.0, -1.0]})
    assert tie["noul"] == 0.801                                    # a tie goes to yes
    plain = req({"state": "s", "questions": {"d": {"type": "noul", "instructions": "x"}}}).questions[0]
    assert set(prof.answer(plain, {"main": [-0.1, -3.0]})) == {"type", "noul", "confidence"}


def test_noul_threshold_alias(prof):
    q = req({"state": "s", "questions": {"d": {"type": "noul", "instructions": "x",
                                               "noul_threshold": 0.5}}}).questions[0]
    a = prof.answer(q, {"main": [-0.1, -3.0]})
    assert a["threshold"] == 0.5 and a["decision"] is True
    q = req({"state": "s", "questions": {"d": {"type": "noul", "instructions": "x",
                                               "noul_threshold": True}}}).questions[0]
    assert "threshold" not in prof.answer(q, {"main": [-0.1, -3.0]})


def test_service_end_to_end(prof):
    from systemone_proxy.service import DecisionService
    A, B, C = prof.label_ids[:3]
    fake = FakeBackend([{A: -0.05, B: -3.0, C: -5.0, 9: -9.0},
                        {A: -0.1, B: -2.6},
                        {A: -4.0, B: -2.0}])                       # C missing → bounded
    out = DecisionService(prof, fake, n_ctx=40960).decide(README_REQ)
    assert out["model"] == "h2o-lightning-4b"
    assert out["usage"]["input_tokens"] == 231 and out["usage"]["backend_calls"] == 3
    assert [c["ids"] for c in fake.calls] == [GOLDEN["ids"][k] for k in ("priority", "all_users", "urgency")]
    assert out["answers"]["priority"]["choice"] == "p1"
    assert [w["code"] for w in out["warnings"]] == ["label_outside_top_n"]


# ---------------------------------------------------------- detect / tests --

@pytest.mark.parametrize("meta,name,want", [
    ({"general.architecture": "qwen35", "general.basename": "h2o-lightning",
      "general.name": "H2O Lightning 4b", "general.size_label": "4B"}, "h2o-lightning-4b.Q8_0.gguf", True),
    ({"general.architecture": "qwen35"}, "h2o-lightning-4b.Q4_K_M.gguf", True),
    ({"general.architecture": "qwen35", "general.basename": "Qwen3.5", "general.size_label": "4B"},
     "Qwen3.5-4B.Q8_0.gguf", False),
    ({"general.architecture": "gemma4", "general.basename": "quyet-1.0", "general.size_label": "large"},
     "Quyet-1.0-Large.Q4_K_M.gguf", False),
    ({"general.architecture": "llama", "general.basename": "h2o-lightning", "general.size_label": "4B"},
     "x.gguf", False),
])
def test_matches(meta, name, want):
    assert H2OLightning4B.matches(meta, name) is want


def test_spec_without_state():
    s = H2OLightning4B.__new__(H2OLightning4B).spec()
    assert set(s) == {"confidence", "labels", "noul_order", "state", "limits", "calls", "extras",
                      "notes", "images"}
    assert set(s["extras"]) == {"threshold", "decision"}


class TokBackend:
    def __init__(self, tok, skew=False):
        self.tok, self.skew = tok, skew

    def tokenize(self, text, parse_special=True):
        ids = self.tok.encode(text, special=parse_special)
        return ids[:-1] + [ids[-1] + 1] if self.skew else ids

    def detokenize(self, ids):
        return self.tok.decode(ids)


def test_selftest(prof, h2o_tok):
    texts = prof.selftest_texts()
    assert texts[0] == GOLDEN["prompts"]["priority"]
    assert h2o_tok.encode(texts[1])[-1] == FIRST_LABEL_IDS["α"]
    assert prof.selftest_labels()[357] == " A"
    assert prof.selftest(TokBackend(h2o_tok)) == []
    assert any("tokenizer mismatch" in p for p in prof.selftest(TokBackend(h2o_tok, skew=True)))

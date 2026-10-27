"""iota-decide-4b v0.3: prompt bytes and ids from letter_decide.py, truncation, readout."""
import json
import math

import pytest

from systemone_proxy.errors import ApiError
from systemone_proxy.profiles import iota_decide as I
from systemone_proxy.profiles.iota_decide import IotaDecide4B
from systemone_proxy.questions import parse_request
from systemone_proxy.service import DecisionService

from conftest import FIXTURES, FakeBackend

GOLDEN = json.loads((FIXTURES / "iota_decide_readme.json").read_text())

# The model card's request
README_REQ = {
    "state": I.README_STATE,
    "questions": {
        "approve": {"type": "noul", "instructions": "Can the refund be issued without approval?"},
        "route": {"type": "choice", "instructions": "Which team handles it?",
                  "criteria": {"billing": "payments and refunds", "tech": "technical faults"}},
    },
}
LABEL_IDS = [357, 417, 351, 414, 458, 426, 469, 462, 353, 604, 710, 436, 380, 443, 496, 387,
             1167, 423, 326, 345, 533, 629, 457, 1543, 783, 1799]


@pytest.fixture(scope="session")
def iota_tok():
    from systemone_proxy import tokenizer
    try:
        return tokenizer.load(IotaDecide4B.tokenizer_source)
    except Exception as e:  # offline, no HF access
        pytest.skip(f"iota-decide tokenizer unavailable: {e}")


@pytest.fixture(scope="session")
def prof(iota_tok):
    return IotaDecide4B(iota_tok)


def req(body):
    return parse_request(body, IotaDecide4B.limits)


# ------------------------------------------------------------------ prompt --

def test_readme_prompts_and_golden_ids(prof):
    r = req(README_REQ)
    for q in r.questions:
        assert I.prompt(I.render_state(README_REQ["state"]), q) == GOLDEN[q.qid]["prompt"]
    built = prof.build(r)
    assert [row.ids for row in built.rows] == [GOLDEN["approve"]["ids"], GOLDEN["route"]["ids"]]
    assert [len(row.ids) for row in built.rows] == [116, 120]
    assert built.input_tokens is None and not built.truncated     # usage: 236, no dedup
    assert built.rows[0].ids[0] == 248045                         # <|im_start|>, no BOS
    assert built.rows[0].ids[-11:] == [248046, 198, 248045, 74455, 198, 248068, 271, 248069, 271,
                                       15666, 25]
    assert built.rows[0].label_ids == [357, 417]                  # A = true


def test_option_texts_use_python_str():
    r = req({"state": "s", "questions": {
        "c": {"type": "choice", "instructions": "pick",
              "criteria": {"x": None, "y": "", "z": 0, "w": {"k": "ü"}, "v": True, "u": "d"}},
        "s": {"type": "score", "instructions": "rate", "criteria": ["lo", None, True, 1.0, {"a": 1}]},
        "n": {"type": "noul", "instructions": "ok?", "criteria": {"true": "holds", "false": ""}},
        "m": {"type": "noul", "instructions": "ok?"},
    }})
    c, s, n, m = r.questions
    assert I.option_texts(c) == ["x", "y", "z", "w: {'k': 'ü'}", "v: True", "u: d"]
    assert I.option_texts(s) == ["lo", "None", "True", "1.0", "{'a': 1}"]
    assert I.option_texts(n) == ["Yes: holds", "No"]
    assert I.option_texts(m) == ["Yes", "No"]
    p = I.prompt("S", n)
    assert "\nQUESTION: ok?\n\nOPTIONS:\nA) Yes: holds\nB) No\n\nAnswer with the letter" in p


def test_state_uses_default_json_separators():
    assert I.render_state({"a": [1, 1.0], "ü": None}) == '{"a": [1, 1.0], "ü": null}'
    assert I.render_state("  raw ") == "  raw "


def test_label_ids(prof, iota_tok):
    assert prof.label_ids == LABEL_IDS
    assert [iota_tok.decode([i]) for i in prof.label_ids] == [" " + c for c in I.LETTERS]
    assert iota_tok.encode("<|audio_start|>", special=True) == [248070]   # v0.3's extra tokens


def test_26_options_and_no_more(prof):
    body = {"state": "s", "questions": {"q": {"type": "score", "instructions": "x",
                                              "criteria": [str(i) for i in range(26)]}}}
    assert prof.build(req(body)).rows[0].label_ids == LABEL_IDS
    body["questions"]["q"]["criteria"].append("26")
    with pytest.raises(ApiError):
        req(body)


# --------------------------------------------------------------- truncation --

def test_a_long_prompt_is_cut_as_the_vendor_does(prof, iota_tok):
    state = " ".join(f"w{i}" for i in range(12000))
    body = {"state": state, "questions": {"q": {"type": "noul", "instructions": "Is it long?"}}}
    built = prof.build(req(body))
    [row] = built.rows
    assert len(row.ids) == 16287                                  # worked out from the vendor code
    q = req(body).questions[0]
    n = len(iota_tok.encode(I.prompt(state, q), special=True))
    keep = max(0, len(state) - int(len(state) * (n - 16384) / n) - 200)
    cut = state[: keep // 2] + "\n[...]\n" + state[len(state) - keep // 2:]
    assert row.ids == iota_tok.encode(I.prompt(cut, q), special=True)[-16384:]
    assert built.truncated == {"q"} and built.warnings[0]["code"] == "state_truncated"


def test_the_character_cut_on_a_mixed_script_state(prof):
    state = "".join("日本語のテキスト" if i % 3 else "plain english words " for i in range(9000))
    built = prof.build(req({"state": state, "questions": {"q": {"type": "noul",
                                                               "instructions": "Is it long?"}}}))
    assert len(built.rows[0].ids) == 16371                        # worked out from the vendor code
    assert built.rows[0].ids[0] == 248045                         # under 16384: nothing cut at the front


# ------------------------------------------------------------------ answers --

T = math.exp(-0.04101192206144333)


def sm(lp):
    e = [math.exp(v / T) for v in lp]
    return [v / sum(e) for v in e]


def test_one_temperature_and_max_p_confidence(prof):
    approve, route = req(README_REQ).questions
    assert T == pytest.approx(0.9598176868741493, rel=1e-15)
    a = prof.answer(route, {"main": [-0.3, -1.5]})
    p = sm([-0.3, -1.5])
    assert a["choice"] == "billing" and a["confidence"] == pytest.approx(max(p), rel=1e-12)
    assert list(a["probabilities"].values()) == pytest.approx(p, rel=1e-12)
    s = prof.answer(req({"state": "s", "questions": {"u": {
        "type": "score", "instructions": "x", "criteria": ["Low", None, "High"]}}}).questions[0],
        {"main": [-2.0, -1.0, -0.4]})
    ps = sm([-2.0, -1.0, -0.4])
    assert s["score"] == pytest.approx(ps[1] + 2 * ps[2])
    assert s["confidence"] == pytest.approx(max(ps))
    assert s["legend"] == {"0": "Low", "1": None, "2": "High"}


def test_noul_floor(prof):
    approve = req(README_REQ).questions[0]
    assert prof.answer(approve, {"main": [math.log(0.6), math.log(0.4)]})["noul"] == 0.801
    low = prof.answer(approve, {"main": [math.log(0.4), math.log(0.6)]})
    assert low["noul"] == pytest.approx(0.199) and low["confidence"] == pytest.approx(0.801)
    sure = prof.answer(approve, {"main": [0.0, -5.0]})
    assert sure["noul"] == pytest.approx(sm([0.0, -5.0])[0]) and sure["noul"] > 0.99


def test_service_end_to_end(prof):
    A, B = LABEL_IDS[:2]
    fake = FakeBackend([{A: -0.05, B: -3.0}, {A: -0.2, B: -2.0}])
    out = DecisionService(prof, fake, n_ctx=16896).decide(README_REQ)
    assert out["usage"]["input_tokens"] == 236 and out["usage"]["backend_calls"] == 2
    assert out["answers"]["route"]["choice"] == "billing"


# -------------------------------------------------------------- detection --

def test_matches():
    meta = {"general.architecture": "qwen35", "general.basename": "iota-decide",
            "general.version": "v0.3", "general.size_label": "4B"}
    assert IotaDecide4B.matches(meta, "iota-decide-4b-v0.3-Q8_0.gguf")
    assert IotaDecide4B.matches({"general.architecture": "qwen35"}, "iota-decide-4b-v0.3.Q8_0.gguf")
    assert not IotaDecide4B.matches({**meta, "general.version": "v0.2"}, "x.gguf")
    assert not IotaDecide4B.matches({**meta, "general.architecture": "clef"}, "iota-decide-4b.gguf")
    assert not IotaDecide4B.matches({**meta, "general.basename": "Qwen3.5"}, "Qwen3.5-4B.gguf")


def test_spec_needs_no_instance():
    s = IotaDecide4B.spec(object.__new__(IotaDecide4B))
    assert {"confidence", "labels", "noul_order", "state", "limits", "calls", "notes"} <= set(s)

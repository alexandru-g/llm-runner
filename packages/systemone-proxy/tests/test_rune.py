"""Rune 26B-A4B v3: surogate decisions v1 prompts, labels and answer maths.

Fixtures:
* rune_golden_v1.json: surogate's golden.json (csrc/src/testing/serve/fixtures/serve/
  decisions_v1/, @65f6f6f), written by its independent Python reference make_golden.py.
  Its codes past 26 options come from a synthetic tokenizer; the real Gemma 4
  codebook differs from index 207 on (GZ is not a single token).
* rune_ticket_prompts.json: the 'ticket' case's full prompts and token ids.
"""
import json
import math
from pathlib import Path

import pytest

from systemone_proxy.errors import ApiError
from systemone_proxy.profiles.base import gemma4_chat
from systemone_proxy.profiles.rune import (SYSTEM, EXTENDED_SYSTEM, Rune, state_text,
                                           v1_probabilities)
from systemone_proxy.questions import parse_request

from conftest import FakeBackend

FIX = Path(__file__).parent / "fixtures"
GOLDEN = json.loads((FIX / "rune_golden_v1.json").read_text(encoding="utf-8"))
TICKET = json.loads((FIX / "rune_ticket_prompts.json").read_text(encoding="utf-8"))
LETTER_IDS = {"A": 236776, "B": 236799, "C": 236780, "D": 236796, "E": 236788, "Z": 236953}
GZ_INDEX = 26 + 6 * 26 + 25      # where the synthetic codebook (with GZ) and Gemma's part


@pytest.fixture(scope="module")
def rune(gemma_tok):
    return Rune(gemma_tok)


@pytest.fixture
def rune_t1(gemma_tok):
    prof = Rune(gemma_tok)
    prof.temperatures = {"choice": 1.0, "score": 1.0, "noul": 1.0}   # golden answers are at T = 1
    return prof


def representable(case):
    """Golden questions the unified contract accepts (string, non-empty
    instructions; score levels not null)."""
    body = json.loads(case["body"])
    for gq in case["questions"]:
        q = body["questions"][gq["name"]]
        if isinstance(q["instructions"], str) and q["instructions"].strip():
            yield body, gq


def golden_questions():
    out = []
    for case in GOLDEN["cases"]:
        for body, gq in representable(case):
            out.append(pytest.param(case, body, gq, id=f"{case['name']}:{gq['name']}"))
    return out


# -------------------------------------------------------------- prompts --

def test_system_prompts():
    assert SYSTEM == GOLDEN["cases"][0]["questions"][0]["system"]
    assert EXTENDED_SYSTEM.endswith("exactly one option code. Do not explain or generate reasoning.")


@pytest.mark.parametrize("case", GOLDEN["cases"], ids=lambda c: c["name"])
def test_state_text_matches_golden(case):
    assert state_text(json.loads(case["body"])["state"]) == case["state_text"]


@pytest.mark.parametrize("case,body,gq", golden_questions())
def test_branch_matches_golden(rune, case, body, gq):
    one = {"state": body["state"], "questions": {gq["name"]: body["questions"][gq["name"]]}}
    req = parse_request(one, Rune.limits)
    system, branch, ids = rune.branch(req.questions[0])
    want = gq["branch"]
    n = len(gq["labels"])
    if n > GZ_INDEX:     # real Gemma codes past GZ: compare the shared part, then check ours
        mine, theirs = branch.split("\n"), want.split("\n")
        head = 3 + GZ_INDEX           # QUESTION:, instructions, OPTIONS:, then option lines
        assert mine[:head] == theirs[:head] and mine[-1] == theirs[-1]
        assert "GZ:" not in branch and f"{rune.codes[n - 1]}: " in branch
    else:
        assert branch == want
        assert rune.codes[:n] == gq["labels"]
    assert system == gq["system"]
    assert len(ids) == n and len(set(ids)) == n


@pytest.mark.parametrize("name", ["sentiment", "refund", "urgency"])
def test_ticket_full_token_ids(rune, gemma_template, name):
    body = json.loads(GOLDEN["cases"][0]["body"])
    req = parse_request(body, Rune.limits)
    built = rune.build(req)
    row = next(r for r in built.rows if r.qid == name)
    want = TICKET["questions"][name]
    assert row.ids == want["ids"] and row.label_ids == want["label_ids"]
    # The text is exactly what Gemma 4's own jinja template renders.
    q = next(q for q in req.questions if q.qid == name)
    system, branch, _ = rune.branch(q)
    user = state_text(req.state) + branch
    rendered = gemma_template([{"role": "system", "content": system},
                               {"role": "user", "content": user}])
    assert rendered == want["prompt"] == gemma4_chat(system, user)
    # <bos> once; ends …\n<|turn>model\n<|channel>thought\n<channel|>; no prefill.
    assert row.ids[:4] == [2, 105, 9731, 107] and row.ids.count(2) == 1
    assert row.ids[-8:] == [107, 105, 4368, 107, 100, 45518, 107, 101]


def test_refund_prompt_text(rune):
    want = ("<bos><|turn>system\n" + SYSTEM + "<turn|>\n<|turn>user\nSHARED STATE (JSON string):\n"
            "\"Customer wrote: the package arrived late and damaged, I want my money back.\"\n\n"
            "QUESTION:\nDoes the customer ask for a refund?\nOPTIONS:\nA: No refund is requested\n"
            "B: A refund is requested\nAnswer with one option letter only.<turn|>\n"
            "<|turn>model\n<|channel>thought\n<channel|>")
    assert TICKET["questions"]["refund"]["prompt"] == want
    assert len(TICKET["questions"]["refund"]["ids"]) == 124
    assert rune.selftest_texts()[0] == want


def test_labels_are_bare_letters(rune, gemma_tok):
    for letter, i in LETTER_IDS.items():
        assert rune.code_ids[ord(letter) - 65] == i
        assert gemma_tok.encode(" " + letter) != [i]        # the space-prefixed token is another one
    assert rune.codes[:28] == [*map(chr, range(65, 91)), "AA", "AB"]
    assert "GZ" not in rune.codes and rune.codes[GZ_INDEX] == "HA"
    assert rune.selftest_labels()[236776] == "A"


def test_noul_order_and_defaults(rune):
    req = parse_request({"state": "s", "questions": {
        "n": {"type": "noul", "instructions": "Is it?", "criteria": {"true": "Yes"}}}}, Rune.limits)
    _, branch, _ = rune.branch(req.questions[0])
    assert branch.endswith("OPTIONS:\nA: false\nB: Yes\nAnswer with one option letter only.")


def test_non_ascii_state_raw(rune):
    assert state_text({"zpráva": "Dobrý den"}) == 'SHARED STATE (JSON string):\n{"zpráva": "Dobrý den"}\n\n'


def test_rejects_nan_and_null_score_level(rune):
    with pytest.raises(ApiError):
        rune.build(parse_request({"state": {"x": float("nan")}, "questions": {
            "n": {"type": "noul", "instructions": "q"}}}, Rune.limits))
    with pytest.raises(ApiError):
        rune.build(parse_request({"state": "s", "questions": {
            "s": {"type": "score", "instructions": "q", "criteria": ["low", None]}}}, Rune.limits))


def test_extended_question_uses_codes(rune):
    crit = {f"o{i}": f"option {i}" for i in range(30)}
    req = parse_request({"state": "s", "questions": {
        "q": {"type": "choice", "instructions": "Which?", "criteria": crit}}}, Rune.limits)
    built = rune.build(req)
    system, branch, ids = rune.branch(req.questions[0])
    assert system == EXTENDED_SYSTEM and branch.endswith("AD: option 29\nAnswer with one option code only.")
    assert built.rows[0].label_ids == rune.code_ids[:30]


# --------------------------------------------------------------- answers --

@pytest.mark.parametrize("case,body,gq", golden_questions())
def test_answers_match_golden(rune_t1, case, body, gq):
    one = {"state": body["state"], "questions": {gq["name"]: body["questions"][gq["name"]]}}
    q = parse_request(one, Rune.limits).questions[0]
    got = rune_t1.answer(q, {"main": gq["logits"]})
    want = gq["answer"]
    if want["type"] == "noul":
        assert got["noul"] == want["noul"]
        assert got["confidence"] == max(want["noul"], 1 - want["noul"])
        return
    assert got["probabilities"] == want["probabilities"]       # bit for bit
    assert got["confidence"] == want["confidence"]
    if want["type"] == "choice":
        assert got["choice"] == want["choice"]
    else:
        assert got["score"] == want["score"] and got["legend"] == want["legend"]


def test_ties_case(rune_t1):
    case = next(c for c in GOLDEN["cases"] if c["name"] == "ties")
    body = json.loads(case["body"])
    for gq in case["questions"]:
        q = parse_request({"state": body["state"], "questions": {gq["name"]: body["questions"][gq["name"]]}},
                          Rune.limits).questions[0]
        got = rune_t1.answer(q, {"main": gq["logits"]})
        for k, v in gq["answer"].items():
            assert got[k] == v, (gq["name"], k)


def test_default_temperature_two(rune):
    assert Rune.temperatures == {"choice": 2.0, "score": 2.0, "noul": 2.0}
    req = parse_request({"state": "s", "questions": {
        "n": {"type": "noul", "instructions": "q"},
        "c": {"type": "choice", "instructions": "q", "criteria": {"x": None, "y": None, "z": None}}}},
        Rune.limits)
    noul = rune.answer(req.questions[0], {"main": [9.0, -1.25]})     # [false, true]
    assert noul["noul"] == pytest.approx(1 / (1 + math.exp(10.25 / 2)))
    ch = rune.answer(req.questions[1], {"main": [-1.0, 3.0, 3.0]})
    assert ch["choice"] == "y"                                       # first of the tie
    assert ch["probabilities"] == dict(zip("xyz", v1_probabilities([-1.0, 3.0, 3.0], 2.0)))


# --------------------------------------------------------------- matching --

def test_matches():
    meta = {"general.architecture": "gemma4", "general.basename": "rune", "general.name": "Rune Bf16 Hf",
            "general.size_label": "128x2.6B", "general.base_model.0.name": "Gemma 4 26B A4B It"}
    assert Rune.matches(meta, "rune-26b-a4b-v3-IQ3_S-dyn.gguf")
    assert Rune.matches(meta, "rune-26b-a4b-v3-Q8_0.gguf")
    assert Rune.matches(meta, "rune-26b-a4b-IQ3_M.gguf")
    assert not Rune.matches(meta, "rune-26b-a4b-v1-Q4_K_M.gguf")
    assert not Rune.matches(meta, "Rune_V2.Q8_0.gguf")
    assert not Rune.matches({**meta, "general.architecture": "gemma3"}, "rune-v3.gguf")
    assert not Rune.matches({**meta, "general.basename": "quyet-1.0"}, "rune-v3.gguf")
    # Livesport's Q8_0: stripped header, identified by its file name only
    stripped = {"general.architecture": "gemma4", "general.name": "Hf", "general.size_label": "128x2.6B"}
    assert Rune.matches(stripped, "rune-26b-a4b-v3-Q8_0.gguf")
    assert not Rune.matches(stripped, "rune-26b-a4b-v1-Q8_0.gguf")
    assert not Rune.matches(stripped, "gemma-4-26b-a4b-it-Q8_0.gguf")


def test_spec_without_instance():
    spec = Rune.spec(object.__new__(Rune))
    assert spec["noul_order"] == "A = false, B = true"
    assert any("ensure_ascii" in n or "json.dumps(state)" in n for n in spec["notes"])
    assert any("thinking" in n and "order_averaging" in n for n in spec["notes"])


# ------------------------------------------------------------ end to end --

def test_service_end_to_end(rune):
    from systemone_proxy.service import DecisionService
    a, b, c = (LETTER_IDS[x] for x in "ABC")
    fake = FakeBackend([{a: -9.0, b: -2.0, c: -0.01, 7: -12.0},
                        {a: -0.01, b: -6.0},
                        {a: -3.0, b: -0.5}])          # C missing → bounded by the min returned
    svc = DecisionService(rune, fake, n_ctx=16384)
    out = svc.decide(json.loads(GOLDEN["cases"][0]["body"]))
    assert out["model"] == "rune-26b-a4b-v3"
    assert list(out["answers"]) == ["sentiment", "refund", "urgency"]
    assert out["answers"]["sentiment"]["choice"] == "negative"
    assert out["answers"]["refund"]["noul"] < 0.1
    assert [call["ids"] for call in fake.calls] == [TICKET["questions"][k]["ids"]
                                                    for k in ("sentiment", "refund", "urgency")]
    assert any(w["code"] == "label_outside_top_n" for w in out["warnings"])

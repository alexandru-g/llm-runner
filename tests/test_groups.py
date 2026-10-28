"""Several group names per running copy: `run --group a,b`, `group: [a, b]`."""
from __future__ import annotations

import json

import pytest
from click.testing import CliRunner

from llm_runner import apply as apply_mod
from llm_runner import cli
from llm_runner import gateway as gw
from llm_runner import process as proc
from llm_runner.cluster import OPENAI, Instance, pick

from test_apply import QWEN, _apply, _write, host  # noqa: F401  (fixture)
from test_cluster import _Fake, _call, _node, _rec, _serve

ASSISTANT = "assistant"
GEMMA = "gemma-4-26b-a4b-it"


def test_pick_by_any_of_the_groups():
    def inst(id, node, prio, groups):
        info = {"api": OPENAI, "status": "ready", "port": 1, "group": groups[0]}
        if len(groups) > 1:
            info["groups"] = groups
        return Instance(id, node, prio, False, info)

    insts = [inst("gemma-26b-q4", "orin", 1, [ASSISTANT, GEMMA]),
             inst("gemma-31b-q4", "rtx", 0, [ASSISTANT, "gemma-4-31b-it"])]
    assert [c.id for c in pick(insts, ASSISTANT, OPENAI)] == ["gemma-31b-q4", "gemma-26b-q4"]
    assert [c.id for c in pick(insts, GEMMA, OPENAI)] == ["gemma-26b-q4"]
    assert [c.id for c in pick(insts, "gemma-4-31b-it", OPENAI)] == ["gemma-31b-q4"]
    # A node before 1.6 sends only "group": it answers to that one.
    old = Instance("old", "lab", 0, False, {"api": OPENAI, "status": "ready", "group": "x"})
    assert old.groups == ["x"] and pick([old], "x", OPENAI)[0].id == "old"


@pytest.fixture
def pa(tmp_path):
    """orin's Gemma 26B answers to `assistant` and its own group; rtx's 31B
    joins `assistant` (and is preferred, priority 0)."""
    rtx_m, orin_m = _serve(_Fake, "rtx"), _serve(_Fake, "orin")
    lab, lab_srv = _node(tmp_path, "lab", 5, [], serve=True)
    url = f"http://127.0.0.1:{lab.port}"
    rtx, rtx_srv = _node(tmp_path, "rtx", 0, [_rec("gemma-31b", rtx_m.server_address[1],
                                                   group=f"{ASSISTANT},gemma-4-31b-it")],
                         registry_url=url)
    orin, orin_srv = _node(tmp_path, "orin", 1, [_rec("gemma-26b", orin_m.server_address[1],
                                                      group=f"{ASSISTANT},{GEMMA}")],
                           registry_url=url)
    for _ in range(2):
        for g in (lab, rtx, orin):
            g.heartbeat_once()
    yield {"lab": lab, "rtx": rtx, "orin": orin, "tmp": tmp_path}
    for s in (rtx_m, orin_m, lab_srv, rtx_srv, orin_srv):
        s.shutdown()


def test_routes_and_lists_every_group(pa):
    lab = pa["lab"]
    who = _call(lab, "POST", "/v1/chat/completions", {"model": ASSISTANT})[1]
    assert who == {"who": "rtx", "model": "gemma-31b"}
    who = _call(lab, "POST", "/v1/chat/completions", {"model": GEMMA})[1]
    assert who == {"who": "orin", "model": "gemma-26b"}
    assert _call(lab, "GET", f"/m/{ASSISTANT}/props")[1]["who"] == "rtx"
    assert _call(lab, "POST", "/v1/chat/completions",
                 {"model": f"{ASSISTANT}@orin"})[1]["who"] == "orin"

    by_id = {m["id"]: m["llmctl"] for m in _call(lab, "GET", "/v1/models")[1]["data"]}
    assert set(by_id) == {"gemma-31b", "gemma-26b", ASSISTANT}
    a = by_id[ASSISTANT]
    assert a["group"] == ASSISTANT and a["members"] == ["gemma-26b", "gemma-31b"]
    assert "groups" not in a
    assert [(n["node"], n["model"]) for n in a["nodes"]] == [("rtx", "gemma-31b"),
                                                            ("orin", "gemma-26b")]
    # Each member shows its first group as `group` (what 1.5 readers know) and all in `groups`.
    assert by_id["gemma-26b"]["group"] == ASSISTANT
    assert by_id["gemma-26b"]["groups"] == [ASSISTANT, GEMMA]


def test_assistant_falls_back_to_orin_when_rtx_is_gone(pa):
    rtx, lab, orin = pa["rtx"], pa["lab"], pa["orin"]
    (pa["tmp"] / "rtx.json").write_text("[]")
    for g in (rtx, lab, orin):
        g.heartbeat_once()
    who = _call(lab, "POST", "/v1/chat/completions", {"model": ASSISTANT})[1]
    assert who == {"who": "orin", "model": "gemma-26b"}
    # One member left: reachable as `assistant`, listed once (as itself).
    ids = {m["id"] for m in _call(orin, "GET", "/v1/models")[1]["data"]}
    assert ids == {"gemma-26b"}


# ------------------------------------------------------------- the CLI --

def _run(*args):
    return CliRunner().invoke(cli.cli, ["run", "qwen3-v1", "--no-menu", "--port", "8101", *args])


def test_run_records_several_groups(host):  # noqa: F811
    r = _run("--group", f"{ASSISTANT},qwen3,{ASSISTANT}")
    assert r.exit_code == 0, r.output
    [rec] = proc.list_running()
    assert rec.group == f"{ASSISTANT},qwen3"                     # repeats dropped
    assert proc.autostart_find(rec.name).group == rec.group     # and restored
    [row] = json.loads(CliRunner().invoke(cli.cli, ["ps", "--json"]).output)["models"]
    assert row["group"] == ASSISTANT and row["groups"] == [ASSISTANT, "qwen3"]
    m = gw.LocalModels(proc.STATE_FILE)
    [model] = m.models()
    info = m.describe(model)["llmctl"]
    assert info["group"] == ASSISTANT and info["groups"] == [ASSISTANT, "qwen3"]


def test_one_group_keeps_the_old_shape(host):  # noqa: F811
    assert _run("--group", "qwen3").exit_code == 0
    [row] = json.loads(CliRunner().invoke(cli.cli, ["ps", "--json"]).output)["models"]
    assert row["group"] == "qwen3" and "groups" not in row


def test_bad_group_lists_are_refused(host):  # noqa: F811
    for bad in ("a,,b", "a,-b", "a, b"):
        r = _run("--group", bad)
        assert r.exit_code != 0 and not proc.list_running(), bad


def test_apply_takes_a_list(tmp_path, host):  # noqa: F811
    f = _write(tmp_path, "serve.yaml", QWEN.replace("group: qwen3", f"group: [{ASSISTANT}, qwen3]"))
    [e] = apply_mod.load([f])
    assert e.run_argv("qwen3-v1")[e.run_argv("qwen3-v1").index("--group") + 1] == \
        f"{ASSISTANT},qwen3"
    r, out = _apply(f)
    assert r.exit_code == 0, r.output
    assert proc.list_running()[0].group == f"{ASSISTANT},qwen3"
    r, out = _apply(f)
    assert [a["action"] for a in out["actions"]] == ["unchanged"]
    bad = _write(tmp_path, "bad.yaml", QWEN.replace("group: qwen3", "group: [ok, 'no way']"))
    with pytest.raises(apply_mod.ApplyError) as err:
        apply_mod.load([bad])
    assert "list of names" in err.value.problems[0]

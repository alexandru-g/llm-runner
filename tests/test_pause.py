"""`llmctl pause` / `resume`: down on purpose, launch remembered."""
from __future__ import annotations

import json

from click.testing import CliRunner

from llm_runner import cli
from llm_runner import process as proc

from test_apply import host  # noqa: F401  (fixture)
from test_supervisor import w  # noqa: F401  (fixture)


def _cli(*args):
    return CliRunner().invoke(cli.cli, list(args))


def _run(port="8101", *args):
    return _cli("run", "qwen3-v1", "--no-menu", "--port", port, *args)


def _events(tmp_path):
    path = tmp_path / "state" / "events.jsonl"
    return [json.loads(x) for x in path.read_text().splitlines()]


def test_pause_keeps_the_launch_and_resume_brings_it_back(tmp_path, host):  # noqa: F811
    assert _run("8101", "--n-parallel", "4").exit_code == 0
    [rec] = proc.list_running()

    r = _cli("pause")                               # the one running model
    assert r.exit_code == 0, r.output
    assert f"Paused {rec.name}" in r.output and not proc.list_running()
    spec = proc.autostart_find(rec.name)
    assert spec.paused and "--n-parallel" in spec.run_args

    ps = _cli("ps")
    assert rec.name in ps.output and "paused" in ps.output
    [row] = json.loads(_cli("ps", "--json").output)["models"]
    assert row["state"] == "paused" and row["pid"] is None and row["port"] == 8101

    # nothing else brings it back
    [(_, status)] = proc.restore()
    assert status == "paused" and not proc.list_running()
    assert _cli("pause", rec.name).output.startswith(f"{rec.name} is already paused")

    r = _cli("resume")                              # the one paused model
    assert r.exit_code == 0, r.output
    [back] = proc.list_running()
    assert back.name == rec.name and back.port == 8101 and back.pid != rec.pid
    assert back.cmd == rec.cmd
    assert not proc.autostart_find(rec.name).paused
    assert [e["event"] for e in _events(tmp_path)][-2:] == ["paused", "started"]
    assert _events(tmp_path)[-1]["by"] == "resume"

    r = _cli("resume", rec.name)
    assert r.exit_code != 0 and "running, not paused" in r.output


def test_stop_forgets_a_paused_model(tmp_path, host):  # noqa: F811
    assert _run().exit_code == 0
    [rec] = proc.list_running()
    assert _cli("pause", rec.name).exit_code == 0
    r = _cli("stop", rec.name)
    assert r.exit_code == 0, r.output
    assert "was paused" in r.output and proc.autostart_find(rec.name) is None
    r = _cli("resume")
    assert r.exit_code != 0 and "No paused models" in r.output


def test_a_paused_port_is_not_handed_out(host, monkeypatch):  # noqa: F811
    assert _run("8080").exit_code == 0
    [rec] = proc.list_running()
    assert _cli("pause", rec.name).exit_code == 0
    assert cli._next_free_port(8080) == 8081


def test_sync_keeps_paused_entries(host):  # noqa: F811
    assert _run().exit_code == 0
    [rec] = proc.list_running()
    assert _cli("pause", rec.name).exit_code == 0
    kept, dropped = proc.autostart_sync()
    assert [s.name for s in kept] == [rec.name] and not dropped


def test_the_supervisor_leaves_a_paused_model_alone(w):  # noqa: F811
    w.add("qwen")
    w.s.tick()
    w.specs["qwen"].paused = True        # `llmctl pause` marks it before the kill
    w.kill("qwen")
    w.run(10)
    assert w.restored == [] and w.s.state("qwen") is None


def test_unpaused_entries_write_no_key(host):  # noqa: F811
    """llmctl before pause/resume would restore a paused model anyway, but
    entries that aren't paused stay exactly as it wrote them."""
    assert _run().exit_code == 0
    assert all("paused" not in r for r in json.loads(proc.AUTOSTART_FILE.read_text()))


def test_ls_is_list(host):  # noqa: F811
    assert _cli("ls").output == _cli("list").output
    assert "\n  ls " not in _cli("--help").output


def test_start_is_run(host):  # noqa: F811
    r = _cli("start", "qwen3-v1", "--no-menu", "--port", "8101")
    assert r.exit_code == 0, r.output
    assert [p.name for p in proc.list_running()] == ["qwen3-v1"]
    assert "\n  start " not in _cli("--help").output

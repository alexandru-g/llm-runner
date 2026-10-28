"""`run --owner`: who started a model, in ps, events, the gateway and restarts."""
from __future__ import annotations

import json

from click.testing import CliRunner

from llm_runner import apply as apply_mod
from llm_runner import cli
from llm_runner import gateway as gw
from llm_runner import process as proc

from test_apply import QWEN, _apply, _write, host  # noqa: F401  (fixture)


def _run(*args, env=None):
    return CliRunner().invoke(cli.cli, ["run", "qwen3-v1", "--no-menu", "--port", "8101",
                                        *args], env=env)


def _events(tmp_path):
    path = tmp_path / "state" / "events.jsonl"
    return [json.loads(x) for x in path.read_text().splitlines()]


def test_owner_is_recorded_shown_and_kept(tmp_path, host):  # noqa: F811
    r = _run("--owner", "A1")
    assert r.exit_code == 0, r.output
    [rec] = proc.list_running()
    assert rec.owner == "A1" and proc.autostart_find(rec.name).owner == "A1"
    [started] = [e for e in _events(tmp_path) if e["event"] == "started"]
    assert started["owner"] == "A1"

    ps = CliRunner().invoke(cli.cli, ["ps"])
    assert "OWNER" in ps.output and " A1 " in ps.output
    rows = json.loads(CliRunner().invoke(cli.cli, ["ps", "--json"]).output)["models"]
    assert rows[0]["owner"] == "A1"
    ev = CliRunner().invoke(cli.cli, ["events"]).output
    assert "[owner A1]" in ev

    # restore (boot, the supervisor) brings it back with its owner
    proc.stop(rec.name, keep_autostart=True)
    [(spec, status)] = proc.restore([rec.name])
    assert status.startswith("started") and proc.list_running()[0].owner == "A1"


def test_owner_from_the_environment_survives_a_rerun(tmp_path, host):  # noqa: F811
    r = _run(env={"LLMCTL_OWNER": "PS6 supervisor"})
    assert r.exit_code == 0, r.output
    spec = proc.autostart_find(proc.list_running()[0].name)
    assert spec.owner == "PS6 supervisor"
    assert spec.run_args[spec.run_args.index("--owner") + 1] == "PS6 supervisor"


def test_no_owner_writes_no_key(tmp_path, host):  # noqa: F811
    """llmctl before 1.5 can't read an "owner" key in its state files."""
    assert _run().exit_code == 0
    for f in (proc.STATE_FILE, proc.AUTOSTART_FILE):
        assert all("owner" not in r for r in json.loads(f.read_text())), f
    assert proc.list_running()[0].owner is None


def test_records_with_unknown_keys_still_load(tmp_path, monkeypatch):
    monkeypatch.setattr(proc, "STATE_FILE", tmp_path / "running.json")
    (tmp_path / "running.json").write_text(json.dumps([{
        "name": "m", "pid": 1, "port": 8080, "started_at": 0.0, "model_path": "",
        "log_file": "", "from_the_future": 1}]))
    assert [p.name for p in proc._load()] == ["m"]


def test_bad_owner_tags_are_refused(host):  # noqa: F811
    for tag in ("", "-x", "a" * 65, "tab\there"):
        r = _run("--owner", tag)
        assert r.exit_code != 0 and not proc.list_running(), tag


def test_apply_ignores_a_different_owner(tmp_path, host):  # noqa: F811
    f = _write(tmp_path, "serve.yaml", QWEN.replace("    port: 8101", "    port: 8101\n    owner: A0"))
    r, out = _apply(f)
    assert r.exit_code == 0, r.output
    assert proc.list_running()[0].owner == "A0"
    r = CliRunner().invoke(cli.cli, ["apply", str(f), "--no-wait", "--json"],
                           env={"LLMCTL_OWNER": "A3"})
    assert r.exit_code == 0, r.output
    assert [a["action"] for a in json.loads(r.stdout)["actions"]] == ["unchanged"]
    assert apply_mod.without_owner(["--ctx-size", "8", "--owner", "A1", "--np", "4"]) == [
        "--ctx-size", "8", "--np", "4"]


def test_the_gateway_lists_the_owner(tmp_path, host):  # noqa: F811
    assert _run("--owner", "A1").exit_code == 0
    models = gw.LocalModels(proc.STATE_FILE)
    [m] = models.models()
    assert models.describe(m)["llmctl"]["owner"] == "A1"


def test_stop_forgets_a_crashed_model(tmp_path, host):  # noqa: F811
    """A model that died (and that the supervisor keeps restarting) is
    taken out of autostart by `stop`, instead of 'No running model'."""
    import os
    import signal
    import time
    assert _run("--owner", "A1").exit_code == 0
    [rec] = proc.list_running()
    os.killpg(rec.pid, signal.SIGKILL)
    for _ in range(40):
        if not proc.list_running():
            break
        time.sleep(0.05)
    (proc.STATE_DIR / "supervisor.json").write_text(json.dumps(
        {"models": {rec.name: {"state": "crashed", "restarts": 5}}}))
    r = CliRunner().invoke(cli.cli, ["stop", rec.name])
    assert r.exit_code == 0, r.output
    assert "Removed" in r.output and "crashed" in r.output
    assert proc.autostart_find(rec.name) is None
    [ev] = [e for e in _events(tmp_path) if e["event"] == "stopped"]
    assert ev["owner"] == "A1" and "was crashed" in ev["detail"]
    r = CliRunner().invoke(cli.cli, ["stop", rec.name])          # nothing left to stop
    assert r.exit_code != 0 and "No running model" in r.output

"""`run --json` and `ps --json`: what gpuq (and other programs) read instead of
llmctl's state files."""
from __future__ import annotations

import json

from click.testing import CliRunner

from llm_runner import cli
from llm_runner import process as proc

from test_apply import host  # noqa: F401  (fixture)


def _run(*args):
    return CliRunner().invoke(cli.cli, ["run", "qwen3-v1", "--no-menu", "--port", "8101",
                                        *args])


def _ps():
    r = CliRunner().invoke(cli.cli, ["ps", "--json"])
    assert r.exit_code == 0, r.output
    return {m["name"]: m for m in json.loads(r.stdout)["models"]}


def test_run_json_prints_only_the_model_on_stdout(host):  # noqa: F811
    r = _run("--json", "--owner", "gpuq-12 A3")
    assert r.exit_code == 0, r.output
    out = json.loads(r.stdout)                      # nothing else on stdout
    [rec] = proc.list_running()
    assert out["pid"] == rec.pid and out["port"] == 8101 and out["backend"] == "vllm"
    assert out["owner"] == "gpuq-12 A3" and out["log"] == rec.log_file
    assert out["ready"] is None and out["error"] is None     # not waited for
    assert "Started qwen3-v1" in r.stderr
    row = _ps()["qwen3-v1"]
    assert {k: row[k] for k in ("pid", "port", "owner", "runtime")} == \
        {k: out[k] for k in ("pid", "port", "owner", "runtime")}


def test_run_json_is_not_remembered_for_restarts(host):  # noqa: F811
    assert _run("--json").exit_code == 0
    assert "--json" not in proc.autostart_find("qwen3-v1").run_args


def test_run_json_wait(host, monkeypatch):  # noqa: F811
    monkeypatch.setattr(cli, "_wait_healthy", lambda rec, timeout: True)
    r = _run("--json", "--wait")
    assert r.exit_code == 0 and json.loads(r.stdout)["ready"] is True


def test_run_json_wait_failure_still_reports(host, monkeypatch):  # noqa: F811
    monkeypatch.setattr(cli, "_wait_healthy", lambda rec, timeout: False)
    r = _run("--json", "--wait")
    assert r.exit_code == 1
    out = json.loads(r.stdout)
    assert out["ready"] is False and "not ready after" in out["error"]


def test_run_without_json_is_unchanged(host):  # noqa: F811
    r = _run()
    assert r.exit_code == 0 and r.stdout.startswith("Started qwen3-v1")


def test_ps_lists_a_dead_model_until_it_is_stopped(host):  # noqa: F811
    # Died, and the supervisor hasn't looked yet (or the gateway is off):
    # still listed, so a reader can tell it from a stop.
    assert _run().exit_code == 0
    rec = proc.list_running()[0]
    proc.stop(rec.name, timeout=0, keep_autostart=True)
    row = _ps()["qwen3-v1"]
    assert row["pid"] is None and row["state"] == "down" and row["port"] == 8101
    text = CliRunner().invoke(cli.cli, ["ps"]).output
    assert "down" in text
    assert CliRunner().invoke(cli.cli, ["stop", "qwen3-v1"]).exit_code == 0
    assert "qwen3-v1" not in _ps()


def test_ps_keeps_the_supervisors_down_states(host, tmp_path):  # noqa: F811
    assert _run().exit_code == 0
    rec = proc.list_running()[0]
    proc.stop(rec.name, timeout=0, keep_autostart=True)
    (proc.STATE_DIR / "supervisor.json").write_text(json.dumps({"models": {
        "qwen3-v1": {"state": "restarting", "restarts": 1, "next_restart_in": 4.0}}}))
    row = _ps()["qwen3-v1"]
    assert row["state"] == "restarting" and row["next_restart_in"] == 4.0

"""Planned restarts (holds), model events and the last log of a stopped model."""
from __future__ import annotations

import json
import os
import subprocess
import time

from click.testing import CliRunner

from llm_runner import cli
from llm_runner import process as proc
from llm_runner import supervisor as sup

from test_apply import QWEN, _apply, _write, host  # noqa: F401  (fixture)
from test_gateway import _call, env  # noqa: F401  (fixture)
from test_supervisor import w  # noqa: F401  (fixture)

_POPEN = subprocess.Popen          # the real one: the host fixture patches it


def _events(state):
    path = state / "events.jsonl"
    return [json.loads(x) for x in path.read_text().splitlines()] if path.exists() else []


# ------------------------------------------------------------------ process --

def test_holds_lapse_and_release(tmp_path, monkeypatch):
    monkeypatch.setattr(proc, "STATE_DIR", tmp_path)
    proc.hold("qwen", "llmctl apply: x", group="qwen3")
    proc.hold("old", "y", ttl=-1)
    assert set(proc.holds()) == {"qwen"} and proc.holds()["qwen"]["group"] == "qwen3"
    proc.release("qwen")
    assert proc.holds() == {}


def test_the_same_event_for_the_same_pid_is_written_once(tmp_path):
    assert proc.record_event("q", "ready", "a", state_dir=tmp_path, pid=7, by="run")
    assert not proc.record_event("q", "ready", "b", state_dir=tmp_path, pid=7, by="supervisor")
    assert proc.record_event("other", "ready", "", state_dir=tmp_path, pid=7)
    assert proc.record_event("q", "ready", "", state_dir=tmp_path, pid=8)
    [first, _, last] = _events(tmp_path)
    assert first["by"] == "run" and first["pid"] == 7 and last["pid"] == 8
    assert {"ts", "model", "event", "detail"} <= set(first)


def test_last_log_picks_the_newest_of_that_name_only(tmp_path, monkeypatch):
    monkeypatch.setattr(proc, "LOG_DIR", tmp_path)
    for i, n in enumerate(["qwen-8101-100.log", "qwen-8101-200.log", "qwen-big-8102-300.log"]):
        (tmp_path / n).write_text(n)
        os.utime(tmp_path / n, (1000 + i, 1000 + i))
    assert proc.last_log("qwen").name == "qwen-8101-200.log"
    assert proc.last_log("nope") is None


# --------------------------------------------------------------- supervisor --

def test_a_held_model_is_not_a_crash_and_is_new_afterwards(w, monkeypatch, tmp_path):  # noqa: F811
    held: dict = {}
    monkeypatch.setattr(proc, "holds", lambda state_dir=None: held)
    w.add("qwen")
    w.s.tick()
    assert w.s.state("qwen").seen_ready
    held["qwen"] = {"reason": "apply"}
    w.kill("qwen")
    w.run(30)
    assert w.restored == [] and w.s.state("qwen") is None
    w.healthy.discard("qwen")
    w.add("qwen", healthy=False)               # the new instance, which never gets ready
    held.clear()
    w.s.tick()
    w.kill("qwen")
    w.run(30)
    assert w.restored == []                    # a failed start is not restarted
    [ev] = [e for e in _events(tmp_path) if e["event"] == "start_failed"]
    assert ev["model"] == "qwen" and ev["log"] == "l" and ev["by"] == "supervisor"


def test_crash_events_name_the_dead_pid(w, tmp_path):  # noqa: F811
    w.add("qwen")
    w.s.tick()
    pid = w.live["qwen"].pid
    w.kill("qwen")
    w.s.tick()
    ev = [e for e in _events(tmp_path) if e["event"] == sup.RESTARTING][-1]
    assert ev["pid"] == pid


# ------------------------------------------------------------------ gateway --

def test_a_held_model_gets_503_with_retry_after_not_404(env, tmp_path, monkeypatch):  # noqa: F811
    monkeypatch.setattr(proc, "STATE_DIR", tmp_path / "st")
    monkeypatch.setattr(proc, "AUTOSTART_FILE", tmp_path / "st" / "autostart.json")
    status, out = _call(env["port"], "POST", "/v1/chat/completions", {"model": "qwen"})
    assert status == 404
    proc.hold("qwen", "llmctl apply: restart: +qwen3-v2", group="qwen3")
    for name in ("qwen", "qwen3"):
        status, out = _call(env["port"], "POST", "/v1/chat/completions", {"model": name})
        assert status == 503 and out["error"]["code"] == "model_unavailable", out
        assert "being restarted" in out["error"]["message"]
    status, out = _call(env["port"], "POST", "/m/qwen/v1/chat/completions", {})
    assert status == 503
    proc.release("qwen")
    status, out = _call(env["port"], "POST", "/v1/chat/completions", {"model": "qwen"})
    assert status == 404


def test_retry_after_is_sent_for_held_models(env, tmp_path, monkeypatch):  # noqa: F811
    import http.client
    monkeypatch.setattr(proc, "STATE_DIR", tmp_path / "st")
    proc.hold("qwen", "apply")
    c = http.client.HTTPConnection("127.0.0.1", env["port"], timeout=5)
    c.request("POST", "/v1/chat/completions", body=json.dumps({"model": "qwen"}),
              headers={"Content-Type": "application/json"})
    r = c.getresponse()
    r.read()
    assert r.status == 503 and r.getheader("Retry-After") == "30"


# -------------------------------------------------------------------- apply --

def test_apply_restart_holds_the_model_and_records_events(tmp_path, host, monkeypatch):  # noqa: F811
    f = _write(tmp_path, "serve.yaml", QWEN)
    _apply(f)
    seen = []
    real_stop = proc.stop

    def stop(name, **kw):
        seen.append((dict(proc.holds()), kw, [s.name for s in proc.autostart_list()]))
        return real_stop(name, **kw)
    monkeypatch.setattr(proc, "stop", stop)
    f.write_text(f.read_text().replace("qwen3-v1", "qwen3-v2"))
    r, out = _apply(f)
    assert r.exit_code == 0, r.output
    [(held, kw, manifest)] = seen
    assert "qwen" in held and held["qwen"]["group"] == "qwen3" and kw["keep_autostart"]
    assert manifest == ["qwen"]
    assert proc.holds() == {}                         # released afterwards
    evs = [(e["event"], e.get("by")) for e in _events(proc.STATE_DIR)]
    assert evs == [("started", "apply"), ("stopped", "apply"), ("started", "apply")]
    stopped = [e for e in _events(proc.STATE_DIR) if e["event"] == "stopped"][0]
    assert stopped["reason"].startswith("restart: ") and "+qwen3-v2" in stopped["reason"]
    assert stopped["pid"] and stopped["port"] == 8101


def test_a_failed_start_leaves_an_event_and_its_log(tmp_path, host, monkeypatch):  # noqa: F811
    def popen(argv, **kw):
        kw["stdout"].write(b"ValueError: Free memory on device is less than desired\n")
        return _POPEN(["true"], start_new_session=True)
    monkeypatch.setattr(proc.subprocess, "Popen", popen)
    f = _write(tmp_path, "serve.yaml", QWEN)
    r = CliRunner().invoke(cli.cli, ["apply", str(f), "--wait", "--json"])
    assert r.exit_code == 1
    ev = _events(proc.STATE_DIR)[-1]
    assert ev["event"] == "start_failed" and ev["by"] == "apply"
    assert ev["reason"] == "exited before it was ready" and ev["log"].endswith(".log")
    time.sleep(0.1)
    r = CliRunner().invoke(cli.cli, ["logs", "qwen"])
    assert r.exit_code == 0, r.output
    assert "Free memory on device" in r.stdout and "is not running" in r.stderr
    r = CliRunner().invoke(cli.cli, ["logs", "nope"])
    assert r.exit_code == 2 and "has no log" in r.output


def test_an_engine_still_loading_gets_model_unavailable(tmp_path):
    import http.client
    import threading
    from http.server import BaseHTTPRequestHandler

    from llm_runner import gateway as gw
    from test_gateway import _record, _serve

    class Loading(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def _503(self):
            body = b'{"error":{"code":503,"message":"Loading model","type":"unavailable_error"}}'
            self.send_response(503)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        do_GET = do_POST = _503

    up = _serve(Loading)
    state = tmp_path / "running.json"
    state.write_text(json.dumps([_record("qwen", up.server_address[1])]))
    httpd = gw.make_server(0, "127.0.0.1", gw.Gateway(gw.LocalModels(state)))
    threading.Thread(target=httpd.serve_forever, args=(0.05,), daemon=True).start()
    try:
        c = http.client.HTTPConnection("127.0.0.1", httpd.server_address[1], timeout=5)
        c.request("POST", "/v1/chat/completions", body=json.dumps({"model": "qwen"}),
                  headers={"Content-Type": "application/json"})
        r = c.getresponse()
        out = json.loads(r.read())
        assert r.status == 503 and out["error"]["code"] == "model_unavailable"
        assert "loading" in out["error"]["message"] and r.getheader("Retry-After") == "10"
    finally:
        up.shutdown()
        httpd.shutdown()


def test_an_instance_replaced_between_ticks_is_new(w, tmp_path):  # noqa: F811
    w.add("qwen")
    w.s.tick()
    w.kill("qwen")
    w.healthy.discard("qwen")
    w.add("qwen", healthy=False)          # `apply --no-wait`, faster than a tick
    w.s.tick()
    assert w.s.state("qwen").state == sup.STARTING
    w.kill("qwen")                        # and it never gets ready
    w.run(30)
    assert w.restored == []
    assert [e["event"] for e in _events(tmp_path)] == ["ready", "start_failed"]
    w.add("qwen")
    w.s.tick()
    assert _events(tmp_path)[-1]["event"] == "ready"

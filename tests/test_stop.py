"""`llmctl stop` ends the gateway's requests for the model at once (issue #1)."""
from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import threading
import time

import pytest
from click.testing import CliRunner

from llm_runner import cli
from llm_runner import gateway as gw
from llm_runner import process as proc

from test_gateway import END, PIECE, SSE_HEAD, _ask, _call, _record, _Silent


@pytest.fixture
def stop_on(tmp_path):
    """A gateway in front of a _Silent model; ``stop()`` marks its engine as
    being stopped, the way `proc.stop` does before its first signal."""
    servers = []

    def start(model, pid=None):
        state = tmp_path / "running.json"
        state.write_text(json.dumps([_record("silent", model.port, pid=pid)]))
        httpd = gw.make_server(0, "127.0.0.1", gw.Gateway(gw.LocalModels(state)))
        servers.append(httpd)
        threading.Thread(target=httpd.serve_forever, args=(0.05,), daemon=True).start()

        def stop(hold_pid=os.getpid()):
            (tmp_path / "holds.json").write_text(json.dumps({"silent": {
                "reason": "llmctl stop", "by": "stop", "pid": hold_pid,
                "until": time.time() + 60}}))
        return httpd.server_address[1], stop

    yield start
    for s in servers:
        s.shutdown()


def _events(d):
    path = d / "events.jsonl"
    return [json.loads(x) for x in path.read_text().splitlines()] if path.exists() else []


def _read_all(client):
    client.settimeout(5)
    data = b""
    while chunk := client.recv(65536):
        data += chunk
        if data.endswith(END):
            break
    return data


def _last_event(data):
    body = data.split(b"\r\n\r\n", 1)[1]
    events = [x for x in body.split(b"\n\n") if b"data: " in x]
    return json.loads(events[-1][events[-1].index(b"data: ") + 6:])


def test_a_stop_ends_a_silent_stream_with_an_error_at_once(stop_on, tmp_path, monkeypatch):
    monkeypatch.setattr(proc, "STATE_DIR", tmp_path)
    monkeypatch.setattr(gw, "CUT_EVENT_AFTER", 0.2)
    model = _Silent(SSE_HEAD + PIECE)              # one piece, then silence
    port, stop = stop_on(model)
    client = _ask(port)
    data = b""
    while b"data: piece" not in data:
        data += client.recv(65536)
    t0 = time.monotonic()
    stop()
    data += _read_all(client)
    took = time.monotonic() - t0
    client.close()
    assert took < 1.0, took
    err = _last_event(data)["error"]
    assert err["code"] == "model_unavailable" and "stopped" in err["message"]
    assert model.hung_up.wait(3)                    # the engine's slot is let go too
    deadline = time.time() + 3
    while not _events(tmp_path) and time.time() < deadline:
        time.sleep(0.05)
    [ev] = _events(tmp_path)
    assert ev["event"] == "requests_cut" and ev["requests"] == 1 and ev["model"] == "silent"


def test_a_request_waiting_for_the_engine_gets_503(stop_on, tmp_path, monkeypatch):
    monkeypatch.setattr(proc, "STATE_DIR", tmp_path)
    model = _Silent()                               # never answers
    port, stop = stop_on(model)
    client = _ask(port)
    assert model.asked.wait(3)
    stop()
    data = _read_all_head(client)
    client.close()
    assert data.startswith(b"HTTP/1.1 503")
    assert b"model_unavailable" in data and b"being stopped" in data
    assert b"Retry-After" not in data               # a stopped model isn't coming back


def _read_all_head(client):
    client.settimeout(5)
    data = b""
    while b"}}" not in data:
        chunk = client.recv(65536)
        if not chunk:
            break
        data += chunk
    return data


def test_a_new_request_during_the_grace_gets_503_without_reaching_the_engine(
        stop_on, tmp_path, monkeypatch):
    monkeypatch.setattr(proc, "STATE_DIR", tmp_path)
    model = _Silent()
    port, stop = stop_on(model)
    stop()
    status, out = _call(port, "POST", "/v1/chat/completions",
                        {"model": "silent", "messages": []})
    assert status == 503 and out["error"]["code"] == "model_unavailable"
    assert "being stopped" in out["error"]["message"]
    assert not model.asked.is_set()


def test_an_engine_closing_its_stream_on_sigterm_still_reads_as_stopped(
        stop_on, tmp_path, monkeypatch):
    # Ends the stream cleanly but without [DONE], as an engine shutting down
    # may, before the watch's next look at the holds.
    monkeypatch.setattr(proc, "STATE_DIR", tmp_path)
    port, stop = stop_on(_Silent(SSE_HEAD + PIECE, END, after=0.15))
    client = _ask(port)
    data = b""
    while b"data: piece" not in data:
        data += client.recv(65536)
    stop()
    data += _read_all(client)
    client.close()
    assert _last_event(data)["error"]["code"] == "model_unavailable"


def test_a_hold_for_another_pid_cuts_nothing(stop_on, tmp_path, monkeypatch):
    # `apply` holds the name across a restart; the new engine is a new pid.
    monkeypatch.setattr(proc, "STATE_DIR", tmp_path)
    wait = 3 * gw.WATCH_EVERY_MS / 1000
    port, stop = stop_on(_Silent(SSE_HEAD + PIECE, PIECE + END, after=wait))
    stop(hold_pid=os.getpid() + 1)
    client = _ask(port)
    data = _read_all(client)
    client.close()
    assert data.count(b"data: piece") == 2 and b"error" not in data


# ------------------------------------------------------------------ process --

_STUBBORN = "import signal, time; signal.signal(signal.SIGTERM, signal.SIG_IGN); time.sleep(60)"


def _running(tmp_path, name="m"):
    p = subprocess.Popen([sys.executable, "-c", _STUBBORN], start_new_session=True)
    time.sleep(0.2)                     # past interpreter start, SIGTERM ignored
    rec = proc.RunningProcess(name=name, pid=p.pid, port=1, started_at=0, model_path="x",
                              log_file="l")
    proc._save([rec])
    return p, rec


def test_stop_timeout_0_kills_at_once_and_its_hold_lapses_soon(tmp_path):
    p, rec = _running(tmp_path)
    seen = []
    real = proc._terminate
    proc_terminate = lambda target, timeout: (seen.append(proc.holds().get("m")),  # noqa: E731
                                              real(target, timeout))
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(proc, "_terminate", proc_terminate)
        t0 = time.monotonic()
        proc.stop("m", timeout=0)
    assert time.monotonic() - t0 < 2
    assert p.wait(3) == -signal.SIGKILL
    [hold] = seen
    assert hold["pid"] == rec.pid and hold["by"] == "stop"      # marked before any signal
    left = proc.holds()["m"]["until"] - time.time()             # and lapsing soon after
    assert 0 < left <= proc.STOP_LINGER


def test_stop_joins_an_apply_hold_and_leaves_it(tmp_path):
    p, rec = _running(tmp_path)
    proc.hold("m", "llmctl apply: new flags")
    proc.stop("m", timeout=0, keep_autostart=True)
    p.wait(3)
    h = proc.holds()["m"]
    assert h["reason"] == "llmctl apply: new flags" and h["pid"] == rec.pid


def test_cli_stop_timeout(tmp_path):
    p, _ = _running(tmp_path)
    r = CliRunner().invoke(cli.cli, ["stop", "m", "--timeout", "0"])
    assert r.exit_code == 0, r.output
    assert p.wait(3) == -signal.SIGKILL
    assert CliRunner().invoke(cli.cli, ["stop", "m", "--timeout", "-1"]).exit_code == 2


def test_an_engine_breaking_its_stream_on_sigterm_reads_as_stopped(
        stop_on, tmp_path, monkeypatch):
    monkeypatch.setattr(proc, "STATE_DIR", tmp_path)
    monkeypatch.setattr(gw, "CUT_EVENT_AFTER", 0.2)
    # Drops the connection without the last chunk, as llama-server may on SIGTERM.
    port, stop = stop_on(_Silent(SSE_HEAD + PIECE, b"", after=0.1))
    client = _ask(port)
    data = b""
    while b"data: piece" not in data:
        data += client.recv(65536)
    stop()
    data += _read_all(client)
    client.close()
    err = _last_event(data)["error"]
    assert err["code"] == "model_unavailable" and "was stopped" in err["message"]
    deadline = time.time() + 3
    while not _events(tmp_path) and time.time() < deadline:
        time.sleep(0.05)
    assert [e["event"] for e in _events(tmp_path)] == ["requests_cut"]


def test_cut_counts_are_logged_when_the_gateway_exits(tmp_path, monkeypatch):
    monkeypatch.setattr(gw, "CUT_EVENT_AFTER", 30)
    local = gw.LocalModels(tmp_path / "running.json")
    local.count_cut("m", 7)
    local.count_cut("m", 7)
    local.flush_cuts()
    local.flush_cuts()                              # nothing left to log twice
    [ev] = _events(tmp_path)
    assert ev["requests"] == 2 and ev["pid"] == 7


def test_an_exiting_gateway_waits_for_requests_being_cut(tmp_path, monkeypatch):
    local = gw.LocalModels(tmp_path / "running.json")
    with local.relaying("m", 7):
        assert not local.cutting()                  # running, nobody stopping it
        (tmp_path / "holds.json").write_text(json.dumps(
            {"m": {"by": "stop", "pid": 7, "until": time.time() + 60}}))
        assert local.cutting()
    assert not local.cutting()


def test_a_second_stop_soon_after_has_its_own_hold(tmp_path):
    p, _ = _running(tmp_path)
    proc.stop("m", timeout=0)
    p.wait(3)
    p2, rec2 = _running(tmp_path)                 # run again within the linger
    proc.stop("m", timeout=0)
    p2.wait(3)
    h = proc.holds()["m"]
    assert h["pid"] == rec2.pid and h["by"] == "stop"

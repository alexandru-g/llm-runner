"""When each model was last used: the gateway's requests, `run`'s starts,
and `llmctl list --sort used`."""
from __future__ import annotations

import json
import threading
import time

from click.testing import CliRunner

from llm_runner import cli, registry, usage
from llm_runner import gateway as gw
from llm_runner.cluster import HOP_HEADER, TOKEN_HEADER
from llm_runner.config import Cluster

from test_apply import host  # noqa: F401  (fixture)
from test_gateway import _call, _record, _serve, _Upstream


def test_merge_keeps_the_later_time_and_adds_requests(tmp_path):
    usage.merge({"/m/a.gguf": {"last_request": 100.0, "requests": 2, "name": "a"}}, tmp_path)
    usage.merge({"/m/a.gguf": {"last_request": 50.0, "requests": 3}}, tmp_path)   # older
    usage.note_start("/m/a.gguf", "a", when=120.0, state_dir=tmp_path)
    e = usage.load(tmp_path)["/m/a.gguf"]
    assert e == {"last_request": 100.0, "requests": 5, "last_started": 120.0, "name": "a"}
    assert usage.last_used(e) == 120.0 and usage.last_used(None) is None


def test_tracker_writes_now_and_then_and_on_flush(tmp_path):
    t = usage.Tracker(tmp_path, every=3600)
    t.note("/m/a.gguf", "a", when=10.0)
    t.note("/m/a.gguf", "a", when=20.0)
    assert usage.load(tmp_path) == {}                 # not due yet
    t.flush()
    assert usage.load(tmp_path)["/m/a.gguf"]["requests"] == 2
    now = usage.Tracker(tmp_path, every=0)
    now.note("/m/b.gguf", "b")
    assert usage.load(tmp_path)["/m/b.gguf"]["requests"] == 1


def test_gateway_records_what_its_models_answer(tmp_path):
    model = _serve(_Upstream, "glm")
    state = tmp_path / "running.json"
    state.write_text(json.dumps([_record("glm-ocr", model.server_address[1])]))
    local = gw.LocalModels(state)
    httpd = gw.make_server(0, "127.0.0.1", gw.Gateway(
        local, Cluster("orin", 1, None, None, "t0k", True)))
    threading.Thread(target=httpd.serve_forever, args=(0.05,), daemon=True).start()
    port = httpd.server_address[1]
    try:
        assert _call(port, "POST", "/v1/chat/completions",
                     {"model": "glm-ocr", "messages": []})[0] == 200
        assert _call(port, "POST", "/v1/chat/completions",          # lab forwarded it here
                     {"model": "glm-ocr", "messages": []},
                     headers={HOP_HEADER: "lab", TOKEN_HEADER: "t0k"})[0] == 200
        _call(port, "GET", "/v1/models")                             # not a use
        _call(port, "POST", "/v1/chat/completions", {"model": "nope", "messages": []})
        local.usage.flush()
    finally:
        httpd.shutdown()
        model.shutdown()
    rec = usage.load(tmp_path)
    assert list(rec) == ["/models/glm-ocr.gguf"]
    e = rec["/models/glm-ocr.gguf"]
    assert e["name"] == "glm-ocr" and e["requests"] == 2         # the hop counts here
    assert time.time() - e["last_request"] < 30


def test_run_records_its_start(host):  # noqa: F811
    r = CliRunner().invoke(cli.cli, ["run", "qwen3-v1", "--no-menu", "--port", "8101"])
    assert r.exit_code == 0, r.output
    [(path, e)] = usage.load().items()
    assert path.endswith("qwen3-v1") and e["name"] == "qwen3-v1"
    assert time.time() - e["last_started"] < 30


def test_list_sorts_by_last_use_with_events_as_the_fallback(host, tmp_path):  # noqa: F811
    from llm_runner import process as proc
    models = registry.MODELS_DIR
    usage.merge({str(models / "qwen3-v2"): {"last_request": time.time() - 7200}})
    (proc.STATE_DIR / "events.jsonl").write_text(json.dumps(
        {"ts": time.strftime("%Y-%m-%dT%H:%M:%S%z", time.localtime(time.time() - 300)),
         "model": "qwen3-v1", "event": "started"}) + "\n")
    r = CliRunner().invoke(cli.cli, ["list", "-t"])
    assert r.exit_code == 0, r.output
    rows = [line.split() for line in r.output.splitlines() if line.startswith("qwen3")]
    assert [row[0] for row in rows] == ["qwen3-v1", "qwen3-v2"]       # 5 min, then 2 h
    assert "5m ago" in r.output and "2h ago" in r.output
    r = CliRunner().invoke(cli.cli, ["list"])
    assert [line.split()[0] for line in r.output.splitlines()
            if line.startswith("qwen3")] == ["qwen3-v1", "qwen3-v2"]  # by key


def test_ago():
    now = 1_000_000_000.0
    assert usage.ago(None) == "never" and usage.ago(now - 5, now) == "just now"
    assert usage.ago(now - 600, now) == "10m ago" and usage.ago(now - 7200, now) == "2h ago"
    assert usage.ago(now - 86400 * 3, now) == "3d ago"
    assert usage.ago(now - 86400 * 90, now) == time.strftime(
        "%Y-%m-%d", time.localtime(now - 86400 * 90))

"""Cluster routing: selection logic, the registry table, and three gateways
(lab registry, rtx priority 0, orin priority 1) talking over loopback."""
from __future__ import annotations

import dataclasses
import http.client
import json
import os
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from llm_runner import cluster as cl
from llm_runner import gateway as gw
from llm_runner.cluster import OPENAI, Instance, RegistryTable, Unroutable, pick
from llm_runner.config import Cluster

TOKEN = "s3cret"


def inst(id, node, priority=0, local=False, status="ready", vision=False, api=OPENAI,
         group=None):
    return Instance(id, node, priority, local,
                    {"api": api, "status": status, "vision": vision, "port": 1,
                     "group": group})


# ------------------------------------------------------------------ routing --

def test_order_ready_then_priority_then_local():
    rtx = inst("quyet", "rtx", 0)
    orin = inst("quyet", "orin", 1, local=True)
    assert [c.node for c in pick([orin, rtx], "quyet", OPENAI)] == ["rtx", "orin"]
    loading = inst("quyet", "rtx", 0, status="loading")
    assert [c.node for c in pick([orin, loading], "quyet", OPENAI)] == ["orin", "rtx"]
    tie = inst("quyet", "rtx", 1)
    assert pick([tie, orin], "quyet", OPENAI)[0].node == "orin"


def test_pin_prefix_and_errors():
    insts = [inst("quyet-1.0-large.q8_0", "rtx", 0), inst("quyet-1.0-large.q8_0", "orin", 1),
             inst("gemma-4-26b-q4", "orin", 1), inst("gemma-4-26b-q8", "rtx", 0)]
    assert [c.node for c in pick(insts, "quyet", OPENAI)] == ["rtx", "orin"]   # one id, two nodes
    assert [c.node for c in pick(insts, "quyet@orin", OPENAI)] == ["orin"]
    with pytest.raises(Unroutable) as e:
        pick(insts, "gemma-4-26b-q4@rtx", OPENAI)
    assert e.value.status == 404 and "orin" in e.value.message
    with pytest.raises(Unroutable) as e:
        pick(insts, "gemma-4-26b", OPENAI)                                  # two ids
    assert e.value.code == "model_ambiguous"


def test_groups():
    g = "gemma-4-26b-a4b-it"
    insts = [inst("gemma-4-26b-a4b-it-ud-q4_k_m", "orin", 1, group=g),
             inst("gemma-4-26b-a4b-it-q8_0", "rtx", 0, group=g),
             inst("gemma-4-31b-it-q4_k_m", "rtx", 0, group="gemma-4-31b-it"),
             inst("h2o-lightning-4b.q8_0", "rtx", 0, group="h2o-lightning-4b")]
    by_group = pick(insts, g, OPENAI)
    assert [(c.id, c.node) for c in by_group] == [("gemma-4-26b-a4b-it-q8_0", "rtx"),
                                                  ("gemma-4-26b-a4b-it-ud-q4_k_m", "orin")]
    # A prefix of both quants means the group; of two groups, it is ambiguous.
    assert pick(insts, "gemma-4-26b", OPENAI) == by_group
    with pytest.raises(Unroutable) as e:
        pick(insts, "gemma-4", OPENAI)
    assert e.value.code == "model_ambiguous"
    assert [c.node for c in pick(insts, g + "@orin", OPENAI)] == ["orin"]
    # Exact ids still mean exactly that model; a lone member answers to its group.
    assert [c.node for c in pick(insts, "gemma-4-26b-a4b-it-ud-q4_k_m", OPENAI)] == ["orin"]
    assert pick(insts, "h2o-lightning-4b", OPENAI)[0].id == "h2o-lightning-4b.q8_0"


def test_default_group(tmp_path):
    from llm_runner.registry import default_group
    (tmp_path / "gemma-4-26b-a4b-it").mkdir()
    q4 = tmp_path / "gemma-4-26b-a4b-it" / "gemma-4-26b-a4b-it-UD-Q4_K_M.gguf"
    q4.touch()
    loose = tmp_path / "translategemma-27b-it.Q4_K_M.gguf"
    loose.touch()
    assert default_group(q4, tmp_path) == "gemma-4-26b-a4b-it"
    assert default_group(loose, tmp_path) == ""
    assert default_group(tmp_path / "gemma-4-26b-a4b-it", tmp_path) == ""   # a checkpoint dir


def test_vision_filter():
    insts = [inst("h2o", "rtx", 0, vision=False), inst("h2o", "orin", 1, vision=True)]
    assert [c.node for c in pick(insts, "h2o", OPENAI, needs_vision=True)] == ["orin"]
    with pytest.raises(Unroutable) as e:
        pick(insts[:1], "h2o", OPENAI, needs_vision=True)
    assert e.value.code == "vision_unavailable"


@pytest.mark.parametrize("payload, expected", [
    ({"messages": [{"role": "user", "content": [
        {"type": "text", "text": "hi"},
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,AA"}}]}]}, True),
    ({"messages": [{"role": "user", "content": "hi"}]}, False),
    ({"state": "x", "images": ["AA"], "questions": {}}, True),
    ({"state": "x", "images": [], "questions": {}}, False),
    ({"prompt": {"prompt_string": "x", "multimodal_data": ["AA"]}}, True),
    (None, False),
])
def test_has_images(payload, expected):
    assert cl.has_images(payload) is expected


@pytest.mark.parametrize("status, body, expected", [
    (400, b'{"error":{"message":"the request exceeds the available context size, try increasing it"}}', True),
    (400, b"This model's maximum context length is 8192 tokens", True),
    (422, b'{"error": {"code": "input_too_long"}}', True),
    (400, b"'messages' is required", False),
    (500, b"exceeds the available context size", False),
])
def test_context_errors(status, body, expected):
    assert cl.is_context_error(status, body) is expected


# ----------------------------------------------------------------- registry --

def _hb(node, port=9000, **kw):
    return {"protocol": cl.PROTOCOL, "version": "0.9.0", "node": node, "priority": 0,
            "port": port, "models": [], **kw}


def test_registry_table(monkeypatch):
    t = RegistryTable("0.9.0")
    assert t.heartbeat(_hb("rtx"), "192.168.8.10")[0] == 200
    status, reply = t.heartbeat(_hb("orin"), "192.168.8.20")
    assert status == 200 and [n["node"] for n in reply["nodes"]] == ["rtx"]
    assert reply["nodes"][0]["url"] == "http://192.168.8.10:9000"
    assert reply["node_url"] == "http://192.168.8.20:9000" and reply["warnings"] == []
    # Another machine under a name in use.
    status, reply = t.heartbeat(_hb("rtx"), "192.168.8.99")
    assert status == 409 and reply["error"]["code"] == "node_name_taken"
    # Old protocol / other version.
    assert t.heartbeat(_hb("old", protocol=0), "10.0.0.1")[0] == 409
    status, reply = t.heartbeat(_hb("orin", version="0.8.0"), "192.168.8.20")
    assert "orin 0.8.0" in reply["warnings"][0]
    # Silence drops a node; a goodbye drops it at once.
    t.goodbye("orin")
    assert [n["node"] for n in t.snapshot()] == ["rtx"]
    monkeypatch.setattr(cl, "NODE_TIMEOUT", -1)
    assert t.snapshot() == []


# ------------------------------------------------------- three live gateways --

class _Fake(BaseHTTPRequestHandler):
    """A model server: answers who it is; `long` prompts hit its context limit."""
    protocol_version = "HTTP/1.1"

    def log_message(self, *a):
        pass

    def _reply(self, status, payload):
        body = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/health":
            return self._reply(200, {"status": "ok"})
        self._reply(200, {"who": self.server.label, "path": self.path})

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        if body.get("long") and self.server.label == "rtx":
            return self._reply(400, {"error": {
                "message": "the request exceeds the available context size, try increasing it"}})
        if body.get("stream"):
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Connection", "close")
            self.end_headers()
            for i in range(3):
                self.wfile.write(f"data: {self.server.label} {i}\n\n".encode())
                self.wfile.flush()
                time.sleep(0.02)
            self.close_connection = True
            return
        self._reply(200, {"who": self.server.label, "model": body.get("model")})


def _serve(handler_or_httpd, label=None):
    srv = handler_or_httpd if isinstance(handler_or_httpd, ThreadingHTTPServer) \
        else ThreadingHTTPServer(("127.0.0.1", 0), handler_or_httpd)
    srv.daemon_threads = True
    srv.label = label
    threading.Thread(target=srv.serve_forever, args=(0.05,), daemon=True).start()
    return srv


def _rec(name, port, vision=False, group=""):
    cmd = ["llama-server", "--ctx-size", "8192"] + (["--mmproj", "m"] if vision else [])
    return {"name": name, "pid": os.getpid(), "port": port, "started_at": 1.7e9,
            "model_path": f"/m/{name}.gguf", "log_file": "", "cmd": cmd,
            "backend": "llama", "env": {}, "group": group}


def _node(tmp, name, priority, records, registry_url=None, serve=False, token=TOKEN):
    state = tmp / f"{name}.json"
    state.write_text(json.dumps(records))
    g = gw.Gateway(gw.LocalModels(state), Cluster(name, priority, None, registry_url, token, serve))
    httpd = gw.make_server(0, "127.0.0.1", g)
    g.cluster = dataclasses.replace(g.cluster, node_advertise=f"http://127.0.0.1:{g.port}")
    _serve(httpd)
    return g, httpd


@pytest.fixture
def cluster(tmp_path):
    rtx_q = _serve(_Fake, "rtx")
    orin_q = _serve(_Fake, "orin")
    orin_ocr = _serve(_Fake, "orin-ocr")
    lab, lab_srv = _node(tmp_path, "lab", 5, [], serve=True)
    url = f"http://127.0.0.1:{lab.port}"
    rtx, rtx_srv = _node(tmp_path, "rtx", 0,
                         [_rec("quyet", rtx_q.server_address[1], vision=True),
                          _rec("gemma-q8", rtx_q.server_address[1], group="gemma")],
                         registry_url=url)
    orin, orin_srv = _node(tmp_path, "orin", 1,
                           [_rec("quyet", orin_q.server_address[1]),
                            _rec("glm-ocr", orin_ocr.server_address[1]),
                            _rec("gemma-q4", orin_q.server_address[1], group="gemma")],
                           registry_url=url)
    for _ in range(2):
        for g in (lab, rtx, orin):
            g.heartbeat_once()
    yield {"lab": lab, "rtx": rtx, "orin": orin, "rtx_q": rtx_q, "tmp": tmp_path}
    for s in (rtx_q, orin_q, orin_ocr, lab_srv, rtx_srv, orin_srv):
        s.shutdown()


def _call(g, method, path, body=None, headers=None):
    conn = http.client.HTTPConnection("127.0.0.1", g.port, timeout=5)
    conn.request(method, path, body=None if body is None else json.dumps(body),
                 headers={"Content-Type": "application/json", **(headers or {})})
    resp = conn.getresponse()
    raw = resp.read()
    conn.close()
    return resp.status, json.loads(raw) if raw else None


def test_models_cover_the_cluster(cluster):
    status, out = _call(cluster["orin"], "GET", "/v1/models")
    by_id = {m["id"]: m["llmctl"] for m in out["data"]}
    assert set(by_id) == {"quyet", "glm-ocr", "gemma-q8", "gemma-q4", "gemma"}
    assert by_id["gemma"]["members"] == ["gemma-q4", "gemma-q8"]
    assert [(n["node"], n["model"]) for n in by_id["gemma"]["nodes"]] == \
        [("rtx", "gemma-q8"), ("orin", "gemma-q4")]
    q = by_id["quyet"]
    assert q["node"] == "rtx" and q["local"] is True          # here too, but rtx wins
    assert [n["node"] for n in q["nodes"]] == ["rtx", "orin"]
    status, out = _call(cluster["rtx"], "GET", "/v1/models")
    ocr = next(m["llmctl"] for m in out["data"] if m["id"] == "glm-ocr")
    assert ocr["node"] == "orin" and ocr["local"] is False
    # The registry runs no models but routes like any node.
    status, out = _call(cluster["lab"], "GET", "/v1/models")
    assert {m["id"] for m in out["data"]} >= {"quyet", "glm-ocr", "gemma"}


def test_routes_by_priority_pin_and_remote(cluster):
    orin, rtx, lab = cluster["orin"], cluster["rtx"], cluster["lab"]
    assert _call(orin, "POST", "/v1/chat/completions", {"model": "quyet"})[1]["who"] == "rtx"
    out = _call(orin, "POST", "/v1/chat/completions", {"model": "quyet@orin"})[1]
    assert out == {"who": "orin", "model": "quyet"}                 # pin stripped
    assert _call(rtx, "POST", "/v1/chat/completions", {"model": "glm"})[1]["who"] == "orin-ocr"
    assert _call(lab, "POST", "/v1/chat/completions", {"model": "quyet"})[1]["who"] == "rtx"
    assert _call(rtx, "GET", "/m/glm-ocr/props")[1] == {"who": "orin-ocr", "path": "/props"}
    assert _call(rtx, "GET", "/m/quyet@orin/props")[1]["who"] == "orin"


def test_streams_across_a_hop(cluster):
    conn = http.client.HTTPConnection("127.0.0.1", cluster["rtx"].port, timeout=5)
    conn.request("POST", "/v1/chat/completions",
                 body=json.dumps({"model": "glm-ocr", "stream": True}),
                 headers={"Content-Type": "application/json"})
    data = conn.getresponse().read()
    assert data == b"".join(f"data: orin-ocr {i}\n\n".encode() for i in range(3))


def test_failover_on_context_and_down(cluster):
    orin = cluster["orin"]
    # rtx refuses the long prompt; orin's own copy takes it.
    assert _call(orin, "POST", "/v1/chat/completions",
                 {"model": "quyet", "long": True})[1]["who"] == "orin"
    # rtx's model server dies: its gateway says 502 and orin falls back.
    cluster["rtx_q"].shutdown()
    cluster["rtx_q"].server_close()
    assert _call(orin, "POST", "/v1/chat/completions", {"model": "quyet"})[1]["who"] == "orin"


def test_images_go_where_vision_is(cluster):
    img = {"model": "quyet", "messages": [{"role": "user", "content": [
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,AA"}}]}]}
    assert _call(cluster["orin"], "POST", "/v1/chat/completions", img)[1]["who"] == "rtx"
    status, out = _call(cluster["orin"], "POST", "/v1/chat/completions",
                        {**img, "model": "quyet@orin"})
    assert status == 400 and out["error"]["code"] == "vision_unavailable"


def test_hops_need_the_token(cluster):
    rtx = cluster["rtx"]
    status, out = _call(rtx, "POST", "/v1/chat/completions", {"model": "quyet"},
                        headers={cl.HOP_HEADER: "evil", cl.TOKEN_HEADER: "nope"})
    assert status == 401
    status, _ = _call(cluster["lab"], "GET", "/registry/v1/nodes")
    assert status == 401
    status, out = _call(cluster["lab"], "GET", "/registry/v1/nodes",
                        headers={cl.TOKEN_HEADER: TOKEN})
    assert status == 200 and [n["node"] for n in out["nodes"]] == ["rtx", "orin", "lab"]
    # A hop is only routed locally: rtx doesn't pass glm-ocr on.
    status, out = _call(rtx, "POST", "/v1/chat/completions", {"model": "glm-ocr"},
                        headers={cl.HOP_HEADER: "orin", cl.TOKEN_HEADER: TOKEN})
    assert status == 404


def test_wrong_token_and_goodbye(cluster, tmp_path):
    lab_url = f"http://127.0.0.1:{cluster['lab'].port}"
    intruder, srv = _node(tmp_path, "intruder", 0, [], registry_url=lab_url, token="guess")
    intruder.heartbeat_once()
    assert "token" in intruder.view.error and intruder.view.nodes() == []
    srv.shutdown()
    status, out = _call(cluster["rtx"], "GET", "/cluster/v1/status")
    assert out["registry_ok"] and {n["node"] for n in out["nodes"]} == {"orin", "lab"}
    cluster["orin"].goodbye()
    cluster["rtx"].heartbeat_once()
    status, out = _call(cluster["rtx"], "GET", "/v1/models")
    assert {m["id"] for m in out["data"]} == {"quyet", "gemma-q8"}


def test_failover_when_a_remote_model_is_gone(cluster):
    # rtx stopped its quyet since its last heartbeat: orin still lists it there.
    rtx = cluster["rtx"]
    rtx.local.state_file.write_text("[]")
    os.utime(rtx.local.state_file, (time.time() + 5, time.time() + 5))
    assert _call(cluster["orin"], "POST", "/v1/chat/completions",
                 {"model": "quyet"})[1]["who"] == "orin"


def test_registry_sees_a_goodbye_at_once(cluster):
    cluster["orin"].goodbye()
    status, out = _call(cluster["lab"], "GET", "/v1/models")   # no heartbeat in between
    q = next(m for m in out["data"] if m["id"] == "quyet")
    assert [n["node"] for n in q["llmctl"]["nodes"]] == ["rtx"]


def test_group_routes_each_member_by_its_own_name(cluster):
    orin = cluster["orin"]
    out = _call(orin, "POST", "/v1/chat/completions", {"model": "gemma"})[1]
    assert out == {"who": "rtx", "model": "gemma-q8"}
    out = _call(orin, "POST", "/v1/chat/completions", {"model": "gemma@orin"})[1]
    assert out == {"who": "orin", "model": "gemma-q4"}
    # rtx's q8 refuses the long prompt; orin's q4 gets it under its own name.
    out = _call(orin, "POST", "/v1/chat/completions", {"model": "gemma", "long": True})[1]
    assert out == {"who": "orin", "model": "gemma-q4"}
    assert _call(cluster["rtx"], "GET", "/m/gemma/props")[1]["who"] == "rtx"

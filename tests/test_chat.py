"""`llmctl chat` against a fake gateway that speaks the chat and image APIs."""
import base64
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import click
import pytest
from click.testing import CliRunner

from llm_runner import chat
from llm_runner import cli
from llm_runner import config as config_mod
from llm_runner import process as proc
from llm_runner import speedtest as st

PNG = b"\x89PNG\r\n\x1a\nfake"


class Fake:
    def __init__(self):
        self.requests: list[tuple[str, dict | bytes]] = []
        self.headers: list = []
        self.models: list[dict] = []
        self.status = 200
        self.error = "boom"
        self.pieces = [("reasoning_content", "let me think"), ("content", "Hel"),
                       ("content", "lo")]
        self.finish = "stop"
        self.images = [PNG]


def _handler(fake: Fake):
    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def _json(self, obj, code=200):
            data = json.dumps(obj).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            if self.path == "/v1/models":
                self._json({"object": "list", "data": fake.models})
            else:
                self._json({"error": "not found"}, 404)

        def do_POST(self):
            raw = self.rfile.read(int(self.headers["Content-Length"]))
            fake.headers.append(self.headers)
            if self.path == "/v1/images/edits":
                fake.requests.append((self.path, raw))
            else:
                fake.requests.append((self.path, json.loads(raw)))
            if fake.status != 200:
                self._json({"error": {"message": fake.error}}, fake.status)
            elif self.path == "/v1/chat/completions":
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.end_headers()
                for key, piece in fake.pieces:
                    ev = {"choices": [{"delta": {key: piece}, "finish_reason": None}]}
                    self.wfile.write(f"data: {json.dumps(ev)}\n\n".encode())
                ev = {"choices": [{"delta": {}, "finish_reason": fake.finish}],
                      "usage": {"prompt_tokens": 12, "completion_tokens": 3},
                      "timings": {"predicted_per_second": 40.0}}
                self.wfile.write(f"data: {json.dumps(ev)}\n\n".encode())
                self.wfile.write(b"data: [DONE]\n\n")
            else:
                self._json({"data": [{"b64_json": base64.b64encode(b).decode()}
                                     for b in fake.images]})
    return H


@pytest.fixture
def server():
    fake = Fake()
    srv = ThreadingHTTPServer(("127.0.0.1", 0), _handler(fake))
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield fake, srv.server_address[1]
    srv.shutdown()


@pytest.fixture(autouse=True)
def _local(monkeypatch):
    monkeypatch.setattr(st.config_mod, "local_url",
                        lambda port, path="", bind=None: f"http://127.0.0.1:{port}{path}")


def _entry(name, node="rtx", api="openai", backend="llama", status="ready", local=True,
           context=8192, owner=None, nodes=None, **info):
    copies = nodes or [{"node": node, "priority": 0, "local": local, "status": status,
                        "context": context}]
    return {"id": name, "object": "model", "owned_by": "llmctl",
            "llmctl": {"node": copies[0]["node"], "local": local, "status": status, "api": api,
                       "backend": backend, "context": context, "owner": owner, "nodes": copies,
                       **info}}


LLM = _entry("llm")
PICS = _entry("pics", node="orin", api="images", backend="omni", local=False, context=None)


def _model(name="served-id", kind=st.TEXT, backend="llama", node="rtx", context=8192):
    return chat.Model(name, name, node, backend, kind, context=context)


def _text(port, **kw):
    return chat.TextSession(_model(), port, **kw)


def _image(port, tmp_path, **kw):
    return chat.ImageSession(_model(kind=st.IMAGE, backend="omni", context=None), port,
                             out=tmp_path, **kw)


@pytest.fixture
def gateway(server, monkeypatch):
    """The fake server as this machine's gateway, listing LLM and PICS."""
    fake, port = server
    fake.models = [LLM, PICS]
    monkeypatch.setattr(config_mod, "get_gateway_port", lambda: port)
    return fake, port


# ---------------------------------------------------------------- who chats --

@pytest.mark.parametrize("entry,word", [
    (_entry("a", api="kserve", backend="onnx"), "computer-vision"),
    (_entry("a", api="systemone", backend="decider", decision=True), "decision endpoint"),
    (_entry("a", api="systemone", backend="decide", decision=True), "decision endpoint"),
    (_entry("a", embeddings=True), "embedding"),
    (_entry("a", decision=True), "decision model"),
    (_entry("a", api="something-new"), "something-new"),
])
def test_refuses_what_does_not_chat(entry, word):
    assert word in chat.refusal(entry)


@pytest.mark.parametrize("entry", [_entry("a"), _entry("a", backend="vllm"),
                                   _entry("a", backend="sglang"),
                                   _entry("a", api="images", backend="omni")])
def test_llms_and_diffusion_chat(entry):
    assert chat.refusal(entry) is None


def test_choices_offer_every_node_this_machines_first():
    both = _entry("both", local=False, nodes=[
        {"node": "rtx", "local": True, "status": "ready", "context": 4096},
        {"node": "orin", "local": False, "status": "ready", "context": 32768}])
    group = _entry("grp", members=["llm", "both"])
    vec = _entry("vec", embeddings=True)
    rows = chat.choices([PICS, both, group, vec, LLM])
    assert [(m.ref, m.node, m.kind) for m in rows] == [
        ("both@rtx", "rtx", st.TEXT), ("llm", "rtx", st.TEXT),
        ("both@orin", "orin", st.TEXT), ("pics", "orin", st.IMAGE)]
    assert rows[2].context == 32768


def test_resolve_names_groups_and_pins():
    both = _entry("both", group="fam", owner="A1", nodes=[
        {"node": "rtx", "local": True, "status": "ready", "context": 4096},
        {"node": "orin", "local": False, "status": "loading", "context": 32768}])
    entries = [both, PICS]
    m = chat.resolve(entries, "both")
    assert (m.ref, m.node, m.context, m.owner) == ("both", "rtx", 4096, "A1")
    m = chat.resolve(entries, "both@orin")
    assert (m.ref, m.node, m.status, m.owner) == ("both@orin", "orin", "loading", None)
    assert chat.resolve(entries, "fam").id == "both"
    with pytest.raises(click.UsageError, match="isn't running on node 'lab'.*rtx, orin"):
        chat.resolve(entries, "both@lab")
    with pytest.raises(click.UsageError, match="No running model matches 'nope'.*both, pics"):
        chat.resolve(entries, "nope")
    with pytest.raises(click.UsageError, match="it is an embedding"):
        chat.resolve([_entry("vec", embeddings=True)], "vec")


# ------------------------------------------------------------------ gateway --

def test_gateway_off_says_how_to_turn_it_on():
    with pytest.raises(click.UsageError, match="config gateway"):
        chat.models_on(None)


def test_gateway_down_says_how_to_start_it():
    with pytest.raises(click.UsageError, match="gateway start"):
        chat.models_on(1)


# ------------------------------------------------------------------- picker --

def test_picker_offers_only_models_that_chat(gateway, monkeypatch):
    import questionary
    fake, port = gateway
    fake.models = [_entry("llm", owner="A1"), PICS, _entry("vec", embeddings=True),
                   _entry("det", api="kserve", backend="onnx"),
                   _entry("dec", api="systemone", backend="decide", decision=True)]
    shown = {}

    def select(message, choices, **kw):
        shown["titles"] = [c.title for c in choices]
        return type("Q", (), {"ask": lambda _: choices[1].value})()

    monkeypatch.setattr(questionary, "select", select)
    m = cli._chat_target(None, True, port)
    assert (m.id, m.node, m.kind) == ("pics", "orin", st.IMAGE)
    assert len(shown["titles"]) == 2
    assert "rtx" in shown["titles"][0] and "owner=A1" in shown["titles"][0]
    assert "orin" in shown["titles"][1] and "image" in shown["titles"][1]


def test_picker_cancelled_aborts(gateway, monkeypatch):
    import questionary
    _, port = gateway
    monkeypatch.setattr(questionary, "select",
                        lambda *a, **k: type("Q", (), {"ask": lambda _: None})())
    with pytest.raises(click.Abort):
        cli._chat_target(None, True, port)


def test_only_one_able_is_used_without_asking(gateway):
    fake, port = gateway
    fake.models = [_entry("orin-llm", node="orin", local=False), _entry("vec", embeddings=True)]
    m = cli._chat_target(None, False, port)
    assert (m.ref, m.node) == ("orin-llm", "orin")


def test_several_without_a_terminal_must_be_named(gateway):
    _, port = gateway
    with pytest.raises(click.UsageError, match="name one: llm, pics"):
        cli._chat_target(None, False, port)


def test_nothing_able_says_why_each_is_not(gateway):
    fake, port = gateway
    fake.models = [_entry("vec", embeddings=True), _entry("det", api="kserve")]
    with pytest.raises(click.UsageError) as e:
        cli._chat_target(None, True, port)
    assert "vec: an embedding model" in e.value.message and "det: a computer" in e.value.message


def test_nothing_running(gateway):
    fake, port = gateway
    fake.models = []
    with pytest.raises(click.UsageError, match="No running models"):
        cli._chat_target(None, True, port)


def test_a_model_reported_down_is_still_tried(gateway):
    """A model busy generating can miss the gateway's health probe."""
    fake, _ = gateway
    busy = _entry("busy", status="down")
    fake.models = [busy]
    r = CliRunner().invoke(cli.cli, ["chat", "busy", "-m", "hi"])
    assert r.exit_code == 0, r.output
    assert "reports busy on rtx as down; trying anyway" in r.stderr
    assert r.stdout == "Hello\n"
    r = CliRunner().invoke(cli.cli, ["chat", "-m", "hi"])
    assert r.exit_code == 0 and "trying anyway" in r.stderr


def test_the_picker_shows_a_model_that_is_not_ready(gateway, monkeypatch):
    import questionary
    fake, port = gateway
    fake.models = [_entry("slow", status="loading"), LLM]
    shown = {}

    def select(message, choices, **kw):
        shown["titles"] = [c.title for c in choices]
        return type("Q", (), {"ask": lambda _: choices[0].value})()

    monkeypatch.setattr(questionary, "select", select)
    cli._chat_target(None, True, port)
    assert any("(loading)" in t for t in shown["titles"])
    assert not any("(ready)" in t for t in shown["titles"])


def test_a_pid_names_a_model_on_this_machine(gateway, monkeypatch):
    _, port = gateway
    rec = type("R", (), {"name": "llm"})()
    monkeypatch.setattr(proc, "find", lambda i: rec)
    assert cli._chat_target("12345", False, port).id == "llm"


def test_naming_one_that_cannot_chat_explains(gateway):
    fake, _ = gateway
    fake.models = [_entry("dec", api="systemone", backend="decide", decision=True)]
    r = CliRunner().invoke(cli.cli, ["chat", "dec"])
    assert r.exit_code == 2 and "it is a decision endpoint" in r.output
    r = CliRunner().invoke(cli.cli, ["chat", "gateway"])
    assert r.exit_code == 2 and "No running model matches" in r.output


# --------------------------------------------------------------------- text --

def test_conversation_is_kept_and_reasoning_is_not(server, capsys):
    fake, port = server
    s = _text(port, system="be brief", params={"temp": 0.2, "max-tokens": 9})
    assert s.send("hi") and s.send("again")
    _, first = fake.requests[0]
    _, second = fake.requests[1]
    assert first["model"] == "served-id" and first["stream"] is True
    assert first["temperature"] == 0.2 and first["max_tokens"] == 9
    assert [m["role"] for m in second["messages"]] == ["system", "user", "assistant", "user"]
    assert second["messages"][2]["content"] == "Hello"
    out, err = capsys.readouterr()
    assert out == "Hello\nHello\n"
    assert "let me think" in err
    assert "prompt 12/8,192 · 3 tokens · 40.0 t/s" in err


def test_a_refused_message_leaves_the_conversation_as_it_was(server, capsys):
    fake, port = server
    s = _text(port)
    assert s.send("one")
    fake.status, fake.error = 400, "the request exceeds the available context size"
    assert not s.send("too long")
    assert [m["content"] for m in s.history] == ["one", "Hello"]
    out, err = capsys.readouterr()
    assert "/clear starts over" in err and out == "Hello\n"
    fresh = _text(port)
    assert not fresh.send("too long on its own")
    assert "/clear" not in capsys.readouterr().err
    fake.status = 200
    s.command("/retry")
    assert fake.requests[-1][1]["messages"][-1]["content"] == "too long"
    assert [m["content"] for m in s.history] == ["one", "Hello", "too long", "Hello"]


def test_retry_replaces_the_last_answer(server):
    fake, port = server
    s = _text(port)
    s.send("q")
    s.command("/retry")
    assert [m["content"] for m in fake.requests[-1][1]["messages"]] == ["q"]
    assert [m["role"] for m in s.history] == ["user", "assistant"]


def test_cut_off_and_empty_replies_are_reported(server, capsys):
    fake, port = server
    s = _text(port)
    fake.finish = "length"
    assert s.send("q")
    assert "[cut off" in capsys.readouterr().err
    fake.pieces, fake.finish = [], "stop"
    assert not s.send("q2")
    assert "sent no reply" in capsys.readouterr().err
    assert len(s.history) == 2


def test_ctrl_c_keeps_what_came(server, capsys):
    fake, port = server
    t = _model().target(port)
    got = []

    def emit(kind, piece):
        if piece == "lo":
            raise KeyboardInterrupt
        got.append(piece)

    reply = chat.stream_reply(t, "served-id", [{"role": "user", "content": "x"}], {}, emit)
    assert reply.stopped and reply.text == "Hello"      # it was added, then the key came
    assert "Hel" in got


def test_ctrl_c_before_anything_came_forgets_the_message(server, monkeypatch, capsys):
    _, port = server
    s = _text(port)

    def stopped(*a, **k):
        return chat.Reply(stopped=True)

    monkeypatch.setattr(chat, "stream_reply", stopped)
    assert not s.send("q")
    assert s.history == [] and s.unanswered == "q"


def test_dead_server_is_a_message_not_a_traceback(capsys):
    s = chat.TextSession(_model(), 1)
    assert not s.send("q")
    assert "no answer from port 1" in capsys.readouterr().err


def test_stream_error_event_is_reported(server, monkeypatch, capsys):
    fake, port = server
    fake.pieces = []
    real = st._sse

    def with_error(resp):
        yield 0.0, {"error": {"message": "slot died"}}
        yield from real(resp)

    monkeypatch.setattr(st, "_sse", with_error)
    s = _text(port)
    assert not s.send("q")
    assert "slot died" in capsys.readouterr().err


def test_set_unset_system_clear(server, capsys):
    fake, port = server
    s = _text(port)
    s.command("/set top_p 0.5")
    s.command("/set stop [\"x\"]")
    s.command("/set model other")
    assert "chat sets 'model' itself" in capsys.readouterr().err
    s.command("/system be kind")
    s.send("q")
    body = fake.requests[-1][1]
    assert body["top_p"] == 0.5 and body["stop"] == ["x"] and body["model"] == "served-id"
    assert body["messages"][0] == {"role": "system", "content": "be kind"}
    s.command("/unset top_p")
    s.command("/clear")
    s.send("r")
    body = fake.requests[-1][1]
    assert "top_p" not in body and len(body["messages"]) == 2


# ------------------------------------------------------------------- images --

def test_prompt_generates_and_saves(server, tmp_path, capsys):
    fake, port = server
    s = _image(port, tmp_path, params={"size": "512x512", "steps": 4, "negative": "blur",
                                       "cfg": 4})
    assert s.send("a fox")
    path, body = fake.requests[0]
    assert path == "/v1/images/generations"
    assert body["prompt"] == "a fox" and body["model"] == "served-id"
    assert body["num_inference_steps"] == 4 and body["negative_prompt"] == "blur"
    assert body["true_cfg_scale"] == 4 and body["response_format"] == "b64_json"
    out, err = capsys.readouterr()
    saved = out.strip()
    assert saved.endswith(f"-{body['seed']}.png") and saved.startswith(str(tmp_path))
    assert "served-id-" in saved
    assert (tmp_path / saved.rsplit("/", 1)[1]).read_bytes() == PNG
    assert f"seed {body['seed']}" in err


def test_seed_is_new_unless_set_and_several_images_get_numbers(server, tmp_path, capsys):
    fake, port = server
    s = _image(port, tmp_path, params={"n": 2})
    fake.images = [PNG, b"\xff\xd8jpeg"]
    s.send("a")
    s.send("a")
    assert fake.requests[0][1]["seed"] != fake.requests[1][1]["seed"]
    lines = capsys.readouterr().out.split()
    assert lines[0].endswith("-1.png") and lines[1].endswith("-2.jpg")
    s.command("/set seed 7")
    s.send("a")
    assert fake.requests[-1][1]["seed"] == 7


def test_reference_makes_an_edit_and_dropping_it_generates(server, tmp_path, capsys):
    fake, port = server
    ref = tmp_path / "in.png"
    ref.write_bytes(b"REFERENCE")
    s = _image(port, tmp_path, params={"steps": 4})
    s.command(f"/image {ref}")
    assert s.send("make it blue")
    path, raw = fake.requests[-1]
    assert path == "/v1/images/edits"
    assert b'name="prompt"\r\n\r\nmake it blue' in raw
    assert b'name="model"' not in raw
    assert fake.headers[-1]["X-Model"] == "served-id"
    assert b'name="num_inference_steps"\r\n\r\n4' in raw
    assert b'name="image"; filename="in.png"\r\nContent-Type: image/png\r\n\r\nREFERENCE' in raw
    s.command("/image")
    s.send("a fox")
    assert fake.requests[-1][0] == "/v1/images/generations"
    s.command("/image /no/such/file")
    assert "No such file" in capsys.readouterr().err
    assert s.references == []


def _spawned(monkeypatch):
    calls = []

    def popen(argv, **kw):
        calls.append((argv, kw))

    monkeypatch.setattr(chat.subprocess, "Popen", popen)
    monkeypatch.setattr(chat, "opener", lambda: "/usr/bin/xdg-open")
    return calls


def test_each_image_opens_in_the_viewer_detached(server, tmp_path, monkeypatch):
    fake, port = server
    calls = _spawned(monkeypatch)
    fake.images = [PNG, b"\xff\xd8jpeg"]
    s = _image(port, tmp_path, preview=True, params={"n": 2})
    assert s.send("a fox")
    assert [c[0][0] for c in calls] == ["/usr/bin/xdg-open"] * 2
    assert all(Path(c[0][1]).is_absolute() and Path(c[0][1]).is_file() for c in calls)
    assert all(c[1]["start_new_session"] and c[1]["stdout"] == chat.subprocess.DEVNULL
               for c in calls)


def test_retries_and_edits_open_too(server, tmp_path, monkeypatch):
    _, port = server
    calls = _spawned(monkeypatch)
    ref = tmp_path / "in.png"
    ref.write_bytes(b"REFERENCE")
    s = _image(port, tmp_path, preview=True)
    s.send("a fox")
    s.command("/retry")
    s.command(f"/image {ref}")
    s.send("make it blue")
    assert len(calls) == 3


def test_nothing_opens_when_preview_is_off_or_generation_fails(server, tmp_path, monkeypatch):
    fake, port = server
    calls = _spawned(monkeypatch)
    assert _image(port, tmp_path, preview=False).send("a fox")
    assert calls == []
    fake.status = 500
    assert not _image(port, tmp_path, preview=True).send("a fox")
    assert calls == []


def test_a_viewer_that_wont_start_leaves_the_image_saved(server, tmp_path, monkeypatch, capsys):
    _, port = server
    monkeypatch.setattr(chat, "opener", lambda: "/nope/xdg-open")
    s = _image(port, tmp_path, preview=True)
    assert s.send("a fox")
    out, err = capsys.readouterr()
    assert out.strip() and Path(out.strip()).is_file()
    assert "Can't open images" in err and s.preview is False


@pytest.mark.parametrize("platform,env,which,want", [
    ("linux", {"DISPLAY": ":0"}, "/usr/bin/xdg-open", "/usr/bin/xdg-open"),
    ("linux", {"WAYLAND_DISPLAY": "wayland-0"}, "/usr/bin/xdg-open", "/usr/bin/xdg-open"),
    ("linux", {}, "/usr/bin/xdg-open", None),                   # plain SSH
    ("linux", {"DISPLAY": ":0"}, None, None),                   # no xdg-utils
    ("darwin", {}, "/usr/bin/open", "/usr/bin/open"),
])
def test_opener(monkeypatch, platform, env, which, want):
    monkeypatch.setattr(chat.sys, "platform", platform)
    monkeypatch.delenv("DISPLAY", raising=False)
    monkeypatch.delenv("WAYLAND_DISPLAY", raising=False)
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    monkeypatch.setattr(chat.shutil, "which", lambda name: which)
    assert chat.opener() == want


def test_image_failures_are_messages(server, tmp_path, capsys):
    fake, port = server
    s = _image(port, tmp_path)
    fake.status, fake.error = 400, "bad size"
    assert not s.send("a")
    assert "bad size" in capsys.readouterr().err
    fake.status, fake.images = 200, []
    assert not s.send("a")
    assert "returned no image" in capsys.readouterr().err


# --------------------------------------------------------------------- repl --

def _lines(*items):
    it = iter(items)

    def read():
        item = next(it, EOFError)
        if isinstance(item, type) and issubclass(item, BaseException):
            raise item
        return item
    return read


def test_repl_sends_runs_commands_and_leaves(server, capsys):
    fake, port = server
    s = _text(port)
    chat.repl(s, _lines("  ", "hi", KeyboardInterrupt, "//etc is a path", "/nope", "/exit", "x"))
    sent = [b["messages"][-1]["content"] for _, b in fake.requests]
    assert sent == ["hi", "/etc is a path"]
    err = capsys.readouterr().err
    assert "Unknown command /nope" in err and "ctx 8,192" in err


@pytest.mark.parametrize("keys,want", [
    ("sent\r", "sent"),
    ("typed ahead\n", "typed ahead"),      # Enter during a reply, not in raw mode
    ("first\x1b\rsecond\r", "first\nsecond"),
])
def test_the_prompt_sends_on_enter_and_adds_a_line_on_alt_enter(keys, want):
    from prompt_toolkit.application import create_app_session
    from prompt_toolkit.input import create_pipe_input
    from prompt_toolkit.output import DummyOutput
    with create_pipe_input() as pipe, create_app_session(input=pipe, output=DummyOutput()):
        pipe.send_text(keys)
        assert chat._reader()() == want


def test_repl_ends_on_ctrl_d(server):
    _, port = server
    chat.repl(_text(port), _lines(EOFError))


def test_help_lists_each_kinds_commands(server, tmp_path, capsys):
    _, port = server
    _text(port).command("/help")
    assert "/system" in capsys.readouterr().err
    _image(port, tmp_path).command("/help")
    err = capsys.readouterr().err
    assert "/image FILE" in err and "/system" not in err


# ---------------------------------------------------------------------- cli --

def test_requests_name_the_model_as_typed(gateway):
    fake, _ = gateway
    fake.models = [_entry("both", nodes=[
        {"node": "rtx", "local": True, "status": "ready", "context": 4096},
        {"node": "orin", "local": False, "status": "ready", "context": 32768}])]
    r = CliRunner().invoke(cli.cli, ["chat", "both@orin", "-m", "hi"])
    assert r.exit_code == 0, r.output
    assert fake.requests[0][1]["model"] == "both@orin"
    assert "32,768" in r.stderr


def test_one_shot_prints_only_the_answer_on_stdout(gateway):
    r = CliRunner().invoke(cli.cli, ["chat", "llm", "-m", "hi", "-t", "0.1", "--seed", "3",
                                     "--set", "top_p=0.9", "--size", "512x512"])
    assert r.exit_code == 0, r.output
    assert r.stdout == "Hello\n"
    assert "--size doesn't apply" in r.stderr


def test_one_shot_sends_the_flags(gateway):
    fake, _ = gateway
    CliRunner().invoke(cli.cli, ["chat", "llm", "-m", "hi", "-t", "0.1", "--max-tokens", "5",
                                 "--seed", "3", "--set", "top_p=0.9", "-s", "sys"])
    body = fake.requests[0][1]
    assert body["temperature"] == 0.1 and body["max_tokens"] == 5
    assert body["seed"] == 3 and body["top_p"] == 0.9
    assert body["messages"][0]["content"] == "sys"


def test_piped_stdin_is_one_message(gateway):
    fake, _ = gateway
    r = CliRunner().invoke(cli.cli, ["chat", "llm"], input="from a pipe\n")
    assert r.exit_code == 0, r.output
    assert fake.requests[0][1]["messages"] == [{"role": "user", "content": "from a pipe"}]
    r = CliRunner().invoke(cli.cli, ["chat", "llm"], input="")
    assert r.exit_code == 2 and "Nothing on stdin" in r.output


def test_one_shot_failure_exits_nonzero(gateway):
    fake, _ = gateway
    fake.status = 500
    assert CliRunner().invoke(cli.cli, ["chat", "llm", "-m", "hi"]).exit_code == 1


def test_image_on_another_node_is_saved_here(gateway, tmp_path):
    fake, _ = gateway
    r = CliRunner().invoke(cli.cli, ["chat", "pics", "-m", "a fox", "-o", str(tmp_path / "new"),
                                     "--steps", "4", "--size", "768x768", "-t", "0.5"])
    assert r.exit_code == 0, r.output
    assert r.stdout.strip().startswith(str(tmp_path / "new"))
    assert "--temperature doesn't apply" in r.stderr
    body = fake.requests[0][1]
    assert body["size"] == "768x768" and body["num_inference_steps"] == 4


def _one_image(tmp_path, *flags):
    return CliRunner().invoke(cli.cli, ["chat", "pics", "-m", "a fox", "-o", str(tmp_path),
                                        *flags])


def test_piped_output_opens_nothing_unless_asked(gateway, tmp_path, monkeypatch):
    calls = _spawned(monkeypatch)
    assert _one_image(tmp_path).exit_code == 0
    assert calls == []
    assert _one_image(tmp_path, "--open").exit_code == 0
    assert len(calls) == 1


def test_no_open_wins_on_a_terminal(gateway, tmp_path, monkeypatch):
    calls = _spawned(monkeypatch)
    assert _one_image(tmp_path, "--no-open").exit_code == 0
    assert calls == []


def test_open_with_no_desktop_says_so(gateway, tmp_path, monkeypatch):
    monkeypatch.setattr(chat, "opener", lambda: None)
    r = _one_image(tmp_path, "--open")
    assert r.exit_code == 0 and "no desktop here" in r.stderr
    assert Path(r.stdout.strip()).is_file()


def test_open_on_a_text_model_is_a_note(gateway):
    r = CliRunner().invoke(cli.cli, ["chat", "llm", "-m", "hi", "--open"])
    assert r.exit_code == 0 and "--open doesn't apply" in r.stderr


def test_bad_flags(gateway):
    r = CliRunner().invoke(cli.cli, ["chat", "llm", "-m", "hi", "--set", "nope"])
    assert r.exit_code == 2 and "KEY=VALUE" in r.output
    r = CliRunner().invoke(cli.cli, ["chat", "llm", "-m", "hi", "--set", "stream=false"])
    assert r.exit_code == 2 and "chat sets 'stream' itself" in r.output
    r = CliRunner().invoke(cli.cli, ["chat", "pics", "-m", "hi", "--size", "huge"])
    assert r.exit_code == 2 and "WxH" in r.output


def test_gateway_off_stops_the_command(monkeypatch):
    monkeypatch.setattr(config_mod, "get_gateway_port", lambda: None)
    r = CliRunner().invoke(cli.cli, ["chat", "llm", "-m", "hi"])
    assert r.exit_code == 2 and "turned off" in r.output

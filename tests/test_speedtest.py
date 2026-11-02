"""`llmctl speedtest` against a fake server that speaks each backend's API."""
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest
from click.testing import CliRunner

from llm_runner import process as proc
from llm_runner import speedtest as st

TOKENS_PER_WORD = 1.2
OVERHEAD = 20


def _prompt_tokens(body: dict) -> int:
    if "messages" in body:
        text = body["messages"][0]["content"]
    elif "prompt" in body:
        text = body["prompt"]
    else:
        text = body["state"]
    words = len(text.split()) - (len(st._TASK.split()) if "messages" in body else 0)
    return round(words * TOKENS_PER_WORD) + OVERHEAD


class Fake:
    """What the fake server answers, and what it was sent."""

    def __init__(self, flavour: str):
        self.flavour = flavour              # llama | vllm | embed | decide | image
        self.requests: list[tuple[str, dict]] = []
        self.busy = 0
        self.spec = [0.0, 0.0]


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
            if self.path == "/health":
                return self._json({"status": "ok"})
            if self.path == "/v1/models":
                return self._json({"data": [{"id": "served-id", "max_model_len": 8192}]})
            if self.path == "/props" and fake.flavour == "llama":
                return self._json({"default_generation_settings": {"n_ctx": 8192}})
            if self.path == "/slots" and fake.flavour == "llama":
                return self._json([{"is_processing": i < fake.busy} for i in range(2)])
            if self.path == "/metrics" and fake.flavour == "vllm":
                text = (f"vllm:num_requests_running{{model_name=\"x\"}} {fake.busy}\n"
                        "vllm:num_requests_waiting{model_name=\"x\"} 0\n"
                        f"vllm:spec_decode_num_accepted_tokens_total {fake.spec[0]}\n"
                        f"vllm:spec_decode_num_draft_tokens_total {fake.spec[1]}\n")
                data = text.encode()
                self.send_response(200)
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                return self.wfile.write(data)
            self._json({"error": "not found"}, 404)

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            fake.requests.append((self.path, body))
            if self.path == "/v1/embeddings":
                n = sum(_prompt_tokens({"prompt": x}) for x in body["input"])
                return self._json({"data": [{"embedding": [0.0]} for _ in body["input"]],
                                   "usage": {"prompt_tokens": n, "total_tokens": n}})
            if self.path == "/v1/systemone":
                q = body["questions"]
                return self._json({"answers": {k: {"type": "noul", "noul": 0.5} for k in q},
                                   "usage": {"input_tokens": _prompt_tokens(body) * len(q),
                                             "output_tokens": 0, "backend_calls": 1}})
            if self.path == "/v1/images/generations":
                return self._json({"data": [{"b64_json": "AAAA"}]})
            prompt = _prompt_tokens(body)
            if prompt > 8192:
                return self._json({"error": {"message": "context exceeded"}}, 400)
            if not body.get("stream"):
                return self._json({"choices": [{"message": {"content": "x"}}],
                                   "usage": {"prompt_tokens": prompt, "completion_tokens": 1}})
            gen = body["max_tokens"]
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()
            key = "text" if self.path == "/v1/completions" else None
            for _ in range(gen):
                choice = {"text": "a"} if key else {"delta": {"content": "a"}}
                self.wfile.write(f"data: {json.dumps({'choices': [choice]})}\n\n".encode())
            last: dict = {"choices": [], "usage": {"prompt_tokens": prompt,
                                                   "completion_tokens": gen}}
            if fake.flavour == "llama":
                last["timings"] = {"prompt_n": prompt, "prompt_per_second": 1000.0,
                                   "predicted_n": gen, "predicted_per_second": 50.0,
                                   "draft_n": 10, "draft_n_accepted": 7}
            else:
                fake.spec[0] += 6
                fake.spec[1] += 10
            self.wfile.write(f"data: {json.dumps(last)}\n\ndata: [DONE]\n\n".encode())

    return H


@pytest.fixture
def server():
    started = []

    def start(flavour):
        fake = Fake(flavour)
        srv = ThreadingHTTPServer(("127.0.0.1", 0), _handler(fake))
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        started.append(srv)
        return fake, srv.server_address[1]

    yield start
    for srv in started:
        srv.shutdown()


@pytest.fixture(autouse=True)
def _local(monkeypatch, tmp_path):
    monkeypatch.setattr(st.config_mod, "local_url",
                        lambda port, path="", bind=None: f"http://127.0.0.1:{port}{path}")
    monkeypatch.setattr(st, "STATE_FILE", tmp_path / "speedtest.json")


def _values(kind, **kw):
    return {**st.defaults_for(kind), **kw}


def test_text_on_llama_uses_server_timings_and_fresh_prompts(server):
    fake, port = server("llama")
    t = st.Target("m", "llama", port, st.TEXT, cmd=("llama-server", "--model-draft", "d"))
    rows, header = st.run(t, _values(st.TEXT, prompt="512,4096", gen=16, runs=2), seed=1)
    assert header["ctx"] == 8192 and header["model"] == "served-id"
    small, big = rows
    assert abs(small["prompt_tokens"] - 512) <= 4 and abs(big["prompt_tokens"] - 4096) <= 8
    assert small["decode_tps"] == 50.0 and small["prefill_tps"] == 1000.0
    assert small["server_timed"] and small["draft_accept"] == pytest.approx(0.7)
    streamed = [b for p, b in fake.requests if b.get("stream")]
    assert all(b["cache_prompt"] is False and b["ignore_eos"] for b in streamed)
    contents = [b["messages"][0]["content"] for b in streamed]
    assert len(set(contents)) == len(contents)      # no prompt repeats: no cache hits
    out = st.render(t, _values(st.TEXT, gen=16, runs=2), rows, header)
    assert "drafter on" in out and "70%" in out and "server timings" in out


def test_text_on_vllm_times_the_stream_and_reads_spec_counters(server):
    fake, port = server("vllm")
    t = st.Target("m", "vllm", port, st.TEXT)
    rows, _ = st.run(t, _values(st.TEXT, prompt="512", gen=32, runs=2, concurrency=2), seed=1)
    [row] = rows
    assert row["ok"] == 4 and not row["server_timed"]
    assert row["decode_tps"] > 0 and row["total_tps"] > 0 and row["ttft_s"] >= 0
    assert row["draft_accept"] == pytest.approx(0.6)
    sent = [b for p, b in fake.requests if b.get("stream")]
    assert all(b["model"] == "served-id" and "cache_prompt" not in b for b in sent)


def test_text_skips_sizes_beyond_the_context(server):
    _, port = server("llama")
    t = st.Target("m", "llama", port, st.TEXT)
    rows, _ = st.run(t, _values(st.TEXT, prompt="512,16384", gen=8, runs=1), seed=1)
    assert rows[0]["ok"] == 1
    assert rows[1]["ok"] == 0 and "doesn't fit" in rows[1]["note"]


def test_decision_model_on_llama_uses_raw_completions(server):
    fake, port = server("llama")
    t = st.Target("m", "llama", port, st.TEXT, completion=True)
    rows, _ = st.run(t, _values(st.TEXT, prompt="512", gen=8, runs=1), seed=1)
    assert rows[0]["ok"] == 1
    assert {p for p, _ in fake.requests} == {"/v1/completions"}


def test_embeddings_report_per_input_tokens(server):
    _, port = server("embed")
    t = st.Target("e", "llama", port, st.EMBED)
    rows, _ = st.run(t, _values(st.EMBED, prompt="512", inputs=4, runs=2), seed=1)
    [row] = rows
    assert abs(row["prompt_tokens"] - 512) <= 4
    assert row["inputs_per_s"] > 0 and row["tokens_per_s"] > row["inputs_per_s"]


def test_decisions_report_per_question_state_tokens(server):
    _, port = server("decide")
    t = st.Target("d", "decide", port, st.DECISION)
    rows, _ = st.run(t, _values(st.DECISION, prompt="1024", questions=4, runs=1), seed=1)
    [row] = rows
    assert abs(row["prompt_tokens"] - 1024) <= 6 and row["decisions_per_s"] > 0


def test_images_time_each_generation(server):
    fake, port = server("image")
    t = st.Target("f", "omni", port, st.IMAGE)
    rows, _ = st.run(t, _values(st.IMAGE, size="512x512", steps=4, runs=2), seed=1)
    [row] = rows
    assert row["ok"] == 2 and row["seconds_per_image"] >= 0
    bodies = [b for p, b in fake.requests if p == "/v1/images/generations"]
    assert len(bodies) == 3                                    # warm-up + 2 runs
    assert all(b["size"] == "512x512" and b["num_inference_steps"] == 4 for b in bodies)


def test_errors_are_reported_not_raised(server):
    _, port = server("llama")
    t = st.Target("m", "llama", port + 1 if port < 65535 else port - 1, st.TEXT)
    rows, _ = st.run(t, _values(st.TEXT, prompt="512", runs=1), seed=1)
    assert rows[0]["ok"] == 0 and "no answer" in rows[0]["errors"][0]


def test_busy_detection(server):
    fake, port = server("llama")
    t = st.Target("m", "llama", port, st.TEXT)
    assert st.active_requests(t) == 0
    fake.busy = 2
    assert st.active_requests(t) == 2
    vfake, vport = server("vllm")
    vfake.busy = 3
    assert st.active_requests(st.Target("v", "vllm", vport, st.TEXT)) == 3
    # A decision endpoint is as busy as the llama-server behind it.
    d = st.Target("d", "decide", 1, st.DECISION,
                  cmd=("systemone-proxy", "serve", "--backend", f"http://127.0.0.1:{port}"))
    assert st.active_requests(d) == 2


def test_kind_of_each_backend():
    assert st.kind_of("llama", ["llama-server", "-m", "x"]) == st.TEXT
    assert st.kind_of("llama", ["llama-server", "--embeddings"]) == st.EMBED
    assert st.kind_of("vllm", ["vllm", "serve", "--runner", "pooling"]) == st.EMBED
    assert st.kind_of("sglang", ["sglang", "serve", "--is-embedding"]) == st.EMBED
    assert st.kind_of("decider", []) == st.DECISION
    assert st.kind_of("decide", []) == st.DECISION
    assert st.kind_of("omni", []) == st.IMAGE


# ------------------------------------------------------------------ profiles --

def test_profiles_fit_the_kind_and_reset_other_rows():
    text = st.Target("m", "llama", 1, st.TEXT)
    image = st.Target("f", "omni", 1, st.IMAGE)
    assert [p.name for p in st.profiles_for(text)] == ["default", "quick", "long", "load"]
    assert [p.name for p in st.profiles_for(image)] == ["default", "quick"]
    quick = st.resolve(st.TEXT, st.find_profile(text, "quick"))
    assert quick == {"prompt": "512", "gen": 128, "concurrency": 1, "runs": 1}
    assert st.resolve(st.IMAGE, st.find_profile(image, "quick")) == {
        "size": "512x512", "steps": None, "runs": 1}
    with pytest.raises(Exception, match="no profile 'long'"):
        st.find_profile(image, "long")


def test_user_profiles_and_last_are_offered_per_kind():
    text = st.Target("m", "llama", 1, st.TEXT)
    st.save_profile("mine", st.TEXT, {"prompt": "2048", "gen": 512})
    st.save_profile("pics", st.IMAGE, {"steps": 20})
    st.remember("m", {"runs": 5})
    names = [p.name for p in st.profiles_for(text)]
    assert names[0] == "last" and "mine" in names and "pics" not in names
    assert st.resolve(st.TEXT, st.find_profile(text, "mine"))["prompt"] == "2048"
    assert st.resolve(st.TEXT, st.find_profile(text, "last"))["runs"] == 5
    assert st.delete_profile("mine") and not st.delete_profile("mine")


class _Scripted:
    def __init__(self, answers):
        self.answers, self.prompts = list(answers), []

    def select(self, message, choices, **kw):
        self.prompts.append((message, [getattr(c, "title", None) for c in choices]))
        answer = self.answers.pop(0)
        return type("Q", (), {"ask": lambda _: answer})()

    def text(self, message, **kw):
        answer = self.answers.pop(0)
        return type("Q", (), {"ask": lambda _: answer})()


def test_menu_picks_a_profile_then_a_row_and_saves(monkeypatch):
    import questionary
    t = st.Target("m", "llama", 1, st.TEXT)
    # Profile row -> "load" (index 3), the Generate row -> 512 (index 3),
    # Save as profile "mine", then Run.
    script = _Scripted(["profile", 3, "gen", 3, "save", "mine", "run"])
    monkeypatch.setattr(questionary, "select", script.select)
    monkeypatch.setattr(questionary, "text", script.text)
    values = st.configure(t, st.profiles_for(t)[0], {})
    assert values == {"prompt": "1024", "gen": 512, "concurrency": 4, "runs": 3}
    assert st.user_profiles()["mine"] == {
        "kind": st.TEXT, "values": {"prompt": "1024", "gen": 512, "concurrency": 4}}
    first_rows = script.prompts[0][1]
    assert any("Prompt tokens" in r for r in first_rows if r)
    assert any("load (modified)" in r for r in script.prompts[4][1] if r)


def test_typed_values_win_over_the_profile(monkeypatch):
    import questionary
    t = st.Target("m", "llama", 1, st.TEXT)
    monkeypatch.setattr(questionary, "select", _Scripted(["run"]).select)
    values = st.configure(t, st.find_profile(t, "quick"), {"gen": 64})
    assert values["gen"] == 64 and values["prompt"] == "512"


# ---------------------------------------------------------------------- cli --

def _record(port, backend="llama", cmd=None):
    return proc.RunningProcess(name="m", pid=1, port=port, started_at=0.0,
                               model_path="/nowhere/m.gguf", log_file="/dev/null",
                               cmd=cmd or ["llama-server"], backend=backend)


def test_cli_json(server, monkeypatch):
    from llm_runner import cli
    _, port = server("llama")
    rec = _record(port)
    monkeypatch.setattr(cli, "_models_running", lambda: [rec])
    r = CliRunner().invoke(cli.cli, ["speedtest", "--profile", "quick", "--gen", "8",
                                     "--size", "512x512", "--json"])
    assert r.exit_code == 0, r.output
    out = json.loads(r.output[r.output.index("{"):])
    assert out["options"] == {"prompt": "512", "gen": 8, "concurrency": 1, "runs": 1}
    assert out["results"][0]["decode_tps"] == 50.0
    assert "--size doesn't apply" in r.output
    assert st.remembered("m") == {"prompt": "512", "gen": 8, "runs": 1}


def test_cli_refuses_a_busy_model_without_force(server, monkeypatch):
    from llm_runner import cli
    fake, port = server("llama")
    fake.busy = 1
    monkeypatch.setattr(cli, "_models_running", lambda: [_record(port)])
    r = CliRunner().invoke(cli.cli, ["speedtest", "--no-menu", "--runs", "1", "--gen", "4",
                                     "--prompt-tokens", "128"])
    assert r.exit_code != 0 and "--force" in r.output
    r = CliRunner().invoke(cli.cli, ["speedtest", "--no-menu", "--runs", "1", "--gen", "4",
                                     "--prompt-tokens", "128", "--force"])
    assert r.exit_code == 0, r.output


def test_cli_profile_list_and_delete():
    from llm_runner import cli
    st.save_profile("mine", st.TEXT, {"gen": 512})
    r = CliRunner().invoke(cli.cli, ["speedtest", "--list-profiles"])
    assert r.exit_code == 0 and "quick" in r.output and "mine" in r.output and "--gen 512" in r.output
    r = CliRunner().invoke(cli.cli, ["speedtest", "--delete-profile", "mine"])
    assert r.exit_code == 0 and "mine" not in st.user_profiles()
    r = CliRunner().invoke(cli.cli, ["speedtest", "--delete-profile", "quick"])
    assert r.exit_code != 0 and "built in" in r.output


# ------------------------------------------------------------ several models --

def _records(monkeypatch, recs):
    from llm_runner import cli
    monkeypatch.setattr(cli, "_models_running", lambda: list(recs))
    monkeypatch.setattr(cli.proc, "find",
                        lambda ident: next((r for r in recs if r.name == ident), None))


def _named(port, name, backend="llama", cmd=None):
    rec = _record(port, backend, cmd)
    rec.name = name
    return rec


def test_cli_compares_several_named_models(server, monkeypatch):
    from llm_runner import cli
    _, lport = server("llama")
    _, vport = server("vllm")
    _records(monkeypatch, [_named(lport, "fast-one"), _named(vport, "other", "vllm")])
    r = CliRunner().invoke(cli.cli, ["speedtest", "fast-one", "other", "--profile", "quick",
                                     "--gen", "8", "--prompt-tokens", "256,16384"])
    assert r.exit_code == 0, r.output
    out = r.output
    assert "speedtest  2 models · text generation" in out
    assert "256 tokens" in out and "vs best: decode t/s, higher is better" in out
    assert "16,384 tokens: no model measured" in out
    section = out[out.index("  256 tokens"):]
    assert "fast-one" in section and "other" in section and "best" in section
    assert "doesn't fit the context" in out                 # 16K on an 8K context
    assert st.remembered("*text") == {"prompt": "256,16384", "gen": 8, "runs": 1}
    assert st.remembered("other") == st.remembered("*text")


def test_cli_picker_ticks_models_and_json_lists_them(server, monkeypatch):
    import questionary
    from llm_runner import cli
    _, a = server("llama")
    _, b = server("llama")
    recs = [_named(a, "a"), _named(b, "b"), _named(b + 1, "c")]
    _records(monkeypatch, recs)
    shown = {}

    def checkbox(message, choices, **kw):
        shown["titles"] = [c.title for c in choices]
        shown["instruction"] = kw.get("instruction")
        return type("Q", (), {"ask": lambda _: [recs[0], recs[1]]})()

    monkeypatch.setattr(questionary, "checkbox", checkbox)
    r = CliRunner().invoke(cli.cli, ["speedtest", "--runs", "1", "--gen", "4",
                                     "--prompt-tokens", "128", "--json"])
    assert r.exit_code == 0, r.output
    assert len(shown["titles"]) == 3 and "space" in shown["instruction"]
    out = json.loads(r.output[r.output.index("{"):])
    assert [m["name"] for m in out["models"]] == ["a", "b"]
    assert all(m["results"][0]["decode_tps"] == 50.0 for m in out["models"])


def test_cli_mixed_kinds_get_their_own_tables(server, monkeypatch):
    from llm_runner import cli
    _, tport = server("llama")
    _, iport = server("image")
    _records(monkeypatch, [_named(tport, "chat"), _named(iport, "pics", "omni")])
    r = CliRunner().invoke(cli.cli, ["speedtest", "chat", "pics", "--profile", "quick",
                                     "--gen", "4"])
    assert r.exit_code == 0, r.output
    assert "speedtest  chat" in r.output and "speedtest  pics" in r.output
    assert "--gen doesn't apply to pics" in r.output


def test_cli_skips_a_model_that_is_down(server, monkeypatch):
    from llm_runner import cli
    _, port = server("llama")
    _records(monkeypatch, [_named(port, "up"), _named(port + 1 if port < 65535 else port - 1,
                                                      "down")])
    r = CliRunner().invoke(cli.cli, ["speedtest", "up", "down", "--profile", "quick",
                                     "--gen", "4"])
    assert r.exit_code == 0, r.output
    assert "skipping down" in r.output and "speedtest  up" in r.output


def test_compare_ranks_images_by_lowest_time():
    t1, t2 = st.Target("a", "omni", 1, st.IMAGE), st.Target("b", "omni", 2, st.IMAGE)
    row = {"label": "512x512", "ok": 2, "failed": 0}
    out = st.render_compare(st.IMAGE, _values(st.IMAGE, size="512x512", steps=4), [
        (t1, [{**row, "seconds_per_image": 2.0}], {}),
        (t2, [{**row, "seconds_per_image": 4.0}], {})])
    a_line = [line for line in out.splitlines() if line.strip().startswith("a ")][-1]
    b_line = [line for line in out.splitlines() if line.strip().startswith("b ")][-1]
    assert a_line.rstrip().endswith("best") and b_line.rstrip().endswith("50%")

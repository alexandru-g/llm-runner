"""`llmctl config bind`: the address the gateway and every backend listen on."""
from pathlib import Path

import pytest
from click.testing import CliRunner

from llm_runner import backends, cli, config as config_mod, process as proc


@pytest.fixture
def cfg(tmp_path, monkeypatch):
    monkeypatch.setattr(config_mod, "CONFIG_DIR", tmp_path)
    monkeypatch.setattr(config_mod, "CONFIG_FILE", tmp_path / "config.json")
    monkeypatch.setattr(proc, "list_running", lambda: [])
    return tmp_path


def _host(cmd):
    return cmd[cmd.index("--host") + 1]


def test_default_is_every_interface(cfg):
    assert config_mod.get_bind_host() == "0.0.0.0"
    assert _host(cli._build_llama_cmd(Path("m.gguf"), None, 8080, 4096, False)) == "0.0.0.0"


def test_config_bind_sets_shows_and_resets(cfg):
    runner = CliRunner()
    out = runner.invoke(cli.cli, ["config", "bind", "127.0.0.1"])
    assert out.exit_code == 0, out.output
    assert config_mod.get_bind_host() == "127.0.0.1"
    assert runner.invoke(cli.cli, ["config", "bind"]).output.strip() == "127.0.0.1"
    assert runner.invoke(cli.cli, ["config", "bind", "--reset"]).exit_code == 0
    assert config_mod.get_bind_host() == "0.0.0.0"
    assert "bind_host" not in config_mod.load()      # the default isn't stored


def test_config_bind_rejects_an_unresolvable_host(cfg):
    out = CliRunner().invoke(cli.cli, ["config", "bind", "no such host!"])
    assert out.exit_code != 0 and config_mod.get_bind_host() == "0.0.0.0"


def test_builders_take_the_host():
    host = "127.0.0.1"
    assert _host(cli._build_llama_cmd(Path("m.gguf"), None, 8080, 4096, False,
                                      embeddings=True, host=host)) == host
    assert _host(cli._build_llama_cmd(Path("m.gguf"), None, 8080, 4096, False,
                                      host=host)) == host
    assert _host(backends.build_vllm_cmd("vllm", Path("m"), name="m", port=8000,
                                         ctx_size=4096, host=host)) == host
    cmd, _ = backends.build_decider_cmd("uvicorn", Path("d"), Path("m"), port=8100, host=host)
    assert _host(cmd) == host


def test_replayed_commands_move_to_the_current_host():
    llama = ["llama-server", "--port", "8080", "--host", "0.0.0.0", "--alias", "m"]
    assert backends.with_bind_host(llama, backends.LLAMA, "127.0.0.1") == [
        "llama-server", "--port", "8080", "--host", "127.0.0.1", "--alias", "m"]
    # The gateway and decision endpoints were once recorded without --host.
    gw = ["python", "-m", "llm_runner", "gateway", "serve", "--port", "9000"]
    assert _host(backends.with_bind_host(gw, backends.GATEWAY, "127.0.0.1")) == "127.0.0.1"
    assert backends.with_bind_host(["x"], backends.LLAMA, "127.0.0.1") == ["x"]


def test_gateway_start_passes_the_host(cfg, monkeypatch):
    config_mod.set_bind_host("127.0.0.1")
    seen = {}
    monkeypatch.setattr(proc, "start", lambda **kw: seen.update(kw))
    cli._start_gateway(9000)
    assert _host(seen["cmd"]) == "127.0.0.1"


def test_gateway_url_follows_the_host(cfg):
    assert "127.0.0.1" not in cli._gateway_url(9000)
    config_mod.set_bind_host("127.0.0.1")
    assert cli._gateway_url(9000) == "http://127.0.0.1:9000/v1"


@pytest.mark.parametrize("typed, url", [
    ("192.168.8.213", "http://192.168.8.213:9000"),
    ("lab:9001", "http://lab:9001"),
    ("[fd00::1]", "http://[fd00::1]:9000"),
    ("http://lab:9000/", "http://lab:9000"),
    ("https://llmctl.example.net", "https://llmctl.example.net"),
    ("https://llmctl.example.net/", "https://llmctl.example.net"),
    ("https://example.net/llmctl", "https://example.net/llmctl"),
])
def test_registry_urls(typed, url):
    assert cli._normalize_url(typed) == url


def test_gateway_restart_only_restarts_a_running_one(monkeypatch):
    monkeypatch.setattr(cli, "_gateway_record", lambda: None)
    started = []
    monkeypatch.setattr(cli, "_start_gateway", lambda port: started.append(port))
    out = CliRunner().invoke(cli.cli, ["gateway", "restart"])
    assert out.exit_code == 0 and "nothing to restart" in out.output and not started


# -- reaching local listeners on a single-address bind ------------------------

@pytest.mark.parametrize("bind, host, url", [
    ("0.0.0.0", "127.0.0.1", "http://127.0.0.1:9000/healthz"),
    ("::", "127.0.0.1", "http://127.0.0.1:9000/healthz"),
    ("localhost", "127.0.0.1", "http://127.0.0.1:9000/healthz"),
    ("127.0.0.1", "127.0.0.1", "http://127.0.0.1:9000/healthz"),
    ("172.19.0.1", "172.19.0.1", "http://172.19.0.1:9000/healthz"),
    ("fd00::1", "fd00::1", "http://[fd00::1]:9000/healthz"),
])
def test_local_host_follows_the_bind_address(cfg, bind, host, url):
    config_mod.set_bind_host(bind)
    assert config_mod.local_host() == host
    assert config_mod.local_url(9000, "/healthz") == url


@pytest.fixture
def listener():
    """An HTTP server on 127.0.0.2 only: 127.0.0.1 is refused, as a gateway
    bound to a Docker bridge address (172.19.0.1) refuses it."""
    import json
    import threading
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    class H(BaseHTTPRequestHandler):
        def do_GET(self):
            body = json.dumps({"status": "ok", "path": self.path}).encode()
            self.send_response(200)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *a):
            pass

    try:
        srv = ThreadingHTTPServer(("127.0.0.2", 0), H)
    except OSError:
        pytest.skip("127.0.0.2 isn't usable here")
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield srv.server_address[1]
    srv.shutdown()
    srv.server_close()


def test_probes_dial_the_bind_address(cfg, listener):
    from llm_runner import supervisor
    assert not supervisor.healthy(listener)               # 127.0.0.1: refused
    config_mod.set_bind_host("127.0.0.2")
    assert proc.port_in_use(listener)
    assert supervisor.healthy(listener)
    assert cli._gateway_get(listener, "/healthz") == {"status": "ok", "path": "/healthz"}


def test_a_decision_endpoint_follows_its_backend_to_the_new_address():
    cmd = ["systemone-proxy", "serve", "--backend", "http://127.0.0.1:8080",
           "--host", "0.0.0.0", "--port", "8180"]
    moved = backends.with_bind_host(cmd, backends.DECIDE, "172.19.0.1")
    assert moved[moved.index("--backend") + 1] == "http://172.19.0.1:8080"
    assert _host(moved) == "172.19.0.1"
    back = backends.with_bind_host(moved, backends.DECIDE, "0.0.0.0")
    assert back[back.index("--backend") + 1] == "http://127.0.0.1:8080"
    llama = ["llama-server", "--port", "8080", "--host", "0.0.0.0"]
    assert backends.with_bind_host(llama, backends.LLAMA, "172.19.0.1") == [
        "llama-server", "--port", "8080", "--host", "172.19.0.1"]


def test_decision_endpoints_are_found_on_any_address(monkeypatch):
    def rec(port, backend_url):
        return proc.RunningProcess(name=f"d{port}", pid=1, port=port, started_at=0.0,
                                   model_path="", log_file="", backend=backends.DECIDE,
                                   cmd=["systemone-proxy", "serve", "--backend", backend_url])
    recs = [rec(8180, "http://127.0.0.1:8080"), rec(8181, "http://172.19.0.1:8080"),
            rec(8182, "http://172.19.0.1:8081")]
    monkeypatch.setattr(proc, "list_running", lambda: recs)
    assert [p.name for p in cli._proxy_attached_to(8080)] == ["d8180", "d8181"]

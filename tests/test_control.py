"""Remote control: `llmctl … --node NAME` runs the command on that node's
gateway, terminal and all (llm_runner.control)."""
from __future__ import annotations

import json
import os
import socket
import struct
import threading

import pytest
from click.testing import CliRunner

from llm_runner import auth, cli, control
from llm_runner import config as config_mod
from llm_runner import gateway as gw
from llm_runner import process as proc
from llm_runner.cluster import TOKEN_HEADER
from llm_runner.config import Cluster


# ------------------------------------------------------------------ --node --

def test_split_node_anywhere_before_double_dash():
    assert control.split_node(["run", "--node", "orin", "-i"]) == ("orin", ["run", "-i"])
    assert control.split_node(["--node=orin", "ps"]) == ("orin", ["ps"])
    assert control.split_node(["run", "m", "-X", "--node x"]) == (None, ["run", "m", "-X",
                                                                         "--node x"])
    assert control.split_node(["run", "--", "--node", "x"])[0] is None


def test_cli_hands_the_command_to_the_node(monkeypatch):
    sent = []
    monkeypatch.setattr(control, "exec_remote", lambda node, argv: sent.append((node, argv)) or 7)
    r = CliRunner().invoke(cli.cli, ["ps", "--node", "orin", "--json"])
    assert sent == [("orin", ["ps", "--json"])] and r.exit_code == 7
    config_mod.update(node_name="rtx")
    r = CliRunner().invoke(cli.cli, ["--node", "rtx", "version"])     # this machine: local
    assert r.exit_code == 0 and "llmctl" in r.output and len(sent) == 1


# -------------------------------------------------------------------- what --

@pytest.mark.parametrize("argv,path", [
    (["run"], ("run",)), (["run", "gemma", "-i", "--ctx-size", "8192"], ("run",)),
    (["start", "gemma"], ("run",)), (["ls", "-t"], ("list",)), (["ps", "--json"], ("ps",)),
    (["decide", "quyet"], ("decide", "start")), (["decide", "spec", "q"], ("decide", "spec")),
    (["stop"], ("stop",)), (["chat"], ("chat",)), (["speedtest"], ("speedtest",)),
    (["download", "unsloth/x"], ("download",)), (["version"], ("version",)),
    (["--version"], ()), (["registry", "status"], ("registry", "status")),
])
def test_commands_that_manage_models_may_run(argv, path):
    assert control.check_argv(argv) == path


@pytest.mark.parametrize("argv,code", [
    (["config", "show"], "command_not_allowed"), (["gateway", "key", "add", "x"],
                                                  "command_not_allowed"),
    (["gateway"], "command_not_allowed"), (["setup", "vllm"], "command_not_allowed"),
    (["registry", "set", "http://x"], "command_not_allowed"),
    (["import", "/etc"], "command_not_allowed"),
    (["run", "gemma", "-X", "--threads 4"], "option_not_allowed"),
    (["run", "gemma", "--env", "PATH=/tmp"], "option_not_allowed"),
    (["run", "--ft", "/tmp/x.gguf"], "option_not_allowed"),
    (["run", "gemma", "--image", "evil:latest"], "option_not_allowed"),
    (["chat", "--out", "/home"], "option_not_allowed"),
    (["nosuch"], "invalid_command"), ([], "invalid_command"),
])
def test_set_up_commands_and_code_running_options_are_refused(argv, code):
    with pytest.raises(control.Refused) as e:
        control.check_argv(argv)
    assert e.value.code == code


def test_apply_files_come_along_without_args_or_env(tmp_path):
    clean = "models:\n  g:\n    model: gemma\n    port: 8081\n    ctx: 8192\n"
    out = control.stage_files(("apply",), ["apply", "my.yaml", "--dry-run"],
                              {"my.yaml": clean}, tmp_path / "w")
    assert out[0] == "apply" and out[1].endswith("0-my.yaml") and out[2] == "--dry-run"
    assert (tmp_path / "w" / "0-my.yaml").read_text() == clean
    for bad in ("    args: [--threads, '4']\n", "    env: {PATH: /tmp}\n",
                "    image: evil:latest\n"):
        with pytest.raises(control.Refused, match="can't be applied remotely"):
            control.stage_files(("apply",), ["apply", "f.yaml"], {"f.yaml": clean + bad},
                                tmp_path / "w2")
    with pytest.raises(control.Refused, match="Only `apply`"):
        control.stage_files(("run",), ["run"], {"f.yaml": clean}, tmp_path / "w3")


# --------------------------------------------------------------------- who --

class _H(dict):
    def get(self, k, default=None):
        return super().get(k, default)


def _who(headers=None, peer="192.168.8.232", **kw):
    args = dict(token="t0k", policy=auth.policy_from(config_mod.load()),
                nodes=[{"node": "rtx", "url": "http://192.168.8.232:9000"}], node="orin")
    return control.authorize(_H(headers or {}), peer, **{**args, **kw})


def _refused(code, **kw):
    with pytest.raises(control.Refused) as e:
        _who(**kw)
    assert e.value.code == code, e.value.message


def test_off_until_turned_on():
    _refused("control_off", peer="127.0.0.1")
    config_mod.update(remote_control=True)
    assert _who(peer="127.0.0.1") == "local"


def test_a_node_needs_the_token_the_allow_list_and_its_own_address():
    config_mod.update(remote_control=True)
    node = {control.NODE_HEADER: "rtx", TOKEN_HEADER: "t0k"}
    _refused("unauthorized", headers={**node, TOKEN_HEADER: "nope"})
    _refused("node_not_allowed", headers=node)
    config_mod.update(remote_control_from=["rtx"])
    assert _who(headers=node) == "node rtx"
    _refused("wrong_address", headers=node, peer="192.168.8.99")     # claims to be rtx


def test_keys_need_the_control_scope_and_the_lan_is_no_exemption():
    config_mod.update(remote_control=True)
    _refused("control_key_needed")                     # a LAN caller with nothing
    key = auth.new_key()
    config_mod.update(gateway_keys={"ops": auth.hash_key(key)})
    bearer = {"Authorization": f"Bearer {key}"}
    _refused("no_control_scope", headers=bearer)
    config_mod.update(gateway_control_keys=["ops"])
    assert _who(headers=bearer, peer="10.0.0.5") == "key ops"


# ------------------------------------------------------------------ frames --

def test_frames_split_across_reads():
    data = control.frame(control.OUT, b"hello") + control.frame(control.EXIT, struct.pack(">i", 3))
    r = control.Reader()
    got = r.feed(data[:4]) + r.feed(data[4:9]) + r.feed(data[9:])
    assert got == [(control.OUT, b"hello"), (control.EXIT, struct.pack(">i", 3))]


def _frames_until_exit(sock):
    r, out, err = control.Reader(), b"", b""
    while True:
        data = sock.recv(65536)
        assert data, "closed before the exit frame"
        for kind, payload in r.feed(data):
            if kind == control.OUT:
                out += payload
            elif kind == control.ERR:
                err += payload
            elif kind == control.EXIT:
                return struct.unpack(">i", payload)[0], out, err


@pytest.fixture
def child_env(monkeypatch):
    """The commands the node runs see this test's config and state dirs."""
    monkeypatch.setenv("LLM_RUNNER_CONFIG_DIR", str(config_mod.CONFIG_DIR))
    monkeypatch.setenv("LLM_RUNNER_STATE_DIR", str(proc.STATE_DIR))


@pytest.mark.parametrize("use_tty", [False, True])
def test_run_exec_relays_output_and_the_exit_code(child_env, use_tty):
    a, b = socket.socketpair()
    result = {}
    t = threading.Thread(target=lambda: result.update(code=control.run_exec(
        a, ["version"], who="local", use_tty=use_tty, size=(100, 30), term="xterm")))
    t.start()
    code, out, err = _frames_until_exit(b)
    t.join(10)
    assert code == 0 == result["code"] and b"llmctl" in out
    a.close()
    b.close()


def test_run_exec_keeps_stdout_and_stderr_apart_without_a_tty(child_env):
    a, b = socket.socketpair()
    threading.Thread(target=control.run_exec, args=(a, ["stop", "nothing"]),
                     kwargs=dict(who="node rtx", use_tty=False, size=None, term=None),
                     daemon=True).start()
    code, out, err = _frames_until_exit(b)
    assert code == 2 and b"No running" in err and out == b""


# --------------------------------------------------------------- end to end --

@pytest.fixture
def node(tmp_path, child_env):
    """A gateway on loopback, with remote control on."""
    config_mod.update(remote_control=True, node_name="orin")
    state = tmp_path / "running.json"
    state.write_text("[]")
    g = gw.Gateway(gw.LocalModels(state), Cluster("orin", 1, None, None, "t0k", True))
    httpd = gw.make_server(0, "127.0.0.1", g)
    threading.Thread(target=httpd.serve_forever, args=(0.05,), daemon=True).start()
    yield f"http://127.0.0.1:{httpd.server_address[1]}"
    httpd.shutdown()


def test_exec_remote_end_to_end(node, capfd):
    assert control.exec_remote(node, ["version"]) == 0
    out, err = capfd.readouterr()
    assert "llmctl" in out
    assert control.exec_remote(node, ["ps", "--json"]) == 0
    out, _ = capfd.readouterr()
    # (This gateway logs to the same stdout, being in the test's process.)
    assert json.loads(out[out.index("{"):out.rindex("}") + 1])["models"] == []
    assert control.exec_remote(node, ["config", "show"]) == 1
    _, err = capfd.readouterr()
    assert "can't be run remotely" in err


def test_exec_remote_refused_when_off(node, capfd):
    config_mod.update(remote_control=None)
    assert control.exec_remote(node, ["version"]) == 1
    assert "Remote control is off on orin" in capfd.readouterr().err


def test_events_say_who_asked(child_env, monkeypatch):
    monkeypatch.setenv("LLMCTL_REMOTE", "node rtx")
    proc.record_event("m", "stopped", "by stop")
    [line] = (proc.STATE_DIR / "events.jsonl").read_text().splitlines()
    assert json.loads(line)["remote"] == "node rtx"
    monkeypatch.delenv("LLMCTL_REMOTE")
    proc.record_event("m", "started", "by run")
    assert "remote" not in json.loads(
        (proc.STATE_DIR / "events.jsonl").read_text().splitlines()[1])


# ------------------------------------------------------------- the settings --

def test_gateway_control_commands_and_control_keys():
    r = CliRunner()
    assert "Remote control: off" in r.invoke(cli.cli, ["gateway", "control"]).output
    out = r.invoke(cli.cli, ["gateway", "control", "on"]).output
    assert "Remote control: on" in out
    r.invoke(cli.cli, ["gateway", "control", "allow", "rtx", "lab"])
    assert control.settings()["nodes"] == ["lab", "rtx"]
    r.invoke(cli.cli, ["gateway", "control", "deny", "lab"])
    assert control.settings()["nodes"] == ["rtx"]
    res = r.invoke(cli.cli, ["gateway", "key", "add", "ops", "--control"])
    assert res.exit_code == 0 and control.settings()["keys"] == ["ops"]
    assert "ops  (control)" in r.invoke(cli.cli, ["gateway", "key"]).output
    r.invoke(cli.cli, ["gateway", "key", "rm", "ops"])
    assert control.settings()["keys"] == []
    r.invoke(cli.cli, ["gateway", "control", "off"])
    assert not control.settings()["on"] and control.settings()["nodes"] == ["rtx"]
    assert os.environ.get(control.KEY_ENV) is None

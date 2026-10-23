"""The gateway's supervisor: crash detection, backoff, give-up, and states."""
from __future__ import annotations

import json

import pytest

from llm_runner import process as proc
from llm_runner import supervisor as sup


class World:
    """Fake processes, manifest and clock."""

    def __init__(self, monkeypatch, tmp_path):
        self.now = 1000.0
        self.live: dict[str, proc.RunningProcess] = {}
        self.specs: dict[str, proc.LaunchSpec] = {}
        self.healthy: set[str] = set()
        self.restored: list[str] = []
        self.next_pid = 100
        self.fail_restore = False
        monkeypatch.setattr(proc, "list_running", lambda: list(self.live.values()))
        monkeypatch.setattr(proc, "autostart_list", lambda: list(self.specs.values()))
        monkeypatch.setattr(proc, "port_in_use", lambda port: False)
        self.s = sup.Supervisor(tmp_path, health=self._health, restore=self._restore,
                                clock=lambda: self.now)

    def _health(self, port):
        return any(p.port == port and p.name in self.healthy for p in self.live.values())

    def _restore(self, names):
        out = []
        for n in names:
            self.restored.append(n)
            if self.fail_restore:
                out.append((self.specs[n], "error: boom"))
                continue
            self.add(n, healthy=False)
            out.append((self.specs[n], f"started pid={self.live[n].pid} port=1"))
        return out

    def add(self, name, healthy=True, restart="on-failure", restart_max=5):
        self.next_pid += 1
        self.live[name] = proc.RunningProcess(name=name, pid=self.next_pid, port=8000,
                                              started_at=0, model_path="m", log_file="l",
                                              backend="vllm")
        self.specs.setdefault(name, proc.LaunchSpec(name=name, cmd=["vllm"], port=8000,
                                                    model_path="m", backend="vllm",
                                                    restart=restart,
                                                    restart_max=restart_max))
        if healthy:
            self.healthy.add(name)

    def kill(self, name):
        self.live.pop(name)

    def run(self, seconds, step=1.0):
        end = self.now + seconds
        while self.now < end:
            self.s.tick()
            self.now += step


@pytest.fixture
def w(monkeypatch, tmp_path):
    return World(monkeypatch, tmp_path)


def test_backoff_doubles_and_caps():
    assert [sup.backoff(n) for n in range(7)] == [2, 4, 8, 16, 32, 60, 60]


def test_a_crash_after_ready_is_restarted_with_backoff(w, tmp_path):
    w.add("qwen")
    w.s.tick()
    assert w.s.state("qwen").state == sup.READY
    w.kill("qwen")
    w.s.tick()
    t = w.s.state("qwen")
    assert t.state == sup.RESTARTING and w.restored == []
    assert w.s.retry_after("qwen") == 32                 # 2 s backoff + 30
    w.run(2.5)
    assert w.restored == ["qwen"] and w.s.state("qwen").state == sup.STARTING
    w.healthy.add("qwen")
    w.s.tick()
    assert w.s.state("qwen").state == sup.READY and w.s.state("qwen").total_restarts == 1
    events = [json.loads(x)["event"] for x in (tmp_path / "events.jsonl").read_text().splitlines()]
    assert events == ["ready", "restarting", "restarted", "ready"]
    states = sup.read_states(tmp_path)
    assert states["qwen"]["state"] == "ready" and states["qwen"]["restarts"] == 1


def test_it_gives_up_after_restart_max(w):
    w.add("qwen", restart_max=2)
    w.s.tick()
    for _ in range(3):
        w.kill("qwen")
        w.run(70)
        w.healthy.add("qwen") if "qwen" in w.live else None
        w.s.tick()
    assert w.restored == ["qwen", "qwen"]
    assert w.s.state("qwen").state == sup.CRASHED
    assert w.s.retry_after("qwen") is None
    assert [t.name for t in w.s.down()] == ["qwen"]


def test_restart_never_marks_crashed_at_once(w):
    w.add("qwen", restart="never")
    w.s.tick()
    w.kill("qwen")
    w.run(10)
    assert w.restored == [] and w.s.state("qwen").state == sup.CRASHED


def test_a_stop_is_not_a_crash(w):
    w.add("qwen")
    w.s.tick()
    del w.specs["qwen"]                  # `llmctl stop` drops it from autostart first
    w.kill("qwen")
    w.run(10)
    assert w.restored == [] and w.s.state("qwen") is None


def test_a_failed_start_and_boot_entries_are_left_alone(w):
    w.add("oom", healthy=False)          # never got ready
    w.s.tick()
    w.kill("oom")
    w.specs["boot"] = proc.LaunchSpec(name="boot", cmd=["x"], port=1, model_path="m")
    w.run(10)
    assert w.restored == []


def test_a_failed_restart_backs_off_and_counts(w):
    w.add("qwen")
    w.s.tick()
    w.kill("qwen")
    w.fail_restore = True
    w.run(3)
    t = w.s.state("qwen")
    assert w.restored == ["qwen"] and t.state == sup.RESTARTING and t.last_error == "error: boom"
    w.run(3)
    assert w.restored == ["qwen"]        # next try after 4 s
    w.run(2)
    assert w.restored == ["qwen", "qwen"]


def test_ten_minutes_ready_resets_the_count(w):
    w.add("qwen", restart_max=2)
    w.s.tick()
    for _ in range(2):
        w.kill("qwen")
        w.run(70)
        w.healthy.add("qwen")
    w.run(sup.WINDOW + 5, step=5)
    w.kill("qwen")
    w.run(5)
    assert w.restored.count("qwen") == 3 and w.s.state("qwen").state != sup.CRASHED


def test_stop_removes_autostart_before_killing(monkeypatch, tmp_path):
    order = []
    rec = proc.RunningProcess(name="q", pid=999999, port=1, started_at=0, model_path="m",
                              log_file="l")
    monkeypatch.setattr(proc, "find", lambda ident: rec)
    monkeypatch.setattr(proc, "reconcile", list)
    monkeypatch.setattr(proc, "_save", lambda procs: None)
    monkeypatch.setattr(proc, "autostart_remove", lambda name: order.append("autostart"))
    monkeypatch.setattr(proc, "_signal_group", lambda pid, sig: order.append("signal"))
    proc.stop("q")
    assert order[0] == "autostart"


def test_old_manifests_load_and_policy_round_trips(monkeypatch, tmp_path):
    f = tmp_path / "autostart.json"
    monkeypatch.setattr(proc, "AUTOSTART_FILE", f)
    monkeypatch.setattr(proc, "STATE_DIR", tmp_path)
    f.write_text(json.dumps([{"name": "a", "cmd": ["x"], "port": 1, "model_path": "m",
                              "future_key": 1}]))
    [a] = proc.autostart_list()
    assert a.restart == "on-failure" and a.restart_max == 5
    proc.autostart_add(name="b", cmd=["x"], port=2, model_path="m", restart="never",
                       restart_max=3)
    b = proc.autostart_find("b")
    assert (b.restart, b.restart_max) == ("never", 3)


# ---------------------------------------------------------------- gateway --

def test_gateway_answers_503_for_a_down_model_and_lists_it(monkeypatch, tmp_path):
    import http.client
    import threading

    from llm_runner import gateway as gw
    state = tmp_path / "running.json"
    state.write_text("[]")
    specs = [proc.LaunchSpec(name="qwen", cmd=["vllm", "serve", "/m", "--port", "8101"],
                             port=8101, model_path="/m", backend="vllm", group="qwen3"),
             proc.LaunchSpec(name="boot", cmd=["x"], port=8102, model_path="/b",
                             backend="vllm")]
    monkeypatch.setattr(proc, "autostart_list", lambda: specs)
    g = gw.Gateway(gw.LocalModels(state))
    g.supervisor.tracked["qwen"] = sup.Tracked("qwen", state=sup.RESTARTING, seen_ready=True,
                                               total_restarts=2,
                                               next_at=g.supervisor.clock() + 10)
    httpd = gw.make_server(0, "127.0.0.1", g)
    threading.Thread(target=httpd.serve_forever, args=(0.05,), daemon=True).start()

    def call(method, path, body=None):
        c = http.client.HTTPConnection("127.0.0.1", httpd.server_address[1], timeout=5)
        c.request(method, path, body=json.dumps(body) if body else None,
                  headers={"Content-Type": "application/json"})
        r = c.getresponse()
        out = r.status, r.getheader("Retry-After"), json.loads(r.read() or b"null")
        c.close()
        return out
    try:
        _, _, models = call("GET", "/v1/models")
        [m] = models["data"]
        assert m["id"] == "qwen" and m["llmctl"]["status"] == "restarting"
        assert m["llmctl"]["restarts"] == 2
        status, retry, out = call("POST", "/v1/chat/completions", {"model": "qwen"})
        assert status == 503 and out["error"]["code"] == "model_unavailable"
        assert 35 <= int(retry) <= 40
        status, _, out = call("POST", "/v1/chat/completions", {"model": "boot"})
        assert status == 503 and "not running" in out["error"]["message"]
        assert call("POST", "/m/boot/v1/completions", {"x": 1})[0] == 503
        assert call("POST", "/v1/chat/completions", {"model": "other"})[0] == 404
        text = g.metrics.render()
        assert 'llmctl_instance_state{model="qwen",node="%s",backend="vllm",' \
               'state="restarting"} 1' % g.node in text
    finally:
        httpd.shutdown()

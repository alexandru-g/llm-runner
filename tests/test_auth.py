"""API keys on the gateway for callers from outside the local networks."""
import http.client
import json
import threading

import pytest
from click.testing import CliRunner

from llm_runner import auth, cli, config as config_mod
from llm_runner import gateway as gw

KEY = "llmctl-test-key"
ON = auth.Policy(mode=auth.ON, keys={"laptop": auth.hash_key(KEY)})


def _caller(policy, peer, **headers):
    return str(policy.caller(peer, {k.replace("_", "-"): v for k, v in headers.items()}))


def test_off_lets_everyone_in():
    assert not auth.Policy().needs_key("203.0.113.5", {})


def test_local_callers_need_no_key():
    for peer in ("127.0.0.1", "::1", "192.168.8.50", "172.20.0.38", "100.101.1.2",
                 "::ffff:192.168.8.50", "fd7a:115c:a1e0::1"):
        assert not ON.needs_key(peer, {}), peer
    assert ON.needs_key("203.0.113.5", {})
    assert ON.needs_key("2a02:2f04:d808:3200::5", {})


def test_a_proxy_is_judged_by_the_address_it_appended():
    # nginx appends the real client to whatever the client sent.
    assert _caller(ON, "172.20.0.45", X_Forwarded_For="203.0.113.9") == "203.0.113.9"
    assert _caller(ON, "172.20.0.45",
                   X_Forwarded_For="192.168.1.5, 203.0.113.9") == "203.0.113.9"
    assert ON.needs_key("172.20.0.45", {"X-Forwarded-For": "192.168.1.5, 203.0.113.9"})
    assert _caller(ON, "172.20.0.45", X_Real_IP="203.0.113.9") == "203.0.113.9"
    # A LAN client through the proxy is still local.
    assert not ON.needs_key("172.20.0.45", {"X-Forwarded-For": "192.168.8.60"})
    # Garbage forwarded: unknown, so outside.
    assert ON.needs_key("172.20.0.45", {"X-Forwarded-For": "unknown"})


def test_an_outside_peer_cannot_claim_to_be_local():
    assert ON.needs_key("203.0.113.5", {"X-Forwarded-For": "192.168.1.5"})
    assert ON.needs_key("203.0.113.5", {"X-Real-IP": "127.0.0.1"})


def test_a_listed_proxy_without_a_forwarded_address_counts_as_outside():
    p = auth.Policy(mode=auth.ON, proxies=(auth.parse_network("172.20.0.45"),))
    assert p.needs_key("172.20.0.45", {})
    assert not p.needs_key("172.20.0.38", {})              # another container
    assert not p.needs_key("172.20.0.45", {"X-Forwarded-For": "10.0.0.7"})


def test_via_proxy():
    assert ON.via_proxy("172.20.0.45", {"X-Forwarded-For": "192.168.8.247"})
    assert ON.via_proxy("172.20.0.45", {"X-Real-IP": "192.168.8.247"})
    assert not ON.via_proxy("192.168.8.247", {})
    # An outside peer's headers prove nothing; a listed proxy is one without them.
    assert not ON.via_proxy("203.0.113.5", {"X-Forwarded-For": "192.168.1.5"})
    p = auth.Policy(proxies=(auth.parse_network("172.20.0.45"),))
    assert p.via_proxy("172.20.0.45", {}) and not p.via_proxy("172.20.0.38", {})


def test_narrowing_local_to_loopback_wants_keys_from_the_lan():
    p = auth.Policy(mode=auth.ON, local=(auth.parse_network("127.0.0.0/8"),))
    assert p.needs_key("192.168.8.50", {})
    assert not p.needs_key("127.0.0.1", {}) and not p.needs_key("::1", {})


def test_keys_by_bearer_or_x_api_key():
    assert ON.key_name({"Authorization": f"Bearer {KEY}"}) == "laptop"
    assert ON.key_name({"Authorization": f"bearer {KEY}"}) == "laptop"
    assert ON.key_name({"X-Api-Key": KEY}) == "laptop"
    assert ON.key_name({"Authorization": "Bearer nope"}) is None
    assert ON.key_name({}) is None


@pytest.fixture
def gateway(tmp_path):
    (tmp_path / "running.json").write_text("[]")
    policy = {"p": ON}
    httpd = gw.make_server(0, "127.0.0.1", gw.Gateway(gw.LocalModels(tmp_path / "running.json"),
                                                      access=lambda: policy["p"]))
    threading.Thread(target=httpd.serve_forever, args=(0.05,), daemon=True).start()
    yield httpd.server_address[1], policy
    httpd.shutdown()


def _get(port, path, **headers):
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
    conn.request("GET", path, headers=headers)
    resp = conn.getresponse()
    body = json.loads(resp.read() or b"null")
    conn.close()
    return resp.status, body


def test_gateway_wants_a_key_from_outside(gateway):
    port, policy = gateway
    outside = {"X-Forwarded-For": "203.0.113.9"}
    status, body = _get(port, "/v1/models", **outside)
    assert status == 401 and body["error"]["code"] == "invalid_api_key"
    assert _get(port, "/v1/models", **outside, Authorization="Bearer nope")[0] == 401
    assert _get(port, "/v1/models", **outside, Authorization=f"Bearer {KEY}")[0] == 200
    assert _get(port, "/cluster/v1/status", **outside)[0] == 401     # lists LAN IPs
    for path in ("/health", "/healthz", "/readyz"):
        assert _get(port, path, **outside)[0] in (200, 503), path
    # Local callers, and anyone with auth off, need nothing.
    assert _get(port, "/v1/models")[0] == 200
    policy["p"] = auth.Policy()
    assert _get(port, "/v1/models", **outside)[0] == 200


def test_cli_keys_and_settings():
    runner = CliRunner()
    out = runner.invoke(cli.cli, ["gateway", "key", "add", "laptop"])
    assert out.exit_code == 0, out.output
    key = out.stdout.strip().splitlines()[0]
    assert key.startswith("llmctl-")
    stored = config_mod.load()["gateway_keys"]["laptop"]
    assert stored == auth.hash_key(key) and key not in json.dumps(config_mod.load())
    assert runner.invoke(cli.cli, ["gateway", "key", "add", "laptop"]).exit_code != 0

    out = runner.invoke(cli.cli, ["gateway", "auth", "on", "--proxy", "172.20.0.45"])
    assert out.exit_code == 0, out.output
    policy = auth.ConfigPolicy()()
    assert policy.mode == auth.ON and policy.needs_key("172.20.0.45", {})
    assert policy.key_name({"Authorization": f"Bearer {key}"}) == "laptop"
    assert runner.invoke(cli.cli, ["gateway", "auth", "--local", "nope"]).exit_code != 0

    assert runner.invoke(cli.cli, ["gateway", "key", "rm", "laptop"]).exit_code == 0
    assert "gateway_keys" not in config_mod.load()
    assert runner.invoke(cli.cli, ["gateway", "auth", "off", "--no-proxy"]).exit_code == 0
    assert auth.ConfigPolicy()().mode == auth.OFF


def test_config_show_hides_secrets():
    config_mod.update(cluster_token="tok-secret", gateway_upstream_key="llmctl-secret",
                      gateway_keys={"laptop": auth.hash_key("k")})
    out = CliRunner().invoke(cli.cli, ["config", "show"]).output
    assert "tok-secret" not in out and "llmctl-secret" not in out
    assert auth.hash_key("k") not in out and "gateway_keys = laptop" in out

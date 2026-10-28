"""`decide:` in serving files: a model's decision endpoint, applied with it."""
from __future__ import annotations

import json

import pytest
from click.testing import CliRunner

from llm_runner import apply as apply_mod
from llm_runner import backends
from llm_runner import cli
from llm_runner import process as proc
from llm_runner import registry

from test_apply import _write, host  # noqa: F401  (fixture)

H2O = """\
models:
  h2o:
    model: h2o-q8_0
    port: 8102
    group: decider
    n_parallel: 4
    decide:
      port: 8192
      profile: h2o-lightning-4b
"""


@pytest.fixture
def gguf(host, monkeypatch):  # noqa: F811
    folder = registry.MODELS_DIR / "h2o"
    folder.mkdir()
    (folder / "h2o-q8_0.gguf").write_bytes(b"GGUF")
    real = backends.find
    monkeypatch.setattr(backends, "find", lambda b: "/usr/bin/systemone-proxy"
                        if b == backends.DECIDE else real(b))
    return host


def _apply(path, *extra):
    r = CliRunner().invoke(cli.cli, ["apply", str(path), "--no-wait", "--json", *extra])
    return r, (json.loads(r.stdout) if r.stdout.strip().startswith("{") else None)


def _actions(out):
    return [(a["name"], a["action"]) for a in out["actions"]]


def test_the_endpoint_starts_with_its_model_and_then_stays(tmp_path, gguf):
    f = _write(tmp_path, "serve.yaml", H2O)
    r, out = _apply(f)
    assert r.exit_code == 0, r.output
    assert _actions(out) == [("h2o", "start"), ("h2o.decide", "start")]
    assert out["actions"][1]["run"] == ["llmctl", "decide", "start", "h2o", "--port", "8192",
                                        "--profile", "h2o-lightning-4b"]
    d = proc.find("h2o.decide")
    assert d.port == 8192 and d.backend == backends.DECIDE
    assert d.cmd[d.cmd.index("--backend") + 1].endswith(":8102")
    assert d.cmd[d.cmd.index("--concurrency") + 1] == "4"            # the model's slots
    assert d.group == "decider.decide"
    r, out = _apply(f)
    assert _actions(out) == [("h2o", "unchanged"), ("h2o.decide", "unchanged")]


def test_a_changed_endpoint_or_model_restarts_it(tmp_path, gguf):
    f = _write(tmp_path, "serve.yaml", H2O)
    assert _apply(f)[0].exit_code == 0
    g = _write(tmp_path, "serve.yaml", H2O.replace("      profile:", "      cache: request\n"
                                                                    "      profile:"))
    r, out = _apply(g)
    assert r.exit_code == 0, r.output
    assert _actions(out) == [("h2o", "unchanged"), ("h2o.decide", "restart")]
    assert out["actions"][1]["reason"] == "cache off -> request"
    assert proc.find("h2o.decide").cmd[-2:] == ["--cache", "request"]
    g = _write(tmp_path, "serve.yaml", H2O.replace("n_parallel: 4", "n_parallel: 2")
               .replace("      profile:", "      cache: request\n      profile:"))
    r, out = _apply(g)
    assert _actions(out) == [("h2o", "restart"), ("h2o.decide", "restart")]
    assert out["actions"][1]["reason"] == "h2o: restart"


def test_prune_takes_a_pruned_models_endpoint_too(tmp_path, gguf):
    f = _write(tmp_path, "serve.yaml", H2O)
    assert _apply(f)[0].exit_code == 0
    empty = _write(tmp_path, "empty.yaml", "models: {}\n")
    r, out = _apply(empty, "--prune")
    assert r.exit_code == 0, r.output
    assert _actions(out) == [("h2o", "stop"), ("h2o.decide", "stop")]
    assert not proc.list_running()


def test_bad_decide_blocks(tmp_path, gguf):
    for text, why in (
            ("      port: 8192\n      profile: x\n      flavour: y\n", "not a known setting"),
            ("      profile: x\n", "decide.port is required"),
            ("      port: 8102\n", "also used"),
            ("      port: 8192\n      cache: always\n", "off or request")):
        f = _write(tmp_path, "bad.yaml", H2O.split("    decide:\n")[0] + "    decide:\n" + text)
        with pytest.raises(apply_mod.ApplyError) as e:
            apply_mod.load([f])
        assert why in "\n".join(e.value.problems), text


def test_only_llama_models_take_an_endpoint(tmp_path, gguf):
    f = _write(tmp_path, "serve.yaml", "models:\n  q:\n    model: qwen3-v1\n    port: 8101\n"
                                       "    decide:\n      port: 8192\n")
    r, out = _apply(f)
    assert r.exit_code == 3 and "needs a llama-server model" in r.output

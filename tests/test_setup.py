"""`llmctl setup`: per-backend status, and the plans that install engines
(uv tools, venvs) and images, without installing anything."""
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from click.testing import CliRunner

from llm_runner import backends, cli, config as config_mod, containers, engines, process as proc


def _venv(root: Path, *dists: str, python="3.12") -> Path:
    """A venv with ``name==version`` dist-info folders and a bin/python."""
    site = root / "lib" / f"python{python}" / "site-packages"
    site.mkdir(parents=True)
    for d in dists:
        name, version = d.split("==")
        (site / f"{name.replace('-', '_')}-{version}.dist-info").mkdir()
    (root / "bin").mkdir()
    (root / "bin" / "python").write_text("")
    return root


@pytest.fixture
def host(tmp_path, monkeypatch):
    """A home with uv and no engines; cfg and state in tmp."""
    home = tmp_path / "home"
    (home / ".local" / "bin").mkdir(parents=True)
    (home / ".local" / "bin" / "uv").write_text("")
    monkeypatch.setattr(Path, "home", lambda: home)
    monkeypatch.setattr(config_mod, "CONFIG_DIR", tmp_path / "cfg", raising=False)
    monkeypatch.setattr(config_mod, "load", lambda: dict(_cfg))
    monkeypatch.setattr(config_mod, "update", lambda **kw: _cfg.update(kw))
    monkeypatch.setattr(proc, "STATE_DIR", tmp_path / "state")
    found: dict[str, str] = {}
    monkeypatch.setattr(backends, "find_engine", lambda b: found.get(b))
    monkeypatch.setattr(backends, "find_llama_server", lambda: None)
    monkeypatch.setattr(backends, "find_decider_dir", lambda: None)
    monkeypatch.setattr(backends, "is_jetson", lambda: False)
    monkeypatch.setattr(engines, "backends_gpu", lambda: True)
    monkeypatch.setattr(containers, "docker_bin", lambda: "/usr/bin/docker")
    monkeypatch.setattr(cli.images, "local_image", lambda ref, state: None)
    _cfg.clear()
    return SimpleNamespace(home=home, found=found, uv=str(home / ".local" / "bin" / "uv"),
                           cfg=_cfg)


_cfg: dict = {}


def test_versions_come_from_dist_info(tmp_path):
    v = _venv(tmp_path / "v", "onnxruntime_gpu==1.31.1", "vllm-omni==0.30.0")
    assert engines.dist_version(v, "onnxruntime-gpu") == "1.31.1"
    assert engines.dist_version(v, "vllm_omni") == "0.30.0"
    assert engines.dist_version(v, "onnxruntime") is None


def test_status_rows(host):
    tool = _venv(host.home / ".local/share/uv/tools/vllm", "vllm==0.29.0")
    host.found["vllm"] = str(tool / "bin" / "vllm")
    sg = _venv(host.home / "sg", "sglang==0.5.21")
    host.found["sglang"] = str(sg / "bin" / "sglang")
    ort = _venv(host.home / ".venvs/onnxruntime", "onnxruntime==1.31.1")
    host.found["onnx"] = str(ort / "bin" / "python")
    rows = {r.backend: r for r in engines.status(proc.STATE_DIR, {})}
    assert [r for r in rows] == list(engines.SETUP_BACKENDS)
    assert (rows["vllm"].state, rows["vllm"].version) == ("ok", "0.29.0")
    assert rows["sglang"].state == "differs" and "--upgrade" in rows["sglang"].note
    assert rows["omni"].state == "missing" and rows["omni"].note == "llmctl setup omni"
    assert rows["onnx"].state == "ok" and rows["onnx"].note == "onnxruntime"
    assert rows["llama"].state == "missing" and rows["decider"].state == "missing"
    assert rows["triton"].state == "missing" and rows["triton"].note == "llmctl setup triton"


def test_plan_a_uv_tool(host):
    [step] = engines.plan_engine("vllm")
    assert step.argv == [host.uv, "tool", "install", "--python", "3.12", "--torch-backend",
                         "auto", "vllm==0.29.0", "--with", "transformers==5.16.1"]
    sg = engines.plan_engine("sglang", find_links=Path("/w"))[0].argv
    assert sg[sg.index("--find-links") - 2:sg.index("--find-links") + 2] == [
        "--offline", "--no-index", "--find-links", "/w"]
    assert sg.count("--with") == 2


def test_an_installed_engine_is_kept(host):
    tool = _venv(host.home / "t", "vllm==0.28.1")
    host.found["vllm"] = str(tool / "bin" / "vllm")
    with pytest.raises(engines.SetupError, match="0.28.1 is installed.*--upgrade"):
        engines.plan_engine("vllm")
    assert engines.plan_engine("vllm", upgrade=True)[0].argv[3] == "--python"
    ok = _venv(host.home / "o", "vllm==0.29.0")
    host.found["vllm"] = str(ok / "bin" / "vllm")
    assert engines.plan_engine("vllm") == []


def test_plan_a_venv(host):
    steps = engines.plan_engine("omni")
    venv = host.home / ".venvs" / "vllm-omni"
    assert [s.argv for s in steps] == [
        [host.uv, "venv", "--python", "3.12", str(venv)],
        [host.uv, "pip", "install", "-p", str(venv / "bin/python"), "--torch-backend", "auto",
         "vllm==0.30.0", "vllm-omni==0.30.0"]]


@pytest.mark.parametrize("kw, packages, extra", [
    ({"gpu": True, "jetson": False}, ["onnxruntime-gpu==1.31.1", "numpy"], []),
    ({"gpu": False, "jetson": False}, ["onnxruntime==1.31.1", "numpy"], []),
    ({"gpu": True, "jetson": False, "cpu": True}, ["onnxruntime==1.31.1", "numpy"], []),
    ({"gpu": True, "jetson": True}, ["onnxruntime==1.31.1", "numpy"], []),
    ({"gpu": True, "jetson": True, "index_url": "https://idx/jp6"},
     ["onnxruntime-gpu", "numpy"], ["--extra-index-url", "https://idx/jp6"]),
])
def test_plan_onnx_picks_its_build(host, kw, packages, extra):
    install = engines.plan_engine("onnx", **kw)[-1].argv
    assert install[-len(packages):] == packages
    assert all(a in install for a in extra) and "--torch-backend" not in install


def test_switching_onnx_builds_removes_the_other(host):
    venv = _venv(host.home / ".venvs" / "onnxruntime", "onnxruntime==1.31.1")
    host.found["onnx"] = str(venv / "bin" / "python")
    steps = engines.plan_engine("onnx", upgrade=True, gpu=True, jetson=False)
    assert [s.argv[1:3] for s in steps] == [["pip", "uninstall"], ["pip", "install"]]
    assert steps[0].argv[-1] == "onnxruntime"


def test_no_uv_no_plan(host, monkeypatch):
    (host.home / ".local" / "bin" / "uv").unlink()
    monkeypatch.setattr(engines.shutil, "which", lambda name: None)
    with pytest.raises(engines.SetupError, match="uv not found"):
        engines.plan_engine("vllm")


def test_plan_an_image(host, monkeypatch):
    ref = containers.default_image("triton")
    assert [s.argv for s in engines.plan_image("triton", proc.STATE_DIR, {}, set_runtime=False)] \
        == [["/usr/bin/docker", "pull", ref]]
    steps = engines.plan_image("vllm", proc.STATE_DIR, {})
    assert steps[-1].config == {"runtime": {"vllm": {
        "kind": "docker", "image": containers.default_image("vllm")}}}
    monkeypatch.setattr(cli.images, "local_image", lambda r, s: (r, "sha256:x"))
    assert engines.plan_image("triton", proc.STATE_DIR, {}, set_runtime=False) == []
    cfg = {"runtime": {"vllm": {"kind": "docker", "image": containers.default_image("vllm")}}}
    assert engines.plan_image("vllm", proc.STATE_DIR, cfg) == []
    arc = engines.plan_image("llama", proc.STATE_DIR, {}, archive=Path("/a/l.tar"))
    assert arc[0].archive == Path("/a/l.tar") and arc[1].config


def test_a_jetson_takes_the_igpu_triton(host, monkeypatch):
    monkeypatch.setattr(backends, "is_jetson", lambda: True)
    igpu = containers.expected_images()["triton-igpu"]["image"]
    assert containers.default_image("triton") == igpu
    assert containers.default_image("vllm") == containers.expected_images()["vllm"]["image"]
    assert engines.plan_image("triton", proc.STATE_DIR, {}, set_runtime=False)[0].argv[-1] == igpu


# -------------------------------------------------------------------- CLI --

def test_setup_lists_backends(host):
    out = CliRunner().invoke(cli.cli, ["setup"])
    assert out.exit_code == 0, out.output
    lines = out.output.splitlines()
    assert lines[0].split() == ["BACKEND", "STATE", "VERSION", "TESTED", "WHERE"]
    assert [ln.split()[0] for ln in lines[1:]] == list(engines.SETUP_BACKENDS)


def test_setup_dry_run_runs_nothing(host, monkeypatch):
    monkeypatch.setattr(cli.subprocess, "run", lambda *a, **k: pytest.fail("ran"))
    out = CliRunner().invoke(cli.cli, ["setup", "omni", "--dry-run"])
    assert out.exit_code == 0, out.output
    assert "1. create the venv" in out.output and "vllm-omni==0.30.0" in out.output


def test_setup_runs_the_plan_and_checks(host, monkeypatch):
    ran = []

    def run(argv, **kw):
        ran.append(argv)
        if argv[1:3] == ["tool", "install"]:
            _venv(host.home / ".local/share/uv/tools/vllm", "vllm==0.29.0")
            host.found["vllm"] = str(host.home / ".local/share/uv/tools/vllm/bin/vllm")
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr(cli.subprocess, "run", run)
    monkeypatch.setattr(cli, "_require_online", lambda what: None)
    out = CliRunner().invoke(cli.cli, ["setup", "vllm", "--yes"])
    assert out.exit_code == 0, out.output
    assert ran[0][:3] == [host.uv, "tool", "install"] and "vllm: ok, 0.29.0" in out.output
    again = CliRunner().invoke(cli.cli, ["setup", "vllm"])
    assert "already set up" in again.output


def test_setup_runtime_docker_sets_the_runtime(host, monkeypatch):
    monkeypatch.setattr(cli, "_require_online", lambda what: None)
    monkeypatch.setattr(cli.subprocess, "run", lambda argv, **kw: SimpleNamespace(returncode=0))
    pulled = {"done": False}
    monkeypatch.setattr(cli.images, "local_image",
                        lambda ref, state: (ref, "sha256:1") if pulled["done"] else None)
    real = engines.plan_image

    def plan(*a, **kw):
        steps = real(*a, **kw)
        pulled["done"] = True
        return steps
    monkeypatch.setattr(engines, "plan_image", plan)
    out = CliRunner().invoke(cli.cli, ["setup", "llama", "--runtime", "docker", "--yes"])
    assert out.exit_code == 0, out.output
    assert host.cfg["runtime"]["llama"]["kind"] == "docker"
    assert "llama: ok" in out.output


def test_setup_llama_and_decider_are_checked_only(host):
    out = CliRunner().invoke(cli.cli, ["setup", "llama"])
    assert out.exit_code == 0 and "doesn't install it" in out.output
    out = CliRunner().invoke(cli.cli, ["setup", "triton", "--runtime", "host"])
    assert out.exit_code != 0 and "container only" in out.output
    out = CliRunner().invoke(cli.cli, ["setup", "onnx", "--runtime", "docker"])
    assert out.exit_code != 0 and "no container image" in out.output


def test_engines_json_pins_every_installable_engine():
    pins = engines.pins()
    assert set(pins) == set(engines.INSTALLABLE)
    for name, p in pins.items():
        assert p["kind"] in ("uv-tool", "venv") and p["version"], name
        assert any(p["version"] in pkg for pkg in p["packages"]), name
    assert json.loads(Path(engines.__file__).with_name("engines.json").read_text())[
        "schema"] == "llmctl-engines/1"

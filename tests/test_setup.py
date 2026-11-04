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


# ------------------------------------------------------- Jetson Orin, JetPack 6 --

def _deb() -> dict:
    return engines._manifest()["orin_compat"]


@pytest.fixture
def orin(host, monkeypatch):
    monkeypatch.setattr(backends, "is_jetson", lambda: True)
    monkeypatch.setattr(backends, "is_orin_jetpack6", lambda: True)
    monkeypatch.setattr(backends, "host_compiler",
                        lambda max_major: ("/usr/bin/gcc", "/usr/bin/g++"))
    monkeypatch.delenv("XDG_DATA_HOME", raising=False)
    return host


def _unpacked(host) -> Path:
    d = host.home / ".local/share/llm-runner/cuda-compat-orin/usr/local/cuda-13.5/compat_orin"
    d.mkdir(parents=True)
    return d


def test_orin_takes_its_own_versions_and_no_torch_backend(orin):
    steps = engines.plan_engine("vllm", gpu=True)
    assert "CUDA 13.5" in steps[0].what and steps[0].call is not None
    argv = steps[1].argv
    assert "vllm==0.33.0" in argv and "--torch-backend" not in argv and "--with" not in argv
    argv = engines.plan_engine("omni")[-1].argv
    assert "vllm==0.32.0" in argv and "vllm-omni==0.32.0" in argv
    assert "--torch-backend" not in argv


def test_other_hosts_keep_their_pins(host):
    argv = engines.plan_engine("vllm")[0].argv
    assert "vllm==0.29.0" in argv and "--torch-backend" in argv
    assert not any(s.call for s in engines.plan_engine("omni"))


def test_sglang_and_onnx_get_no_compat_on_an_orin(orin):
    assert not any(s.call for s in engines.plan_engine("sglang"))
    assert not any(s.call for s in engines.plan_engine("onnx", gpu=True, jetson=True))


def test_orin_engine_in_place_but_no_compat_only_adds_it(orin):
    orin.found["vllm"] = str(_venv(orin.home / ".local/share/uv/tools/vllm",
                                   "vllm==0.33.0") / "bin" / "vllm")
    row = engines._engine_status("vllm")
    assert row.state == "configure" and row.tested == "0.33.0"
    steps = engines.plan_engine("vllm")
    assert len(steps) == 1 and steps[0].call is not None
    _unpacked(orin)
    assert engines._engine_status("vllm").state == "ok"
    assert engines.plan_engine("vllm") == []


def test_orin_without_a_compiler_is_told_before_installing(orin, monkeypatch):
    monkeypatch.setattr(backends, "host_compiler", lambda max_major: None)
    with pytest.raises(engines.SetupError, match="build-essential"):
        engines.plan_engine("vllm")


def _fake_dpkg(monkeypatch):
    def run(argv, **kw):
        assert argv[:2] == ["dpkg-deb", "-x"]
        (Path(argv[3]) / "usr/local/cuda-13.5/compat_orin").mkdir(parents=True)
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr(engines.subprocess, "run", run)


def test_compat_is_taken_offline_checked_and_unpacked(orin, tmp_path, monkeypatch):
    import hashlib
    links = tmp_path / "links"
    links.mkdir()
    (links / _deb()["file"]).write_bytes(b"package")
    monkeypatch.setitem(_deb(), "sha256", hashlib.sha256(b"package").hexdigest())
    _fake_dpkg(monkeypatch)
    engines.install_orin_compat(links)
    assert backends.orin_compat_dir().name == "compat_orin"
    assert [p.name for p in backends.orin_compat_root().parent.iterdir()] == ["cuda-compat-orin"]


def test_a_compat_package_with_the_wrong_checksum_is_refused(orin, tmp_path, monkeypatch):
    links = tmp_path / "links"
    links.mkdir()
    (links / _deb()["file"]).write_bytes(b"tampered")
    _fake_dpkg(monkeypatch)
    with pytest.raises(engines.SetupError, match="SHA256"):
        engines.install_orin_compat(links)
    assert backends.orin_compat_dir() is None


def test_compat_missing_from_the_offline_folder_says_so(orin, tmp_path):
    with pytest.raises(engines.SetupError, match="can't get"):
        engines.install_orin_compat(tmp_path)


def test_launch_env_on_an_orin_only_touches_vllm_and_omni(orin, monkeypatch):
    compat = _unpacked(orin)
    monkeypatch.setenv("LD_LIBRARY_PATH", "/inherited")
    assert backends.launch_env(False, "vllm")["LD_LIBRARY_PATH"] == f"{compat}:/inherited"
    assert backends.launch_env(False, "omni")["LD_LIBRARY_PATH"] == f"{compat}:/inherited"
    assert "LD_LIBRARY_PATH" not in backends.launch_env(False, "llama")
    assert "LD_LIBRARY_PATH" not in backends.launch_env(False, "onnx")


def test_launch_env_elsewhere_has_no_compat(host):
    _unpacked(host)
    assert "LD_LIBRARY_PATH" not in backends.launch_env(False, "vllm")


def test_a_cache_vllm_cannot_write_gets_replaced(host, monkeypatch):
    monkeypatch.delenv("XDG_CACHE_HOME", raising=False)
    assert "VLLM_CACHE_ROOT" not in backends.launch_env(False, "vllm")
    cache = host.home / ".cache" / "vllm"
    cache.mkdir(parents=True)
    assert "VLLM_CACHE_ROOT" not in backends.launch_env(False, "vllm")
    monkeypatch.setattr(backends.os, "access", lambda p, mode: False)
    want = str(host.home / ".cache" / "llm-runner" / "vllm")
    assert backends.launch_env(False, "vllm")["VLLM_CACHE_ROOT"] == want
    assert backends.launch_env(False, "omni")["VLLM_CACHE_ROOT"] == want
    assert "VLLM_CACHE_ROOT" not in backends.launch_env(False, "llama")


def test_jetson_memory_comes_from_meminfo_when_smi_has_none(monkeypatch):
    monkeypatch.setattr(backends, "is_jetson", lambda: True)
    monkeypatch.setattr(backends, "_system_memory_mib", lambda: (50000, 62828))

    def smi(*a, **k):
        return SimpleNamespace(stdout="[N/A], [N/A]\n")

    monkeypatch.setattr("subprocess.run", smi)
    assert backends.gpu_memory_mib() == (50000, 62828)
    # (50000 - 1536 - 6144) / 62828: the system shares this memory.
    assert backends.default_gpu_mem("vllm")[0] == 0.67


def test_system_memory_parses_meminfo():
    free, total = backends._system_memory_mib()
    assert 0 < free <= total


TAG = "llmctl/vllm-orin:0.33.0"
BUILT = "sha256:" + "c" * 64


def test_orin_builds_its_image_and_pins_it_by_id(orin, monkeypatch):
    present: dict[str, dict] = {}
    monkeypatch.setattr(containers, "inspect_image", lambda ref: present.get(ref))
    monkeypatch.setattr(cli.images, "local_image", lambda ref, state: (ref, ref))
    steps = engines.plan_image("vllm", proc.STATE_DIR, {})
    assert len(steps) == 2 and "build" in steps[0].what
    argv = steps[0].argv
    assert argv[:4] == ["/usr/bin/docker", "build", "-t", TAG]
    assert "VLLM_VERSION=0.33.0" in argv and f"COMPAT_SHA256={_deb()['sha256']}" in argv
    assert (Path(argv[-1]) / "Dockerfile").is_file()

    present[TAG] = {"Id": BUILT}
    steps[1].call()
    ref = f"llmctl/vllm-orin@{BUILT}"
    assert orin.cfg["runtime"]["vllm"] == {"kind": "docker", "image": ref}
    assert containers.is_pinned(ref)
    assert cli.images.loaded(proc.STATE_DIR)[ref]["image_id"] == BUILT
    assert engines.plan_image("vllm", proc.STATE_DIR, orin.cfg) == []
    row = engines.image_status("vllm", proc.STATE_DIR, orin.cfg["runtime"]["vllm"])
    assert (row.state, row.version, row.where) == ("ok", "0.33.0", TAG)


def test_orin_image_present_is_not_rebuilt_unless_upgrading(orin, monkeypatch):
    monkeypatch.setattr(containers, "inspect_image", lambda ref: {"Id": BUILT})
    monkeypatch.setattr(cli.images, "local_image", lambda ref, state: (ref, ref))
    steps = engines.plan_image("vllm", proc.STATE_DIR, {})
    assert len(steps) == 1 and steps[0].call is not None
    assert "build" in engines.plan_image("vllm", proc.STATE_DIR, {}, upgrade=True)[0].argv


def test_orin_has_no_published_vllm_image(orin, monkeypatch):
    assert containers.default_image("vllm") is None
    assert containers.default_image("llama") is not None
    with pytest.raises(engines.SetupError, match="no archive"):
        engines.plan_image("vllm", proc.STATE_DIR, {}, archive=Path("x.tar"))
    row = engines.image_status("vllm", proc.STATE_DIR, {})
    assert row.state == "missing" and "--runtime docker" in row.note


def test_setup_on_an_orin_ends_with_a_gpu_check(orin, monkeypatch):
    tool = _venv(orin.home / ".local/share/uv/tools/vllm", "vllm==0.33.0")
    orin.found["vllm"] = str(tool / "bin" / "vllm")
    (tool / "bin" / "vllm").write_text("")
    _unpacked(orin)
    calls = []
    monkeypatch.setattr(engines, "plan_engine",
                        lambda *a, **k: [engines.Step("x", call=lambda: None)])

    def run(argv, **kw):
        calls.append((argv, kw.get("env")))
        return SimpleNamespace(returncode=0, stdout="2.14.0+cu130 Orin 64\n", stderr="")

    monkeypatch.setattr(engines.subprocess, "run", run)
    monkeypatch.setattr(engines, "status", lambda *a: [engines.Status("vllm", "ok")])
    r = CliRunner().invoke(cli.cli, ["setup", "vllm", "-y"])
    assert r.exit_code == 0, r.output
    assert "GPU check passed: 2.14.0+cu130 Orin 64" in r.output
    assert calls[0][0][0] == str(tool / "bin" / "python")
    assert "compat_orin" in calls[0][1]["LD_LIBRARY_PATH"]


def test_a_failed_gpu_check_fails_setup(orin, monkeypatch):
    orin.found["omni"] = str(_venv(orin.home / ".venvs/vllm-omni", "vllm-omni==0.32.0")
                             / "bin" / "vllm")
    calls = []

    def run(argv, **kw):
        calls.append(argv)
        return SimpleNamespace(returncode=1, stdout="", stderr="CUDA error: no kernel image\n")

    monkeypatch.setattr(engines.subprocess, "run", run)
    with pytest.raises(engines.SetupError, match="no kernel image"):
        engines.check_gpu("omni", proc.STATE_DIR)
    assert "import vllm_omni" in calls[0][-1]


def test_gpu_check_of_an_image_runs_it(orin, monkeypatch):
    monkeypatch.setattr(containers, "gpu_args", lambda g: ["--gpus", "all"])
    monkeypatch.setattr(cli.images, "local_image", lambda ref, state: ("byid", "byid"))
    seen = {}

    def run(argv, **kw):
        seen["argv"] = argv
        return SimpleNamespace(returncode=0, stdout="2.14.0+cu130 Orin 64\n", stderr="")

    monkeypatch.setattr(engines.subprocess, "run", run)
    engines.check_gpu("vllm", proc.STATE_DIR, image="llmctl/vllm-orin@" + BUILT)
    assert seen["argv"][:2] == ["/usr/bin/docker", "run"]
    assert "byid" in seen["argv"] and "--gpus" in seen["argv"]


def test_the_dockerfile_ships_in_the_package():
    pin = engines.pins()["vllm"]["orin"]["image"]
    folder = Path(engines.__file__).with_name("docker") / pin["dockerfile"]
    text = (folder / "Dockerfile").read_text()
    for arg in ("VLLM_VERSION", "UV_VERSION", "COMPAT_URL", "COMPAT_SHA256"):
        assert f"ARG {arg}" in text
    assert "g++" in text

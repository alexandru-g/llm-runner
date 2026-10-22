"""Container runtime: the docker argv, runtime choice, lifecycle, caches."""
import json
from pathlib import Path

import pytest
from click.testing import CliRunner

from llm_runner import backends
from llm_runner import cli
from llm_runner import containers
from llm_runner import gateway as gw
from llm_runner import process as proc
from llm_runner import registry

DIGEST = "sha256:" + "c" * 64
IMAGE = f"vllm/vllm-openai@{DIGEST}"


@pytest.fixture(autouse=True)
def _host(monkeypatch):
    monkeypatch.setattr(containers, "docker_bin", lambda: "/usr/bin/docker")
    monkeypatch.setattr(containers, "has_cdi", lambda: True)
    monkeypatch.setattr(containers, "gpu_arch", lambda index=0: "sm_89")
    monkeypatch.setattr(containers.os, "getuid", lambda: 1000)
    monkeypatch.setattr(containers.os, "getgid", lambda: 1000)


def _vllm_cmd(model: Path, host="127.0.0.1", port=8101, extra=()):
    return backends.build_vllm_cmd("vllm", model, name="qwen", port=port, ctx_size=4096,
                                   host=host, extra=list(extra))


def _argv(tmp_path, *, host="127.0.0.1", gpus="all", extra=(), env=None, cache=None):
    models = tmp_path / "models"
    model = models / "qwen3-0.6b"
    model.mkdir(parents=True, exist_ok=True)
    rt = {"kind": "docker", "image": IMAGE, "digest": DIGEST, "gpus": gpus,
          "container": "llmctl-qwen-8101"}
    return containers.docker_argv(
        name="qwen", cmd=_vllm_cmd(model, host=host, extra=extra), port=8101,
        backend="vllm", model_path=str(model), env=env or {}, runtime=rt,
        models_dir=models, cache_dir=cache), models, model


def test_docker_argv_pins_mounts_publishes_and_isolates(tmp_path):
    argv, models, model = _argv(tmp_path, env={"HF_HUB_OFFLINE": "1"},
                                cache=tmp_path / "cache")
    s = " ".join(argv)
    assert argv[:6] == ["/usr/bin/docker", "run", "--rm", "--init", "--pull", "never"]
    assert f"-v {model}:/model:ro" in s and f"-v {models}:/models:ro" in s
    assert f"-v {tmp_path / 'cache'}:/cache" in s
    assert "-p 127.0.0.1:8101:8101" in s          # the bind host only
    assert "--network bridge" in s and "--user 1000:1000" in s
    assert "--device nvidia.com/gpu=all" in s
    assert "-e HF_HUB_OFFLINE=1" in s
    i = argv.index("--entrypoint")
    assert argv[i + 1:i + 3] == ["vllm", IMAGE]
    inner = argv[i + 3:]
    assert inner[:2] == ["serve", "/model"]
    assert inner[inner.index("--host") + 1] == "0.0.0.0"
    assert str(tmp_path) not in " ".join(inner)


def test_other_models_dir_paths_map_to_slash_models(tmp_path):
    models = tmp_path / "models"
    draft = models / "qwen3-0.6b-draft"
    argv, _, _ = _argv(tmp_path, extra=["--speculative-config",
                                        json.dumps({"model": str(draft)})])
    assert '{"model": "/models/qwen3-0.6b-draft"}' in argv


def test_a_path_prefix_only_matches_whole_components(tmp_path):
    out = containers._sub_path("/m/qwen-2/x /m/qwen/y /m/qwen", "/m/qwen", "/model")
    assert out == "/m/qwen-2/x /model/y /model"


def test_gpu_selection_cdi_and_legacy(tmp_path, monkeypatch):
    argv, _, _ = _argv(tmp_path, gpus="0,1")
    assert " ".join(argv).count("--device nvidia.com/gpu=") == 2
    monkeypatch.setattr(containers, "has_cdi", lambda: False)
    argv, _, _ = _argv(tmp_path, gpus="1")
    assert argv[argv.index("--gpus") + 1] == '"device=1"'
    assert containers.parse_gpus(" 0, 2") == "0,2" and containers.parse_gpus(None) == "all"
    with pytest.raises(ValueError):
        containers.parse_gpus("first")


def test_llama_gguf_mounts_its_folder(tmp_path):
    models = tmp_path / "models"
    gguf = models / "gemma" / "gemma-Q4.gguf"
    gguf.parent.mkdir(parents=True)
    cmd = ["llama-server", "-m", str(gguf), "--mmproj", str(gguf.parent / "mmproj.gguf"),
           "--host", "0.0.0.0", "--port", "8080"]
    argv = containers.docker_argv(
        name="gemma", cmd=cmd, port=8080, backend="llama", model_path=str(gguf), env={},
        runtime={"kind": "docker", "image": "ghcr.io/ggml-org/llama.cpp@" + DIGEST,
                 "digest": DIGEST, "container": "c"}, models_dir=models, cache_dir=None)
    i = argv.index("--entrypoint")
    assert argv[i + 1] == "/app/llama-server"
    assert argv[i + 3:] == ["-m", "/model/gemma-Q4.gguf", "--mmproj", "/model/mmproj.gguf",
                            "--host", "0.0.0.0", "--port", "8080"]
    assert "-p" in argv and argv[argv.index("-p") + 1] == "8080:8080"
    assert "--shm-size" not in argv


def test_container_env_has_no_host_toolchain(monkeypatch):
    monkeypatch.setattr(backends, "toolchain_env", lambda: {"CC": "/usr/bin/gcc"})
    env = containers.container_env(True, "vllm", {"X": "1"}, with_cache=True)
    assert "CC" not in env and env["HF_HUB_OFFLINE"] == "1" and env["X"] == "1"
    assert env["VLLM_CACHE_ROOT"] == "/cache/vllm" and env["VLLM_NO_USAGE_STATS"] == "1"


def test_shipped_images_are_pinned():
    imgs = containers.expected_images()
    assert {"vllm", "llama"} <= set(imgs)
    for e in imgs.values():
        assert containers.is_pinned(e["image"]), e


# ------------------------------------------------------------ choosing it --

@pytest.fixture
def fake_start(monkeypatch, tmp_path):
    started = {}
    monkeypatch.setattr(proc, "start", lambda **kw: started.update(kw) or proc.RunningProcess(
        name=kw["name"], pid=1, port=kw["port"], started_at=0, model_path=kw["model_path"],
        log_file="l", cmd=kw["cmd"], backend=kw["backend"], runtime=kw["runtime"]))
    monkeypatch.setattr(cli, "_ensure_gateway", lambda: None)
    monkeypatch.setattr(cli, "_port_in_use", lambda port: False)
    monkeypatch.setattr(backends, "default_gpu_mem", lambda backend=None: None)
    monkeypatch.setattr(backends, "find_engine", lambda backend: None)   # not on the host
    ckpt = tmp_path / "qwen3"
    ckpt.mkdir()
    (ckpt / "config.json").write_text('{"architectures": ["Qwen3ForCausalLM"]}')
    (ckpt / "model.safetensors").write_bytes(b"")
    return started, ckpt


def test_run_in_docker_needs_no_host_engine(fake_start, monkeypatch):
    started, ckpt = fake_start
    monkeypatch.setattr(containers, "inspect_image", lambda ref: {"RepoDigests": [IMAGE]})
    r = CliRunner().invoke(cli.cli, ["run", "--ft", str(ckpt), "--runtime", "docker",
                                     "--image", IMAGE, "--gpu", "0", "--port", "8101"])
    assert r.exit_code == 0, r.output
    rt = started["runtime"]
    assert rt == {"kind": "docker", "gpus": "0", "image": IMAGE, "digest": DIGEST}
    assert started["cmd"][0] == "vllm"
    assert "--runtime docker" in " ".join(started["run_args"])


def test_run_refuses_a_tag_and_a_missing_image(fake_start, monkeypatch):
    _, ckpt = fake_start
    r = CliRunner().invoke(cli.cli, ["run", "--ft", str(ckpt), "--image", "vllm/vllm-openai:latest"])
    assert r.exit_code != 0 and "not pinned by digest" in r.output
    monkeypatch.setattr(containers, "inspect_image", lambda ref: None)
    r = CliRunner().invoke(cli.cli, ["run", "--ft", str(ckpt), "--image", IMAGE])
    assert r.exit_code != 0 and "never pulls" in r.output


def test_configured_runtime_applies(fake_start, monkeypatch):
    started, ckpt = fake_start
    monkeypatch.setattr(containers, "inspect_image", lambda ref: {"RepoDigests": []})
    r = CliRunner().invoke(cli.cli, ["config", "runtime", "vllm", "docker", "--image", IMAGE,
                                     "--network", "llm-egress"])
    assert r.exit_code == 0, r.output
    r = CliRunner().invoke(cli.cli, ["run", "--ft", str(ckpt)])
    assert r.exit_code == 0, r.output
    assert started["runtime"]["image"] == IMAGE and started["runtime"]["network"] == "llm-egress"
    # --runtime host still wins per launch (and then needs the host engine).
    monkeypatch.setattr(backends, "find_engine", lambda backend: "/usr/bin/vllm")
    r = CliRunner().invoke(cli.cli, ["run", "--ft", str(ckpt), "--runtime", "host"])
    assert r.exit_code == 0 and started["runtime"] == {"kind": "host", "gpus": "all"}


def test_kv_cache_mem_maps_to_vllm_and_hybrids_warn(fake_start, monkeypatch):
    started, ckpt = fake_start
    monkeypatch.setattr(backends, "find_engine", lambda backend: "/usr/bin/vllm")
    r = CliRunner().invoke(cli.cli, ["run", "--ft", str(ckpt), "--kv-cache-mem", "6G"])
    assert r.exit_code == 0, r.output
    cmd = started["cmd"]
    assert cmd[cmd.index("--kv-cache-memory-bytes") + 1] == "6G"
    monkeypatch.setattr(backends, "hf_is_hybrid", lambda path: True)
    r = CliRunner().invoke(cli.cli, ["run", "--ft", str(ckpt)])
    assert "hybrid model without a fixed KV size" in r.output
    r = CliRunner().invoke(cli.cli, ["run", "--ft", str(ckpt), "--kv-cache-mem", "lots"])
    assert r.exit_code != 0


# --------------------------------------------------------------- lifecycle --

def test_start_runs_docker_and_records_the_runtime(tmp_path, monkeypatch):
    calls = {}

    class Popen:
        pid = 4242

        def __init__(self, argv, **kw):
            calls["argv"], calls["env"] = argv, kw["env"]

    monkeypatch.setattr(proc, "LOG_DIR", tmp_path / "logs")
    monkeypatch.setattr(proc, "STATE_DIR", tmp_path)
    monkeypatch.setattr(proc, "reconcile", list)
    saved = {}
    monkeypatch.setattr(proc, "_save", lambda procs: saved.update(procs=procs))
    monkeypatch.setattr(proc, "autostart_add", lambda **kw: saved.update(auto=kw))
    monkeypatch.setattr(proc.subprocess, "Popen", Popen)
    stopped = []
    monkeypatch.setattr(containers, "stop_container", lambda n, timeout=20: stopped.append(n))
    models = tmp_path / "models"
    model = models / "qwen3"
    model.mkdir(parents=True)
    monkeypatch.setattr(registry, "MODELS_DIR", models)
    rec = proc.start(name="qwen", cmd=_vllm_cmd(model), port=8101, model_path=str(model),
                     backend="vllm", env={"A": "1"},
                     runtime={"kind": "docker", "image": IMAGE, "digest": DIGEST, "gpus": "all"})
    assert calls["argv"][0] == "/usr/bin/docker" and "-e A=1" in " ".join(calls["argv"])
    assert "A" not in calls["env"]                      # it goes to the container
    assert rec.runtime["container"] == "llmctl-qwen-8101" and stopped == ["llmctl-qwen-8101"]
    assert rec.cmd == _vllm_cmd(model)                   # host form kept
    assert rec.runtime["cache_dir"].startswith(str(tmp_path / "cache" / "qwen"))
    log = Path(rec.log_file).read_text()
    assert f"runtime=docker image={IMAGE} digest={DIGEST}" in log
    assert "cache: cold start" in log
    assert proc.chosen_runtime(rec.runtime) == {"kind": "docker", "image": IMAGE,
                                                "digest": DIGEST, "gpus": "all"}


def test_stop_stops_the_container(monkeypatch):
    rec = proc.RunningProcess(name="q", pid=999999, port=1, started_at=0, model_path="m",
                              log_file="l", runtime={"kind": "docker", "container": "llmctl-q-1"})
    monkeypatch.setattr(proc, "find", lambda ident: rec)
    monkeypatch.setattr(proc, "reconcile", list)
    monkeypatch.setattr(proc, "_save", lambda procs: None)
    monkeypatch.setattr(proc, "autostart_remove", lambda name: None)
    stopped = []
    monkeypatch.setattr(containers, "stop_container", lambda n, timeout=20: stopped.append(n))
    proc.stop("q")
    assert stopped == ["llmctl-q-1"]


def test_describe_and_ps_report_the_image(tmp_path):
    state = tmp_path / "running.json"
    state.write_text(json.dumps([{"name": "qwen", "pid": 1, "port": 9, "backend": "vllm",
                                  "started_at": 0, "model_path": "", "cmd": [],
                                  "runtime": {"kind": "docker", "image": IMAGE,
                                              "digest": DIGEST}}]))
    local = gw.LocalModels(state)
    [m] = local.models()                       # pid 1 is always alive
    assert m.runtime == ("docker", IMAGE, DIGEST)
    info = local.describe(m)["llmctl"]
    assert info["runtime"] == "docker" and info["image"] == IMAGE and info["image_digest"] == DIGEST
    assert cli._runtime_label({"kind": "docker", "image": IMAGE, "digest": DIGEST}) == \
        "docker vllm-openai@sha256:cccccccccccc"
    assert cli._runtime_label(None) == "host"


# ------------------------------------------------------------------ caches --

def _installed(tmp_path, caches):
    folder = tmp_path / "models" / "qwen3-v2"
    (folder / "cache" / "vllm" / "torch_compile_cache").mkdir(parents=True)
    (folder / "cache" / "vllm" / "torch_compile_cache" / "k").write_text("kernel")
    (folder / ".llmctl-source.json").write_text(json.dumps({"source": "install",
                                                            "caches": caches}))
    return folder


def _cache(**kw):
    return {"kind": "vllm", "path": "cache/vllm", "engine": "vllm",
            "engine_version": "0.29.0", "image": IMAGE, "gpu_arch": "sm_89", **kw}


def test_a_matching_digest_seeds_the_cache(tmp_path):
    folder = _installed(tmp_path, [_cache()])
    d, rule, line = containers.prepare_cache(
        state_dir=tmp_path / "state", name="qwen", model_path=str(folder), backend="vllm",
        runtime={"kind": "docker", "image": IMAGE, "digest": DIGEST})
    assert rule == "digest" and (d / "torch_compile_cache" / "k").read_text() == "kernel"
    assert "seeded" in line and "cache rule: digest" in line
    # Later starts reuse what's there.
    (d / "torch_compile_cache" / "k").write_text("warmer")
    d2, _, line = containers.prepare_cache(
        state_dir=tmp_path / "state", name="qwen", model_path=str(folder), backend="vllm",
        runtime={"kind": "docker", "image": IMAGE, "digest": DIGEST})
    assert d2 == d and (d / "torch_compile_cache" / "k").read_text() == "warmer"


def test_a_container_never_uses_the_engine_version_rule(tmp_path):
    folder = _installed(tmp_path, [_cache(image="vllm/vllm-openai@sha256:" + "d" * 64)])
    other = "sha256:" + "e" * 64
    d, rule, line = containers.prepare_cache(
        state_dir=tmp_path / "state", name="qwen", model_path=str(folder), backend="vllm",
        runtime={"kind": "docker", "image": "x@" + other, "digest": other,
                 "engine_version": "0.29.0"})
    assert rule == "none" and "cold start" in line
    assert not (d / "torch_compile_cache").exists()


def test_the_host_falls_back_to_engine_version(tmp_path, monkeypatch):
    folder = _installed(tmp_path, [_cache()])
    monkeypatch.setattr(containers, "engine_version", lambda backend: "0.29.0")
    d, rule, line = containers.prepare_cache(
        state_dir=tmp_path / "state", name="qwen", model_path=str(folder), backend="vllm",
        runtime={"kind": "host"})
    assert rule == "engine_version" and "engine_version (dev fallback)" in line


def test_the_gpu_arch_must_match(tmp_path, monkeypatch):
    folder = _installed(tmp_path, [_cache(gpu_arch="sm_87")])
    _, rule, _ = containers.prepare_cache(
        state_dir=tmp_path / "state", name="qwen", model_path=str(folder), backend="vllm",
        runtime={"kind": "docker", "image": IMAGE, "digest": DIGEST})
    assert rule == "none"


def test_a_host_run_without_a_shipped_cache_keeps_the_shared_one(tmp_path):
    folder = tmp_path / "plain"
    folder.mkdir()
    assert containers.prepare_cache(state_dir=tmp_path, name="q", model_path=str(folder),
                                    backend="vllm", runtime={"kind": "host"})[0] is None


def test_export_then_manifest_then_seed(tmp_path, monkeypatch):
    from llm_runner import bundle
    state = tmp_path / "state"
    model = tmp_path / "models" / "qwen3"
    model.mkdir(parents=True)
    (model / "config.json").write_text("{}")
    rt = {"kind": "docker", "image": IMAGE, "digest": DIGEST, "engine_version": "0.29.0"}
    d, _, _ = containers.prepare_cache(state_dir=state, name="qwen", model_path=str(model),
                                       backend="vllm", runtime=rt)
    (d / "vllm" / "torch_compile_cache").mkdir(parents=True)
    (d / "vllm" / "torch_compile_cache" / "graph").write_text("compiled")
    monkeypatch.setattr(proc, "STATE_DIR", state)
    monkeypatch.setattr(proc, "find", lambda name: None)
    r = CliRunner().invoke(cli.cli, ["cache", "export", "qwen", str(model), "--json"])
    assert r.exit_code == 0, r.output
    entry = json.loads(r.stdout)["cache"]
    assert entry == {"kind": "vllm", "path": "cache/vllm", "engine": "vllm",
                     "engine_version": "0.29.0", "image": IMAGE, "gpu_arch": "sm_89"}
    data = bundle.build_manifest(model, name="qwen3-v2", model_id="Qwen/Qwen3", revision="r",
                                 license="apache-2.0", fmt="hf")
    assert data["caches"] == [entry]
    (model / bundle.MANIFEST_FILE).write_text(json.dumps(data))
    bundle.install(model, tmp_path / "installed")
    d2, rule, _ = containers.prepare_cache(
        state_dir=tmp_path / "state2", name="qwen", model_path=str(tmp_path / "installed" / "qwen3-v2"),
        backend="vllm", runtime=rt)
    assert rule == "digest" and (d2 / "vllm" / "torch_compile_cache" / "graph").exists()

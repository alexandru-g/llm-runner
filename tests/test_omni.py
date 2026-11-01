"""The omni backend: diffusers pipelines (FLUX.1) served by vLLM-Omni for
image generation — discovery, download, engine lookup, argv, routing."""
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
from click.testing import CliRunner

from llm_runner import backends, cli, config as config_mod, gateway as gw, process as proc
from llm_runner.cluster import IMAGES, OPENAI, Instance, Unroutable, pick
from llm_runner.registry import discover_models

GIB = 1024 ** 3


def _arg(cmd, flag):
    return cmd[cmd.index(flag) + 1]


def _weights(path: Path, size: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:          # sparse: the size without the disk
        f.truncate(size)


def _flux(root: Path, name="FLUX.1-schnell", *, dit=GIB, t5=GIB, single_file=False) -> Path:
    """A FLUX.1 diffusers layout: model_index.json, a config.json + weights
    per component, tokenizers and a scheduler without weights."""
    d = root / name
    d.mkdir(parents=True, exist_ok=True)
    (d / "model_index.json").write_text(json.dumps({
        "_class_name": "FluxPipeline", "_diffusers_version": "0.30.0",
        "scheduler": ["diffusers", "FlowMatchEulerDiscreteScheduler"],
        "text_encoder": ["transformers", "CLIPTextModel"],
        "text_encoder_2": ["transformers", "T5EncoderModel"],
        "tokenizer": ["transformers", "CLIPTokenizer"],
        "tokenizer_2": ["transformers", "T5TokenizerFast"],
        "transformer": ["diffusers", "FluxTransformer2DModel"],
        "vae": ["diffusers", "AutoencoderKL"]}))
    for comp, size in (("transformer", dit), ("text_encoder_2", t5),
                       ("text_encoder", 1024), ("vae", 1024)):
        (d / comp).mkdir()
        (d / comp / "config.json").write_text("{}")
        _weights(d / comp / "model.safetensors", size)
    for comp in ("scheduler", "tokenizer", "tokenizer_2"):
        (d / comp).mkdir()
        (d / comp / "config.json").write_text("{}")
    if single_file:
        _weights(d / "flux1-schnell.safetensors", 1024)
    return d


# ------------------------------------------------------------- detection --

def test_a_diffusers_pipeline_is_one_omni_model(tmp_path):
    _flux(tmp_path)
    llm = tmp_path / "Qwen3-8B"
    llm.mkdir()
    (llm / "config.json").write_text("{}")
    (llm / "model.safetensors").write_text("")
    specs = discover_models(tmp_path)
    assert set(specs) == {"flux.1-schnell", "qwen3-8b"}      # not transformer/, vae/, ...
    flux = specs["flux.1-schnell"]
    assert flux.backend == backends.OMNI and flux.model_path == tmp_path / "FLUX.1-schnell"
    assert not flux.has_vision and not flux.decision
    assert specs["qwen3-8b"].backend == backends.VLLM


def test_components_and_detection(tmp_path):
    d = _flux(tmp_path, dit=3 * GIB, t5=2 * GIB)
    comps = backends.diffusers_components(d)
    assert set(comps) == {"transformer", "text_encoder", "text_encoder_2", "vae"}
    assert comps["transformer"] == 3 * GIB
    assert backends.is_diffusers_dir(d) and backends.infer_backend(d) == backends.OMNI
    assert not backends.is_diffusers_dir(d / "transformer")    # a part, not a pipeline
    empty = tmp_path / "index-only"
    empty.mkdir()
    (empty / "model_index.json").write_text('{"transformer": ["diffusers", "X"]}')
    assert not backends.is_diffusers_dir(empty)                # no weights: nothing to serve
    (empty / "model_index.json").write_text("not json")
    assert backends.diffusers_components(empty) == {}


# ----------------------------------------------------------------- argv --

def test_omni_cmd():
    cmd = backends.build_omni_cmd("/v/bin/vllm", Path("/m/flux"), name="flux", port=8090,
                                  offload=["--diffusion-offload-config", "{}"],
                                  host="127.0.0.1", extra=["--cache-backend", "tea_cache"])
    assert cmd[:4] == ["/v/bin/vllm", "serve", "/m/flux", "--omni"]
    assert _arg(cmd, "--served-model-name") == "flux" and _arg(cmd, "--port") == "8090"
    assert _arg(cmd, "--host") == "127.0.0.1"
    assert _arg(cmd, "--diffusion-offload-config") == "{}"
    assert cmd[-2:] == ["--cache-backend", "tea_cache"]


@pytest.mark.parametrize("extra", [["--enable-cpu-offload"],
                                   ["--diffusion-offload-config={\"mode\":\"module\"}"]])
def test_an_offload_flag_in_extra_replaces_the_default(extra):
    cmd = backends.build_omni_cmd("vllm", Path("/m"), name="f", port=1,
                                  offload=["--diffusion-offload-config", "{}"], extra=extra)
    assert "{}" not in cmd and cmd[-1:] == extra


@pytest.mark.parametrize("free_gib, components", [
    (40, None),                         # everything fits: no offload
    (20, ["dit"]),                      # the 4090 case: stream the DiT
    (8, ["dit", "text_encoder"]),       # not even the rest fits
])
def test_default_offload_follows_free_vram(tmp_path, monkeypatch, free_gib, components):
    d = _flux(tmp_path, dit=22 * GIB, t5=9 * GIB)
    monkeypatch.setattr(backends, "gpu_memory_mib",
                        lambda device=0: (free_gib * 1024, 24564))
    args, why = backends.default_omni_offload(d)
    if components is None:
        assert args == [] and "fit" in why
    else:
        assert args[0] == "--diffusion-offload-config"
        assert json.loads(args[1]) == {"mode": "layer", "components": components}


def test_unknown_vram_offloads_the_dit(tmp_path, monkeypatch):
    monkeypatch.setattr(backends, "gpu_memory_mib", lambda device=0: None)
    args, why = backends.default_omni_offload(_flux(tmp_path))
    assert json.loads(args[1])["components"] == ["dit"] and "unknown" in why


# --------------------------------------------------------------- engine --

def _exe(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("#!/bin/sh\n")
    path.chmod(0o755)
    return path


@pytest.fixture
def no_engines(tmp_path, monkeypatch):
    for var in ("LLM_RUNNER_OMNI_BIN", "LLM_RUNNER_VLLM_BIN", "XDG_DATA_HOME"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("PATH", str(tmp_path / "empty"))
    monkeypatch.setenv("UV_TOOL_DIR", str(tmp_path / "uv-tools"))
    monkeypatch.setattr(Path, "home", lambda: tmp_path / "home")
    return tmp_path


def test_omni_never_takes_plain_vllm(no_engines, monkeypatch):
    t = no_engines
    _exe(t / "bin" / "vllm")
    _exe(t / "uv-tools" / "vllm" / "bin" / "vllm")
    monkeypatch.setenv("PATH", str(t / "bin"))
    assert backends.find_engine(backends.VLLM) == str(t / "bin" / "vllm")
    assert backends.find_engine(backends.OMNI) is None
    venv = _exe(t / "home" / ".venvs" / "vllm-omni" / "bin" / "vllm")
    assert backends.find_engine(backends.OMNI) == str(venv)
    tool = _exe(t / "uv-tools" / "vllm-omni" / "bin" / "vllm")
    assert backends.find_engine(backends.OMNI) == str(tool)
    pinned = _exe(t / "pinned" / "vllm")
    config_mod.set_engine_bin(backends.OMNI, pinned)
    assert backends.find_engine(backends.OMNI) == str(pinned)
    monkeypatch.setenv("LLM_RUNNER_OMNI_BIN", str(venv))
    assert backends.find_engine(backends.OMNI) == str(venv)


def test_config_omni_bin_and_hint(no_engines):
    venv = _exe(no_engines / "home" / ".venvs" / "vllm-omni" / "bin" / "vllm")
    runner = CliRunner()
    assert f"auto-detected {venv}" in runner.invoke(cli.cli, ["config", "omni-bin"]).output
    hint = backends.missing_hint(backends.OMNI)
    assert "~/.venvs/vllm-omni" in hint and "vllm-omni==" in hint and "vllm==" in hint


# ------------------------------------------------------------------- run --

@pytest.fixture
def launched(monkeypatch):
    monkeypatch.setattr(backends, "find", lambda b: f"/bin/{b}")
    monkeypatch.setattr(backends, "gpu_memory_mib", lambda device=0: (24000, 24564))
    monkeypatch.setattr(cli, "_port_in_use", lambda p: False)
    monkeypatch.setattr(cli, "_ensure_gateway", lambda: None)
    started = {}
    monkeypatch.setattr(proc, "start", lambda **kw: started.update(kw) or proc.RunningProcess(
        name=kw["name"], pid=1, port=kw["port"], started_at=0, model_path=kw["model_path"],
        log_file="/dev/null", cmd=kw["cmd"], backend=kw["backend"]))
    return started


def test_run_a_flux_pipeline(tmp_path, launched):
    d = _flux(tmp_path, dit=22 * GIB, t5=9 * GIB)
    out = CliRunner().invoke(cli.cli, ["run", "--ft", str(d), "--name", "flux",
                                       "--port", "8090"])
    assert out.exit_code == 0, out.output
    cmd = launched["cmd"]
    assert launched["backend"] == "omni" and cmd[:5] == ["/bin/omni", "serve", str(d),
                                                         "--omni", "--host"]
    assert json.loads(_arg(cmd, "--diffusion-offload-config"))["components"] == ["dit"]
    assert "--max-model-len" not in cmd and "--alias" not in cmd
    assert "/v1/images/generations" in out.output and "warning" not in out.output


def test_run_warns_about_text_model_flags(tmp_path, launched):
    d = _flux(tmp_path)
    out = CliRunner().invoke(cli.cli, ["run", "--ft", str(d), "--ctx-size", "8192",
                                       "--tool-parser", "hermes", "-X", "--enable-cpu-offload"])
    assert out.exit_code == 0, out.output
    assert "text-model-only option(s) ignored for omni: --ctx-size" in out.output
    assert "vllm/sglang-only option(s) ignored for omni: --tool-parser" in out.output
    cmd = launched["cmd"]
    assert "--enable-cpu-offload" in cmd and "--diffusion-offload-config" not in cmd


@pytest.mark.parametrize("backend", ["vllm", "sglang", "llama"])
def test_a_pipeline_runs_only_on_omni(tmp_path, launched, backend):
    out = CliRunner().invoke(cli.cli, ["run", "--ft", str(_flux(tmp_path)),
                                       "--backend", backend])
    assert out.exit_code != 0 and "only --backend omni" in out.output


def test_omni_needs_a_pipeline(tmp_path, launched):
    ckpt = tmp_path / "Qwen3-8B"
    ckpt.mkdir()
    (ckpt / "config.json").write_text("{}")
    (ckpt / "model.safetensors").write_text("")
    out = CliRunner().invoke(cli.cli, ["run", "--ft", str(ckpt), "--backend", "omni"])
    assert out.exit_code != 0 and "not a diffusers pipeline" in out.output


# -------------------------------------------------------------- download --

def test_download_skips_the_single_file_copies(tmp_path, monkeypatch):
    import huggingface_hub
    files = ["model_index.json", "flux1-schnell.safetensors", "ae.safetensors",
             "schnell_grid.jpeg", "README.md",
             "transformer/config.json",
             "transformer/diffusion_pytorch_model-00001-of-00003.safetensors",
             "text_encoder_2/model-00001-of-00002.safetensors", "vae/config.json",
             "vae/diffusion_pytorch_model.safetensors"]
    info = SimpleNamespace(siblings=[SimpleNamespace(rfilename=f, size=1) for f in files])
    got = {}

    def snapshot(**kw):
        got.update(kw)
        target = Path(kw["local_dir"])
        _flux(target.parent, target.name)
    monkeypatch.setattr(huggingface_hub, "snapshot_download", snapshot)
    target = tmp_path / "models" / "flux.1-schnell"
    cli._snapshot_hf("black-forest-labs/FLUX.1-schnell", info, target, revalidate=False)
    allowed = set(got["allow_patterns"])
    assert "flux1-schnell.safetensors" not in allowed and "ae.safetensors" not in allowed
    assert "transformer/diffusion_pytorch_model-00001-of-00003.safetensors" in allowed
    assert {"model_index.json", "schnell_grid.jpeg", "README.md"} <= allowed
    source = json.loads((target / ".llmctl-source.json").read_text())
    assert source["format"] == "hf" and source["repo_id"] == "black-forest-labs/FLUX.1-schnell"


def test_a_plain_checkpoint_download_keeps_ignore_patterns(tmp_path, monkeypatch):
    import huggingface_hub
    files = ["config.json", "model.safetensors", "pytorch_model.bin"]
    info = SimpleNamespace(siblings=[SimpleNamespace(rfilename=f, size=1) for f in files])
    got = {}

    def snapshot(**kw):
        got.update(kw)
        d = Path(kw["local_dir"])
        (d / "config.json").write_text("{}")
        (d / "model.safetensors").write_text("")
    monkeypatch.setattr(huggingface_hub, "snapshot_download", snapshot)
    cli._snapshot_hf("Qwen/Qwen3-8B", info, tmp_path / "qwen3-8b", revalidate=False)
    assert "allow_patterns" not in got and "*.bin" in got["ignore_patterns"]


def test_update_resolves_a_pipeline_to_its_folder(tmp_path, monkeypatch):
    d = _flux(tmp_path)
    (d / ".llmctl-source.json").write_text(json.dumps(
        {"repo_id": "black-forest-labs/FLUX.1-schnell", "format": "hf", "files": []}))
    monkeypatch.setattr(cli, "discover_models", lambda: discover_models(tmp_path))
    repo, folder, known = cli._resolve_update_target("flux.1-schnell")
    assert (repo, folder, known) == ("black-forest-labs/FLUX.1-schnell", d, None)


# --------------------------------------------------------------- gateway --

def _inst(id, api):
    return Instance(id, "rtx", 0, True, {"api": api, "status": "ready", "port": 1})


def test_images_route_apart_from_chat():
    chat, flux = _inst("gemma", OPENAI), _inst("flux", IMAGES)
    # A chat call without "model" still finds the one chat model ...
    assert [c.id for c in pick([chat, flux], None, OPENAI)] == ["gemma"]
    # ... and an image call the one image model, even one carrying images.
    assert [c.id for c in pick([chat, flux], None, IMAGES)] == ["flux"]
    with pytest.raises(Unroutable) as e:
        pick([chat], "gemma", IMAGES)
    assert e.value.code == "model_not_found"
    with pytest.raises(Unroutable) as e:
        pick([chat], None, IMAGES)
    assert e.value.code == "no_models" and "image model" in e.value.message


def test_gateway_describes_an_image_model(tmp_path, monkeypatch):
    cmd = backends.build_omni_cmd("vllm", tmp_path, name="flux", port=8090)
    m = gw.Model(name="flux", pid=os.getpid(), port=8090, backend="omni", started_at=0,
                 model_path=str(tmp_path), cmd=tuple(cmd))
    assert m.api == IMAGES
    local = gw.LocalModels(tmp_path / "running.json")
    monkeypatch.setattr(local, "status", lambda m: "ready")
    info = local.describe(m)["llmctl"]
    assert info["api"] == "images" and info["backend"] == "omni"
    assert info["vision"] is False and info["decision"] is False and "context" not in info

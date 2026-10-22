"""The sglang backend, engine discovery and the CUDA toolchain vllm/sglang
launches are pinned to."""
import os
from pathlib import Path

import pytest
from click.testing import CliRunner

from llm_runner import backends, cli, config as config_mod, gateway as gw, process as proc


# The real probe; conftest stubs ``backends.toolchain_env`` for every test.
_toolchain_env = backends.toolchain_env


def _arg(cmd, flag):
    return cmd[cmd.index(flag) + 1]


def _sglang(**kw):
    return backends.build_sglang_cmd("sglang", Path("/m/qwen3-8b"), name="qwen3-8b",
                                     port=8081, ctx_size=16384, **kw)


# ------------------------------------------------------------------ argv --

def test_sglang_cmd_basics():
    cmd = _sglang(gpu_mem=0.8, n_parallel=4, host="127.0.0.1")
    assert cmd[:4] == ["sglang", "serve", "--model-path", "/m/qwen3-8b"]
    assert _arg(cmd, "--served-model-name") == "qwen3-8b"
    assert _arg(cmd, "--context-length") == "16384"
    assert _arg(cmd, "--mem-fraction-static") == "0.8"
    assert _arg(cmd, "--max-running-requests") == "4"
    assert _arg(cmd, "--host") == "127.0.0.1"
    assert "--enable-metrics" in cmd               # the gateway reads the queue from it
    assert "--kv-cache-dtype" not in cmd and "--speculative-algorithm" not in cmd


def test_sglang_cmd_options():
    cmd = _sglang(kv_quant="q8_0", tool_parser="qwen3_coder", reasoning_parser="qwen3",
                  trust_remote_code=True, extra=["--attention-backend", "triton"])
    assert _arg(cmd, "--kv-cache-dtype") == "fp8_e4m3"
    assert _arg(cmd, "--tool-call-parser") == "qwen3_coder"
    assert _arg(cmd, "--reasoning-parser") == "qwen3"
    assert "--trust-remote-code" in cmd
    assert cmd[-2:] == ["--attention-backend", "triton"]


def test_sglang_embeddings_drop_chat_options():
    cmd = _sglang(embeddings=True, tool_parser="hermes", draft_path=Path("/d"))
    assert "--is-embedding" in cmd
    assert "--tool-call-parser" not in cmd and "--speculative-algorithm" not in cmd


@pytest.mark.parametrize("draft, spec_type, algo", [
    (Path("/d"), "draft-mtp", "STANDALONE"),
    (Path("/d"), "eagle3", "EAGLE3"),
    (None, "mtp", "NEXTN"),
])
def test_sglang_speculative_chain(draft, spec_type, algo):
    cmd = _sglang(draft_path=draft, spec_type=spec_type, spec_n_max=3)
    assert _arg(cmd, "--speculative-algorithm") == algo
    assert _arg(cmd, "--speculative-num-steps") == "3"
    assert _arg(cmd, "--speculative-eagle-topk") == "1"
    assert _arg(cmd, "--speculative-num-draft-tokens") == "4"
    assert ("--speculative-draft-model-path" in cmd) == (draft is not None)


def test_sglang_ngram_and_no_speculation():
    cmd = _sglang(spec_type="ngram", spec_n_max=8)
    assert _arg(cmd, "--speculative-algorithm") == "NGRAM"
    assert _arg(cmd, "--speculative-num-draft-tokens") == "8"
    assert "--speculative-num-steps" not in cmd
    assert "--speculative-algorithm" not in _sglang()      # draft-mtp needs a drafter


def test_sglang_replay_moves_to_the_current_host():
    cmd = backends.with_bind_host(_sglang(), backends.SGLANG, "127.0.0.1")
    assert _arg(cmd, "--host") == "127.0.0.1"


def test_gpu_mem_default_leaves_sglang_more_headroom(monkeypatch):
    monkeypatch.setattr(backends, "gpu_memory_mib", lambda device=0: (24000, 24564))
    vllm, _ = backends.default_gpu_mem(backends.VLLM)
    sglang, _ = backends.default_gpu_mem(backends.SGLANG)
    assert vllm == 0.9 and sglang == 0.85
    monkeypatch.setattr(backends, "gpu_memory_mib", lambda device=0: (8000, 24564))
    assert backends.default_gpu_mem(backends.SGLANG)[0] == 0.2


# ------------------------------------------------------------- discovery --

def _exe(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("#!/bin/sh\n")
    path.chmod(0o755)
    return path


@pytest.fixture
def no_engines(tmp_path, monkeypatch):
    """Nothing on PATH, an empty home and uv tool dir."""
    for var in ("LLM_RUNNER_VLLM_BIN", "LLM_RUNNER_SGLANG_BIN", "XDG_DATA_HOME"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("PATH", str(tmp_path / "empty"))
    monkeypatch.setenv("UV_TOOL_DIR", str(tmp_path / "uv-tools"))
    monkeypatch.setattr(Path, "home", lambda: tmp_path / "home")
    return tmp_path


def test_find_engine_order(no_engines, monkeypatch):
    t = no_engines
    assert backends.find_engine(backends.SGLANG) is None
    venv = _exe(t / "home" / ".venvs" / "sglang" / "bin" / "sglang")
    assert backends.find_engine(backends.SGLANG) == str(venv)
    tool = _exe(t / "uv-tools" / "sglang" / "bin" / "sglang")
    assert backends.find_engine(backends.SGLANG) == str(tool)
    on_path = _exe(t / "bin" / "sglang")
    monkeypatch.setenv("PATH", str(t / "bin"))
    assert backends.find_engine(backends.SGLANG) == str(on_path)
    pinned = _exe(t / "pinned" / "sglang")
    config_mod.set_engine_bin(backends.SGLANG, pinned)
    assert backends.find_engine(backends.SGLANG) == str(pinned)
    monkeypatch.setenv("LLM_RUNNER_SGLANG_BIN", str(tool))
    assert backends.find_engine(backends.SGLANG) == str(tool)
    assert backends.find_engine(backends.VLLM) is None       # each engine its own


def test_config_engine_bin_command(no_engines):
    tool = _exe(no_engines / "uv-tools" / "sglang" / "bin" / "sglang")
    runner = CliRunner()
    assert f"auto-detected {tool}" in runner.invoke(cli.cli, ["config", "sglang-bin"]).output
    out = runner.invoke(cli.cli, ["config", "sglang-bin", str(tool)])
    assert out.exit_code == 0 and config_mod.load()["sglang_bin"] == str(tool)
    assert runner.invoke(cli.cli, ["config", "sglang-bin", "--clear"]).exit_code == 0
    assert "sglang_bin" not in config_mod.load()


def test_missing_hint_names_uv():
    assert "uv tool install" in backends.missing_hint(backends.SGLANG)
    assert "conda" not in backends.missing_hint(backends.VLLM)


# ------------------------------------------------------------- toolchain --

def test_nvcc_max_gcc_reads_the_version_check(tmp_path):
    crt = tmp_path / "targets" / "x86_64-linux" / "include" / "crt"
    crt.mkdir(parents=True)
    (crt / "host_config.h").write_text(
        "#if defined(__GNUC__)\n#if __GNUC__ > 14\n#error unsupported GNU version!\n#endif\n")
    assert backends.nvcc_max_gcc(tmp_path) == 14
    assert backends.nvcc_max_gcc(tmp_path / "nothing") is None


@pytest.fixture
def compilers(tmp_path, monkeypatch):
    """Fake system bin dir: g++ 16, g++-15 and g++-13 (with gccs), where
    g++-15 has no cc1plus, plus a lone g++-12 with no gcc-12."""
    d = tmp_path / "usr-bin"
    versions = {"g++": 16, "g++-15": 15, "g++-13": 13, "g++-12": 12}
    for name in versions:
        _exe(d / name)
        if name != "g++-12":
            _exe(d / name.replace("g++", "gcc"))
    monkeypatch.setattr(backends, "_SYSTEM_BIN_DIRS", (d,))
    monkeypatch.setattr(backends, "_gcc_major", lambda p: versions[p.name])
    monkeypatch.setattr(backends, "_has_cc1plus", lambda p: p.name != "g++-15")
    return d


def test_host_compiler_takes_the_newest_nvcc_accepts(compilers):
    d = compilers
    assert backends.host_compiler(16) == (str(d / "gcc"), str(d / "g++"))
    assert backends.host_compiler(None) == (str(d / "gcc"), str(d / "g++"))
    # 15 can't compile C++ here, so 13 it is; 12 has no matching gcc.
    assert backends.host_compiler(15) == (str(d / "gcc-13"), str(d / "g++-13"))
    assert backends.host_compiler(12) is None


def test_toolchain_env_pins_the_compilers(compilers, tmp_path, monkeypatch):
    cuda = tmp_path / "cuda"
    _exe(cuda / "bin" / "nvcc")
    (cuda / "include" / "crt").mkdir(parents=True)
    (cuda / "include" / "crt" / "host_config.h").write_text("#if __GNUC__ > 13\n")
    monkeypatch.setenv("CUDA_HOME", str(cuda))
    _toolchain_env.cache_clear()
    try:
        env = _toolchain_env()
    finally:
        _toolchain_env.cache_clear()
    gxx = str(compilers / "g++-13")
    assert env == {"CUDA_HOME": str(cuda), "CC": str(compilers / "gcc-13"),
                   "CXX": gxx, "NVCC_CCBIN": gxx}


def test_launch_env_adds_the_toolchain_for_hf_engines_only(monkeypatch):
    monkeypatch.setattr(backends, "toolchain_env", lambda: {"CXX": "/usr/bin/g++"})
    assert backends.launch_env(False, backends.SGLANG)["CXX"] == "/usr/bin/g++"
    assert backends.launch_env(False, backends.VLLM)["CXX"] == "/usr/bin/g++"
    assert "CXX" not in backends.launch_env(False, backends.LLAMA)
    assert "CXX" not in backends.launch_env(False)


def test_sglang_health_is_a_cheap_check():
    flag = "SGLANG_ENABLE_HEALTH_ENDPOINT_GENERATION"
    assert backends.launch_env(False, backends.SGLANG)[flag] == "0"
    assert flag not in backends.launch_env(False, backends.VLLM)


def test_dash_e_overrides_the_toolchain(tmp_path, monkeypatch):
    monkeypatch.setattr(backends, "toolchain_env", lambda: {"CXX": "/usr/bin/g++"})
    monkeypatch.setattr(proc, "LOG_DIR", tmp_path)
    monkeypatch.setattr(proc, "reconcile", list)
    monkeypatch.setattr(proc, "_save", lambda procs: None)
    monkeypatch.setattr(proc, "autostart_add", lambda **kw: None)
    seen = {}

    class Popen:
        pid = 1

        def __init__(self, cmd, env, **kw):
            seen.update(env)

    monkeypatch.setattr(proc.subprocess, "Popen", Popen)
    proc.start(name="m", cmd=["sglang"], port=8081, model_path="m", backend="sglang",
               env={"CXX": "/opt/g++"})
    assert seen["CXX"] == "/opt/g++"


# ------------------------------------------------------------------- run --

def test_run_on_sglang(tmp_path, monkeypatch):
    ckpt = tmp_path / "Qwen3-8B"
    ckpt.mkdir()
    (ckpt / "config.json").write_text("{}")
    (ckpt / "model.safetensors").write_text("")
    monkeypatch.setattr(backends, "find", lambda b: f"/bin/{b}")
    monkeypatch.setattr(backends, "gpu_memory_mib", lambda device=0: (24000, 24564))
    monkeypatch.setattr(cli, "_port_in_use", lambda p: False)
    monkeypatch.setattr(cli, "_ensure_gateway", lambda: None)
    started = {}
    monkeypatch.setattr(proc, "start", lambda **kw: started.update(kw) or proc.RunningProcess(
        name=kw["name"], pid=1, port=kw["port"], started_at=0, model_path=kw["model_path"],
        log_file="/dev/null", cmd=kw["cmd"], backend=kw["backend"]))
    out = CliRunner().invoke(cli.cli, ["run", "--ft", str(ckpt), "--backend", "sglang",
                                       "--name", "qwen", "--port", "8090",
                                       "--tool-parser", "hermes", "--ctx-size", "8192"])
    assert out.exit_code == 0, out.output
    assert "warning" not in out.output                  # --tool-parser applies to sglang
    cmd = started["cmd"]
    assert started["backend"] == "sglang" and cmd[:2] == ["/bin/sglang", "serve"]
    assert _arg(cmd, "--context-length") == "8192"
    assert _arg(cmd, "--mem-fraction-static") == "0.85"
    assert _arg(cmd, "--tool-call-parser") == "hermes"


def test_sglang_rejects_4bit_kv(tmp_path, monkeypatch):
    ckpt = tmp_path / "m"
    ckpt.mkdir()
    (ckpt / "config.json").write_text("{}")
    (ckpt / "model.safetensors").write_text("")
    monkeypatch.setattr(backends, "find", lambda b: f"/bin/{b}")
    out = CliRunner().invoke(cli.cli, ["run", "--ft", str(ckpt), "--backend", "sglang",
                                       "--kv-quant", "q4_0"])
    assert out.exit_code != 0 and "4-bit" in out.output


# --------------------------------------------------------------- gateway --

def test_gateway_describes_an_sglang_model(tmp_path, monkeypatch):
    m = gw.Model(name="qwen", pid=os.getpid(), port=8081, backend="sglang", started_at=0,
                 model_path=str(tmp_path), cmd=tuple(_sglang(n_parallel=8, embeddings=True)))
    local = gw.LocalModels(tmp_path / "running.json")
    monkeypatch.setattr(local, "status", lambda m: "ready")
    info = local.describe(m)["llmctl"]
    assert info["context"] == 16384 and info["parallel"] == 8 and info["embeddings"]

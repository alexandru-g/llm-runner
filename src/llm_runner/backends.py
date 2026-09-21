"""Serving backends: how to find each server binary and build its argv.

Three backends are supported:

``llama``
    llama.cpp's ``llama-server`` serving a single GGUF. The argv builder for
    it lives in ``cli._build_llama_cmd`` (it predates this module and carries
    a lot of tuning history in its comments).

``vllm``
    ``vllm serve`` on a Hugging Face-format checkpoint directory (safetensors
    + ``config.json``). vLLM is a Python package, usually installed in its own
    venv/conda env rather than on PATH, so :func:`find_vllm` looks in the
    common places and the location can be pinned with ``llmctl config
    vllm-bin`` or ``$LLM_RUNNER_VLLM_BIN``.

``decider``
    Mapika/decider's ``decider.serve`` (uvicorn) on a decider checkpoint
    directory — an HF checkpoint that also carries ``decider_config.json``.
    It speaks the Jev wire (``POST /v1/systemone``: typed questions in,
    probabilities out; no text generation). The package lives in its own
    clone + venv; point llmctl at it with ``llmctl config decider-dir`` or
    ``$LLM_RUNNER_DECIDER_DIR``. The default launch profile is sized for a
    GPU shared with a llama-server (graphs ≤ 8 × 1024 tokens, batch ≤ 8,
    fp8 linears, raw model temperature) — see :func:`build_decider_cmd`.
"""
from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

LLAMA = "llama"
VLLM = "vllm"
DECIDER = "decider"
BACKENDS = (LLAMA, VLLM, DECIDER)


# ---------------------------------------------------------------- binaries --

def find_llama_server() -> str | None:
    return shutil.which("llama-server")


def _conda_roots() -> list[Path]:
    roots: list[Path] = []
    conda_exe = os.environ.get("CONDA_EXE")
    if conda_exe:
        roots.append(Path(conda_exe).resolve().parents[1])
    home = Path.home()
    roots += [home / "miniconda3", home / "anaconda3", home / "miniforge3",
              home / "mambaforge", home / ".conda"]
    seen: set[Path] = set()
    out: list[Path] = []
    for r in roots:
        if r not in seen and r.exists():
            seen.add(r)
            out.append(r)
    return out


def find_vllm() -> str | None:
    """Locate the ``vllm`` CLI.

    Order: ``$LLM_RUNNER_VLLM_BIN`` → ``config.json`` ``vllm_bin`` → PATH →
    a conda/venv env literally named ``vllm`` → any conda env that has it.
    Returns an absolute path (or None) so the recorded launch argv keeps
    working from a systemd unit whose PATH lacks the env.
    """
    env = os.environ.get("LLM_RUNNER_VLLM_BIN")
    if env:
        p = Path(env).expanduser()
        return str(p) if p.is_file() else None

    from llm_runner import config
    configured = config.get_vllm_bin()
    if configured and configured.is_file():
        return str(configured)

    on_path = shutil.which("vllm")
    if on_path:
        return on_path

    home = Path.home()
    candidates = [home / ".venvs" / "vllm" / "bin" / "vllm",
                  home / "venvs" / "vllm" / "bin" / "vllm",
                  home / "vllm" / ".venv" / "bin" / "vllm"]
    for root in _conda_roots():
        candidates.append(root / "envs" / "vllm" / "bin" / "vllm")
    for root in _conda_roots():
        envs = root / "envs"
        if envs.is_dir():
            candidates += sorted(envs.glob("*/bin/vllm"))
    for c in candidates:
        if c.is_file() and os.access(c, os.X_OK):
            return str(c.resolve())
    return None


def find_decider_dir() -> Path | None:
    """The decider clone: ``$LLM_RUNNER_DECIDER_DIR`` → config ``decider_dir``.
    A dir counts when it holds the package (``decider/serve.py``)."""
    env = os.environ.get("LLM_RUNNER_DECIDER_DIR")
    if env:
        p = Path(env).expanduser()
        return p if (p / "decider" / "serve.py").is_file() else None
    from llm_runner import config
    configured = config.get_decider_dir()
    if configured and (configured / "decider" / "serve.py").is_file():
        return configured
    return None


def find_decider() -> str | None:
    """``uvicorn`` inside the decider clone's venv (``.venv*/bin/uvicorn``);
    it is what ``scripts/serve.sh`` upstream execs. Absolute, like vllm."""
    root = find_decider_dir()
    if root is None:
        return None
    for venv in sorted(root.glob(".venv*")):
        bin_ = venv / "bin" / "uvicorn"
        if bin_.is_file() and os.access(bin_, os.X_OK):
            return str(bin_.resolve())
    return None


def find(backend: str) -> str | None:
    if backend == LLAMA:
        return find_llama_server()
    if backend == DECIDER:
        return find_decider()
    return find_vllm()


def missing_hint(backend: str) -> str:
    if backend == LLAMA:
        return ("`llama-server` binary not found on PATH. Install llama.cpp "
                "(https://github.com/ggml-org/llama.cpp) and ensure llama-server "
                "is on PATH.")
    if backend == DECIDER:
        return ("decider clone not found. `git clone https://github.com/Mapika/decider`, "
                "create its venv (`uv venv --python 3.12 .venv312 && uv pip install "
                "-p .venv312/bin/python -e \".[serve]\"`) and point llmctl at it with "
                "`llmctl config decider-dir /path/to/decider` (or $LLM_RUNNER_DECIDER_DIR).")
    return ("`vllm` CLI not found. Install it in its own env (e.g. "
            "`conda create -n vllm python=3.12 && conda activate vllm && "
            "pip install vllm`) — llmctl finds an env named `vllm` "
            "automatically — or point at it with `llmctl config vllm-bin "
            "/path/to/env/bin/vllm` (or $LLM_RUNNER_VLLM_BIN).")


# ------------------------------------------------------------ model format --

# Weight files vLLM can load from a checkpoint directory. A dir with
# ``config.json`` but none of these is a tokenizer-only / adapter repo.
_HF_WEIGHT_GLOBS = ("*.safetensors", "*.bin", "*.pt", "*.pth")


def is_hf_model_dir(path: Path) -> bool:
    """True if ``path`` looks like a Hugging Face checkpoint vLLM can serve."""
    if not (path.is_dir() and (path / "config.json").is_file()):
        return False
    return any(next(path.glob(g), None) is not None for g in _HF_WEIGHT_GLOBS)


def hf_dir_size(path: Path) -> int:
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file())


def hf_has_vision(path: Path) -> bool:
    """Best-effort: multimodal checkpoints ship a preprocessor / vision config."""
    if (path / "preprocessor_config.json").is_file():
        return True
    try:
        cfg = json.loads((path / "config.json").read_text())
    except (OSError, json.JSONDecodeError):
        return False
    return "vision_config" in cfg or any(
        "ForConditionalGeneration" in a for a in cfg.get("architectures", []))


def is_decider_model_dir(path: Path) -> bool:
    """A decider checkpoint: an HF checkpoint dir that also ships
    ``decider_config.json`` (fitted temperature, version, option limits)."""
    return is_hf_model_dir(path) and (path / "decider_config.json").is_file()


def infer_backend(path: Path) -> str:
    """``decider`` for a decider checkpoint dir, ``vllm`` for any other HF
    checkpoint dir, ``llama`` for anything else (GGUF)."""
    if path.is_dir():
        return DECIDER if is_decider_model_dir(path) else VLLM
    return LLAMA


# -------------------------------------------------------------- vllm argv --

def gpu_memory_mib(device: int = 0) -> tuple[int, int] | None:
    """(free, total) MiB for ``device`` via nvidia-smi, or None if unavailable."""
    import subprocess
    try:
        out = subprocess.run(
            ["nvidia-smi", f"--id={device}", "--query-gpu=memory.free,memory.total",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5, check=True).stdout
        free, total = (int(x) for x in out.strip().splitlines()[0].split(","))
        return free, total
    except (OSError, subprocess.SubprocessError, ValueError, IndexError):
        return None


# vLLM's own default for --gpu-memory-utilization (0.9 / 0.92 depending on
# version). It checks the fraction against *total* VRAM at startup and aborts
# if that much isn't free, so we never ask for more than is actually free.
VLLM_DEFAULT_GPU_MEM = 0.9
_GPU_MEM_HEADROOM_MIB = 768  # CUDA context, torch.compile workspace, other tenants


def default_gpu_mem() -> tuple[float, str] | None:
    """Pick --gpu-memory-utilization from the free VRAM right now.

    Returns ``(fraction, reason)`` or None when nvidia-smi is unavailable (then
    vLLM's default applies). The fraction is what's free minus headroom, capped
    at vLLM's default and floored at 0.05, rounded down to 2 decimals so the
    check doesn't fail on rounding.
    """
    mem = gpu_memory_mib()
    if mem is None:
        return None
    free, total = mem
    usable = max(free - _GPU_MEM_HEADROOM_MIB, 0)
    frac = min(VLLM_DEFAULT_GPU_MEM, int(usable / total * 100) / 100)
    frac = max(frac, 0.05)
    return frac, f"{free} MiB of {total} MiB free"

# llama.cpp KV-cache quant names → vLLM --kv-cache-dtype. vLLM has no 4-bit
# KV cache; q4_0 is rejected in the CLI before we get here.
_KV_DTYPE = {"f16": "auto", "q8_0": "fp8"}

# Speculative-decoding methods vLLM accepts in --speculative-config. The
# llama-style default "draft-mtp" is not one of them: with a --draft model we
# use "draft_model", otherwise no speculation.
VLLM_SPEC_METHODS = ("ngram", "eagle", "eagle3", "mtp", "draft_model")


def build_vllm_cmd(vllm_bin: str, model_path: Path, *, name: str, port: int,
                   ctx_size: int, kv_quant: str = "f16",
                   gpu_mem: float | None = None,
                   n_parallel: int | None = None,
                   embeddings: bool = False,
                   draft_path: Path | None = None,
                   spec_type: str = "draft-mtp", spec_n_max: int = 3,
                   tool_parser: str | None = None,
                   reasoning_parser: str | None = None,
                   trust_remote_code: bool = False,
                   extra: list[str] | None = None) -> list[str]:
    cmd = [
        vllm_bin, "serve", str(model_path),
        "--host", "0.0.0.0",
        "--port", str(port),
        # Stable model id for /v1 clients (otherwise it's the full path).
        "--served-model-name", name,
        "--max-model-len", str(ctx_size),
    ]
    if gpu_mem is not None:
        cmd += ["--gpu-memory-utilization", f"{gpu_mem:g}"]
    if n_parallel is not None:
        cmd += ["--max-num-seqs", str(n_parallel)]
    dtype = _KV_DTYPE.get(kv_quant, "auto")
    if dtype != "auto":
        cmd += ["--kv-cache-dtype", dtype]
    if embeddings:
        cmd += ["--runner", "pooling"]
    else:
        if tool_parser:
            cmd += ["--enable-auto-tool-choice", "--tool-call-parser", tool_parser]
        if reasoning_parser:
            cmd += ["--reasoning-parser", reasoning_parser]
        spec = _speculative_config(draft_path, spec_type, spec_n_max)
        if spec:
            cmd += ["--speculative-config", json.dumps(spec, separators=(",", ":"))]
    if trust_remote_code:
        cmd.append("--trust-remote-code")
    if extra:
        cmd += extra
    return cmd


def _speculative_config(draft_path: Path | None, spec_type: str,
                        n: int) -> dict | None:
    if draft_path is not None:
        method = spec_type if spec_type in ("eagle", "eagle3") else "draft_model"
        return {"method": method, "model": str(draft_path),
                "num_speculative_tokens": n}
    if spec_type in ("ngram", "mtp"):
        cfg: dict = {"method": spec_type, "num_speculative_tokens": n}
        if spec_type == "ngram":
            cfg["prompt_lookup_max"] = 4
        return cfg
    return None


# ------------------------------------------------------------ decider argv --

# Launch profile for a decision server sharing the GPU with a llama-server.
# Upstream defaults capture CUDA graphs up to 32 × 1536 tokens (~20 GB peak
# for a 2B model); these keep decider-2b at ~5.6 GB loaded / ~6.1 GB peak
# (measured 2026-09-21 on a 4090, batch 4) and let the long tail (> 1024
# tokens) run eagerly one or two rows at a time. `-E KEY=VALUE` overrides
# any of them; `-E DECIDER_FP8=1` trades ~0.7 GB for half the throughput
# (fp8 without torch.compile is slower on Ada).
DECIDER_DEFAULT_ENV = {
    "DECIDER_COMPILE": "0",              # graphs already give the batched speed; compile costs a 4-minute start
    "DECIDER_FP8": "0",                  # bf16 linears: 50 dec/s vs 27 for fp8 e4m3 on the 4090
    "DECIDER_MAX_BATCH": "8",            # rows per forward = the client's concurrency (--n-parallel)
    "DECIDER_WARMUP_MAX_B": "8",         # graphs captured up to this batch size
    "DECIDER_GRAPH_MAX_T": "1024",       # ... and this length; longer rows run eagerly
    "DECIDER_MAX_FWD_TOKENS": "4608",    # padded tokens per eager forward: caps the long-row activations
    "DECIDER_TEMPERATURE": "1.0",        # raw model; the client applies its own fitted scaling
    "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
}


def build_decider_cmd(uvicorn_bin: str, decider_dir: Path, model_path: Path, *,
                      port: int, n_parallel: int | None = None,
                      extra: list[str] | None = None) -> tuple[list[str], dict[str, str]]:
    """``(argv, env)`` for ``decider.serve``. The model and every knob travel
    in the environment (that is the server's interface); ``--app-dir`` makes
    the package importable without a cwd change."""
    cmd = [uvicorn_bin, "decider.serve:app", "--app-dir", str(decider_dir),
           "--host", "0.0.0.0", "--port", str(port)]
    if extra:
        cmd += extra
    env = dict(DECIDER_DEFAULT_ENV)
    env["DECIDER_MODEL"] = str(model_path)
    if n_parallel is not None:
        env["DECIDER_MAX_BATCH"] = str(n_parallel)
        env["DECIDER_WARMUP_MAX_B"] = str(n_parallel)
    return cmd, env

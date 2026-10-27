"""Serving backends: how to find each server binary and build its argv.

Four backends are supported:

``llama``
    llama.cpp's ``llama-server`` serving a single GGUF. The argv builder for
    it lives in ``cli._build_llama_cmd`` (it predates this module and carries
    a lot of tuning history in its comments).

``vllm``
    ``vllm serve`` on a Hugging Face-format checkpoint directory (safetensors
    + ``config.json``). vLLM is a Python package with its own CUDA torch, so
    it lives in its own venv — ``uv tool install vllm`` — which
    :func:`find_engine` finds; the location can be pinned with ``llmctl
    config vllm-bin`` or ``$LLM_RUNNER_VLLM_BIN``.

``sglang``
    ``sglang serve`` on the same checkpoint directories, installed and found
    the same way (``uv tool install sglang``, ``llmctl config sglang-bin``,
    ``$LLM_RUNNER_SGLANG_BIN``). Never picked automatically: a checkpoint
    runs on it with ``--backend sglang``.

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

import functools
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from urllib.parse import urlsplit

from llm_runner import config

LLAMA = "llama"
VLLM = "vllm"
SGLANG = "sglang"
DECIDER = "decider"
# Not a model server: the systemone-proxy started by `llmctl decide` in front
# of a running llama-server.
DECIDE = "decide"
# Not a model server either: the `llmctl gateway` router that puts every
# running model behind one port. Started with the first model.
GATEWAY = "gateway"
BACKENDS = (LLAMA, VLLM, SGLANG, DECIDER)
# Engines that serve a Hugging Face checkpoint dir behind an OpenAI /v1 API.
# They take the same llmctl options and are interchangeable for a model.
HF_ENGINES = (VLLM, SGLANG)


# Set for every process llmctl starts. vLLM sends usage stats home unless told
# not to; huggingface_hub (in vLLM and decider) reports telemetry the same way.
NO_TELEMETRY_ENV = {
    "VLLM_NO_USAGE_STATS": "1",
    "VLLM_DO_NOT_TRACK": "1",
    "DO_NOT_TRACK": "1",
    "HF_HUB_DISABLE_TELEMETRY": "1",
}
# Added in offline mode: load models from disk only, never ask the hub.
OFFLINE_ENV = {"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"}


# SGLang's /health generates a token by default and answers after ~1 s, the
# gateway's whole probe timeout, so models flapped to "down". Off, it is a
# cheap readiness check (503 while starting) like llama-server's and vLLM's.
SGLANG_ENV = {"SGLANG_ENABLE_HEALTH_ENDPOINT_GENERATION": "0"}


def launch_env(offline: bool, backend: str | None = None) -> dict[str, str]:
    """The environment llmctl sets for ``backend`` (``-E`` still wins): no
    telemetry, offline mode, and for vllm/sglang the CUDA toolchain their
    kernel JIT compiles with (:func:`toolchain_env`)."""
    return {**NO_TELEMETRY_ENV, **(OFFLINE_ENV if offline else {}),
            **(toolchain_env() if backend in HF_ENGINES else {}),
            **(SGLANG_ENV if backend == SGLANG else {})}


def with_bind_host(cmd: list[str], backend: str, host: str) -> list[str]:
    """``cmd`` set to listen on ``host``. A recorded command keeps the address
    it was launched with, so replaying one (restore, restart) rewrites it to
    the current ``llmctl config bind``."""
    cmd = list(cmd)
    if "--host" in cmd[:-1]:
        cmd[cmd.index("--host") + 1] = host
    elif backend in (DECIDE, GATEWAY):
        # Recorded before llmctl passed --host; both default to 0.0.0.0.
        cmd += ["--host", host]
    if backend == DECIDE and "--backend" in cmd[:-1]:
        # The llama-server behind it moved to the same address.
        i = cmd.index("--backend") + 1
        port = urlsplit(cmd[i]).port
        if port is not None:
            cmd[i] = config.local_url(port, bind=host)
    return cmd


# ---------------------------------------------------------------- binaries --

def find_llama_server() -> str | None:
    return shutil.which("llama-server")


# $LLM_RUNNER_<X>_BIN overrides discovery of each Python-packaged engine.
_BIN_ENV = {VLLM: "LLM_RUNNER_VLLM_BIN", SGLANG: "LLM_RUNNER_SGLANG_BIN"}


def bin_env_var(backend: str) -> str:
    return _BIN_ENV[backend]


def _uv_tool_dir() -> Path:
    """Where ``uv tool install`` puts each tool's venv."""
    env = os.environ.get("UV_TOOL_DIR")
    if env:
        return Path(env).expanduser()
    xdg = os.environ.get("XDG_DATA_HOME")
    return (Path(xdg).expanduser() if xdg else Path.home() / ".local" / "share") / "uv" / "tools"


def find_engine(backend: str) -> str | None:
    """Locate the ``vllm`` / ``sglang`` CLI.

    Order: ``$LLM_RUNNER_<X>_BIN`` → ``config.json`` ``<x>_bin`` → PATH → the
    venv ``uv tool install <x>`` made → a venv at ``~/.venvs/<x>``,
    ``~/venvs/<x>`` or ``~/<x>/.venv``. Returns an absolute path (or None)
    so the recorded launch argv keeps working from a systemd unit whose PATH
    lacks ``~/.local/bin``.
    """
    env = os.environ.get(_BIN_ENV[backend])
    if env:
        p = Path(env).expanduser()
        return str(p) if p.is_file() else None

    configured = config.get_engine_bin(backend)
    if configured and configured.is_file():
        return str(configured)

    on_path = shutil.which(backend)
    if on_path:
        return str(Path(on_path).absolute())

    home = Path.home()
    for c in (_uv_tool_dir() / backend / "bin" / backend,
              home / ".venvs" / backend / "bin" / backend,
              home / "venvs" / backend / "bin" / backend,
              home / backend / ".venv" / "bin" / backend):
        if c.is_file() and os.access(c, os.X_OK):
            return str(c)
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


def find_systemone_proxy() -> str | None:
    """The ``systemone-proxy`` CLI (packages/systemone-proxy in this repo):
    ``$LLM_RUNNER_SYSTEMONE_PROXY`` → PATH → the package's own ``.venv``
    in a source checkout. Absolute, so systemd restores find it."""
    env = os.environ.get("LLM_RUNNER_SYSTEMONE_PROXY")
    if env:
        p = Path(env).expanduser()
        return str(p.resolve()) if p.is_file() else None
    on_path = shutil.which("systemone-proxy")
    if on_path:
        return str(Path(on_path).resolve())
    local = (Path(__file__).resolve().parents[2] / "packages" / "systemone-proxy"
             / ".venv" / "bin" / "systemone-proxy")
    if local.is_file() and os.access(local, os.X_OK):
        return str(local)
    return None


def find(backend: str) -> str | None:
    if backend == LLAMA:
        return find_llama_server()
    if backend == DECIDER:
        return find_decider()
    if backend == DECIDE:
        return find_systemone_proxy()
    if backend == GATEWAY:
        return sys.executable    # it is llmctl itself (python -m llm_runner)
    return find_engine(backend)


def missing_hint(backend: str) -> str:
    if backend == LLAMA:
        return ("`llama-server` binary not found on PATH. Install llama.cpp "
                "(https://github.com/ggml-org/llama.cpp) and ensure llama-server "
                "is on PATH.")
    if backend == DECIDE:
        return ("`systemone-proxy` not found. Install it from this repo with "
                "`uv tool install ./packages/systemone-proxy` (re-run ./install.sh), "
                "or point at it with $LLM_RUNNER_SYSTEMONE_PROXY.")
    if backend == DECIDER:
        return ("decider clone not found. `git clone https://github.com/Mapika/decider`, "
                "create its venv (`uv venv --python 3.12 .venv312 && uv pip install "
                "-p .venv312/bin/python -e \".[serve]\"`) and point llmctl at it with "
                "`llmctl config decider-dir /path/to/decider` (or $LLM_RUNNER_DECIDER_DIR).")
    return (f"`{backend}` CLI not found. Install it in its own venv with "
            f"`uv tool install --torch-backend auto {backend}` (llmctl finds it "
            f"there), or point at another one with `llmctl config {backend}-bin "
            f"/path/to/venv/bin/{backend}` (or ${_BIN_ENV[backend]}).")


# --------------------------------------------------------------- toolchain --

# vLLM and SGLang JIT-compile CUDA kernels (FlashInfer, sgl-kernel) on first
# use. nvcc hands host code to $NVCC_CCBIN, else the first `gcc` on PATH;
# FlashInfer passes `-ccbin $CC` and links with $CXX. Left to the inherited
# environment that can be a compiler without C++ (conda base's gcc: "cannot
# execute 'cc1plus'") or one newer than nvcc accepts, so launches pin all
# of them to a system GCC that nvcc supports.
_SYSTEM_BIN_DIRS = (Path("/usr/bin"), Path("/usr/local/bin"))
_CUDA_ROOTS = (Path("/usr/local/cuda"), Path("/opt/cuda"))


def find_nvcc() -> Path | None:
    """nvcc from ``$CUDA_HOME``/``$CUDA_PATH`` → PATH → /usr/local/cuda,
    /opt/cuda."""
    for var in ("CUDA_HOME", "CUDA_PATH"):
        root = os.environ.get(var)
        if root and (Path(root) / "bin" / "nvcc").is_file():
            return Path(root) / "bin" / "nvcc"
    on_path = shutil.which("nvcc")
    if on_path:
        return Path(on_path).resolve()
    return next((r / "bin" / "nvcc" for r in _CUDA_ROOTS
                 if (r / "bin" / "nvcc").is_file()), None)


def nvcc_max_gcc(cuda_root: Path) -> int | None:
    """Newest GCC major this CUDA's nvcc accepts, read from the version check
    in its ``crt/host_config.h`` (``#if __GNUC__ > N`` → error). None when
    the header is missing or the check isn't there."""
    for header in [cuda_root / "include" / "crt" / "host_config.h",
                   *sorted(cuda_root.glob("targets/*/include/crt/host_config.h"))]:
        try:
            m = re.search(r"#if\s+__GNUC__\s*>\s*(\d+)", header.read_text(errors="replace"))
        except OSError:
            continue
        if m:
            return int(m.group(1))
    return None


def _gcc_major(compiler: Path) -> int | None:
    try:
        out = subprocess.run([str(compiler), "-dumpversion"], capture_output=True,
                             text=True, timeout=5, check=True).stdout
        return int(out.strip().split(".")[0])
    except (OSError, subprocess.SubprocessError, ValueError):
        return None


def _has_cc1plus(gxx: Path) -> bool:
    """Whether this g++ can compile C++ (``-print-prog-name`` echoes the bare
    name back when cc1plus isn't installed)."""
    try:
        out = subprocess.run([str(gxx), "-print-prog-name=cc1plus"], capture_output=True,
                             text=True, timeout=5, check=True).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return False
    return os.path.isabs(out) and os.path.isfile(out)


def host_compiler(max_major: int | None) -> tuple[str, str] | None:
    """``(gcc, g++)`` from the system bin dirs: the newest ``g++``/``g++-N``
    with a matching ``gcc`` that can compile C++ and is no newer than
    ``max_major``. Only system dirs, so an env's compiler on PATH never wins."""
    found: list[tuple[int, int, Path, Path]] = []
    for d in _SYSTEM_BIN_DIRS:
        for gxx in [d / "g++", *sorted(d.glob("g++-[0-9]*"))]:
            gcc = d / gxx.name.replace("g++", "gcc", 1)
            if not (gxx.is_file() and gcc.is_file()):
                continue
            major = _gcc_major(gxx)
            if major is None or (max_major is not None and major > max_major):
                continue
            # Prefer the unversioned name at equal versions (it's the default).
            found.append((major, gxx.name == "g++", gxx, gcc))
    for _, _, gxx, gcc in sorted(found, reverse=True):
        if _has_cc1plus(gxx):
            return str(gcc), str(gxx)
    return None


@functools.cache
def toolchain_env() -> dict[str, str]:
    """``CUDA_HOME``, ``CC``, ``CXX`` and ``NVCC_CCBIN`` for vllm/sglang
    launches; empty without nvcc (no JIT to feed). Compilers are left out
    when no system GCC fits, and then the inherited ones apply."""
    nvcc = find_nvcc()
    if nvcc is None:
        return {}
    cuda_root = nvcc.parent.parent
    env = {"CUDA_HOME": str(cuda_root)}
    pair = host_compiler(nvcc_max_gcc(cuda_root))
    if pair:
        gcc, gxx = pair
        env |= {"CC": gcc, "CXX": gxx, "NVCC_CCBIN": gxx}
    return env


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


def hf_is_hybrid(path: Path) -> bool:
    """True for a hybrid checkpoint: some layers keep a recurrent (mamba /
    linear-attention) state instead of a KV cache, e.g. Qwen3.5's
    ``layer_types`` of ``linear_attention`` + ``full_attention``. VLM wrappers
    keep that under ``text_config``; Nemotron-H spells it as a
    ``hybrid_override_pattern`` with ``M`` for mamba layers."""
    try:
        cfg = json.loads((path / "config.json").read_text())
    except (OSError, json.JSONDecodeError):
        return False
    for c in (cfg, cfg.get("text_config")):
        if not isinstance(c, dict):
            continue
        types = c.get("layer_types")
        if isinstance(types, list) and any(
                isinstance(t, str) and ("linear" in t or "mamba" in t) for t in types):
            return True
        if "M" in str(c.get("hybrid_override_pattern", "")):
            return True
    return False


def _gguf_general_tags(path: Path) -> list[str]:
    """``general.tags`` from a GGUF header, or [] if absent/unreadable. Only
    the leading ``general.*`` block is read (a few KB), never the tensors."""
    import struct
    scalar = {0: "B", 1: "b", 2: "H", 3: "h", 4: "I", 5: "i", 6: "f", 7: "?",
              10: "Q", 11: "q", 12: "d"}
    try:
        with path.open("rb") as f:
            def read(fmt: str):
                n = struct.calcsize(fmt)
                return struct.unpack(fmt, f.read(n))

            def string() -> str:
                (n,) = read("<Q")
                return f.read(n).decode("utf-8", "replace")

            def value(t: int):
                if t == 8:
                    return string()
                if t == 9:
                    et, n = read("<IQ")
                    return [value(et) for _ in range(n)]
                return read("<" + scalar[t])[0]

            if f.read(4) != b"GGUF":
                return []
            (version,) = read("<I")
            if version < 2:
                return []
            _, n_kv = read("<QQ")
            for _ in range(n_kv):
                key = string()
                (t,) = read("<I")
                if not key.startswith("general."):
                    return []  # writers emit general.* first; tags would be there
                v = value(t)
                if key == "general.tags":
                    return [str(x) for x in v] if isinstance(v, list) else []
    except (OSError, struct.error, KeyError, ValueError):
        pass
    return []


# Weights that are an ordinary LLM checkpoint but are meant to be read as a
# calibrated decision (letter-logprob readout over a vendor-specific prompt,
# e.g. Quyet), not chatted with.
_DECISION_SIDECARS = ("quyet_config.json",)
# GGUF ``general.tags`` (copied from the model card) that mark one. Generic
# ones like "classification" / "calibration" are left out on purpose.
_DECISION_TAGS = {"decision-model", "decision-making", "typed-decisions",
                  "typed-classification", "jevbench"}
# File names of known decision-model GGUFs, for headers without tags (e.g.
# Livesport's rune-26b-a4b-v3-Q8_0.gguf carries neither tags nor a basename).
_DECISION_FILENAMES = re.compile(
    r"^(rune-26b-a4b|xor-26b-a4b|bobcat-flash|h2o-lightning|quyet-1\.0)", re.I)


def is_decision_model(path: Path) -> bool:
    """Best-effort: a decision-model checkpoint dir (vendor config sidecar) or
    GGUF (one of ``_DECISION_TAGS`` in its header, or the sidecar beside it)."""
    folder = path if path.is_dir() else path.parent
    if any((folder / s).is_file() for s in _DECISION_SIDECARS):
        return True
    if path.suffix != ".gguf":
        return False
    tags = _gguf_general_tags(path)
    if tags:
        return bool(_DECISION_TAGS & set(tags))
    return bool(_DECISION_FILENAMES.match(path.name))


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
# (headroom MiB, cap) per engine. vLLM's fraction covers everything it
# allocates, so the headroom is just the CUDA context, torch.compile workspace
# and other tenants. SGLang's --mem-fraction-static covers only weights + KV
# pool; activations and CUDA graphs come out of what is left, so it needs
# more of it. Its cap leaves the same share free on an idle card.
_GPU_MEM_POLICY = {VLLM: (768, VLLM_DEFAULT_GPU_MEM), SGLANG: (3072, 0.88)}


def default_gpu_mem(backend: str = VLLM) -> tuple[float, str] | None:
    """Pick ``backend``'s GPU memory fraction from the free VRAM right now.

    Returns ``(fraction, reason)`` or None when nvidia-smi is unavailable (then
    the engine's own default applies). The fraction is what's free minus the
    engine's headroom, capped and floored at 0.05, rounded down to 2 decimals
    so vLLM's free-memory check doesn't fail on rounding.
    """
    mem = gpu_memory_mib()
    if mem is None:
        return None
    free, total = mem
    headroom, cap = _GPU_MEM_POLICY[backend]
    usable = max(free - headroom, 0)
    frac = min(cap, int(usable / total * 100) / 100)
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
                   host: str = "0.0.0.0",
                   extra: list[str] | None = None) -> list[str]:
    cmd = [
        vllm_bin, "serve", str(model_path),
        "--host", host,
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


# ------------------------------------------------------------- sglang argv --

# llama.cpp KV-cache quant names → SGLang --kv-cache-dtype. Its 4-bit KV
# (nvfp4 / fp4_mx_block16) needs Blackwell, so q4_0 is rejected like vLLM's.
_SGLANG_KV_DTYPE = {"f16": "auto", "q8_0": "fp8_e4m3"}
# llmctl --spec-type → --speculative-algorithm. With a --draft model, eagle /
# eagle3 name the drafter's kind and anything else is a plain small model
# (STANDALONE); without one, mtp is the checkpoint's own MTP head (NEXTN).
_SGLANG_SPEC_DRAFT = {"eagle": "EAGLE", "eagle3": "EAGLE3"}
_SGLANG_SPEC_SELF = {"mtp": "NEXTN", "ngram": "NGRAM"}
# Mamba-state slots SGLang holds per running request on a hybrid model
# (radix cache on, no speculation). It sizes that pool from memory, not from
# --max-running-requests, and then caps the requests to fit — 0.5.22 logs
# "max_running_requests is capped to 6 by the mamba state cache
# (max_mamba_cache_size=34, 5 state slots per request)". SGLang-internal
# (kv_cache_configurator._calculate_mamba_ratio), so it may change.
SGLANG_MAMBA_SLOTS_PER_REQUEST = 5
# Flags that size the mamba pool themselves: if -X has one, leave it be.
_SGLANG_MAMBA_POOL_FLAGS = ("--max-mamba-cache-size", "--mamba-full-memory-ratio")


def sglang_mamba_cache_size(n_parallel: int) -> int:
    """``--max-mamba-cache-size`` that admits ``n_parallel`` requests: their
    slots plus 20% (at least 4) headroom for spec decoding or a bumped ratio."""
    slots = n_parallel * SGLANG_MAMBA_SLOTS_PER_REQUEST
    return slots + max(4, -(-slots // 5))


def build_sglang_cmd(sglang_bin: str, model_path: Path, *, name: str, port: int,
                     ctx_size: int, kv_quant: str = "f16",
                     gpu_mem: float | None = None,
                     n_parallel: int | None = None,
                     embeddings: bool = False,
                     draft_path: Path | None = None,
                     spec_type: str = "draft-mtp", spec_n_max: int = 3,
                     tool_parser: str | None = None,
                     reasoning_parser: str | None = None,
                     trust_remote_code: bool = False,
                     host: str = "0.0.0.0",
                     extra: list[str] | None = None) -> list[str]:
    """``sglang serve`` with the same options as :func:`build_vllm_cmd`."""
    cmd = [
        sglang_bin, "serve",
        "--model-path", str(model_path),
        "--host", host,
        "--port", str(port),
        "--served-model-name", name,
        "--context-length", str(ctx_size),
        # Prometheus /metrics: the gateway reads the queue depth from it.
        "--enable-metrics",
    ]
    if gpu_mem is not None:
        cmd += ["--mem-fraction-static", f"{gpu_mem:g}"]
    if n_parallel is not None:
        cmd += ["--max-running-requests", str(n_parallel)]
        # On hybrid models that alone doesn't lift SGLang's mamba-pool cap.
        if not _sets_any(extra, _SGLANG_MAMBA_POOL_FLAGS) and hf_is_hybrid(model_path):
            cmd += ["--max-mamba-cache-size", str(sglang_mamba_cache_size(n_parallel))]
    dtype = _SGLANG_KV_DTYPE.get(kv_quant, "auto")
    if dtype != "auto":
        cmd += ["--kv-cache-dtype", dtype]
    if embeddings:
        cmd.append("--is-embedding")
    else:
        if tool_parser:
            cmd += ["--tool-call-parser", tool_parser]
        if reasoning_parser:
            cmd += ["--reasoning-parser", reasoning_parser]
        cmd += _sglang_spec_args(draft_path, spec_type, spec_n_max)
    if trust_remote_code:
        cmd.append("--trust-remote-code")
    if extra:
        cmd += extra
    return cmd


def _sets_any(extra: list[str] | None, flags: tuple[str, ...]) -> bool:
    """True if ``extra`` passes any of ``flags`` (``--f v`` or ``--f=v``)."""
    return any(a == f or a.startswith(f + "=") for a in extra or () for f in flags)


def _sglang_spec_args(draft_path: Path | None, spec_type: str, n: int) -> list[str]:
    if draft_path is not None:
        algo = _SGLANG_SPEC_DRAFT.get(spec_type, "STANDALONE")
        args = ["--speculative-algorithm", algo,
                "--speculative-draft-model-path", str(draft_path)]
    elif spec_type in _SGLANG_SPEC_SELF:
        algo = _SGLANG_SPEC_SELF[spec_type]
        args = ["--speculative-algorithm", algo]
    else:
        return []
    if algo == "NGRAM":
        return args + ["--speculative-num-draft-tokens", str(n)]
    # A chain of n drafted tokens: n steps, one candidate each, n + 1 verified.
    return args + ["--speculative-num-steps", str(n), "--speculative-eagle-topk", "1",
                   "--speculative-num-draft-tokens", str(n + 1)]


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
                      host: str = "0.0.0.0",
                      extra: list[str] | None = None) -> tuple[list[str], dict[str, str]]:
    """``(argv, env)`` for ``decider.serve``. The model and every knob travel
    in the environment (that is the server's interface); ``--app-dir`` makes
    the package importable without a cwd change."""
    cmd = [uvicorn_bin, "decider.serve:app", "--app-dir", str(decider_dir),
           "--host", host, "--port", str(port)]
    if extra:
        cmd += extra
    env = dict(DECIDER_DEFAULT_ENV)
    env["DECIDER_MODEL"] = str(model_path)
    if n_parallel is not None:
        env["DECIDER_MAX_BATCH"] = str(n_parallel)
        env["DECIDER_WARMUP_MAX_B"] = str(n_parallel)
    return cmd, env

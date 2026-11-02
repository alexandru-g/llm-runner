"""`llmctl setup`: what each backend needs on this host, and how to install it.

The Python engines get a venv of their own each, at the versions in
``engines.json`` (what this llmctl was tested with): vllm and sglang as uv
tools, vLLM-Omni and ONNX Runtime as venvs under ``~/.venvs``, where
:func:`backends.find_engine` looks. Container backends (triton always,
vllm/sglang/llama with ``--runtime docker``) get the image pinned in
``images.json``, pulled by digest or loaded from an archive, and the runtime
setting. llama.cpp (built from source, with host-specific flags) and decider
(a clone of another repo) are only checked.

Everything here returns plans (:class:`Step` lists) and status rows; the
CLI shows a plan, asks, and runs it. Nothing installs on import or status.
"""
from __future__ import annotations

import functools
import json
import re
import shutil
import subprocess
from dataclasses import dataclass, field
from importlib import resources
from pathlib import Path

from llm_runner import backends, containers
from llm_runner.backends import DECIDER, LLAMA, OMNI, ONNX, SGLANG, TRITON, VLLM

# The order `llmctl setup` lists them in.
SETUP_BACKENDS = (LLAMA, VLLM, SGLANG, OMNI, DECIDER, ONNX, TRITON)
INSTALLABLE = (VLLM, SGLANG, OMNI, ONNX)


class SetupError(Exception):
    """A setup llmctl can't plan here (no uv, no docker, ...)."""


@functools.cache
def pins() -> dict:
    """``engines.json`` as shipped: per engine, how llmctl installs it."""
    try:
        data = json.loads(resources.files("llm_runner").joinpath("engines.json").read_text())
    except (OSError, json.JSONDecodeError, ModuleNotFoundError):
        return {}
    return data.get("engines", {}) if isinstance(data, dict) else {}


def uv_bin() -> str | None:
    """uv: the standalone install in ~/.local/bin first (a conda env's uv on
    PATH would tie the venvs to that env), then PATH."""
    local = Path.home() / ".local" / "bin" / "uv"
    return str(local) if local.is_file() else shutil.which("uv")


def _norm(dist: str) -> str:
    return re.sub(r"[-_.]+", "_", dist).lower()


def dist_version(venv: Path, dist: str) -> str | None:
    """``dist``'s version in ``venv``, from its ``*.dist-info`` folder name
    (importing the engine to ask takes seconds)."""
    want = _norm(dist)
    for info in venv.glob("lib/python*/site-packages/*.dist-info"):
        name, _, version = info.name[:-len(".dist-info")].rpartition("-")
        if _norm(name) == want:
            return version
    return None


def pinned_venv(pin: dict) -> Path:
    """A pin's venv, with ``~`` as :func:`backends.find_engine` reads it."""
    v = pin["venv"]
    return Path.home() / v[2:] if v.startswith("~/") else Path(v)


def venv_of(binary: str) -> Path:
    """The venv a ``<venv>/bin/<exe>`` belongs to (not resolved: a venv's
    ``python`` is a symlink out of it)."""
    return Path(binary).parent.parent


# ------------------------------------------------------------------ status --

@dataclass
class Status:
    backend: str
    state: str                    # ok, missing, differs, configure, found
    version: str | None = None
    tested: str | None = None
    where: str | None = None
    note: str = ""


def _engine_status(backend: str) -> Status:
    pin = pins().get(backend) or {}
    binary = backends.find_engine(backend)
    tested = pin.get("version")
    if binary is None:
        return Status(backend, "missing", tested=tested, note=f"llmctl setup {backend}")
    dists = pin.get("dist") or backend
    version, variant = None, None
    # ~/.local/bin/vllm links into its uv tool's venv; a venv's own python
    # links out of it, so try the path as found first, then resolved.
    for venv in dict.fromkeys((venv_of(binary), venv_of(str(Path(binary).resolve())))):
        for d in [dists] if isinstance(dists, str) else dists:
            version = dist_version(venv, d)
            if version:
                variant = d
                break
        if version:
            break
    state = "ok" if version and version.split("+")[0] == tested else "differs"
    note = "" if state == "ok" else (f"llmctl setup {backend} --upgrade" if version else
                                     "version unknown")
    if backend == ONNX and variant:
        note = (variant + ("; " + note if note else "")).strip()
    return Status(backend, state, version, tested, binary, note)


def _llama_status() -> Status:
    binary = backends.find_llama_server()
    if binary is None:
        return Status(LLAMA, "missing", note="build llama.cpp (README, install step 2)")
    try:
        out = subprocess.run([binary, "--version"], capture_output=True, text=True,
                             timeout=10)
        m = re.search(r"build (\d+)(?:, commit (\w+))?", out.stdout + out.stderr)
        version = f"b{m.group(1)}" + (f" ({m.group(2)})" if m.group(2) else "") if m else None
    except (OSError, subprocess.TimeoutExpired):
        version = None
    return Status(LLAMA, "found", version, None, binary)


def _decider_status() -> Status:
    d = backends.find_decider_dir()
    if d is None:
        return Status(DECIDER, "missing", note="clone decider (README, install step 4)")
    return Status(DECIDER, "found" if backends.find_decider() else "missing",
                  where=str(d), note="" if backends.find_decider() else "no .venv*/bin/uvicorn")


def image_status(backend: str, state_dir: Path, configured: dict) -> Status:
    """The container image ``backend`` would run here, and whether it's present."""
    from llm_runner import images
    ref = (configured.get("image") if configured.get("kind") == containers.DOCKER
           else None) or containers.default_image(backend)
    entry = containers.expected_images().get(containers.image_key(backend)) or {}
    if not ref:
        return Status(backend, "missing", note="no image in images.json")
    found = images.local_image(ref, state_dir) if containers.docker_bin() else None
    label = f"{entry.get('tag', '')} ({ref.split('@')[0]})".strip()
    if found is None:
        return Status(backend, "missing", entry.get("engine_version"), None, label,
                      f"llmctl setup {backend}" + ("" if backend == TRITON else
                                                   " --runtime docker"))
    return Status(backend, "ok", entry.get("engine_version"), None, label,
                  "" if ref == containers.default_image(backend) else "configured image")


def status(state_dir: Path, cfg: dict) -> list[Status]:
    """One row per backend: what this host has for it."""
    rows = []
    for b in SETUP_BACKENDS:
        configured = containers.runtime_from_config(b, cfg)
        if b == TRITON or configured.get("kind") == containers.DOCKER:
            rows.append(image_status(b, state_dir, configured))
        elif b == LLAMA:
            rows.append(_llama_status())
        elif b == DECIDER:
            rows.append(_decider_status())
        else:
            rows.append(_engine_status(b))
    return rows


# -------------------------------------------------------------------- plans --

@dataclass
class Step:
    what: str                     # one line for the plan
    argv: list[str] = field(default_factory=list)       # a command, or
    config: dict | None = None    # llmctl config to set: {"runtime": {backend: entry}}
    archive: Path | None = None   # an image archive to `images load`


def _uv_source(find_links: Path | None) -> list[str]:
    return ["--offline", "--no-index", "--find-links", str(find_links)] if find_links else []


def plan_engine(backend: str, *, upgrade: bool = False, cpu: bool = False,
                index_url: str | None = None, find_links: Path | None = None,
                gpu: bool | None = None, jetson: bool | None = None) -> list[Step]:
    """Steps that put ``backend``'s Python engine on this host at the tested
    version. [] when it's there already (and ``upgrade`` not asked); a
    different version is left alone unless ``upgrade``."""
    pin = pins().get(backend)
    if not pin:
        raise SetupError(f"{backend} has no host install in engines.json")
    uv = uv_bin()
    if uv is None:
        raise SetupError("uv not found; install it first: "
                         "https://docs.astral.sh/uv/getting-started/installation/")
    current = _engine_status(backend)
    if current.state == "ok" and not upgrade:
        return []
    if current.state == "differs" and not upgrade:
        raise SetupError(f"{backend} {current.version or '(unknown version)'} is installed at "
                         f"{current.where}; llmctl was tested with {pin['version']}. "
                         f"`llmctl setup {backend} --upgrade` installs that.")
    torch = ["--torch-backend", pin["torch_backend"]] if pin.get("torch_backend") else []
    src = _uv_source(find_links)
    if pin["kind"] == "uv-tool":
        argv = [uv, "tool", "install", "--python", pin["python"], *torch, *src,
                *pin["packages"]]
        for w in pin.get("with") or ():
            argv += ["--with", w]
        return [Step(f"install {' '.join(pin['packages'])} as a uv tool", argv)]
    venv = pinned_venv(pin)
    packages, extra = list(pin["packages"]), []
    steps = []
    if backend == ONNX:
        gpu = backends_gpu() if gpu is None else gpu
        jetson = backends.is_jetson() if jetson is None else jetson
        if jetson and index_url:
            packages, extra = list(pin["jetson_packages"]), ["--extra-index-url", index_url]
        elif cpu or not gpu or jetson:
            packages = list(pin["cpu_packages"])
        # onnxruntime and onnxruntime-gpu both install the `onnxruntime` module.
        other = "onnxruntime-gpu" if packages[0].startswith("onnxruntime==") else "onnxruntime"
        if (venv / "bin" / "python").exists() and dist_version(venv, other):
            steps.append(Step(f"remove {other} from {venv}",
                              [uv, "pip", "uninstall", "-p", str(venv / "bin" / "python"),
                               other]))
    if not (venv / "bin" / "python").exists():
        steps.insert(0, Step(f"create the venv {venv} (Python {pin['python']})",
                             [uv, "venv", "--python", pin["python"], str(venv)]))
    steps.append(Step(f"install {' '.join(packages)} into {venv}",
                      [uv, "pip", "install", "-p", str(venv / "bin" / "python"), *torch,
                       *src, *extra, *packages]))
    return steps


def backends_gpu() -> bool:
    from llm_runner import cv
    return cv.gpu() is not None


def plan_image(backend: str, state_dir: Path, cfg: dict, *, archive: Path | None = None,
               set_runtime: bool = True) -> list[Step]:
    """Steps that put ``backend``'s pinned image on this host (pulled by
    digest, or loaded from ``archive``) and point its runtime at it."""
    from llm_runner import images
    if containers.docker_bin() is None:
        raise SetupError("the docker CLI is not on PATH")
    configured = containers.runtime_from_config(backend, cfg)
    ref = containers.default_image(backend)
    if not ref:
        raise SetupError(f"images.json has no image for {backend}")
    steps = []
    if archive is not None:
        steps.append(Step(f"load {archive.name} (checked against images.json)",
                          archive=archive))
    elif images.local_image(ref, state_dir) is None:
        steps.append(Step(f"pull {ref}", [containers.docker_bin(), "pull", ref]))
    want = {"kind": containers.DOCKER, "image": ref}
    have = (cfg.get("runtime") or {}).get(backend) or {}
    if set_runtime and (configured.get("kind") != containers.DOCKER
                        or have.get("image") not in (None, ref)):
        steps.append(Step(f"llmctl config runtime {backend} docker  ({ref.split('@')[0]})",
                          config={"runtime": {backend: want}}))
    return steps


def check_onnx(python: str) -> tuple[str | None, list[str]]:
    """ONNX Runtime's version and execution providers in that venv."""
    try:
        out = subprocess.run(
            [python, "-I", "-c", "import json, numpy, onnxruntime as o; "
             "print(json.dumps([o.__version__, o.get_available_providers()]))"],
            capture_output=True, text=True, timeout=120)
        version, providers = json.loads(out.stdout.strip().splitlines()[-1])
        return version, providers
    except (OSError, subprocess.TimeoutExpired, ValueError, IndexError):
        return None, []

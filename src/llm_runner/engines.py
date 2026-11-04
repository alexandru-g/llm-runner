"""`llmctl setup`: what each backend needs on this host, and how to install it.

The Python engines get a venv of their own each, at the versions in
``engines.json`` (what this llmctl was tested with): vllm and sglang as uv
tools, vLLM-Omni and ONNX Runtime as venvs under ``~/.venvs``, where
:func:`backends.find_engine` looks. Container backends (triton always,
vllm/sglang/llama with ``--runtime docker``) get the image pinned in
``images.json``, pulled by digest or loaded from an archive, and the runtime
setting. llama.cpp (built from source, with host-specific flags) and decider
(a clone of another repo) are only checked.

A Jetson Orin on JetPack 6 takes its own versions (the ``orin`` block of a pin)
and NVIDIA's ``cuda-compat-orin``, which lets PyPI's CUDA 13 wheels run on its
CUDA 12.6 driver. vLLM's image is built there, as no published one runs on it.

Everything here returns plans (:class:`Step` lists) and status rows; the
CLI shows a plan, asks, and runs it. Nothing installs on import or status.
"""
from __future__ import annotations

import functools
import hashlib
import json
import os
import re
import shutil
import subprocess
import tempfile
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from importlib import resources
from pathlib import Path
from typing import Callable

from llm_runner import backends, containers
from llm_runner.backends import DECIDER, LLAMA, OMNI, ONNX, SGLANG, TRITON, VLLM

# The order `llmctl setup` lists them in.
SETUP_BACKENDS = (LLAMA, VLLM, SGLANG, OMNI, DECIDER, ONNX, TRITON)
INSTALLABLE = (VLLM, SGLANG, OMNI, ONNX)


class SetupError(Exception):
    """A setup llmctl can't plan here (no uv, no docker, ...)."""


@functools.cache
def _manifest() -> dict:
    try:
        data = json.loads(resources.files("llm_runner").joinpath("engines.json").read_text())
    except (OSError, json.JSONDecodeError, ModuleNotFoundError):
        return {}
    return data if isinstance(data, dict) else {}


def pins() -> dict:
    """``engines.json`` as shipped: per engine, how llmctl installs it."""
    return _manifest().get("engines", {})


def pin_for(backend: str) -> dict | None:
    """``backend``'s pin on this host: with its ``orin`` block on an Orin on JetPack 6."""
    pin = pins().get(backend)
    if pin and "orin" in pin and backends.is_orin_jetpack6():
        return {**pin, **pin["orin"]}
    return pin


def _needs_orin_compat(backend: str) -> bool:
    return "orin" in (pins().get(backend) or {}) and backends.is_orin_jetpack6()


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
    pin = pin_for(backend) or {}
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
    if state == "ok" and _needs_orin_compat(backend) and backends.orin_compat_dir() is None:
        state, note = "configure", f"no CUDA 13 libraries: llmctl setup {backend}"
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
    pin = pin_for(backend) or {}
    built = pin.get("image")
    if built and (not ref or ref.split("@")[0] == built["tag"].split(":")[0]):
        found = images.local_image(ref, state_dir) if ref and containers.docker_bin() else None
        if found is None:
            return Status(backend, "missing", tested=pin["version"],
                          note=f"llmctl setup {backend} --runtime docker")
        return Status(backend, "ok", pin["version"], pin["version"], built["tag"])
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
    call: Callable[[], None] | None = None     # something llmctl does itself


def _uv_source(find_links: Path | None) -> list[str]:
    return ["--offline", "--no-index", "--find-links", str(find_links)] if find_links else []


def plan_engine(backend: str, *, upgrade: bool = False, cpu: bool = False,
                index_url: str | None = None, find_links: Path | None = None,
                gpu: bool | None = None, jetson: bool | None = None) -> list[Step]:
    """Steps that put ``backend``'s Python engine on this host at the tested
    version. [] when it's there already (and ``upgrade`` not asked); a
    different version is left alone unless ``upgrade``."""
    pin = pin_for(backend)
    if not pin:
        raise SetupError(f"{backend} has no host install in engines.json")
    uv = uv_bin()
    if uv is None:
        raise SetupError("uv not found; install it first: "
                         "https://docs.astral.sh/uv/getting-started/installation/")
    compat = plan_orin_compat(find_links) if _needs_orin_compat(backend) else []
    current = _engine_status(backend)
    if current.state in ("ok", "configure") and not upgrade:
        return compat
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
        return [*compat, Step(f"install {' '.join(pin['packages'])} as a uv tool", argv)]
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
    return [*compat, *steps]


def plan_orin_compat(find_links: Path | None = None) -> list[Step]:
    """Unpack ``cuda-compat-orin`` for this user. Needs no root, and only
    llmctl's vLLM and Omni launches see it. vLLM compiles kernels at startup, so
    a compiler is checked here rather than failing in the first model's log."""
    if backends.host_compiler(None) is None:
        raise SetupError("vLLM compiles kernels with gcc and g++ at startup, and they aren't "
                         "installed: sudo apt install build-essential")
    if backends.orin_compat_dir() is not None:
        return []
    c = _manifest()["orin_compat"]
    return [Step(f"unpack NVIDIA's CUDA {c['version']} libraries for JetPack 6 into "
                 f"{backends.orin_compat_root()}\n     from {c['file']}",
                 call=lambda: install_orin_compat(find_links))]


def install_orin_compat(find_links: Path | None = None) -> None:
    """Fetch the package (or take it from ``find_links``), check its SHA256 against
    ``engines.json`` and unpack it into place."""
    c = _manifest()["orin_compat"]
    root = backends.orin_compat_root()
    root.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=root.parent) as tmp:
        deb = Path(tmp) / c["file"]
        try:
            if find_links:
                shutil.copyfile(find_links / c["file"], deb)
            else:
                with urllib.request.urlopen(c["url"], timeout=60) as r, deb.open("wb") as f:
                    shutil.copyfileobj(r, f)
        except (OSError, urllib.error.URLError) as e:
            raise SetupError(f"can't get {c['file']}: {e}") from None
        digest = hashlib.sha256(deb.read_bytes()).hexdigest()
        if digest != c["sha256"]:
            raise SetupError(f"{c['file']} has SHA256 {digest}; engines.json expects "
                             f"{c['sha256']}.")
        out = Path(tmp) / "unpacked"
        try:
            subprocess.run(["dpkg-deb", "-x", str(deb), str(out)], check=True)
        except (OSError, subprocess.CalledProcessError) as e:
            raise SetupError(f"can't unpack {c['file']}: {e}") from None
        shutil.rmtree(root, ignore_errors=True)
        os.replace(out, root)


def backends_gpu() -> bool:
    from llm_runner import cv
    return cv.gpu() is not None


def plan_image(backend: str, state_dir: Path, cfg: dict, *, archive: Path | None = None,
               set_runtime: bool = True, upgrade: bool = False) -> list[Step]:
    """Steps that put ``backend``'s pinned image on this host (pulled by
    digest, loaded from ``archive``, or built where nothing published runs)
    and point its runtime at it."""
    from llm_runner import images
    if containers.docker_bin() is None:
        raise SetupError("the docker CLI is not on PATH")
    pin = pin_for(backend) or {}
    if pin.get("image"):
        if archive is not None:
            raise SetupError(f"{backend}'s image is built on this host; there's no archive "
                             "to load.")
        return _plan_built_image(backend, pin, state_dir, cfg, upgrade)
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


def _plan_built_image(backend: str, pin: dict, state_dir: Path, cfg: dict,
                      upgrade: bool) -> list[Step]:
    """Build ``pin``'s image here unless its tag is present, and pin the runtime
    to the result. The tag names the version, and keeps ``docker image prune``
    off the image."""
    from llm_runner import images
    built, c = pin["image"], _manifest()["orin_compat"]
    tag = built["tag"]
    steps = []
    if upgrade or containers.inspect_image(tag) is None:
        args = {"VLLM_VERSION": pin["version"], "UV_VERSION": built["uv"],
                "COMPAT_URL": c["url"], "COMPAT_SHA256": c["sha256"]}
        steps.append(Step(f"build {tag} here (about 8 minutes; downloads about 10 GB)",
                          [containers.docker_bin(), "build", "-t", tag,
                           *[a for k, v in args.items() for a in ("--build-arg", f"{k}={v}")],
                           str(Path(__file__).with_name("docker") / built["dockerfile"])]))
    have = (cfg.get("runtime") or {}).get(backend) or {}
    info = None if steps else containers.inspect_image(tag)
    pinned = info and f"{tag.split(':')[0]}@{info.get('Id')}"
    if steps or have.get("kind") != containers.DOCKER or have.get("image") != pinned \
            or images.local_image(pinned, state_dir) is None:
        steps.append(Step(f"llmctl config runtime {backend} docker  ({tag})",
                          call=lambda: pin_built_image(backend, tag, state_dir)))
    return steps


def pin_built_image(backend: str, tag: str, state_dir: Path) -> None:
    """Point ``backend``'s runtime at what ``tag`` is now, by its ID."""
    from llm_runner import config as config_mod
    from llm_runner import images
    info = containers.inspect_image(tag)
    if info is None or not info.get("Id"):
        raise SetupError(f"{tag} isn't on this host.")
    ref = f"{tag.split(':')[0]}@{info['Id']}"
    images.record_built(state_dir, ref, info["Id"])
    runtimes = dict(config_mod.load().get("runtime") or {})
    runtimes[backend] = {"kind": containers.DOCKER, "image": ref}
    config_mod.update(runtime=runtimes)


_GPU_PROBE = ("import torch; x = torch.randn(64, 64, device='cuda'); "
              "print(torch.__version__, torch.cuda.get_device_name(0), (x @ x).shape[0])")


def check_gpu(backend: str, state_dir: Path, *, image: str | None = None) -> str:
    """Run a tensor operation on the GPU with what ``backend`` will launch with,
    and say what ran. A torch without kernels for the GPU still reports CUDA as
    available, so only an operation proves it. Raises :class:`SetupError`."""
    from llm_runner import images
    code = _GPU_PROBE + ("; import vllm_omni" if backend == OMNI else "")
    if image is not None:
        found = images.local_image(image, state_dir)
        argv = [containers.docker_bin() or "docker", "run", "--rm", "--pull", "never",
                *containers.gpu_args("all"), "--entrypoint", "python",
                found[0] if found else image, "-I", "-c", code]
        env = None
    else:
        binary = backends.find_engine(backend)
        if binary is None:
            raise SetupError(f"{backend} isn't installed.")
        python = venv_of(str(Path(binary).resolve())) / "bin" / "python"
        argv = [str(python), "-I", "-c", code]
        env = {**os.environ, **backends.launch_env(False, backend)}
    try:
        out = subprocess.run(argv, capture_output=True, text=True, timeout=300, env=env)
    except (OSError, subprocess.TimeoutExpired) as e:
        raise SetupError(f"the GPU check didn't run: {e}") from None
    if out.returncode != 0:
        tail = "\n".join(out.stderr.strip().splitlines()[-4:])
        raise SetupError(f"{backend} can't use the GPU:\n{tail}")
    return out.stdout.strip().splitlines()[-1]


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

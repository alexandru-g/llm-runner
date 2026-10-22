"""Container runtime: model servers run in Docker from an image pinned by digest.

A backend runs on the host (the default, as before) or in a container, set
per backend with ``llmctl config runtime`` or per launch with ``run
--runtime docker``. The engine command llmctl records stays the one the host
would run (paths, ``--host``, ``--port``), so ``ps``, the gateway and
``restart`` read it as before. :func:`docker_argv` wraps it when the server
starts:

- the image is pinned by digest; llmctl never pulls;
- GPUs come through CDI (``--device nvidia.com/gpu=N``) when the host has
  CDI specs for them, else ``--gpus``;
- the model's folder is mounted read-only at ``/model``, and ``MODELS_DIR``
  at ``/models`` (drafters, other files). The fixed ``/model`` path keeps
  vLLM's compile-cache key the same on every host and for every version
  folder, since vLLM hashes the model path into it;
- the port is published on the bind host only, and the engine listens on
  ``0.0.0.0`` inside the container;
- a per-model writable cache is mounted at ``/cache`` (see :func:`prepare_cache`).

``docker run`` stays attached, so the tracked PID, the log file, ``stop``
and ``--wait`` work exactly as for host processes.
"""
from __future__ import annotations

import functools
import json
import os
import re
import shutil
import subprocess
from importlib import resources
from pathlib import Path

from llm_runner import backends
from llm_runner.backends import LLAMA, SGLANG, VLLM

HOST = "host"
DOCKER = "docker"
RUNTIMES = (HOST, DOCKER)
# Backends that can run in a container.
CONTAINER_BACKENDS = (VLLM, SGLANG, LLAMA)
CONTAINER_MODEL = "/model"
CONTAINER_MODELS = "/models"
CONTAINER_CACHE = "/cache"
# The engine binary inside each image, when images.json doesn't say.
_DEFAULT_ENTRYPOINT = {VLLM: ["vllm"], SGLANG: ["sglang"], LLAMA: ["/app/llama-server"]}
# What the cache directories inside the container are. Every tool that
# caches compiled kernels is pointed below /cache so one export carries all.
_CACHE_ENV = {
    "HOME": f"{CONTAINER_CACHE}/home",
    "USER": "llmctl",      # torch's getpass.getuser() fails for a uid without a passwd entry
    "XDG_CACHE_HOME": f"{CONTAINER_CACHE}/xdg",
    "VLLM_CACHE_ROOT": f"{CONTAINER_CACHE}/vllm",
    "TRITON_CACHE_DIR": f"{CONTAINER_CACHE}/triton",
    "TORCHINDUCTOR_CACHE_DIR": f"{CONTAINER_CACHE}/inductor",
    "FLASHINFER_WORKSPACE_BASE": f"{CONTAINER_CACHE}/flashinfer",
}
CACHE_DESCRIPTOR = "llmctl-cache.json"


class ContainerError(Exception):
    """A container launch that can't start (bad image, no docker)."""


# ------------------------------------------------------------------ images --

@functools.cache
def expected_images() -> dict:
    """``images.json`` shipped with this llmctl: per backend, the image ref
    (pinned by digest) this version was tested with."""
    try:
        text = resources.files("llm_runner").joinpath("images.json").read_text()
        data = json.loads(text)
    except (OSError, json.JSONDecodeError, ModuleNotFoundError):
        return {}
    return data.get("images", {}) if isinstance(data, dict) else {}


def default_image(backend: str) -> str | None:
    entry = expected_images().get(backend)
    return entry.get("image") if isinstance(entry, dict) else None


def entrypoint(backend: str, image: str) -> list[str]:
    entry = expected_images().get(backend)
    if isinstance(entry, dict) and entry.get("entrypoint"):
        return list(entry["entrypoint"])
    return list(_DEFAULT_ENTRYPOINT[backend])


def is_pinned(ref: str) -> bool:
    return bool(re.search(r"@sha256:[0-9a-f]{64}$", ref))


def pinned_digest(ref: str) -> str | None:
    m = re.search(r"@(sha256:[0-9a-f]{64})$", ref)
    return m.group(1) if m else None


def docker_bin() -> str | None:
    return shutil.which("docker")


def inspect_image(ref: str) -> dict | None:
    """``docker image inspect`` of a local image, or None if it isn't there."""
    d = docker_bin()
    if d is None:
        return None
    r = subprocess.run([d, "image", "inspect", ref], capture_output=True, text=True)
    if r.returncode != 0:
        return None
    try:
        return json.loads(r.stdout)[0]
    except (json.JSONDecodeError, IndexError):
        return None


def resolve_digest(ref: str) -> str:
    """The digest ``ref`` is pinned to, checked against the local image.

    A ref pinned by digest is its own answer once the image is present. A
    tag (allowed only with ``--allow-tag``) resolves to its repo digest, or
    to the image id for a local build that was never pushed."""
    info = inspect_image(ref)
    if info is None:
        raise ContainerError(
            f"image {ref} is not on this host. llmctl never pulls: `docker pull {ref}` "
            "(or `docker load` it from the bundle) first.")
    pinned = pinned_digest(ref)
    if pinned:
        return pinned
    repo = ref.rsplit(":", 1)[0] if ":" in ref.rsplit("/", 1)[-1] else ref
    for rd in info.get("RepoDigests") or []:
        name, _, digest = rd.partition("@")
        if name.endswith(repo.split("/", 1)[-1]):
            return digest
    return info.get("Id", "")


# --------------------------------------------------------------------- GPUs --

_CDI_DIRS = (Path("/etc/cdi"), Path("/var/run/cdi"), Path("/run/cdi"))


@functools.cache
def has_cdi() -> bool:
    """Whether the host has CDI specs for NVIDIA GPUs (``nvidia-ctk cdi generate``)."""
    for d in _CDI_DIRS:
        try:
            for f in d.iterdir():
                if f.suffix in (".yaml", ".yml", ".json") and "nvidia.com/gpu" in f.read_text():
                    return True
        except OSError:
            continue
    return False


def parse_gpus(value: str | None) -> str:
    """``all`` or a comma list of indices, normalised; raises ValueError."""
    if value is None or value.strip().lower() in ("", "all"):
        return "all"
    parts = [p.strip() for p in value.split(",")]
    if not all(p.isdigit() for p in parts):
        raise ValueError(f"--gpu takes 'all' or GPU indices like 0 or 0,1, not {value!r}")
    return ",".join(str(int(p)) for p in parts)


def gpu_args(gpus: str) -> list[str]:
    if has_cdi():
        ids = ["all"] if gpus == "all" else gpus.split(",")
        return [a for i in ids for a in ("--device", f"nvidia.com/gpu={i}")]
    return ["--gpus", "all" if gpus == "all" else f'"device={gpus}"']


@functools.cache
def gpu_arch(index: int = 0) -> str | None:
    """``sm_89`` for an Ada GPU, from nvidia-smi's compute capability."""
    smi = shutil.which("nvidia-smi")
    if smi is None:
        return None
    try:
        out = subprocess.run([smi, "--query-gpu=compute_cap", "--format=csv,noheader"],
                             capture_output=True, text=True, timeout=10).stdout
    except (OSError, subprocess.TimeoutExpired):
        return None
    caps = [c.strip() for c in out.splitlines() if c.strip()]
    if index >= len(caps):
        return None
    return "sm_" + caps[index].replace(".", "")


def first_gpu(gpus: str) -> int:
    return 0 if gpus == "all" else int(gpus.split(",")[0])


# ------------------------------------------------------------------ runtime --

def runtime_from_config(backend: str, cfg: dict) -> dict:
    """The configured runtime for ``backend`` (``config.json`` ``runtime``)."""
    entry = (cfg.get("runtime") or {}).get(backend) or {}
    kind = entry.get("kind", HOST)
    out = {"kind": kind}
    if kind == DOCKER:
        out["image"] = entry.get("image") or default_image(backend)
        if entry.get("network"):
            out["network"] = entry["network"]
    return out


def container_name(name: str, port: int) -> str:
    safe = re.sub(r"[^A-Za-z0-9_.-]", "-", name).strip("-.") or "model"
    return f"llmctl-{safe}-{port}"


def _sub_path(arg: str, prefix: str, repl: str) -> str:
    # A path prefix only where it ends a path component: /m/qwen must not
    # rewrite /m/qwen-2.
    return re.sub(re.escape(prefix) + r"(?=$|/|[\"',\s}\]])", repl, arg)


def container_cmd(cmd: list[str], *, backend: str, image: str, model_path: str,
                  models_dir: Path) -> list[str]:
    """The engine argv as it runs inside the container."""
    model = Path(model_path)
    model_dir = model if model.is_dir() else model.parent
    inner = backends.with_bind_host(cmd, backend, "0.0.0.0")
    out = entrypoint(backend, image)[1:] + inner[1:]
    out = [_sub_path(a, str(model_dir), CONTAINER_MODEL) for a in out]
    if _mounts_models_dir(models_dir, model_dir):
        out = [_sub_path(a, str(models_dir), CONTAINER_MODELS) for a in out]
    return out


def _mounts_models_dir(models_dir: Path, model_dir: Path) -> bool:
    return models_dir.is_dir() and models_dir != model_dir


def bind_host_of(cmd: list[str]) -> str:
    if "--host" in cmd[:-1]:
        return cmd[cmd.index("--host") + 1]
    return "0.0.0.0"


def docker_argv(*, name: str, cmd: list[str], port: int, backend: str,
                model_path: str, env: dict[str, str], runtime: dict,
                models_dir: Path, cache_dir: Path | None) -> list[str]:
    """The ``docker run`` argv for one model server."""
    d = docker_bin() or "docker"
    image = runtime["image"]
    model = Path(model_path)
    model_dir = model if model.is_dir() else model.parent
    host = bind_host_of(cmd)
    # --pull never: a restore after the image was removed fails loudly
    # instead of fetching something from the network.
    argv = [d, "run", "--rm", "--init", "--pull", "never",
            "--name", runtime["container"],
            "--label", f"io.llmctl.name={name}",
            "--label", f"io.llmctl.port={port}",
            "--user", f"{os.getuid()}:{os.getgid()}",
            "--network", runtime.get("network") or "bridge",
            "-p", f"{host}:{port}:{port}" if host not in ("0.0.0.0", "") else f"{port}:{port}",
            "-v", f"{model_dir}:{CONTAINER_MODEL}:ro"]
    if _mounts_models_dir(models_dir, model_dir):
        argv += ["-v", f"{models_dir}:{CONTAINER_MODELS}:ro"]
    if cache_dir is not None:
        argv += ["-v", f"{cache_dir}:{CONTAINER_CACHE}"]
    if backend in (VLLM, SGLANG):
        # Worker processes share tensors through /dev/shm.
        argv += ["--shm-size", "8g"]
    argv += gpu_args(runtime.get("gpus") or "all")
    for k, v in env.items():
        argv += ["-e", f"{k}={v}"]
    ep = entrypoint(backend, image)
    argv += ["--entrypoint", ep[0], image]
    argv += container_cmd(cmd, backend=backend, image=image, model_path=model_path,
                          models_dir=models_dir)
    return argv


def container_env(offline: bool, backend: str, env: dict[str, str],
                  with_cache: bool) -> dict[str, str]:
    """Environment inside the container. No host toolchain: the image brings
    its own compiler."""
    base = {**backends.NO_TELEMETRY_ENV, **(backends.OFFLINE_ENV if offline else {}),
            **(backends.SGLANG_ENV if backend == SGLANG else {})}
    if with_cache:
        base.update(_CACHE_ENV)
    return {**base, **env}


def wait_for_docker(timeout: float) -> bool:
    """Whether ``docker info`` answers within ``timeout`` seconds."""
    import time
    d = docker_bin()
    if d is None:
        return False
    deadline = time.time() + timeout
    while True:
        if subprocess.run([d, "info"], capture_output=True).returncode == 0:
            return True
        if time.time() >= deadline:
            return False
        time.sleep(2)


def stop_container(name: str, timeout: int = 20) -> None:
    """``docker stop`` then ``rm -f``; quiet if it's already gone."""
    d = docker_bin()
    if d is None:
        return
    subprocess.run([d, "stop", "-t", str(timeout), name], capture_output=True)
    subprocess.run([d, "rm", "-f", name], capture_output=True)


# -------------------------------------------------------------------- cache --

def cache_root(state_dir: Path) -> Path:
    return state_dir / "cache"


def engine_version(backend: str) -> str | None:
    """The host engine's version, read from its venv's package metadata
    (importing vllm to ask takes seconds)."""
    binary = backends.find_engine(backend) if backend in (VLLM, SGLANG) else None
    if not binary:
        return None
    venv = Path(binary).resolve().parent.parent
    for info in venv.glob(f"lib/python*/site-packages/{backend}-*.dist-info"):
        m = re.match(rf"{backend}-(.+)\.dist-info", info.name)
        if m:
            return m.group(1)
    return None


def shipped_caches(model_path: str) -> tuple[Path, list[dict]]:
    """The caches an installed model folder ships (from its sidecar)."""
    from llm_runner.bundle import read_source
    model = Path(model_path)
    folder = model if model.is_dir() else model.parent
    caches = read_source(folder).get("caches") or []
    return folder, [c for c in caches if isinstance(c, dict)]


def match_cache(caches: list[dict], *, kind: str, digest: str | None,
                engine_ver: str | None, arch: str | None) -> tuple[dict | None, str]:
    """The shipped cache to seed from and the rule that matched.

    ``digest`` (container runtime) is the only rule a container accepts:
    the image digest and GPU arch must both match. ``engine_version`` is
    the host runtime's fallback, for development only (it doesn't pin torch
    or CUDA)."""
    for c in caches:
        if c.get("kind") != kind or (arch and c.get("gpu_arch") != arch):
            continue
        if digest is not None:
            if pinned_digest(c.get("image") or "") == digest:
                return c, "digest"
        elif engine_ver and c.get("engine_version") == engine_ver:
            return c, "engine_version"
    return None, "none"


def _descriptor(*, backend: str, runtime: dict, arch: str | None, ver: str | None,
                model_folder: str) -> dict:
    docker = runtime.get("kind") == DOCKER
    digest = runtime.get("digest") or ""
    return {"kind": backend, "path": None, "engine": backend,
            "engine_version": runtime.get("engine_version") if docker else ver,
            "image": f"{runtime['image'].split('@')[0]}@{digest}"
            if docker and digest.startswith("sha256:") else None,
            "gpu_arch": arch, "model_folder": model_folder}


def prepare_cache(*, state_dir: Path, name: str, model_path: str, backend: str,
                  runtime: dict) -> tuple[Path | None, str, str]:
    """The writable cache dir for this launch, seeded from the model's
    shipped cache when one matches. Returns ``(dir, rule, log line)``.

    Containers always get one (per model name, folder and image digest).
    Host vllm runs get one only when the model ships a cache; otherwise they
    keep the engines' shared defaults (``~/.cache``) as before. The dir
    holds a ``llmctl-cache.json`` saying what built it, which ``cache
    export`` turns into the manifest's ``caches`` entry."""
    docker = runtime.get("kind") == DOCKER
    # llama-server compiles nothing at start; vllm and sglang JIT and cache.
    if backend == LLAMA or (backend != VLLM and not docker):
        return None, "none", ""
    folder, caches = shipped_caches(model_path)
    if not docker and not caches:
        return None, "none", ""
    gpus = runtime.get("gpus") or "all"
    arch = gpu_arch(first_gpu(gpus))
    digest = runtime.get("digest") if docker else None
    ver = None if docker else engine_version(backend)
    tag = (digest or "").removeprefix("sha256:")[:12] if docker else f"host-{ver or 'unknown'}"
    target = cache_root(state_dir) / re.sub(r"[^A-Za-z0-9_.-]", "-", name) / f"{folder.name}-{tag}"
    chosen, rule = (match_cache(caches, kind="vllm", digest=digest, engine_ver=ver, arch=arch)
                    if backend == VLLM else (None, "none"))
    descriptor = _descriptor(backend=backend, runtime=runtime, arch=arch, ver=ver,
                             model_folder=folder.name)
    if target.is_dir() and any(p.name != CACHE_DESCRIPTOR for p in target.iterdir()):
        line = f"cache: reusing {target}"
    elif chosen is None:
        target.mkdir(parents=True, exist_ok=True)
        why = ("no shipped cache" if not caches else
               "shipped cache built for " + ", ".join(
                   f"{c.get('image') or c.get('engine_version')} {c.get('gpu_arch')}"
                   for c in caches))
        line = f"cache: cold start in {target} ({why}; this run: {digest or ver} {arch})"
    else:
        src = folder / chosen["path"]
        tmp = target.with_name(target.name + ".seeding")
        shutil.rmtree(tmp, ignore_errors=True)
        shutil.copytree(src, tmp, ignore=shutil.ignore_patterns(CACHE_DESCRIPTOR))
        shutil.rmtree(target, ignore_errors=True)
        os.replace(tmp, target)
        line = f"cache: seeded {target} from {src}"
    (target / CACHE_DESCRIPTOR).write_text(json.dumps(descriptor, indent=2) + "\n")
    label = "engine_version (dev fallback)" if rule == "engine_version" else rule
    return target, rule, f"{line}; cache rule: {label}"


def find_cache_dirs(state_dir: Path, name: str) -> list[Path]:
    base = cache_root(state_dir) / re.sub(r"[^A-Za-z0-9_.-]", "-", name)
    return sorted(p.parent for p in base.glob(f"*/{CACHE_DESCRIPTOR}"))


def export_cache(cache_dir: Path, out: Path, *, path_in_bundle: str) -> dict:
    """Copy a warm cache to ``out`` (a subtree of a bundle) with its
    descriptor. Returns the manifest ``caches`` entry."""
    try:
        entry = json.loads((cache_dir / CACHE_DESCRIPTOR).read_text())
    except (OSError, json.JSONDecodeError):
        raise ContainerError(f"{cache_dir} has no {CACHE_DESCRIPTOR}; it isn't a cache "
                             "llmctl prepared") from None
    if not any(p.name != CACHE_DESCRIPTOR for p in cache_dir.iterdir()):
        raise ContainerError(f"{cache_dir} is empty; start the model once so it compiles")
    if out.exists() and any(out.iterdir()):
        raise ContainerError(f"{out} is not empty")
    entry["path"] = path_in_bundle
    entry.pop("model_folder", None)
    out.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(cache_dir, out, dirs_exist_ok=True)
    (out / CACHE_DESCRIPTOR).write_text(json.dumps(entry, indent=2) + "\n")
    return entry

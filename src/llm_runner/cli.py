"""CLI for managing background model servers (llama-server / vllm / sglang)."""
from __future__ import annotations

import contextlib
import datetime as dt
import getpass
import json
import os
import re
import shlex
import shutil
import socket
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

import click

from llm_runner import __version__
from llm_runner import backends
from llm_runner import bundle
from llm_runner import containers
from llm_runner import images
from llm_runner import config as config_mod
from llm_runner import process as proc
from llm_runner import registry
from llm_runner.backends import DECIDER, GATEWAY, HF_ENGINES, LLAMA, SGLANG, VLLM
from llm_runner.registry import MODELS_DIR, ModelSpec, discover_models


_port_in_use = proc.port_in_use


def _next_free_port(start: int = 8080) -> int:
    p = start
    while _port_in_use(p):
        p += 1
    return p


def _build_llama_cmd(model_path: Path, mmproj: Path | None, port: int, ctx_size: int,
                     reasoning: bool, *, kv_quant: str = "f16",
                     n_gpu_layers: int = 999, embeddings: bool = False,
                     pooling: str = "cls", batch_size: int | None = None,
                     ubatch_size: int | None = None, draft_path: Path | None = None,
                     spec_type: str = "draft-mtp", spec_n_max: int = 3,
                     n_parallel: int | None = None,
                     cache_ram: int = 0,
                     kv_unified: bool | None = None,
                     host: str = "0.0.0.0",
                     extra: list[str] | None = None) -> list[str]:
    if embeddings:
        # Embedding models (e.g. BERT-style bge-m3) take no chat/generation
        # flags: flash-attn isn't supported, and the prompt-cache / reasoning
        # options are meaningless. Keep it to the embedding essentials.
        # Batch/ubatch default to the context size so a full-length input
        # embeds in a single pass (bge-m3 uses cls pooling).
        batch = batch_size or ctx_size
        ubatch = ubatch_size or ctx_size
        return [
            "llama-server",
            "--model", str(model_path),
            "--n-gpu-layers", str(n_gpu_layers),
            "--ctx-size", str(ctx_size),
            "--batch-size", str(batch),
            "--ubatch-size", str(ubatch),
            "--threads", "8",
            "--port", str(port),
            "--host", host,
            "--embeddings",
            "--pooling", pooling,
            "--metrics",
            *(extra or []),
        ]
    cmd = [
        "llama-server",
        "--model", str(model_path),
        "--n-gpu-layers", str(n_gpu_layers),
        "--ctx-size", str(ctx_size),
        "--flash-attn", "on",
        "--cache-reuse", "256",
        # Host-RAM prompt cache. Off by default: every call saves its state
        # (~75 MB for a short prompt on Gemma 26B), so short, distinct prompts
        # pay ~+33% per call on the RTX (llama.cpp b9780; the enersec v2
        # pipeline once paid +147%) and the cache fills its RAM limit. It pays
        # off when long conversations take turns on a slot: see --cache-ram
        # and the `multi` launch profile.
        "--cache-ram", str(cache_ram),
        "--threads", "8",
        "--port", str(port),
        "--host", host,
        # Prometheus /metrics, re-exported by the gateway (/metrics/backends/…).
        "--metrics",
    ]
    if n_parallel is not None:
        cmd.extend(["--parallel", str(n_parallel)])
    # A non-causal model (vllm-d3) takes each prompt in one physical batch.
    if batch_size is not None:
        cmd.extend(["--batch-size", str(batch_size)])
    if ubatch_size is not None:
        cmd.extend(["--ubatch-size", str(ubatch_size)])
    # With an explicit --parallel N llama-server splits the context N ways per
    # slot (32k/4 = 8k each) and refuses a longer prompt. Unified KV lets the
    # slots share the whole buffer instead: N short requests batch together
    # and a lone long one can still use all of it. Default: on for N > 1.
    if kv_unified is None:
        kv_unified = bool(n_parallel and n_parallel > 1)
    if kv_unified:
        cmd.append("--kv-unified")
    # f16 is llama.cpp's default; only override when quantizing the KV cache.
    # q8_0/q4_0 for the V cache require flash-attn, which is always on above.
    if kv_quant != "f16":
        cmd.extend(["--cache-type-k", kv_quant, "--cache-type-v", kv_quant])
    if not reasoning:
        cmd.extend(["--reasoning-budget", "0"])
    if mmproj and mmproj.exists():
        cmd.extend(["--mmproj", str(mmproj)])
    # Speculative decoding: pair the base model with a drafter (e.g. a Gemma-4
    # MTP head). llama-server runs the drafter to propose up to spec_n_max
    # tokens per step and the base model verifies them in one pass.
    if draft_path and draft_path.exists():
        cmd.extend([
            "--model-draft", str(draft_path),
            "--spec-type", spec_type,
            "--spec-draft-n-max", str(spec_n_max),
        ])
    if extra:
        cmd.extend(extra)
    return cmd


def _resolve_draft(draft_spec: str | None, auto_spec: ModelSpec | None,
                   backend: str = LLAMA) -> Path | None:
    """Resolve the speculative drafter (llama ``--model-draft`` / vllm
    ``--speculative-config`` / sglang ``--speculative-draft-model-path``).

    ``draft_spec`` (from ``--draft``) may be a path (GGUF file, or checkpoint
    dir for vllm/sglang), a registry key, or — vllm/sglang only — a Hugging
    Face repo id the engine fetches itself. When it is omitted, fall back to
    the drafter auto-attached to ``auto_spec`` (a sibling ``*-MTP`` /
    ``*-assistant`` GGUF), if any — llama only, since that file is a GGUF.
    """
    hf = backend in HF_ENGINES
    if draft_spec:
        p = Path(draft_spec).expanduser()
        if p.is_file() or (hf and p.is_dir()):
            return p.resolve()
        specs = discover_models()
        if draft_spec in specs:
            cand = specs[draft_spec]
            # The registry names an engine per format; the format must match.
            if (cand.backend in HF_ENGINES) != hf:
                raise click.BadArgumentUsage(
                    f"--draft '{draft_spec}' is a {cand.backend} model; the "
                    f"drafter must match the {backend} backend.")
            return cand.model_path
        if hf and re.fullmatch(r"[\w.-]+/[\w.-]+", draft_spec):
            return Path(draft_spec)  # HF repo id; the engine downloads it
        raise click.BadArgumentUsage(
            f"--draft '{draft_spec}' is neither an existing model file/dir nor a "
            "known model key."
        )
    if hf:
        return None
    return auto_spec.draft_path if auto_spec else None


def _run_option_args(ctx: click.Context) -> list[str]:
    """Rebuild the option part of the `llmctl run` argv from the params the
    user passed on the command line (defaults are omitted so re-detection —
    e.g. embeddings-by-filename — still applies on a re-run). ``--port`` is
    left out: the manifest pins it separately. So are ``--wait``, ``--menu``
    and ``--profile``: they're about this invocation, not the launch (a
    profile's values are recorded as the flags they set)."""
    src = click.core.ParameterSource.COMMANDLINE
    args: list[str] = []
    for p in ctx.command.params:
        if not isinstance(p, click.Option) or p.name in ("port", "wait", "menu", "profile"):
            continue
        if ctx.get_parameter_source(p.name) != src:
            continue
        val = ctx.params[p.name]
        if p.is_flag:
            if p.secondary_opts:  # --x/--no-x pair: emit whichever side was chosen
                args.append(p.opts[0] if val else p.secondary_opts[0])
            elif val:
                args.append(p.opts[0])
        elif p.multiple:
            for v in val:
                args += [p.opts[0], str(v)]
        elif val is not None:
            args += [p.opts[0], str(val)]
    return args


def _resolve_runtime(backend: str, kind: str | None, image: str | None,
                     allow_tag: bool, gpu: str | None, dns: str | None = None) -> dict:
    """The runtime for one launch: the flags, else ``config runtime``."""
    try:
        gpus = containers.parse_gpus(gpu)
    except ValueError as e:
        raise click.BadOptionUsage("--gpu", str(e)) from None
    try:
        dns_ips = None if dns is None else containers.parse_dns(dns)
    except ValueError as e:
        raise click.BadOptionUsage("--dns", str(e)) from None
    configured = containers.runtime_from_config(backend, config_mod.load())
    if image and kind == containers.HOST:
        raise click.UsageError("--image needs --runtime docker.")
    kind = kind or (containers.DOCKER if image else configured["kind"])
    runtime: dict = {"kind": kind, "gpus": gpus}
    if kind != containers.DOCKER:
        if dns_ips:
            raise click.UsageError("--dns only applies to --runtime docker.")
        return runtime
    if backend not in containers.CONTAINER_BACKENDS:
        raise click.UsageError(f"{backend} has no container runtime; use --runtime host.")
    if containers.docker_bin() is None:
        raise click.UsageError("--runtime docker needs the docker CLI on PATH.")
    ref = image or (configured.get("image") if configured["kind"] == containers.DOCKER
                    else None) or containers.default_image(backend)
    if not ref:
        raise click.UsageError(f"No image for {backend}: pass --image repo@sha256:… or "
                               f"`llmctl config runtime {backend} docker --image …`.")
    if not containers.is_pinned(ref) and not allow_tag:
        raise click.UsageError(f"{ref} is not pinned by digest. Use repo@sha256:…, or "
                               "--allow-tag for development.")
    try:
        digest = containers.resolve_digest(ref, proc.STATE_DIR)
    except containers.ContainerError as e:
        raise click.UsageError(str(e)) from None
    runtime.update(image=ref, digest=digest)
    if configured.get("network"):
        runtime["network"] = configured["network"]
    # --dns '' drops a configured list for this launch.
    if dns_ips if dns_ips is not None else configured.get("dns"):
        runtime["dns"] = dns_ips if dns_ips is not None else configured["dns"]
    expected = containers.expected_images().get(backend) or {}
    if expected.get("image") == ref and expected.get("engine_version"):
        runtime["engine_version"] = expected["engine_version"]
    return runtime


def _require_backend(backend: str) -> str:
    """Return the server binary for ``backend`` or fail with an install hint."""
    binary = backends.find(backend)
    if binary is None:
        raise click.UsageError(backends.missing_hint(backend))
    return binary


# `run` options that only mean something to llama-server, or only to the HF
# engines (vllm, sglang). Passing one to the other side is almost certainly
# a mistake (or a stale `restart` override), so warn instead of silently
# dropping it.
_LLAMA_ONLY_PARAMS = {
    "reasoning": "--reasoning", "n_gpu_layers": "--n-gpu-layers",
    "no_vision": "--no-vision", "kv_unified": "--kv-unified/--no-kv-unified",
    "cache_ram": "--cache-ram", "ctx_checkpoints": "--ctx-checkpoints",
    "pooling": "--pooling",
    "batch_size": "--batch-size", "ubatch_size": "--ubatch-size",
}
_HF_ENGINE_PARAMS = {
    "gpu_mem": "--gpu-mem", "tool_parser": "--tool-parser",
    "reasoning_parser": "--reasoning-parser",
    "trust_remote_code": "--trust-remote-code",
}


def _warn_foreign_flags(ctx: click.Context, backend: str) -> None:
    src = click.core.ParameterSource.COMMANDLINE
    hf = "vllm/sglang"
    if backend == DECIDER:
        # decider takes only --port / --n-parallel / -X / -E; everything
        # engine-specific is meaningless there.
        groups = {LLAMA: _LLAMA_ONLY_PARAMS, hf: _HF_ENGINE_PARAMS}
    else:
        groups = {hf: _HF_ENGINE_PARAMS} if backend == LLAMA else {LLAMA: _LLAMA_ONLY_PARAMS}
    for other, foreign in groups.items():
        used = [flag for name, flag in foreign.items()
                if ctx.get_parameter_source(name) == src]
        if used:
            click.echo(f"warning: {other}-only option(s) ignored for {backend}: "
                       f"{', '.join(used)}", err=True)


def _log_tail(path: str, lines: int = 25) -> str:
    """The last ``lines`` lines of a log, indented; empty if unreadable."""
    try:
        with open(path, "rb") as f:
            f.seek(0, os.SEEK_END)
            f.seek(max(f.tell() - 64 * 1024, 0))
            tail = f.read().decode("utf-8", "replace").splitlines()[-lines:]
    except OSError:
        return ""
    return "\n".join("  " + line for line in tail)


def _wait_healthy(rec: proc.RunningProcess, timeout: float) -> bool:
    """Poll ``/health`` until the server answers 200, the process dies, or
    ``timeout`` elapses. llama-server, vllm and sglang all expose it."""
    url = config_mod.local_url(rec.port, "/health")
    deadline = time.time() + timeout
    spinner = "|/-\\"
    i = 0
    while time.time() < deadline:
        if not proc._alive(rec.pid):
            try:
                os.waitpid(rec.pid, os.WNOHANG)      # it's our child: reap it
            except ChildProcessError:
                pass
            click.echo(f"\rServer exited before becoming healthy. Last lines of "
                       f"{rec.log_file}:", err=True)
            click.echo(_log_tail(rec.log_file), err=True)
            return False
        try:
            with urllib.request.urlopen(url, timeout=2) as resp:
                if resp.status == 200:
                    click.echo("\rReady.                    ")
                    return True
        except (urllib.error.URLError, OSError, TimeoutError):
            pass
        click.echo(f"\rWaiting for {url} {spinner[i % 4]} ", nl=False)
        i += 1
        time.sleep(1.0)
    click.echo(f"\rStill not healthy after {timeout:.0f}s; it may just be slow "
               f"to load — check `llmctl logs {rec.name}`.", err=True)
    return False


@click.group()
@click.version_option(__version__, "--version", "-V", prog_name="llmctl")
def cli() -> None:
    """Manage local model servers (llama-server for GGUF, vllm for HF
    checkpoints)."""


@cli.command()
def version() -> None:
    """Print the llmctl version."""
    click.echo(f"llmctl {__version__}")


@cli.group()
def config() -> None:
    """View and edit persisted configuration."""


@config.command("show")
def config_show() -> None:
    """Show the config file, stored settings, and the effective models dir."""
    stored = config_mod.load()
    exists = config_mod.CONFIG_FILE.exists()
    click.echo(f"Config file: {config_mod.CONFIG_FILE}"
               + ("" if exists else "  (not created yet)"))
    if stored:
        click.echo("\nStored settings:")
        for key, val in stored.items():
            if key in ("cluster_token", "gateway_upstream_key"):
                val = "(set; `llmctl registry token` prints the token)" \
                    if key == "cluster_token" else "(set)"
            elif key == "gateway_keys" and isinstance(val, dict):
                val = ", ".join(sorted(val))
            click.echo(f"  {key} = {val}")
    else:
        click.echo("\n(no settings stored)")
    click.echo(f"\nEffective models dir: {registry._resolve_models_dir()}")
    if os.environ.get("LLM_RUNNER_MODELS_DIR"):
        click.echo("  (overridden by $LLM_RUNNER_MODELS_DIR)")
    click.echo("\nBackends:")
    for backend in backends.BACKENDS:
        found = backends.find(backend)
        click.echo(f"  {backend:<6} {found or '(not found)'}")
    for engine in HF_ENGINES:
        var = backends.bin_env_var(engine)
        if os.environ.get(var):
            click.echo(f"  ({engine} overridden by ${var})")
    if os.environ.get("LLM_RUNNER_DECIDER_DIR"):
        click.echo("  (decider overridden by $LLM_RUNNER_DECIDER_DIR)")
    tc = backends.toolchain_env()
    click.echo("\nCUDA toolchain (vllm/sglang kernel JIT):")
    if not tc:
        click.echo("  (nvcc not found)")
    else:
        limit = backends.nvcc_max_gcc(Path(tc["CUDA_HOME"]))
        click.echo(f"  CUDA_HOME={tc['CUDA_HOME']}"
                   + (f"  (nvcc takes GCC ≤ {limit})" if limit else ""))
        click.echo(f"  CC={tc['CC']}  CXX={tc['CXX']}" if "CC" in tc else
                   "  (no system GCC nvcc accepts — the inherited compilers apply)")
    click.echo("\nRuntimes (`llmctl config runtime`):")
    for backend in containers.CONTAINER_BACKENDS:
        rt = containers.runtime_from_config(backend, stored)
        line = rt["kind"] + (f"  {rt.get('image')}" if rt["kind"] == containers.DOCKER else "")
        click.echo(f"  {backend:<6} {line}")
    click.echo(f"  GPUs via {'CDI' if containers.has_cdi() else '--gpus'}")
    gw_port = config_mod.get_gateway_port()
    click.echo(f"\nGateway: {'off' if gw_port is None else f'port {gw_port}'}")


@config.command("runtime")
@click.argument("backend", required=False,
                type=click.Choice(list(containers.CONTAINER_BACKENDS)))
@click.argument("kind", required=False, type=click.Choice(list(containers.RUNTIMES)))
@click.option("--image", default=None, metavar="REF",
              help="Image for docker, pinned by digest (repo@sha256:…). Default: the "
                   "one this llmctl was tested with (`llmctl images`).")
@click.option("--network", default=None,
              help="Docker network the containers join (default: bridge).")
@click.option("--dns", default=None, metavar="IP[,IP…]",
              help="DNS servers for the containers (docker --dns; default: Docker's). "
                   "On a network without egress, 127.0.0.1 keeps container names "
                   "resolving and stops outside names.")
@click.option("--allow-tag", is_flag=True, default=False,
              help="Accept an image by tag (development only).")
def config_runtime(backend: str | None, kind: str | None, image: str | None,
                   network: str | None, dns: str | None, allow_tag: bool) -> None:
    """Run BACKEND's servers as host processes or Docker containers.

    \b
      llmctl config runtime                       # show
      llmctl config runtime vllm docker           # the tested image
      llmctl config runtime vllm docker --image vllm/vllm-openai@sha256:…
      llmctl config runtime vllm docker --network ss6m --dns 127.0.0.1
      llmctl config runtime vllm host

    Applies to new starts; `run --runtime` overrides it per launch. Running
    servers keep the runtime they started with."""
    cfg = config_mod.load()
    if backend is None:
        for b in containers.CONTAINER_BACKENDS:
            rt = containers.runtime_from_config(b, cfg)
            click.echo(f"{b:<6} {rt['kind']}"
                       + (f"  {rt.get('image')}" if rt["kind"] == containers.DOCKER else "")
                       + (f"  network={rt['network']}" if rt.get("network") else "")
                       + (f"  dns={','.join(rt['dns'])}" if rt.get("dns") else ""))
        return
    if kind is None:
        raise click.UsageError("Give the runtime: host or docker.")
    entry: dict = {"kind": kind}
    if kind == containers.DOCKER:
        ref = image or containers.default_image(backend)
        if not ref:
            raise click.UsageError(f"No tested image for {backend}; pass --image.")
        if not containers.is_pinned(ref) and not allow_tag:
            raise click.UsageError(f"{ref} is not pinned by digest (repo@sha256:…); "
                                   "--allow-tag accepts it for development.")
        entry["image"] = ref
        if network:
            entry["network"] = network
        try:
            if dns_ips := containers.parse_dns(dns):
                entry["dns"] = dns_ips
        except ValueError as e:
            raise click.BadOptionUsage("--dns", str(e)) from None
        if containers.inspect_image(ref) is None:
            click.echo(f"note: {ref} is not on this host yet; `docker pull` or "
                       "`docker load` it before starting a model.", err=True)
    elif image or network or dns:
        raise click.UsageError("--image, --network and --dns only apply to docker.")
    runtimes = dict(cfg.get("runtime") or {})
    runtimes[backend] = entry
    config_mod.update(runtime=runtimes)
    click.echo(f"{backend}: {kind}" + (f"  {entry['image']}" if "image" in entry else ""))


@cli.group("images", invoke_without_command=True)
@click.option("--json", "as_json", is_flag=True, help="Print images.json as shipped.")
@click.option("--protect", is_flag=True,
              help="Tag present images that have no tag, so `docker image prune` "
                   "doesn't delete them.")
@click.pass_context
def images_cmd(ctx: click.Context, as_json: bool, protect: bool) -> None:
    """The container images this llmctl version was tested with, and images
    loaded from archives.

    An image pulled by digest or loaded from an archive has no tag, and
    `docker image prune` deletes such images; --protect tags them
    llmctl/<repo>:<digest>. llmctl never runs an image by that tag."""
    if ctx.invoked_subcommand is not None:
        return
    imgs = containers.expected_images()
    if as_json:
        click.echo(json.dumps({"schema": "llmctl-images/1", "llmctl": __version__,
                               "images": imgs}, indent=2))
        return
    refs = {e["image"]: (backend, e.get("tag", "")) for backend, e in imgs.items()}
    for ref in images.loaded(proc.STATE_DIR):
        refs.setdefault(ref, ("loaded", ""))
    unprotected = 0
    for ref, (label, tag) in refs.items():
        found = images.local_image(ref, proc.STATE_DIR)
        if found is None:
            state = "not on this host"
        else:
            state = "present" + (f", loaded as {found[1][7:19]}" if found[0] != ref else "")
            if images.dangling(ref, proc.STATE_DIR):
                if protect:
                    state += f", now tagged {images.protect(ref, found[1])}"
                else:
                    state += ", no tag (prune would delete it)"
                    unprotected += 1
        click.echo(f"{label:<6} {ref}\n       {tag + '  ' if tag else ''}{state}")
    if unprotected:
        click.echo(f"\n{unprotected} image(s) have no tag; `llmctl images --protect` "
                   "keeps `docker image prune` from deleting them.", err=True)


@images_cmd.command("load")
@click.argument("archive", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option("--json", "as_json", is_flag=True, default=False,
              help="Print one JSON result on stdout; messages go to stderr.")
def images_load(archive: Path, as_json: bool) -> None:
    """Load a `docker save` archive so llmctl can run it offline.

    Before loading, the archive's image is checked against the registry
    chain: the one `llmctl images save` put in the archive, or for a plain
    `docker save` archive, the config digests in images.json. After `docker
    load`, the image's ID must be the archive's config digest (classic
    store) or its manifest digest (containerd store). llmctl records the ID,
    runs the pinned ref by it, and tags the image so prune leaves it.

    \b
    Exit codes: 0 loaded, 3 not a docker save archive or no chain names its
    image, 4 a digest doesn't match, 6 docker unavailable, 1 anything else.
    """
    if not as_json:
        click.echo(f"Checking {archive} ...", err=True)
    try:
        result = images.load(archive, proc.STATE_DIR)
    except bundle.BundleError as e:
        _bundle_fail(e, as_json)
        return
    if as_json:
        click.echo(json.dumps(result, indent=2))
        return
    click.echo(f"Loaded {result['ref']}\n  {result['platform']}, image {result['image_id']} "
               f"(checked through {result['verified_by']}), tagged {result['tag']}")
    if result["platform"] != result["host_platform"]:
        click.echo(f"warning: this host is {result['host_platform']}; the image won't run here.",
                   err=True)


@images_cmd.command("save")
@click.argument("image")
@click.option("-o", "--output", type=click.Path(dir_okay=False, path_type=Path), required=True,
              help="Archive to write.")
@click.option("--platform", "plat", default=None,
              help="Platform of the local image (default: this host's, e.g. linux/amd64).")
@click.option("--json", "as_json", is_flag=True, default=False,
              help="Print one JSON result on stdout; messages go to stderr.")
def images_save(image: str, output: Path, plat: str | None, as_json: bool) -> None:
    """Write IMAGE (repo@sha256:… or a backend name) to an archive that
    `llmctl images load` checks offline.

    It's a `docker save` archive with one more file, llmctl-image.json,
    holding the registry's index and manifest for the image (read with
    skopeo, which must be installed). Plain `docker load` still reads it.

    \b
    Exit codes: 0 written, 3 not pinned by digest, 4 the local image isn't
    the registry's, 6 not on this host / skopeo missing / registry
    unreachable, 1 anything else.
    """
    ref = containers.default_image(image) if image in containers.CONTAINER_BACKENDS else image
    if not ref:
        raise click.BadArgumentUsage(f"No shipped image for {image}.")
    if not as_json:
        click.echo(f"Saving {ref} ...", err=True)
    try:
        result = images.save(ref, output, proc.STATE_DIR, plat)
    except bundle.BundleError as e:
        _bundle_fail(e, as_json)
        return
    if as_json:
        click.echo(json.dumps(result, indent=2))
        return
    click.echo(f"Wrote {result['file']} ({_fmt_size(result['size']).strip()}): {ref}, "
               f"{result['platform']}, config {result['config']}")


def _engine_bin_command(engine: str) -> None:
    """Register `llmctl config <engine>-bin`."""
    @config.command(f"{engine}-bin")
    @click.argument("path", required=False,
                    type=click.Path(dir_okay=False, path_type=Path))
    @click.option("--clear", is_flag=True, help="Forget the stored path.")
    def command(path: Path | None, clear: bool) -> None:
        if clear:
            config_mod.set_engine_bin(engine, None)
            click.echo(f"{engine}_bin cleared")
            return
        if path is None:
            configured = config_mod.get_engine_bin(engine)
            if configured:
                click.echo(str(configured))
            else:
                found = backends.find_engine(engine)
                click.echo(f"(not set; {'auto-detected ' + found if found else engine + ' not found'})")
            return
        if not path.expanduser().is_file():
            click.echo(f"warning: {path} does not exist", err=True)
        saved = config_mod.set_engine_bin(engine, path)
        click.echo(f"{engine}_bin set to {saved}")
        var = backends.bin_env_var(engine)
        if os.environ.get(var):
            click.echo(f"note: ${var} is set and overrides this.", err=True)

    command.help = (
        f"Persist the path to the `{engine}` CLI, or print it when PATH is omitted.\n\n"
        f"Only needed when {engine} lives somewhere llmctl doesn't look: PATH, the "
        f"venv `uv tool install {engine}` made, or ~/.venvs/{engine}. Example:\n\n"
        f"\b\n  llmctl config {engine}-bin ~/envs/{engine}/bin/{engine}")


for _engine in HF_ENGINES:
    _engine_bin_command(_engine)


@config.command("decider-dir")
@click.argument("path", required=False,
                type=click.Path(file_okay=False, path_type=Path))
@click.option("--clear", is_flag=True, help="Forget the stored path.")
def config_decider_dir(path: Path | None, clear: bool) -> None:
    """Persist the Mapika/decider clone dir (its .venv*/bin/uvicorn serves
    decider checkpoints), or print it when PATH is omitted. Example:

      llmctl config decider-dir /fast/ml/jev/decider
    """
    if clear:
        config_mod.set_decider_dir(None)
        click.echo("decider_dir cleared")
        return
    if path is None:
        configured = config_mod.get_decider_dir()
        click.echo(str(configured) if configured else "(not set)")
        return
    resolved = Path(path).expanduser().resolve()
    if not (resolved / "decider" / "serve.py").is_file():
        click.echo(f"warning: {resolved}/decider/serve.py does not exist", err=True)
    saved = config_mod.set_decider_dir(resolved)
    click.echo(f"decider_dir set to {saved}")
    if os.environ.get("LLM_RUNNER_DECIDER_DIR"):
        click.echo("note: $LLM_RUNNER_DECIDER_DIR is set and overrides this.", err=True)


@config.command("gateway")
@click.argument("port", required=False, type=click.IntRange(1, 65535))
@click.option("--off", is_flag=True, help="Don't start the gateway with the first model.")
def config_gateway(port: int | None, off: bool) -> None:
    """Set the all-models gateway port (default 9000), or turn it off.

    The gateway starts with the first `llmctl run` and serves every running
    model on one port. With no PORT, print the current setting.
    """
    if off:
        config_mod.set_gateway_port(None)
        click.echo("gateway off (a running one keeps going: `llmctl gateway stop`)")
        return
    if port is None:
        current = config_mod.get_gateway_port()
        click.echo("off" if current is None else str(current))
        return
    config_mod.set_gateway_port(port)
    click.echo(f"gateway port set to {port}")
    gw = _gateway_record()
    if gw and gw.port != port:
        click.echo(f"note: the running gateway is on {gw.port}; "
                   "`llmctl gateway stop && llmctl gateway start` moves it.", err=True)


@config.command("gateway-log")
@click.argument("fmt", required=False, type=click.Choice(["text", "json"]))
def config_gateway_log(fmt: str | None) -> None:
    """Set the gateway's log format: text (default) or json (one JSON object
    per line, for Vector/Loki/VictoriaLogs). JSON request lines carry
    request_id, trace_id, model, status, timings and token counts, never
    prompt or reply text. Takes effect at `llmctl gateway restart`.
    LLM_RUNNER_GATEWAY_LOG overrides it. With no FMT, print the setting."""
    if fmt is None:
        click.echo(config_mod.load().get("gateway_log_format") or "text")
        return
    config_mod.update(gateway_log_format=None if fmt == "text" else fmt)
    click.echo(f"gateway log format: {fmt} (from the next `llmctl gateway restart`)")


@config.command("gateway-otlp")
@click.argument("endpoint", required=False)
@click.option("--clear", is_flag=True, help="Stop exporting spans.")
def config_gateway_otlp(endpoint: str | None, clear: bool) -> None:
    """Export the gateway's trace spans to an OpenTelemetry collector over
    OTLP/HTTP, e.g. http://lab:4318. Off by default: nothing leaves this
    machine. Trace context (traceparent) reaches the models either way.
    Takes effect at `llmctl gateway restart`; LLM_RUNNER_GATEWAY_OTLP
    overrides it. With no ENDPOINT, print the setting."""
    if clear:
        config_mod.update(gateway_otlp_endpoint=None)
        click.echo("gateway span export: off (from the next `llmctl gateway restart`)")
        return
    if endpoint is None:
        click.echo(config_mod.load().get("gateway_otlp_endpoint") or "off")
        return
    u = urllib.parse.urlsplit(endpoint)
    if u.scheme not in ("http", "https") or not u.hostname:
        raise click.BadParameter("an http(s):// URL, e.g. http://collector:4318",
                                 param_hint="ENDPOINT")
    config_mod.update(gateway_otlp_endpoint=endpoint.rstrip("/"))
    click.echo(f"gateway spans go to {endpoint.rstrip('/')}/v1/traces "
               "(from the next `llmctl gateway restart`)")


@config.command("bind")
@click.argument("host", required=False)
@click.option("--reset", is_flag=True, help=f"Back to {config_mod.DEFAULT_BIND_HOST} "
                                           "(every interface).")
def config_bind(host: str | None, reset: bool) -> None:
    """Set the address the gateway and every backend listen on.

    The default, 0.0.0.0, listens on every interface. 127.0.0.1 keeps
    everything on this machine, behind whatever proxy fronts it. Applies to
    what starts next; running models and the gateway keep their address until
    restarted (`llmctl restart`). With no HOST, print the current setting.
    """
    if reset:
        host = config_mod.DEFAULT_BIND_HOST
    if host is None:
        click.echo(config_mod.get_bind_host())
        return
    try:
        socket.getaddrinfo(host, None)
    except socket.gaierror as e:
        raise click.BadParameter(f"can't resolve '{host}': {e}", param_hint="HOST")
    config_mod.set_bind_host(host)
    click.echo(f"bind address set to {host}")
    if config_mod.get_cluster().enabled and host in ("127.0.0.1", "localhost", "::1"):
        click.echo("warning: this machine is in a cluster; other nodes can't reach a "
                   "gateway on loopback.", err=True)
    if proc.list_running():
        click.echo("note: running processes keep their address until restarted "
                   "(`llmctl restart`).", err=True)


@config.command("offline")
@click.argument("state", required=False, type=click.Choice(["on", "off"]))
def config_offline(state: str | None) -> None:
    """Never reach Hugging Face: on an air-gapped host, or to be sure.

    On: `download` and `update` refuse, and every backend starts with
    HF_HUB_OFFLINE=1 and TRANSFORMERS_OFFLINE=1, so it loads from disk or
    fails rather than trying the network. LLM_RUNNER_OFFLINE=1 (or 0)
    overrides the setting. With no STATE, print whether it is on.
    """
    if state is None:
        on = config_mod.is_offline()
        why = " (from LLM_RUNNER_OFFLINE)" if os.environ.get("LLM_RUNNER_OFFLINE") else ""
        click.echo(("on" if on else "off") + why)
        return
    config_mod.set_offline(state == "on")
    click.echo(f"offline mode {state}")
    if proc.list_running():
        click.echo("note: running models keep their environment until restarted "
                   "(`llmctl restart`).", err=True)


def _require_online(what: str) -> None:
    if config_mod.is_offline():
        raise click.UsageError(
            f"Offline mode is on, so `{what}` won't reach Hugging Face. Copy the model "
            f"into {MODELS_DIR} instead, or `llmctl config offline off`.")


@config.command("models-dir")
@click.argument("path", required=False,
                type=click.Path(file_okay=False, path_type=Path))
@click.option("--create", is_flag=True,
              help="Create the directory if it does not exist.")
def config_models_dir(path: Path | None, create: bool) -> None:
    """Persist the models directory, or print it when PATH is omitted."""
    if path is None:
        configured = config_mod.get_models_dir()
        if configured:
            click.echo(str(configured))
        else:
            click.echo(f"(not set; using {registry._resolve_models_dir()})")
        return
    resolved = Path(path).expanduser().resolve()
    if create:
        resolved.mkdir(parents=True, exist_ok=True)
    elif not resolved.exists():
        click.echo(f"warning: {resolved} does not exist "
                   "(pass --create to make it)", err=True)
    saved = config_mod.set_models_dir(resolved)
    click.echo(f"models_dir set to {saved}")
    if os.environ.get("LLM_RUNNER_MODELS_DIR"):
        click.echo("note: $LLM_RUNNER_MODELS_DIR is set and overrides this.",
                   err=True)


@cli.command("list")
def list_cmd() -> None:
    """List models discovered on disk under MODELS_DIR.

    GGUF files are served by llama-server; Hugging Face checkpoint
    directories (config.json + safetensors) by vllm, or by decider.serve when
    they carry a decider_config.json.
    """
    click.echo(f"Models dir: {MODELS_DIR}")
    specs = discover_models()
    if not specs:
        click.echo("\n(no models found — use `llmctl download <repo>` to fetch one)")
        return
    key_w = max(20, max(len(k) for k in specs))
    sources = {key: _read_source(s.model_path if s.model_path.is_dir() else s.model_path.parent)
               for key, s in specs.items()}
    click.echo("")
    click.echo(f"{'KEY':<{key_w}}  {'BACKEND':<7}  {'SIZE':>8}  {'VISION':<6}  {'DRAFT':<6}  "
               f"{'DECISION':<8}  {'REVISION':<12}  {'LICENSE':<12}  PATH")
    click.echo("-" * (key_w + 95))
    for key, spec in specs.items():
        size = _fmt_size(spec.size).strip()
        vision = "yes" if spec.has_vision else ""
        draft = "yes" if spec.has_draft else ""
        decision = "yes" if spec.decision else ""
        src = sources[key]
        rev = (src.get("revision") or "")[:12]
        lic = (src.get("license") or "")[:12]
        rel = spec.model_path.relative_to(MODELS_DIR) if spec.model_path.is_relative_to(MODELS_DIR) else spec.model_path
        click.echo(f"{key:<{key_w}}  {spec.backend:<7}  {size:>8}  {vision:<6}  {draft:<6}  "
                   f"{decision:<8}  {rev:<12}  {lic:<12}  {rel}")


def _profile_flags(values: dict) -> str:
    """A profile's values as the `run` flags they stand for."""
    opts = {p.name: p for p in run.params if isinstance(p, click.Option)}
    out = []
    for name, value in values.items():
        o = opts.get(name)
        if o is None:
            continue
        if o.is_flag:
            if value:
                out.append(o.opts[0])
        elif value is not None:
            out.append(f"{o.opts[0]} {value}")
    return " ".join(out) or "(defaults)"


@cli.group("profile", invoke_without_command=True)
@click.pass_context
def profile_group(ctx: click.Context) -> None:
    """Launch profiles: named sets of `run` options.

    Pick one in the launch menu's Profile row, or with `llmctl run MODEL
    --profile NAME`. Save your own from the menu (Save as profile…).
    """
    if ctx.invoked_subcommand is None:
        ctx.invoke(profile_list)


@profile_group.command("list")
def profile_list() -> None:
    """List the built-in profiles and yours."""
    from llm_runner import launch
    rows = [("last", "any model", "its last run (launch.json)", ""),
            ("default", "any model", "llmctl's defaults", "")]
    rows += [(p.name, p.audience, p.blurb,
              _profile_flags(p.values)) for p in launch.PROFILES]
    rows += [(n, f"chat on {r.get('backend', '?')}", "yours", _profile_flags(r["values"]))
             for n, r in sorted(launch.user_profiles().items())]
    w = [max(len(r[i]) for r in rows) for i in range(3)]
    click.echo(f"{'NAME':<{w[0]}}  {'FOR':<{w[1]}}  {'USE':<{w[2]}}  SETS")
    for name, for_, use, sets in rows:
        click.echo(f"{name:<{w[0]}}  {for_:<{w[1]}}  {use:<{w[2]}}  {sets}".rstrip())
    click.echo("\nchat: llama or vllm, not decision or embedding models. A profile "
               "resets the rows it doesn't set to their defaults; the group stays.")


@profile_group.command("rm")
@click.argument("name")
def profile_rm(name: str) -> None:
    """Delete one of your profiles."""
    from llm_runner import launch
    if name in launch.RESERVED:
        raise click.UsageError(f"'{name}' is built in.")
    if not launch.delete_profile(name):
        raise click.UsageError(f"No profile '{name}'. Yours: "
                               + (", ".join(launch.user_profiles()) or "(none)"))
    click.echo(f"Deleted profile '{name}'.")


@cli.command()
@click.argument("model", required=False)
@click.option("--backend", type=click.Choice(["auto", LLAMA, VLLM, SGLANG, DECIDER]),
              default="auto", show_default=True,
              help="Serving engine. auto = llama-server for a GGUF, vllm for a "
                   "Hugging Face checkpoint directory. sglang serves the same "
                   "checkpoints and is only used when asked for.")
@click.option("--port", "-p", type=int, default=None,
              help="Listen port. Auto-selected starting at 8080 if omitted or taken.")
@click.option("--ctx-size", "-c", type=int, default=None,
              help="Context size (default: 32768 for chat, 8192 for --embeddings). "
                   "vllm: --max-model-len, sglang: --context-length.")
@click.option("--ft", "ft_path",
              type=click.Path(exists=True, path_type=Path),
              help="Path to a finetuned GGUF file, or an HF checkpoint directory "
                   "for vllm/sglang (bypasses the registry).")
@click.option("--name", "ft_name", default=None,
              help="Override the display name (useful with --ft). For vllm/sglang this "
                   "is also the model id clients send (--served-model-name).")
@click.option("--group", "group_name", default=None,
              help="Second name the gateway routes by, shared by interchangeable "
                   "models (e.g. every quant of one model, across machines). Default: "
                   "the model's folder, e.g. gemma-4-26b-a4b-it. 'none' for no group.")
@click.option("--wait/--no-wait", "wait", default=False,
              help="Block until the server answers /health (or exits). Handy for "
                   "vllm/sglang, which take a minute or more to load and compile.")
@click.option("--menu/--no-menu", "-i", "menu", default=None,
              help="Show the launch-options menu (backend, vision, drafter, reasoning, "
                   "context, KV cache, slots) before starting. On by default when "
                   "MODEL is picked interactively; choices are remembered per model.")
@click.option("--profile", "profile", default=None, metavar="NAME",
              help="Set the launch options from a profile: last, default, fast, "
                   "long, multi, serve, lean, or one saved from the menu (`llmctl "
                   "profile list`). Flags typed alongside it win. With the menu, "
                   "the menu opens on it.")
@click.option("--extra", "-X", "extra", multiple=True,
              help="Extra argument(s) appended verbatim to the server command; "
                   "repeatable, each value is shell-split. e.g. "
                   "-X '--max-num-batched-tokens 8192'.")
@click.option("--env", "-E", "env_pairs", multiple=True, metavar="KEY=VALUE",
              help="Environment variable for the server process; repeatable. "
                   "Remembered for restart/restore. e.g. "
                   "-E VLLM_USE_FLASHINFER_SAMPLER=0.")
@click.option("--runtime", "runtime_kind", type=click.Choice(["host", "docker"]),
              default=None,
              help="Run the server as a host process or in a Docker container. "
                   "Default: `llmctl config runtime <backend>` (host unless set).")
@click.option("--image", default=None, metavar="REF",
              help="Container image (implies --runtime docker), pinned by digest: "
                   "repo@sha256:…. Default: the configured image, else the one this "
                   "llmctl was tested with (`llmctl config runtime`).")
@click.option("--allow-tag", is_flag=True, default=False,
              help="Accept an image by tag instead of digest (development only).")
@click.option("--dns", "dns", default=None, metavar="IP[,IP…]",
              help="DNS servers for the container (docker --dns). Default: `llmctl "
                   "config runtime <backend> --dns`, else Docker's. 127.0.0.1 stops "
                   "outside names resolving; '' drops a configured list.")
@click.option("--gpu", "gpu", default=None, metavar="all|N[,N…]",
              help="GPUs the server may use (default all). Containers get them "
                   "through CDI or --gpus; host processes through "
                   "CUDA_VISIBLE_DEVICES.")
@click.option("--restart", "restart", type=click.Choice(["on-failure", "never"]),
              default=None,
              help="If the server crashes after it was ready: the gateway restarts it "
                   "(on-failure, the default; backoff 2 s…60 s) or leaves it down.")
@click.option("--restart-max", "restart_max", type=click.IntRange(1, 100), default=None,
              help="Give up (state: crashed) after this many restarts in 10 minutes "
                   "(default 5).")
@click.option("--kv-cache-mem", "kv_cache_mem", default=None, metavar="SIZE",
              help="vllm only: fixed KV cache size per GPU (--kv-cache-memory-bytes), "
                   "e.g. 6G, instead of sizing it from free memory. Hybrid models "
                   "(Qwen3.5/3.6) need it for reliable warm restarts.")
@click.option("--reasoning", is_flag=True, default=False,
              help="Enable reasoning. Disabled by default (--reasoning-budget 0). "
                   "llama only; for vllm/sglang see --reasoning-parser.")
@click.option("--kv-quant", type=click.Choice(["f16", "q8_0", "q4_0"]),
              default="f16",
              help="Quantize the KV cache to save VRAM. q8_0 ≈ half the f16 size "
                   "and is near-lossless; q4_0 halves it again with some quality "
                   "cost. Great for fitting big models / long context in VRAM. "
                   "vllm/sglang: q8_0 → an fp8 KV cache (no 4-bit KV there).")
@click.option("--gpu-mem", "gpu_mem", type=click.FloatRange(0.05, 1.0), default=None,
              help="vllm/sglang only: fraction of *total* GPU memory the engine "
                   "may claim (vllm --gpu-memory-utilization, sglang "
                   "--mem-fraction-static). The default is computed from the VRAM "
                   "currently free, minus headroom (more for sglang, whose "
                   "fraction leaves out activations) — pass it to pin a value.")
@click.option("--tool-parser", "tool_parser", default=None,
              help="vllm/sglang only: enable OpenAI tool calling with this parser "
                   "(--tool-call-parser), e.g. hermes, "
                   "llama3_json, mistral, pythonic, qwen3_coder.")
@click.option("--reasoning-parser", "reasoning_parser", default=None,
              help="vllm/sglang only: split reasoning into reasoning_content with this "
                   "parser (--reasoning-parser), e.g. deepseek_r1, qwen3.")
@click.option("--trust-remote-code", is_flag=True, default=False,
              help="vllm/sglang only: allow the checkpoint's custom modeling code.")
@click.option("--n-gpu-layers", "--ngl", "n_gpu_layers", type=int, default=999,
              help="Layers to offload to GPU (default: all). Lower it to spill "
                   "the rest onto CPU when VRAM is tight (slower, but it fits).")
@click.option("--no-vision", is_flag=True, default=False,
              help="Skip the vision projector (mmproj) to free ~1-1.5 GB of VRAM "
                   "on vision models you only use for text.")
@click.option("--draft", "draft_spec", default=None,
              help="Speculative/MTP drafter to pair with MODEL for faster "
                   "generation (llama-server --model-draft). Accepts a path to "
                   "a GGUF or a registry key. If omitted, a drafter sitting "
                   "beside the model (*-MTP / *-assistant / *-draft) is "
                   "auto-attached.")
@click.option("--no-draft", is_flag=True, default=False,
              help="Don't use a speculative drafter even if one is auto-detected "
                   "next to the model.")
@click.option("--spec-type", "spec_type", default="draft-mtp",
              help="Speculative decoding type (llama-server --spec-type). "
                   "Default draft-mtp, for Gemma-4-style MTP drafters. Only "
                   "used when a drafter is attached.")
@click.option("--spec-draft-n-max", "--spec-n-max", "spec_n_max", type=int,
              default=3,
              help="Max draft tokens proposed per step (llama-server "
                   "--spec-draft-n-max). Try 1-6; default 3. Only used when a "
                   "drafter is attached.")
@click.option("--n-parallel", "--np", "n_parallel", type=int, default=None,
              help="Server slots (llama-server --parallel). More slots = more "
                   "concurrent requests, but they split the KV cache and weaken "
                   "prompt-prefix reuse. Defaults to 1 when a drafter is attached "
                   "(MTP + prefix reuse work best single-stream); otherwise "
                   "llama-server auto-selects. vllm: --max-num-seqs (default 256), "
                   "sglang: --max-running-requests.")
@click.option("--kv-unified/--no-kv-unified", "kv_unified", default=None,
              help="Share one KV buffer across all slots (llama-server "
                   "--kv-unified) so each slot may use the full --ctx-size "
                   "rather than ctx/N. Default: on when --n-parallel > 1.")
@click.option("--ctx-checkpoints", "ctx_checkpoints", type=click.IntRange(min=0), default=None,
              help="Context checkpoints llama-server keeps per slot for prompt-cache "
                   "reuse (llama-server --ctx-checkpoints; its default is 32). Each "
                   "can be hundreds of MB on long prompts. Default: 0 (off) for "
                   "decision models, which never reuse the cache; llama-server's "
                   "own default otherwise.")
@click.option("--cache-ram", "cache_ram", type=int, default=0, show_default=True,
              help="Host-RAM prompt cache in MiB (llama-server --cache-ram). "
                   "0 disables it. Worth enabling when several long conversations "
                   "take turns on one slot: each comes back from RAM instead of "
                   "being reprocessed (a 9K-token Gemma switch on the Orin: 16 s "
                   "-> 1.3 s; ~1.1 GB of RAM each). Short, distinct prompts pay "
                   "for the per-call save (+33% with Gemma 26B on the RTX) and "
                   "fill the limit. The `multi` profile sets it.")
@click.option("--embeddings/--no-embeddings", "embeddings", default=None,
              help="Run as an embedding server instead of a chat/completion "
                   "server (drops flash-attn, prompt cache, reasoning). "
                   "Auto-enabled for known embedding models (bge, e5, gte, "
                   "nomic-embed, ...) by filename; pass --embeddings or "
                   "--no-embeddings to force it either way.")
@click.option("--pooling", type=click.Choice(["none", "mean", "cls", "last", "rank"]),
              default="cls",
              help="Pooling strategy for --embeddings mode (default: cls, which "
                   "bge-m3 expects). Ignored without --embeddings.")
@click.option("--batch-size", "-b", type=int, default=None,
              help="llama-server's logical batch size. Default: the context size for "
                   "--embeddings and for non-causal models (vllm-d3), else llama-server's.")
@click.option("--ubatch-size", "-ub", "ubatch_size", type=int, default=None,
              help="llama-server's physical batch size. Default: the context size for "
                   "--embeddings and for non-causal models (vllm-d3), else llama-server's.")
@click.pass_context
def run(ctx: click.Context, model: str | None, backend: str, port: int | None,
        ctx_size: int | None, ft_path: Path | None, ft_name: str | None,
        group_name: str | None, wait: bool, menu: bool | None, profile: str | None, extra: tuple[str, ...], env_pairs: tuple[str, ...],
        runtime_kind: str | None, image: str | None, allow_tag: bool, dns: str | None, gpu: str | None,
        restart: str | None, restart_max: int | None,
        kv_cache_mem: str | None,
        reasoning: bool,
        kv_quant: str, gpu_mem: float | None, tool_parser: str | None,
        reasoning_parser: str | None, trust_remote_code: bool,
        n_gpu_layers: int, no_vision: bool,
        embeddings: bool | None, pooling: str, batch_size: int | None,
        ubatch_size: int | None, draft_spec: str | None, no_draft: bool,
        spec_type: str, spec_n_max: int, n_parallel: int | None,
        cache_ram: int, kv_unified: bool | None, ctx_checkpoints: int | None) -> None:
    """Start MODEL in the background.

    GGUF models run on llama-server; Hugging Face checkpoint directories run
    on vllm (or sglang with --backend sglang), or on decider.serve when they
    carry a decider_config.json. Use --ft <path> to bypass the registry.
    """
    spec: ModelSpec | None = None
    if ft_path:
        if profile:
            raise click.UsageError("--profile needs a model from the registry, not --ft.")
        name = ft_name or f"ft:{ft_path.stem if ft_path.is_file() else ft_path.name}"
        model_path = ft_path.resolve()
        mmproj: Path | None = None
        detected = backends.infer_backend(model_path)
    else:
        specs = discover_models()
        if not specs:
            raise click.UsageError(
                "No models found on disk. Run `llmctl download <repo>` first, "
                "or use --ft <path> to point at a GGUF directly."
            )
        picked = not model
        if picked:
            model = _pick_model_interactively(specs)
        if model not in specs:
            available = "\n  ".join(specs)
            raise click.BadArgumentUsage(
                f"Unknown model '{model}'. Available:\n  {available}"
            )
        spec = specs[model]
        if ((menu or (menu is None and picked)) and sys.stdin.isatty()) or profile:
            from llm_runner import launch
            target = launch.Target(spec, spec.backend if backend == "auto" else backend)
            if (menu or (menu is None and picked)) and sys.stdin.isatty():
                shown = launch.configure(ctx, target, profile)
            else:
                launch.apply_profile(ctx, target, profile)
                shown = True
            if shown:
                # Start over with the chosen values, as if they had been typed.
                return ctx.invoke(run.callback, **{**ctx.params, "model": model,
                                                   "menu": False, "profile": None})
        model_path = spec.model_path
        mmproj = None if no_vision else spec.mmproj_path
        name = ft_name or spec.key
        detected = spec.backend

    # Backend: what the model format implies unless the user forces it. A
    # GGUF *can* be forced onto vllm/sglang (experimental there, needs a
    # tokenizer via -X) but an HF directory can never run on llama-server.
    if backend == "auto":
        backend = detected
    elif backend == LLAMA and model_path.is_dir():
        raise click.UsageError(
            f"'{model_path.name}' is a Hugging Face checkpoint directory; "
            "llama-server only loads GGUF files. Use --backend vllm or sglang.")
    elif backend == DECIDER and not backends.is_decider_model_dir(model_path):
        raise click.UsageError(
            f"'{model_path.name}' is not a decider checkpoint (no decider_config.json).")
    runtime = _resolve_runtime(backend, runtime_kind, image, allow_tag, gpu, dns)
    # A container brings its own engine; the host needn't have one.
    binary = backend if runtime["kind"] == containers.DOCKER else _require_backend(backend)
    _warn_foreign_flags(ctx, backend)
    if kv_cache_mem is not None:
        if backend != VLLM:
            raise click.UsageError("--kv-cache-mem is vllm-only (--kv-cache-memory-bytes).")
        if not re.fullmatch(r"\d+(\.\d+)?[kKmMgGtT]?", kv_cache_mem):
            raise click.BadOptionUsage("--kv-cache-mem",
                                       f"--kv-cache-mem takes a size like 6G, not '{kv_cache_mem}'")
    if backend in HF_ENGINES and kv_quant == "q4_0":
        raise click.UsageError(f"{backend} has no 4-bit KV cache here; use "
                               "--kv-quant q8_0 (→ fp8) or f16.")
    # decider.serve speaks the decision protocol itself; on the other engines a
    # decision model only serves raw completions.
    decision = backend != DECIDER and (spec.decision if spec
                                         else backends.is_decision_model(model_path))
    if decision:
        click.echo(f"note: '{model_path.name}' is a decision model — its answers are "
                   "letter logprobs over a vendor-specific prompt, read by its own "
                   "client. Chat output from it is not meaningful.", err=True)

    # Default embeddings mode by filename when the user didn't force it.
    stem = model_path.stem if model_path.is_file() else model_path.name
    if embeddings is None:
        embeddings = registry.looks_like_embedding_model(stem, model_path)
        if embeddings:
            click.echo(f"Detected embedding model '{stem}' — starting "
                       "in --embeddings mode (use --no-embeddings to override).")

    if port is not None and _port_in_use(port):
        raise click.UsageError(f"Port {port} is already in use.")
    chosen_port = port or _next_free_port()

    # Context defaults differ by mode: embedding models run a short context.
    if ctx_size is None:
        ctx_size = 8192 if embeddings else 32768

    # Speculative drafters don't apply to embedding servers.
    draft_path: Path | None = None
    if embeddings:
        if draft_spec:
            click.echo("Ignoring --draft: speculative decoding doesn't apply to "
                       "embedding servers.", err=True)
    elif not no_draft:
        draft_path = _resolve_draft(draft_spec, spec, backend)

    extra_args = [tok for chunk in extra for tok in shlex.split(chunk)]
    env: dict[str, str] = {}
    for pair in env_pairs:
        key, sep, val = pair.partition("=")
        if not sep or not key:
            raise click.BadOptionUsage("--env", f"--env expects KEY=VALUE, got '{pair}'")
        env[key] = val

    if backend == DECIDER:
        decider_dir = backends.find_decider_dir()
        assert decider_dir is not None  # _require_backend found uvicorn under it
        cmd, launch_env = backends.build_decider_cmd(
            binary, decider_dir, model_path, port=chosen_port,
            n_parallel=n_parallel, host=config_mod.get_bind_host(), extra=extra_args)
        env = {**launch_env, **env}      # -E overrides the profile defaults
    elif backend in HF_ENGINES:
        if gpu_mem is None:
            picked = backends.default_gpu_mem(backend)
            if picked:
                gpu_mem, why = picked
                click.echo(f"Using --gpu-mem {gpu_mem:g} ({why}); pass --gpu-mem "
                           "to override.")
        build = backends.build_vllm_cmd if backend == VLLM else backends.build_sglang_cmd
        cmd = build(
            binary, model_path, name=name, port=chosen_port, ctx_size=ctx_size,
            kv_quant=kv_quant, gpu_mem=gpu_mem, n_parallel=n_parallel,
            embeddings=embeddings, draft_path=draft_path, spec_type=spec_type,
            spec_n_max=spec_n_max, tool_parser=tool_parser,
            reasoning_parser=reasoning_parser,
            trust_remote_code=trust_remote_code, host=config_mod.get_bind_host(),
            extra=extra_args)
        if kv_cache_mem is not None:
            cmd += ["--kv-cache-memory-bytes", kv_cache_mem]
        if backend == VLLM and kv_cache_mem is None and "--kv-cache-memory-bytes" not in \
                " ".join(extra_args) and backends.hf_is_hybrid(model_path):
            click.echo("warning: hybrid model without a fixed KV size; a warm restart can "
                       "run out of memory in chunk_gated_delta_rule. Pass --kv-cache-mem "
                       "(e.g. the size the first start logged as 'Available KV cache "
                       "memory').", err=True)
    else:
        # Known llama.cpp footgun: quantizing the *target* KV cache zeroes MTP /
        # speculative acceptance, so the drafter runs but never gets accepted —
        # you pay its cost for no speedup. Warn loudly rather than silently emit it.
        if draft_path and kv_quant != "f16":
            click.echo(
                f"warning: --kv-quant {kv_quant} quantizes the target KV cache, which "
                "drops MTP/speculative acceptance to ~0% in llama.cpp — the drafter "
                "will run with no speedup. Either keep f16 KV to use MTP (fits less "
                "context), or pass --no-draft to drop the unused drafter and reclaim "
                "its overhead.", err=True)

        # Speculative decoding and prompt-prefix reuse both work best
        # single-stream, so default to one slot when a drafter is attached
        # (override with --n-parallel).
        if n_parallel is None and draft_path is not None:
            n_parallel = 1
            click.echo("Using --n-parallel 1 (best for MTP acceptance + prompt-cache "
                       "reuse); pass --n-parallel N to override.")
        # A non-causal model (vllm-d3) must take each prompt in one physical
        # batch, or llama-server refuses it; let a prompt use the whole context.
        if not embeddings and backends.gguf_is_noncausal(model_path):
            batch_size = batch_size or ctx_size
            ubatch_size = ubatch_size or batch_size
            click.echo(f"Non-causal model: --batch-size {batch_size} --ubatch-size "
                       f"{ubatch_size} (each prompt is processed in one batch).")

        cmd = _build_llama_cmd(model_path, mmproj, chosen_port, ctx_size, reasoning,
                               kv_quant=kv_quant, n_gpu_layers=n_gpu_layers,
                               embeddings=embeddings, pooling=pooling,
                               batch_size=batch_size, ubatch_size=ubatch_size,
                               draft_path=draft_path, spec_type=spec_type,
                               spec_n_max=spec_n_max, n_parallel=n_parallel,
                               cache_ram=cache_ram, kv_unified=kv_unified,
                               host=config_mod.get_bind_host(), extra=extra_args)
        # Report the llmctl name as the model id (as --served-model-name
        # does) instead of the GGUF path, so it matches the gateway's /v1/models.
        if not any(a in ("--alias", "-a") for a in extra_args):
            cmd += ["--alias", name]
        # Decision endpoints never reuse the prompt cache (cache_prompt off), but
        # llama-server still saves up to 32 context checkpoints per slot, each
        # tens to hundreds of MB on a long prompt. That OOM-killed Quyet on the
        # 64 GB Orin, so they default to off there (unless -X sets them).
        if ctx_checkpoints is None and decision and not any(
                a in ("--ctx-checkpoints", "-ctxcp", "--swa-checkpoints") for a in extra_args):
            ctx_checkpoints = 0
        if ctx_checkpoints is not None:
            cmd += ["--ctx-checkpoints", str(ctx_checkpoints)]
    # Remember the llmctl-level invocation (with the interactively-picked model
    # made explicit) so `restart`/`autostart list` can show and re-issue it.
    run_args = ([] if ft_path else [model]) + _run_option_args(ctx)
    if group_name is not None and group_name.lower() != "none" and \
            not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", group_name):
        raise click.UsageError("A group name is letters, digits, '.', '_' and '-'.")
    if group_name is None:
        group = "" if ft_path else registry.default_group(model_path)
    else:
        group = "" if group_name.lower() == "none" else group_name
    if group == name:
        group = ""
    rec = proc.start(name=name, cmd=cmd, port=chosen_port,
                     model_path=str(model_path), run_args=run_args,
                     backend=backend, env=env, group=group, runtime=runtime,
                     restart=restart, restart_max=restart_max)
    by, why = (ctx.obj or {}).get("by", "run"), (ctx.obj or {}).get("reason")
    proc.record_event(rec.name, "started", f"by {by}" + (f", {why}" if why else "")
                      + f"; pid {rec.pid} port {rec.port}", pid=rec.pid, port=rec.port,
                      by=by, reason=why, log=rec.log_file)
    click.echo(f"Started {rec.name}  pid={rec.pid}  port={rec.port}  backend={backend}"
               + (f"  group={group}" if group else ""))
    rt = rec.runtime or {}
    if rt.get("kind") == containers.DOCKER:
        click.echo(f"Runtime: docker {rt['image']}  (digest {rt.get('digest')}, "
                   f"container {rt.get('container')})")
    if rt.get("cache_dir"):
        click.echo(f"Cache:   {rt['cache_dir']}  (shipped cache rule: {rt.get('cache_rule')})")
    _ensure_gateway()
    if env:
        click.echo("Env:     " + " ".join(f"{k}={v}" for k, v in env.items()))
    if draft_path:
        if backend in HF_ENGINES:
            click.echo(f"Speculative drafter: {draft_path.name}  "
                       f"(num_speculative_tokens={spec_n_max})")
        else:
            click.echo(f"Speculative drafter: {draft_path.name}  "
                       f"(--spec-type {spec_type} --spec-draft-n-max {spec_n_max})")
    click.echo(f"Command: {shlex.join(cmd)}")
    click.echo(f"Logs:  {rec.log_file}")
    click.echo(f"Tail:  llmctl logs {rec.name}")
    if wait:
        # vllm/sglang load weights, profile memory and capture CUDA graphs
        # before they listen; llama-server is usually up in seconds.
        timeout = 600 if backend in (*HF_ENGINES, DECIDER) else 120
        t0 = time.time()
        if _wait_healthy(rec, timeout=timeout):
            proc.record_event(rec.name, "ready", f"pid {rec.pid} port {rec.port} after "
                              f"{time.time() - t0:.0f} s", pid=rec.pid, port=rec.port, by=by)
        else:
            why = ("exited before it was ready" if not proc._alive(rec.pid)
                   else f"not ready after {timeout} s")
            proc.record_event(rec.name, "start_failed", f"pid {rec.pid} {why}; log "
                              f"{rec.log_file}", pid=rec.pid, by=by, reason=why,
                              log=rec.log_file)
            raise SystemExit(1)
    elif backend in HF_ENGINES:
        click.echo(f"Note: {backend} takes a minute or more to become ready; use "
                   "--wait to block on /health.")
    elif backend == DECIDER:
        click.echo("Note: decider loads the weights and captures its CUDA graphs "
                   "before it listens (~30 s); use --wait to block on /health.")


@cli.command("ps")
@click.option("--all", "-a", "show_all", is_flag=True,
              help="Every model in the cluster, with the node calls go to (needs the gateway).")
@click.option("--json", "as_json", is_flag=True, help="Machine-readable JSON.")
def ps_cmd(show_all: bool, as_json: bool) -> None:
    """List running models, and crashed ones the gateway is restarting.

    STATE comes from the gateway's supervisor: starting, ready, restarting
    (down, a restart is due) or crashed (down, given up). RESTARTS counts its
    restarts since the gateway started. `llmctl events -f` follows them."""
    if show_all:
        return _ps_cluster()
    from llm_runner import supervisor
    procs = proc.list_running()
    states = supervisor.read_states()
    live = {p.name for p in procs}
    rows = [{"name": p.name, "pid": p.pid, "port": p.port, "backend": p.backend,
             "started_at": p.started_at, "group": p.group or None,
             "model_path": p.model_path, "runtime": p.runtime,
             **_ps_state(states.get(p.name), alive=True)} for p in procs]
    specs = {s.name: s for s in proc.autostart_list()}
    for name, st in states.items():
        if name not in live and st.get("state") in supervisor.DOWN_STATES and name in specs:
            sp = specs[name]
            rows.append({"name": name, "pid": None, "port": sp.port, "backend": sp.backend,
                         "started_at": None, "group": sp.group or None,
                         "model_path": sp.model_path, "runtime": sp.runtime,
                         **_ps_state(st, alive=False)})
    if as_json:
        click.echo(json.dumps({"models": rows}, indent=2))
        return
    if not rows:
        click.echo("(no models running)")
        return
    click.echo(f"{'PID':<8} {'PORT':<6} {'BACKEND':<8} {'STARTED':<20} {'NAME':<28} "
               f"{'STATE':<11} {'RESTARTS':<8} RUNTIME")
    click.echo("-" * 120)
    for r in rows:
        started = (dt.datetime.fromtimestamp(r["started_at"]).strftime("%Y-%m-%d %H:%M:%S")
                   if r["started_at"] else "-")
        state = r["state"] or "-"
        if r["next_restart_in"] is not None:
            state += f" ({r['next_restart_in']:.0f}s)"
        click.echo(f"{r['pid'] or '-':<8} {r['port']:<6} {r['backend']:<8} {started:<20} "
                   f"{r['name']:<28} {state:<11} {r['restarts']:<8} "
                   f"{_runtime_label(r['runtime'])}")
    gw = _gateway_record()
    if gw:
        click.echo(f"\nAll models: {_gateway_url(gw.port)}  (GET /v1/models lists them)")


def _ps_state(st: dict | None, alive: bool) -> dict:
    """``ps`` columns from the supervisor's record (None: not supervised yet,
    e.g. the gateway is off or the model is new)."""
    st = st or {}
    state = st.get("state")
    if alive and state in ("restarting", "crashed"):
        state = "starting"          # restarted since the gateway last wrote
    return {"state": state, "restarts": st.get("restarts", 0),
            "next_restart_in": st.get("next_restart_in") if not alive else None,
            "last_error": st.get("last_error")}


@cli.command("events")
@click.option("-f", "--follow", is_flag=True, help="Keep printing new events.")
@click.option("-n", "lines", type=int, default=20, show_default=True,
              help="How many past events to show.")
@click.option("--json", "as_json", is_flag=True, help="The JSON lines as written.")
def events_cmd(follow: bool, lines: int, as_json: bool) -> None:
    """What happened to models: starts (`run`, `apply`), readiness, failed
    starts (with the log), stops, and the gateway supervisor's deaths,
    restarts and give-ups."""
    path = proc.STATE_DIR / "events.jsonl"

    def show(line: str) -> None:
        if as_json:
            click.echo(line.rstrip())
            return
        try:
            e = json.loads(line)
        except ValueError:
            return
        click.echo(f"{e.get('ts', '')}  {e.get('model', ''):<28} {e.get('event', ''):<15} "
                   f"{e.get('detail', '')}")
    try:
        with open(path) as f:
            for line in f.readlines()[-lines:] if lines > 0 else []:
                show(line)
            while follow:
                line = f.readline()
                if line:
                    show(line)
                else:
                    time.sleep(0.5)
    except FileNotFoundError:
        if not follow:
            click.echo("(no events yet)")
            return
        while not path.exists():
            time.sleep(0.5)
        ctx = click.get_current_context()
        ctx.invoke(events_cmd, follow=True, lines=0, as_json=as_json)
    except KeyboardInterrupt:
        pass


def _runtime_label(rt: dict | None) -> str:
    """``host``, or ``docker vllm-openai@sha256:1a2b3c4d5e6f`` (short digest)."""
    if not rt or rt.get("kind") != containers.DOCKER:
        gpus = (rt or {}).get("gpus")
        return "host" + (f" gpu={gpus}" if gpus and gpus != "all" else "")
    repo = rt.get("image", "").split("@")[0].rsplit("/", 1)[-1].split(":")[0]
    digest = (rt.get("digest") or "").removeprefix("sha256:")[:12]
    return f"docker {repo}@sha256:{digest}"


def _ps_cluster() -> None:
    gw = _gateway_record()
    if gw is None:
        raise click.UsageError("The gateway is not running (it starts with the first model, "
                               "or `llmctl gateway start`).")
    # Groups are listed as models of their own; here each member shows its group.
    models = [m for m in _gateway_get(gw.port, "/v1/models")["data"]
              if "members" not in m["llmctl"]]
    status = _gateway_get(gw.port, "/cluster/v1/status")
    if not models:
        click.echo("(no models running in the cluster)")
    else:
        w = max(len(m["id"]) for m in models)
        gw_ = max([len(m["llmctl"].get("group") or "") for m in models] + [5])
        click.echo(f"  {'MODEL':<{w}}  {'GROUP':<{gw_}}  {'NODE':<12} {'PRIO':<5} "
                   f"{'STATUS':<8} {'PORT':<6} CTX")
        click.echo("-" * (w + gw_ + 50))
        for m in sorted(models, key=lambda m: (m["llmctl"].get("group") or m["id"], m["id"])):
            group = m["llmctl"].get("group") or ""
            for n, inst in enumerate(m["llmctl"]["nodes"]):
                first = n == 0
                node = inst["node"] + (" (here)" if inst["local"] else "")
                click.echo(f"{'→' if first else ' '} {m['id'] if first else '':<{w}}  "
                           f"{group if first else '':<{gw_}}  {node:<12} "
                           f"{inst['priority']:<5} {inst['status']:<8} {inst['port'] or '':<6} "
                           f"{inst['context'] or ''}")
        click.echo("\n→ = where calls go (ready first, then lowest priority, then this machine)."
                   "\nA GROUP name reaches its best member on any node.")
    _echo_cluster_line(status)


def _echo_cluster_line(status: dict) -> None:
    if not status.get("cluster"):
        click.echo("Not in a cluster (`llmctl registry set URL --token T`).")
        return
    reg = status["registry"]
    if status["registry_ok"]:
        click.echo(f"Cluster: node {status['node']} (priority {status['priority']}), "
                   f"registry {reg}, last heartbeat {status['last_heartbeat_age']}s ago")
    else:
        click.echo(f"Cluster: node {status['node']}, registry {reg}: "
                   f"{status['error'] or 'no heartbeat yet'}", err=True)
    for w in status.get("warnings", []):
        click.echo(f"warning: {w}", err=True)


def _gateway_get(port: int, path: str) -> dict:
    try:
        with urllib.request.urlopen(config_mod.local_url(port, path), timeout=5) as r:
            return json.loads(r.read())
    except (urllib.error.URLError, OSError, ValueError) as e:
        raise click.ClickException(f"gateway on port {port} not answering: {e}") from None


@cli.command()
@click.argument("identifier", required=False)
def stop(identifier: str | None) -> None:
    """Stop a running model by NAME or PID.

    With no IDENTIFIER: if exactly one model is running, stop it;
    otherwise show an interactive picker.
    """
    if identifier is None:
        running = _models_running()
        if not running:
            raise click.UsageError("No running models.")
        target = running[0] if len(running) == 1 else _pick_instance_interactively(running)
        identifier = str(target.pid)
    rec = proc.stop(identifier)
    if rec is None:
        raise click.UsageError(f"No running model matches '{identifier}'.")
    click.echo(f"Stopped {rec.name}  pid={rec.pid}  port={rec.port}")
    stopped = [rec]
    # A decision endpoint is useless without its model: take it down too.
    if rec.backend != backends.DECIDE:
        for p in _proxy_attached_to(rec.port):
            proc.stop(str(p.pid))
            click.echo(f"Stopped {p.name}  pid={p.pid}  port={p.port}")
            stopped.append(p)
    for p in stopped:
        if p.backend != GATEWAY:
            proc.record_event(p.name, "stopped", f"by stop; pid {p.pid}", pid=p.pid,
                              port=p.port, by="stop")
    # The gateway came up with the first model and goes down with the last,
    # unless this machine is in a cluster (it still routes to the others).
    gw = _gateway_record()
    if gw and rec.backend != GATEWAY and not _models_running() \
            and not config_mod.get_cluster().enabled:
        proc.stop(str(gw.pid))
        click.echo(f"Stopped gateway  pid={gw.pid}  port={gw.port}  (no models left)")


def _resolve_running_target(identifier: str | None) -> proc.RunningProcess:
    """Find a running instance by NAME/PID, or pick interactively if omitted."""
    if identifier is None:
        running = _models_running()
        if not running:
            raise click.UsageError("No running models.")
        return running[0] if len(running) == 1 else _pick_instance_interactively(running)
    target = proc.find(identifier)
    if target is None:
        raise click.UsageError(f"No running model matches '{identifier}'.")
    return target


@cli.command()
@click.argument("identifier", required=False)
def show(identifier: str | None) -> None:
    """Print the exact server command a running model was started with.

    With no IDENTIFIER: if exactly one model is running, show it; otherwise
    show an interactive picker.
    """
    target = _resolve_running_target(identifier)
    click.echo(f"Name:    {target.name}")
    click.echo(f"PID:     {target.pid}")
    click.echo(f"Port:    {target.port}")
    click.echo(f"Backend: {target.backend}")
    click.echo(f"Model:   {target.model_path}")
    click.echo(f"Log:     {target.log_file}")
    if target.env:
        click.echo("Env:     " + " ".join(f"{k}={v}" for k, v in target.env.items()))
    spec = [f for f in ("--model-draft", "--spec-type", "--spec-draft-n-max",
                        "--speculative-config")
            if f in target.cmd]
    click.echo("Spec:    " + ("yes (" + ", ".join(spec) + ")" if spec else "no"))
    click.echo("\nCommand:")
    click.echo("  " + (shlex.join(target.cmd) if target.cmd else "(not recorded)"))


@cli.command()
@click.argument("identifier", required=False)
@click.option("--lines", "-n", type=int, default=80, help="Tail length.")
@click.option("--follow", "-f", is_flag=True, default=False,
              help="Keep watching for new output (like `tail -f`; Ctrl-C to stop).")
def logs(identifier: str | None, lines: int, follow: bool) -> None:
    """Print the tail of a running model's log.

    With no IDENTIFIER: if exactly one model is running, tail its log;
    otherwise show an interactive picker. A NAME that isn't running shows
    its last log (a failed start, a stopped model).
    """
    if identifier is None:
        running = _models_running()
        if not running:
            raise click.UsageError("No running models.")
        if len(running) == 1:
            target = running[0]
        else:
            target = _pick_instance_interactively(running)
    else:
        target = proc.find(identifier)
        if target is None:
            last = proc.last_log(identifier)
            if last is None:
                raise click.UsageError(f"No running model matches '{identifier}', "
                                       "and it has no log.")
            click.echo(f"({identifier} is not running; its last log, {last}, from "
                       f"{time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(last.stat().st_mtime))})",
                       err=True)
    log_path = Path(target.log_file) if target is not None else last
    if not log_path.exists():
        raise click.UsageError(f"Log file missing: {log_path}")

    with log_path.open("r", errors="replace") as fh:
        tail = fh.read().splitlines()[-lines:]
        click.echo("\n".join(tail))
        if not follow:
            return
        # fh is now at EOF; stream whatever gets appended until interrupted.
        try:
            while True:
                chunk = fh.read()
                if chunk:
                    click.echo(chunk, nl=False)
                else:
                    time.sleep(0.25)
        except KeyboardInterrupt:
            pass


SERVICE_NAME = "llmctl-restore.service"


def _systemd_user_dir() -> Path:
    xdg = os.environ.get("XDG_CONFIG_HOME")
    base = Path(xdg).expanduser() if xdg else Path.home() / ".config"
    return base / "systemd" / "user"


def _build_unit() -> str:
    """Render the systemd user unit that runs `llmctl restore` at boot."""
    llmctl = shutil.which("llmctl") or "llmctl"
    # Pass through llm-runner location overrides so the boot-time service
    # resolves the same models/state as the shell that installed it.
    env_lines = [
        f"Environment={var}={os.environ[var]}"
        for var in ("LLM_RUNNER_MODELS_DIR", "LLM_RUNNER_STATE_DIR",
                    "LLM_RUNNER_CONFIG_DIR", "CUDA_HOME", "CUDA_PATH")
        if os.environ.get(var)
    ]

    # Bake the current PATH/LD_LIBRARY_PATH so llama-server and its shared
    # libs resolve under systemd's otherwise-bare environment. vllm/sglang
    # are recorded by absolute path and get their compilers from llmctl.
    env_lines += [
        f"Environment={var}={os.environ[var]}"
        for var in ("PATH", "LD_LIBRARY_PATH") if os.environ.get(var)
    ]
    exec_start = f"{llmctl} restore"

    env_block = ("\n".join(env_lines) + "\n") if env_lines else ""
    return (
        "[Unit]\n"
        "Description=Restore llm-runner models that were running before reboot\n"
        "After=network-online.target\n"
        "Wants=network-online.target\n"
        "\n"
        "[Service]\n"
        "Type=oneshot\n"
        "RemainAfterExit=yes\n"
        f"{env_block}"
        f"ExecStart={exec_start}\n"
        "\n"
        "[Install]\n"
        "WantedBy=default.target\n"
    )


@cli.command()
@click.argument("names", nargs=-1)
def restore(names: tuple[str, ...]) -> None:
    """Relaunch the models that were running before the last reboot.

    Pass NAMES to restore only those manifest entries. Ports are pinned: an
    entry whose port is taken is reported as an error, never moved.
    """
    # Only demand the binaries the manifest actually needs. A vllm/sglang
    # entry records an absolute binary path, so it doesn't depend on PATH.
    specs = [s for s in proc.autostart_list() if not names or s.name in names]
    in_docker = [s for s in specs if (s.runtime or {}).get("kind") == containers.DOCKER]
    wanted = {s.backend for s in specs if s not in in_docker}
    for backend in sorted(wanted):
        if backends.find(backend) is None:
            click.echo(f"warning: {backends.missing_hint(backend)}", err=True)
    # At boot this user unit can't order itself after the system's docker
    # service, so give the daemon a minute to come up.
    if in_docker and not containers.wait_for_docker(60):
        click.echo("warning: the docker daemon isn't answering; container models will "
                   "fail to start.", err=True)
    results = proc.restore(list(names) or None, host=config_mod.get_bind_host())
    if not results:
        click.echo("Nothing to restore (autostart manifest is empty).")
        return
    failed = False
    for spec, status in results:
        failed |= status.startswith("error")
        click.echo(f"{spec.name:<24}  {status}")
    if _models_running() or config_mod.get_cluster().enabled:
        _ensure_gateway()
    if failed:
        raise SystemExit(1)


@cli.command(context_settings={"ignore_unknown_options": True})
@click.argument("identifier", required=False)
@click.argument("overrides", nargs=-1, type=click.UNPROCESSED)
@click.pass_context
def restart(ctx: click.Context, identifier: str | None,
            overrides: tuple[str, ...]) -> None:
    """Stop IDENTIFIER (if running) and relaunch it with its last `run` args.

    Extra `run` options after the name override the remembered ones and are
    remembered for next time, e.g.:

      llmctl restart gemma-4-26b-a4b-it-ud-q4_k_m --n-parallel 8

    The port is kept the same. Pass --port to move it deliberately.

    With no IDENTIFIER, restart everything that is running, one at a time:
    each model is back and answering before the next one stops (so memory
    never holds two copies), then the decision endpoints, then the gateway.
    """
    if identifier is None:
        if overrides:
            raise click.UsageError("Overrides need a model name: llmctl restart NAME --flag ...")
        return _restart_all(ctx)
    _restart_one(ctx, identifier, overrides)


def _restart_all(ctx: click.Context) -> None:
    running = proc.list_running()
    if not running:
        raise click.UsageError("Nothing is running. Pass a name from `llmctl autostart list`.")
    # Models before the decision endpoints in front of them; the gateway last,
    # so it comes back on the freshly installed llmctl after everything else.
    rank = {backends.DECIDE: 1, GATEWAY: 2}
    order = sorted(running, key=lambda p: (rank.get(p.backend, 0), p.started_at))
    click.echo("Restarting one at a time: " + ", ".join(p.name for p in order))
    failed = []
    for p in order:
        click.echo(f"\n== {p.name}")
        try:
            rec = _restart_one(ctx, p.name, ())
        except click.UsageError as e:
            click.echo(f"skipped: {e.format_message()}", err=True)
            failed.append(p.name)
            continue
        slow = rec is not None and rec.backend in (*HF_ENGINES, DECIDER)
        if rec is None or not _wait_healthy(rec, timeout=600 if slow else 300):
            failed.append(p.name)
    if failed:
        click.echo(f"\nNot back: {', '.join(failed)}", err=True)
        raise SystemExit(1)
    click.echo(f"\nAll {len(order)} restarted.")


def _restart_one(ctx: click.Context, identifier: str,
                 overrides: tuple[str, ...]) -> proc.RunningProcess | None:
    live = proc.find(identifier)
    name = live.name if live else identifier
    spec = proc.autostart_find(name)
    if spec is None:
        raise click.UsageError(
            f"'{name}' is not in the autostart manifest "
            "(see `llmctl autostart list`).")
    if not spec.run_args and overrides:
        raise click.UsageError(
            f"'{name}' was recorded before run args were tracked; re-run it "
            "with `llmctl run …` once, then `restart` can take overrides.")
    # Check the binary before taking the live instance down.
    _require_backend(spec.backend)

    if live:
        proc.stop(name)
        click.echo(f"Stopped {name}  pid={live.pid}  port={live.port}")
        # The listen socket closes with the process, but give the kernel a
        # beat so the pinned port is free again before we rebind it.
        for _ in range(20):
            if not _port_in_use(spec.port):
                break
            time.sleep(0.25)

    if not spec.run_args:
        # Recorded without `run` args (a decision endpoint, the gateway, or a
        # legacy entry): relaunch the resolved server command, on the current
        # bind address.
        cmd = backends.with_bind_host(spec.cmd, spec.backend, config_mod.get_bind_host())
        rec = proc.start(name=spec.name, cmd=cmd, port=spec.port,
                         model_path=spec.model_path, backend=spec.backend,
                         env=spec.env, group=spec.group, restart=spec.restart,
                         restart_max=spec.restart_max)
        click.echo(f"Started {rec.name}  pid={rec.pid}  port={rec.port}")
        return rec

    # Re-issue `llmctl run` with the remembered args; overrides come last so
    # they win, and `run` re-records the merged set in the manifest.
    args = spec.run_cmd()[1:] + list(overrides)
    click.echo(f"Re-running: llmctl run {shlex.join(args)}")
    sub = run.make_context("llmctl run", args, parent=ctx)
    with sub:
        run.invoke(sub)
    return proc.find(name)


class _DecideGroup(click.Group):
    """`llmctl decide MODEL` is shorthand for `llmctl decide start MODEL`."""

    def parse_args(self, ctx: click.Context, args: list[str]) -> list[str]:
        if not args or (args[0] not in self.commands and args[0] not in ("--help", "-h")):
            args = ["start", *args]
        return super().parse_args(ctx, args)


@cli.group(cls=_DecideGroup)
def decide() -> None:
    """Serve a decision model's /v1/systemone API next to its llama-server.

    \b
      llmctl run rune-26b-a4b-v3-iq4_xs-dyn      # the model, as usual
      llmctl decide rune-26b-a4b-v3-iq4_xs-dyn   # + the decision endpoint
      llmctl decide spec rune-26b-a4b-v3-iq4_xs-dyn   # how it answers

    The endpoint is systemone-proxy (packages/systemone-proxy): it rebuilds
    the model vendor's exact prompt, reads the option-letter probabilities
    from llama-server and answers in one format for every supported model.
    """


def _proxy_attached_to(port: int) -> list[proc.RunningProcess]:
    def backend_port(cmd: list[str]) -> int | None:
        if "--backend" not in cmd[:-1]:
            return None
        return urllib.parse.urlsplit(cmd[cmd.index("--backend") + 1]).port
    return [p for p in proc.list_running()
            if p.backend == backends.DECIDE and backend_port(p.cmd) == port]


@decide.command("start")
@click.argument("identifier", required=False)
@click.option("--port", "-p", type=int, default=None,
              help="Port for the decision endpoint (default: first free from 8090).")
@click.option("--profile", default=None,
              help="Force a systemone-proxy profile (default: detected from the GGUF).")
@click.option("--cache", type=click.Choice(["off", "request"]), default="off", show_default=True,
              help="llama-server prompt-cache reuse: off, or within one request only.")
@click.option("--concurrency", type=int, default=None,
              help="Backend calls in flight per request (default: the model's --parallel).")
@click.option("--wait/--no-wait", "wait", default=True, show_default=True,
              help="Wait until the endpoint is up (it self-tests the GGUF first).")
@click.option("--profile", "profile", default=None, metavar="NAME",
              help="Set the launch options from a profile: last, default, fast, "
                   "long, multi, serve, lean, or one saved from the menu (`llmctl "
                   "profile list`). Flags typed alongside it win. With the menu, "
                   "the menu opens on it.")
@click.option("--extra", "-X", "extra", multiple=True,
              help="Extra argument(s) for `systemone-proxy serve`.")
def decide_start(identifier: str | None, port: int | None, profile: str | None, cache: str,
                 concurrency: int | None, wait: bool, extra: tuple[str, ...]) -> None:
    """Start the decision endpoint for a running llama-server model (NAME or PID)."""
    llama = [p for p in proc.list_running() if p.backend == LLAMA]
    if identifier is None:
        flagged = [p for p in llama if backends.is_decision_model(Path(p.model_path))]
        if not flagged:
            raise click.UsageError("No running decision model. `llmctl run <model>` first "
                                   "(`llmctl list` flags them in the DECISION column).")
        target = flagged[0] if len(flagged) == 1 else _pick_instance_interactively(flagged)
    else:
        target = proc.find(identifier)
        if target is None:
            raise click.UsageError(f"No running model matches '{identifier}'. "
                                   f"`llmctl run {identifier}` first.")
        if target.backend != LLAMA:
            raise click.UsageError(f"'{target.name}' runs on {target.backend}; `llmctl decide` "
                                   "needs a llama-server instance.")

    binary = _require_backend(backends.DECIDE)
    if profile is None:
        res = subprocess.run([binary, "detect", target.model_path], capture_output=True, text=True)
        if res.returncode != 0:
            known = subprocess.run([binary, "profiles"], capture_output=True, text=True).stdout
            raise click.UsageError(f"systemone-proxy has no profile for {Path(target.model_path).name}."
                                   f"\nSupported models:\n{known.rstrip()}")
        profile = res.stdout.strip()

    name = f"{target.name}.decide"
    if proc.find(name):
        live = proc.find(name)
        raise click.UsageError(f"{name} is already running on port {live.port}.")
    if port is not None and _port_in_use(port):
        raise click.UsageError(f"Port {port} is already in use.")
    chosen_port = port or _next_free_port(8090)
    if concurrency is None:
        cmd = target.cmd
        concurrency = (int(cmd[cmd.index("--parallel") + 1])
                       if "--parallel" in cmd[:-1] and cmd[cmd.index("--parallel") + 1].isdigit()
                       else 1)
    cmd = [binary, "serve", "--backend", config_mod.local_url(target.port),
           "--host", config_mod.get_bind_host(),
           "--port", str(chosen_port), "--profile", profile,
           "--concurrency", str(concurrency), "--cache", cache,
           *[tok for chunk in extra for tok in shlex.split(chunk)]]
    rec = proc.start(name=name, cmd=cmd, port=chosen_port, model_path=target.model_path,
                     backend=backends.DECIDE,
                     group=f"{base}.decide" if (base := registry.group_of(
                         target.name, target.backend, target.model_path, target.group)) else "")
    click.echo(f"Started {rec.name}  pid={rec.pid}  port={rec.port}  profile={profile}")
    click.echo(f"Endpoint: POST {config_mod.local_url(rec.port)}/v1/systemone  "
               f"(→ llama-server :{target.port})")
    click.echo(f"Contract: llmctl decide spec {target.name}   "
               f"or GET {config_mod.local_url(rec.port)}/v1/spec")
    click.echo(f"Logs:     llmctl logs {rec.name}")
    _ensure_gateway()
    if wait and not _wait_healthy(rec, timeout=300):
        raise SystemExit(1)


@decide.command("spec")
@click.argument("target", required=False)
@click.option("--json", "as_json", is_flag=True, help="Machine-readable JSON.")
def decide_spec(target: str | None, as_json: bool) -> None:
    """Print the request/response contract a client should implement.

    TARGET is a running instance, a model key from `llmctl list`, or a profile
    id from `llmctl decide profiles`. Without TARGET: the common contract only.
    """
    binary = _require_backend(backends.DECIDE)
    args = [binary, "spec"]
    if target:
        live = proc.find(target)
        specs = {} if live else discover_models()
        if live:
            args += ["--gguf", live.model_path]
        elif target in specs:
            args += ["--gguf", str(specs[target].model_path)]
        else:
            args += ["--profile", target]
    if as_json:
        args.append("--json")
    raise SystemExit(subprocess.run(args).returncode)


@decide.command("profiles")
def decide_profiles() -> None:
    """List the decision models systemone-proxy supports."""
    binary = _require_backend(backends.DECIDE)
    raise SystemExit(subprocess.run([binary, "profiles"]).returncode)


def _models_running() -> list[proc.RunningProcess]:
    """Running instances other than the gateway (what pickers offer)."""
    return [p for p in proc.list_running() if p.backend != GATEWAY]


def _gateway_record() -> proc.RunningProcess | None:
    return next((p for p in proc.list_running() if p.backend == GATEWAY), None)


def _gateway_url(port: int) -> str:
    host = config_mod.get_bind_host()
    if host in ("0.0.0.0", "::"):
        host = socket.gethostname()
    elif ":" in host:
        host = f"[{host}]"
    return f"http://{host}:{port}/v1"


def _start_gateway(port: int) -> proc.RunningProcess:
    cmd = [sys.executable, "-m", "llm_runner", "gateway", "serve", "--port", str(port),
           "--host", config_mod.get_bind_host()]
    return proc.start(name="gateway", cmd=cmd, port=port, model_path="", backend=GATEWAY)


def _ensure_gateway() -> None:
    """Start the gateway if it's enabled and not up yet. Never fails a launch."""
    port = config_mod.get_gateway_port()
    if port is None or _gateway_record():
        return
    if _port_in_use(port):
        click.echo(f"note: gateway not started, port {port} is taken "
                   "(`llmctl config gateway PORT` to move it).", err=True)
        return
    rec = _start_gateway(port)
    click.echo(f"Gateway: {_gateway_url(rec.port)}  (all running models; pid={rec.pid})")


def _restart_gateway() -> None:
    """Apply changed cluster settings: restart a running gateway, start one
    that is now wanted, stop one that no longer is."""
    gw = _gateway_record()
    wanted = bool(_models_running()) or config_mod.get_cluster().enabled
    if gw and not wanted:
        proc.stop(str(gw.pid))
        click.echo(f"Stopped gateway  pid={gw.pid}  port={gw.port}")
    elif gw:
        _bounce_gateway(gw)
    elif wanted:
        _ensure_gateway()


def _bounce_gateway(gw: proc.RunningProcess) -> proc.RunningProcess:
    """Stop ``gw`` and start it again on its port, on the installed llmctl."""
    proc.stop(str(gw.pid))
    for _ in range(20):
        if not _port_in_use(gw.port):
            break
        time.sleep(0.25)
    rec = _start_gateway(gw.port)
    click.echo(f"Restarted gateway  pid={rec.pid}  port={rec.port}")
    return rec


def _wait_cluster(timeout: float = 12.0) -> dict | None:
    """The gateway's cluster status once its first heartbeat is in."""
    gw = _gateway_record()
    if gw is None:
        return None
    deadline = time.time() + timeout
    status = None
    while time.time() < deadline:
        try:
            status = _gateway_get(gw.port, "/cluster/v1/status")
            if status["registry_ok"] or status["error"]:
                return status
        except click.ClickException:
            pass
        time.sleep(0.5)
    return status


@cli.group(invoke_without_command=True)
@click.pass_context
def gateway(ctx: click.Context) -> None:
    """One port (default 9000) for every running model.

    \b
      GET  /v1/models                     the running models (+ an `llmctl` block)
      POST /v1/chat/completions ...       routed by the body's "model"
      POST /v1/systemone                  routed to a decision endpoint the same way
      ANY  /m/<name>/<path>               any path of that model (/completion, /props, ...)

    "model" may be an exact name or a unique prefix; with one model running it
    may be left out. It starts with the first `llmctl run`, stops with the
    last `llmctl stop`, and comes back with autostart. `llmctl config gateway
    --off` disables that.

    In a cluster (`llmctl registry`), it lists and routes to the models of
    every node, `model@node` pins one, and it stays up with no local models.
    """
    if ctx.invoked_subcommand is not None:
        return
    gw = _gateway_record()
    if gw is None:
        port = config_mod.get_gateway_port()
        click.echo("Gateway not running" + (" (turned off: `llmctl config gateway 9000`)"
                                             if port is None else
                                             f"; it starts with the next model on port {port}."))
        return
    click.echo(f"Gateway: {_gateway_url(gw.port)}  pid={gw.pid}")
    for p in _models_running():
        click.echo(f"  {p.name:<36}  {p.backend:<8} :{p.port}")
    _echo_cluster_line(_gateway_get(gw.port, "/cluster/v1/status"))


@gateway.command("start")
@click.option("--port", "-p", type=click.IntRange(1, 65535), default=None,
              help="Port (default: `llmctl config gateway`, 9000).")
def gateway_start(port: int | None) -> None:
    """Start the gateway now (it normally starts with the first model)."""
    gw = _gateway_record()
    if gw:
        raise click.UsageError(f"The gateway is already running on port {gw.port}.")
    port = port or config_mod.get_gateway_port() or config_mod.DEFAULT_GATEWAY_PORT
    if _port_in_use(port):
        raise click.UsageError(f"Port {port} is already in use.")
    rec = _start_gateway(port)
    click.echo(f"Gateway: {_gateway_url(rec.port)}  pid={rec.pid}")


@gateway.command("restart")
def gateway_restart() -> None:
    """Restart a running gateway on its port, so it runs the installed llmctl
    (install.sh does this). Does nothing when no gateway is running."""
    gw = _gateway_record()
    if gw is None:
        click.echo("Gateway not running; nothing to restart.")
        return
    rec = _bounce_gateway(gw)
    for _ in range(40):
        try:
            with urllib.request.urlopen(config_mod.local_url(rec.port, "/healthz"), timeout=1):
                return
        except (OSError, ValueError):
            time.sleep(0.25)
    raise click.ClickException(f"The gateway did not come back on port {rec.port}; "
                               f"see `llmctl logs gateway`.")


@gateway.command("stop")
def gateway_stop() -> None:
    """Stop the gateway (the next `llmctl run` starts it again)."""
    gw = _gateway_record()
    if gw is None:
        raise click.UsageError("The gateway is not running.")
    proc.stop(str(gw.pid))
    click.echo(f"Stopped gateway  pid={gw.pid}  port={gw.port}")
    if config_mod.get_cluster().enabled:
        click.echo("note: this machine is out of the cluster until the gateway is back "
                   "(`llmctl gateway start`, the next `llmctl run`, or a restore).", err=True)


@gateway.command("serve", hidden=True)
@click.option("--port", "-p", type=int, default=config_mod.DEFAULT_GATEWAY_PORT)
@click.option("--host", default="0.0.0.0")
def gateway_serve(port: int, host: str) -> None:
    """Run the gateway in the foreground (what `start` launches)."""
    from llm_runner import gateway as gw_mod
    gw_mod.serve(port, host)


def _networks(values: tuple[str, ...], flag: str) -> list[str]:
    from llm_runner import auth
    try:
        return [str(auth.parse_network(v)) for v in values]
    except ValueError as e:
        raise click.BadOptionUsage(flag, f"{flag}: {e}") from None


@gateway.command("auth")
@click.argument("state", required=False, type=click.Choice(["on", "off"]))
@click.option("--local", "local", multiple=True, metavar="CIDR",
              help="A network whose callers need no key (repeat; replaces the list).")
@click.option("--local-default", is_flag=True, help="Back to the default local networks.")
@click.option("--proxy", "proxies", multiple=True, metavar="ADDR",
              help="A reverse proxy's address or network (repeat; replaces the list): "
                   "its requests are judged by the address it forwards, and count as "
                   "outside without one.")
@click.option("--no-proxy", is_flag=True, help="Forget the --proxy list.")
def gateway_auth(state: str | None, local: tuple[str, ...], local_default: bool,
                 proxies: tuple[str, ...], no_proxy: bool) -> None:
    """Require an API key from callers outside the local networks.

    \b
      llmctl gateway key add laptop        # prints the key, once
      llmctl gateway auth on

    With auth on, a call from outside sends `Authorization: Bearer <key>` (what
    OpenAI clients send for their api_key) or `X-Api-Key: <key>`; without
    one it gets 401. Callers on a local network (default: loopback, the
    private ranges, Tailscale's 100.64.0.0/10, IPv6 ULA and link-local) need
    none. Loopback and a single bind address (`llmctl config bind`) are
    always local: llmctl's own calls come from them. A request that a proxy on a local network forwards is judged by
    the client address in its X-Forwarded-For, so a reverse proxy in front
    of the gateway (nginx, Caddy, Traefik) does not make the internet local.
    Health checks (/health, /healthz, /readyz), cluster hops and the
    registry (which check the cluster token) need no key. Changes apply at
    once, no restart. With no arguments, print the settings.
    """
    from llm_runner import auth
    values: dict = {}
    if state is not None:
        values["gateway_auth"] = auth.ON if state == "on" else None
    if local:
        values["gateway_local_networks"] = _networks(local, "--local")
    if local_default:
        values["gateway_local_networks"] = None
    if proxies:
        values["gateway_proxies"] = _networks(proxies, "--proxy")
    if no_proxy:
        values["gateway_proxies"] = None
    if values:
        config_mod.update(**values)
    policy = auth.policy_from(config_mod.load())
    click.echo(f"Auth:     {policy.mode}" + ("  (every caller is let in)"
                                             if policy.mode == auth.OFF else
                                             "  (outside callers need a key)"))
    click.echo("Local:    " + ", ".join(map(str, policy.local)) + "  (+ loopback"
               + "".join(f", {ip}" for ip in policy.own) + ")")
    click.echo("Proxies:  " + (", ".join(map(str, policy.proxies)) or
                               "(none listed; any local peer's X-Forwarded-For is used)"))
    click.echo("Keys:     " + (", ".join(sorted(policy.keys)) or "(none)"))
    if policy.mode == auth.ON and not policy.keys:
        click.echo("warning: no keys yet, so every outside call is refused "
                   "(`llmctl gateway key add NAME`).", err=True)


@gateway.group("key", invoke_without_command=True)
@click.pass_context
def gateway_key(ctx: click.Context) -> None:
    """API keys for callers from outside (see `llmctl gateway auth`)."""
    if ctx.invoked_subcommand is None:
        keys = config_mod.load().get("gateway_keys") or {}
        click.echo("\n".join(sorted(keys)) or "No keys (`llmctl gateway key add NAME`).")
        if config_mod.get_upstream_key():
            click.echo("This gateway's own key for the registry: set (`key set --clear`).")


@gateway_key.command("add")
@click.argument("name")
def gateway_key_add(name: str) -> None:
    """Make a key called NAME and print it. Only its hash is kept, so copy it now."""
    from llm_runner import auth
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", name):
        raise click.UsageError("A key name is letters, digits, '.', '_' and '-'.")
    keys = dict(config_mod.load().get("gateway_keys") or {})
    if name in keys:
        raise click.UsageError(f"There is already a key called {name} "
                               f"(`llmctl gateway key rm {name}` first).")
    key = auth.new_key()
    keys[name] = auth.hash_key(key)
    config_mod.update(gateway_keys=keys)
    click.echo(key)
    click.echo(f"Key {name} added; it is not shown again.", err=True)
    if config_mod.load().get("gateway_auth") != auth.ON:
        click.echo("note: auth is off, so no one needs it yet (`llmctl gateway auth on`).",
                   err=True)


@gateway_key.command("set")
@click.argument("key", required=False)
@click.option("--clear", is_flag=True, help="Forget the key.")
def gateway_key_set(key: str | None, clear: bool) -> None:
    """Give this gateway a key for the registry's gateway (from its `key add`).

    A node that can't reach another node at its LAN address (a laptop away
    from home) sends that node's calls through the registry's gateway, e.g.
    https://llmctl.example.net, as an ordinary client call. This is the key
    it sends, so clients of this machine's gateway need none. With no KEY,
    it is asked for (and not echoed).
    """
    if clear:
        config_mod.update(gateway_upstream_key=None)
        click.echo("Key cleared.")
        return
    if key is None:
        key = click.prompt("Key", hide_input=True).strip()
    if not key:
        raise click.UsageError("The key is empty.")
    config_mod.update(gateway_upstream_key=key)
    click.echo("Key set; relayed calls carry it from now on.")
    c = config_mod.get_cluster()
    if not c.registry_url:
        click.echo("note: this machine reports to no registry, so it has nothing to relay "
                   "through yet (`llmctl registry set URL`).", err=True)


@gateway_key.command("rm")
@click.argument("name")
def gateway_key_rm(name: str) -> None:
    """Revoke the key called NAME (at once, no restart)."""
    keys = dict(config_mod.load().get("gateway_keys") or {})
    if keys.pop(name, None) is None:
        raise click.UsageError(f"No key called {name}.")
    config_mod.update(gateway_keys=keys or None)
    click.echo(f"Key {name} revoked.")


@gateway.command("metrics-token")
@click.option("--clear", is_flag=True,
              help="Forget it: scrapes then need an API key, like other calls.")
def gateway_metrics_token(clear: bool) -> None:
    """Make a scrape-only token for /metrics, /metrics/sd and
    /metrics/backends/*, and print it. Only its hash is kept.

    A monitoring host with it can scrape but can't run models. Like API
    keys it's needed only from outside the local networks, with auth on;
    while none is set, scrapes need an API key. A new token replaces the
    old one.
    """
    from llm_runner import auth
    if clear:
        config_mod.update(gateway_metrics_token=None)
        click.echo("Metrics token cleared; scrapes need an API key when auth asks for one.")
        return
    token = auth.new_key()
    config_mod.update(gateway_metrics_token=auth.hash_key(token))
    click.echo(token)
    click.echo("Metrics token set (it replaces any earlier one and is not shown again). "
               "vmagent: `bearer_token_file` in the scrape and http_sd configs.", err=True)


@cli.group(invoke_without_command=True)
@click.pass_context
def node(ctx: click.Context) -> None:
    """This machine's name and priority in the cluster."""
    if ctx.invoked_subcommand is not None:
        return
    c = config_mod.get_cluster()
    click.echo(f"Name:      {c.node_name}")
    click.echo(f"Priority:  {c.node_priority}  (lower wins when several nodes run a model)")
    click.echo(f"Advertise: {c.node_advertise or '(address the registry sees)'}")
    click.echo(f"Spill:     {'on' if c.spill else 'off'}  (calls to a busy model go to a free "
               "copy elsewhere)")


@node.command("set")
@click.option("--name", default=None, help="Node name (default: the hostname).")
@click.option("--priority", type=int, default=None,
              help="Lower is preferred: e.g. 0 for the fast workstation, 1 for the Orin.")
@click.option("--advertise", default=None,
              help="URL other nodes use to reach this gateway, e.g. http://orin.ts.net:9000 "
                   "(default: the address the registry sees).")
@click.option("--no-advertise", is_flag=True, help="Forget --advertise.")
@click.option("--spill", type=click.Choice(["on", "off"]), default=None,
              help="on (default): a call for a model whose slots are all busy goes to a "
                   "copy with a free slot, if any; off: always the preferred copy.")
def node_set(name: str | None, priority: int | None, advertise: str | None,
             no_advertise: bool, spill: str | None) -> None:
    """Set this machine's name, priority, advertised URL or spill policy."""
    if name is not None and not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", name):
        raise click.UsageError("A node name is letters, digits, '.', '_' and '-'.")
    values: dict = {}
    if name is not None:
        values["node_name"] = name
    if priority is not None:
        values["node_priority"] = priority
    if advertise is not None:
        values["node_advertise"] = _normalize_url(advertise)
    if no_advertise:
        values["node_advertise"] = None
    if spill is not None:
        values["node_spill"] = None if spill == "on" else False
    if not values:
        raise click.UsageError("Nothing to set (--name, --priority, --advertise, --spill).")
    config_mod.update(**values)
    c = config_mod.get_cluster()
    click.echo(f"node {c.node_name}  priority={c.node_priority}"
               + (f"  advertise={c.node_advertise}" if c.node_advertise else "")
               + ("" if c.spill else "  spill=off"))
    if _gateway_record():
        _restart_gateway()


def _require_gateway_on() -> None:
    if config_mod.get_gateway_port() is None:
        raise click.UsageError("The gateway is turned off; a cluster needs it "
                               "(`llmctl config gateway 9000`).")


def _normalize_url(url: str) -> str:
    """A gateway URL from what was typed. A bare host (``lab``, ``lab:9001``)
    is a gateway on the LAN: http, port 9000 unless given. A full URL means
    what it says, so ``https://llmctl.example.net`` is port 443, behind a
    proxy."""
    url = url.strip().rstrip("/")
    if "://" in url:
        return url
    parts = urllib.parse.urlsplit("http://" + url)
    if parts.port is None:
        return f"http://{parts.netloc}:{config_mod.DEFAULT_GATEWAY_PORT}{parts.path}"
    return "http://" + url


@cli.group("registry", invoke_without_command=True)
@click.pass_context
def registry_group(ctx: click.Context) -> None:
    """Join machines into one cluster: every gateway lists and routes to the
    models of all of them.

    \b
      lab$  llmctl registry serve                 # the always-on box; prints a token
      rtx$  llmctl node set --name rtx --priority 0
      rtx$  llmctl registry set lab --token T     # lab:9000
      orin$ llmctl node set --name orin --priority 1
      orin$ llmctl registry set lab --token T
      any$  llmctl ps --all

    Nodes send the registry a heartbeat every 5 s; one silent for 15 s is
    dropped. With no subcommand, show the cluster as this machine sees it.
    """
    if ctx.invoked_subcommand is None:
        ctx.invoke(registry_status)


@registry_group.command("serve")
@click.option("--token", default=None, help="Cluster token (default: keep the current one, "
                                            "or generate one).")
@click.option("--off", is_flag=True, help="Stop being the registry.")
def registry_serve(token: str | None, off: bool) -> None:
    """Make this machine's gateway the registry (it stays up, models or not)."""
    if off:
        config_mod.update(registry_serve=None)
        click.echo("This machine is no longer the registry.")
        _restart_gateway()
        return
    _require_gateway_on()
    import secrets
    token = token or config_mod.get_cluster().token or secrets.token_urlsafe(18)
    config_mod.update(registry_serve=True, cluster_token=token)
    _restart_gateway()
    gw = _gateway_record()
    port = gw.port if gw else config_mod.get_gateway_port()
    from llm_runner.gateway import lan_ip
    click.echo(f"Registry on http://{lan_ip()}:{port}  (node {config_mod.get_cluster().node_name})")
    click.echo("Join the other machines with:")
    click.echo(f"  llmctl registry set {lan_ip()}:{port} --token {token}")
    click.echo("Keep it up across reboots: `llmctl autostart install` (once).")


@registry_group.command("set")
@click.argument("url")
@click.option("--token", default=None, help="The cluster token (`llmctl registry token` on "
                                            "the registry).")
def registry_set(url: str, token: str | None) -> None:
    """Report to the registry at URL: a host (port 9000), host:port, or a full
    URL such as https://llmctl.example.net for a registry behind a proxy."""
    _require_gateway_on()
    url = _normalize_url(url)
    token = token or config_mod.get_cluster().token
    if not token:
        raise click.UsageError("--token is required (`llmctl registry token` on the registry).")
    req = urllib.request.Request(f"{url}/registry/v1/nodes",
                                 headers={"X-Llmctl-Token": token})
    try:
        with urllib.request.urlopen(req, timeout=5) as r:
            nodes = json.loads(r.read())["nodes"]
        click.echo(f"Registry {url}: reachable, {len(nodes)} node(s) "
                   f"({', '.join(n['node'] for n in nodes) or 'none yet'}).")
    except urllib.error.HTTPError as e:
        if e.code == 401:
            raise click.UsageError(f"{url} refused the token.") from None
        if e.code == 404:
            raise click.UsageError(f"{url} is a gateway but not the registry "
                                   "(`llmctl registry serve` there).") from None
        raise click.ClickException(f"{url}: HTTP {e.code}") from None
    except (urllib.error.URLError, OSError, ValueError) as e:
        click.echo(f"warning: {url} not reachable now ({e}); saved anyway, the gateway "
                   "keeps trying.", err=True)
    config_mod.update(registry_url=url, cluster_token=token)
    _restart_gateway()
    status = _wait_cluster()
    if status:
        _echo_cluster_line(status)


@registry_group.command("clear")
def registry_clear() -> None:
    """Leave the cluster (this machine's models stay up, local only)."""
    c = config_mod.get_cluster()
    config_mod.update(registry_url=None,
                      cluster_token=c.token if c.registry_serve else None)
    click.echo("Left the cluster.")
    _restart_gateway()


@registry_group.command("token")
def registry_token() -> None:
    """Print the cluster token (to join another machine)."""
    token = config_mod.get_cluster().token
    if not token:
        raise click.UsageError("No token set (`llmctl registry serve` or `registry set`).")
    click.echo(token)


@registry_group.command("status")
def registry_status() -> None:
    """The cluster as this machine's gateway sees it."""
    c = config_mod.get_cluster()
    if not c.enabled:
        click.echo("Not in a cluster. `llmctl registry set URL --token T` to join one, "
                   "or `llmctl registry serve` to host it.")
        return
    gw = _gateway_record()
    if gw is None:
        raise click.UsageError("The gateway is not running (`llmctl gateway start`).")
    status = _gateway_get(gw.port, "/cluster/v1/status")
    _echo_cluster_line(status)
    if status.get("registry_serve"):
        click.echo("This machine is the registry.")
    click.echo(f"\n  {'NODE':<14} {'PRIO':<5} {'VERSION':<9} {'SEEN':<7} {'URL':<28} MODELS")
    rows = [{"node": status["node"] + " (here)", "priority": status["priority"],
             "version": status["version"], "age": 0, "url": status.get("node_url") or "",
             "models": status["local_models"]}] + status["nodes"]
    for n in sorted(rows, key=lambda r: (r["priority"], r["node"])):
        click.echo(f"  {n['node']:<14} {n['priority']:<5} {n['version']:<9} "
                   f"{str(n['age']) + 's':<7} {n['url']:<28} {', '.join(n['models']) or '-'}")


@cli.group()
def autostart() -> None:
    """Restart the last-running models on boot via a systemd user service."""


def _fmt_run_line(s: proc.LaunchSpec) -> str:
    """The `llmctl run …` line that reproduces a manifest entry, or a hint for
    entries recorded before run args were tracked."""
    if s.backend == backends.DECIDE:
        return f"llmctl decide {s.name.removesuffix('.decide')} --port {s.port}"
    if s.backend == GATEWAY:
        return f"llmctl gateway start --port {s.port}"
    if s.run_args:
        return f"llmctl {shlex.join(s.run_cmd())}"
    return "(run args not recorded — `llmctl run` it again with your flags to capture them)"


@autostart.command("list")
def autostart_list_cmd() -> None:
    """Show the models that `restore` would bring back."""
    specs = proc.autostart_list()
    if not specs:
        click.echo("(autostart manifest empty — start a model to populate it)")
        return
    for s in specs:
        click.echo(f"{s.name:<24}  port={s.port}  backend={s.backend}  {s.model_path}")
        click.echo(f"{'':<24}  {_fmt_run_line(s)}")


@autostart.command("sync")
def autostart_sync_cmd() -> None:
    """Make the manifest mirror the currently running models.

    Drops entries for models that aren't running (crashed, failed to load,
    killed without `llmctl stop`) and adds any running model that's missing.
    """
    kept, dropped = proc.autostart_sync()
    for s in dropped:
        click.echo(f"- {s.name:<24}  port={s.port}  (not running, removed)")
    for s in kept:
        click.echo(f"  {s.name:<24}  port={s.port}")
        click.echo(f"  {'':<24}  {_fmt_run_line(s)}")
    if not kept:
        click.echo("(nothing running — manifest is now empty)")


@autostart.command("install")
def autostart_install() -> None:
    """Install + enable the systemd user service that runs `llmctl restore`."""
    unit_dir = _systemd_user_dir()
    unit_dir.mkdir(parents=True, exist_ok=True)
    unit_path = unit_dir / SERVICE_NAME
    unit_path.write_text(_build_unit())
    click.echo(f"Wrote {unit_path}  (baked in the current PATH / LD_LIBRARY_PATH)")

    # daemon-reload + enable are required; linger lets it run without an active
    # login session (so it actually fires on a headless boot).
    for cmd, required in (
        (["systemctl", "--user", "daemon-reload"], True),
        (["systemctl", "--user", "enable", SERVICE_NAME], True),
        (["loginctl", "enable-linger", getpass.getuser()], False),
    ):
        try:
            subprocess.run(cmd, check=True)
            click.echo(f"  ✓ {' '.join(cmd)}")
        except (subprocess.CalledProcessError, FileNotFoundError) as e:
            lvl = "error" if required else "note"
            click.echo(f"  {lvl}: `{' '.join(cmd)}` failed ({e}) — run it manually.",
                       err=True)
            if cmd[0] == "loginctl":
                # Enabling linger for yourself needs polkit auth, which SSH
                # sessions usually lack; root can always do it.
                click.echo("        Without linger the service only runs after you "
                           "log in. To start it at boot, run:\n"
                           f"          sudo loginctl enable-linger {getpass.getuser()}",
                           err=True)
    click.echo("\nInstalled. Test it now without rebooting:\n"
               f"  systemctl --user start {SERVICE_NAME} && llmctl ps")


@autostart.command("uninstall")
def autostart_uninstall() -> None:
    """Disable + remove the systemd user service (manifest is left intact)."""
    unit_path = _systemd_user_dir() / SERVICE_NAME
    subprocess.run(["systemctl", "--user", "disable", SERVICE_NAME], check=False)
    if unit_path.exists():
        unit_path.unlink()
        click.echo(f"Removed {unit_path}")
    else:
        click.echo("(no unit file installed)")
    subprocess.run(["systemctl", "--user", "daemon-reload"], check=False)


@autostart.command("status")
def autostart_status() -> None:
    """Show the service state and what would be restored."""
    unit_path = _systemd_user_dir() / SERVICE_NAME
    click.echo(f"Unit: {unit_path}"
               + ("" if unit_path.exists() else "  (not installed)"))
    specs = proc.autostart_list()
    click.echo(f"Manifest: {len(specs)} model(s)")
    for s in specs:
        click.echo(f"  {s.name:<24}  port={s.port}")
        click.echo(f"  {'':<24}  {_fmt_run_line(s)}")
    if unit_path.exists():
        subprocess.run(["systemctl", "--user", "status", SERVICE_NAME,
                        "--no-pager"], check=False)


def _fmt_size(n: int | None) -> str:
    if n is None:
        return "?"
    val = float(n)
    for unit in ("B", "K", "M", "G", "T"):
        if val < 1024:
            return f"{val:>6.1f}{unit}"
        val /= 1024
    return f"{val:>6.1f}P"


def _pick_instance_interactively(running: list[proc.RunningProcess]) -> proc.RunningProcess:
    try:
        import questionary
    except ImportError as e:
        raise click.UsageError(
            "questionary not installed. Pass an IDENTIFIER (name or PID) explicitly."
        ) from e
    name_w = max(len(p.name) for p in running)
    choices = [
        questionary.Choice(
            title=f"{p.name:<{name_w}}  pid={p.pid:<6}  port={p.port}",
            value=p,
        )
        for p in running
    ]
    selected = questionary.select(
        "Select a running instance:",
        choices=choices,
        use_shortcuts=False,
    ).ask()
    if selected is None:
        raise click.Abort()
    return selected


def _pick_model_interactively(specs: dict[str, ModelSpec],
                              prompt: str = "Select a model to run:") -> str:
    try:
        import questionary
    except ImportError as e:
        raise click.UsageError(
            "questionary not installed. Pass a MODEL key explicitly, or reinstall."
        ) from e
    key_w = max(len(k) for k in specs)
    choices = [
        questionary.Choice(
            title=f"{key:<{key_w}}  {spec.backend:<5}  "
                  f"{_fmt_size(spec.size).strip():>8}  "
                  f"{'[vision]' if spec.has_vision else '        '}  "
                  f"{'[mtp]' if spec.has_draft else '     '}  "
                  f"{'[decision]' if spec.decision else ''}",
            value=key,
        )
        for key, spec in specs.items()
    ]
    selected = questionary.select(
        prompt,
        choices=choices,
        use_shortcuts=False,
    ).ask()
    if selected is None:
        raise click.Abort()
    return selected


def _derive_subdir(repo_id: str) -> str:
    name = repo_id.split("/")[-1]
    name = re.sub(r"[-_.]?gguf$", "", name, flags=re.IGNORECASE)
    return name.lower()


# Provenance sidecar: records which HF repo (and files) a model folder came
# from, so `llmctl update` can re-check it without the user retyping the repo.
_SOURCE_FILE = ".llmctl-source.json"


def _read_source(folder: Path) -> dict:
    p = folder / _SOURCE_FILE
    if not p.exists():
        return {}
    try:
        data = json.loads(p.read_text())
    except (json.JSONDecodeError, OSError):
        return {}
    return data if isinstance(data, dict) else {}


def _write_source(folder: Path, repo_id: str, rfilenames: list[str],
                  fmt: str = "gguf") -> None:
    data = _read_source(folder)
    files = sorted(set(data.get("files", [])) | set(rfilenames))
    (folder / _SOURCE_FILE).write_text(
        json.dumps({"repo_id": repo_id, "format": fmt, "files": files}, indent=2)
        + "\n")


# Repo files a vllm checkpoint snapshot never needs. When safetensors are
# present the legacy PyTorch pickles are skipped too (see _snapshot_hf).
_HF_IGNORE_ALWAYS = ("*.gguf", "*.msgpack", "*.h5", "*.ot", "*.onnx", "onnx/*",
                     "original/*", "consolidated*", ".gitattributes")
_HF_IGNORE_IF_SAFETENSORS = ("*.bin", "*.pt", "*.pth")


def _snapshot_hf(repo_id: str, info, target_dir: Path, *, revalidate: bool) -> None:
    """Pull a Hugging Face checkpoint (safetensors + config/tokenizer) into
    ``target_dir`` for vllm. ``snapshot_download`` is incremental: files whose
    etag already matches are skipped, so re-running doubles as `update`."""
    from huggingface_hub import snapshot_download

    names = [s.rfilename for s in info.siblings]
    ignore = list(_HF_IGNORE_ALWAYS)
    if any(n.endswith(".safetensors") for n in names):
        ignore += _HF_IGNORE_IF_SAFETENSORS
    elif not any(n.endswith(_HF_IGNORE_IF_SAFETENSORS) for n in names):
        raise click.UsageError(
            f"{repo_id} has no safetensors or PyTorch weights — nothing vllm "
            "can serve.")

    from fnmatch import fnmatch
    wanted = [s for s in info.siblings
              if not any(fnmatch(s.rfilename, pat) for pat in ignore)]
    total = sum(getattr(s, "size", None) or 0 for s in wanted)
    target_dir.mkdir(parents=True, exist_ok=True)
    click.echo(f"Target: {target_dir}  (vllm / Hugging Face checkpoint)")
    verb = "Revalidating" if revalidate else "Downloading"
    click.echo(f"  ↓ {verb} {len(wanted)} files ({_fmt_size(total).strip()}) "
               f"from {repo_id} ...")
    try:
        snapshot_download(repo_id=repo_id, local_dir=str(target_dir),
                          ignore_patterns=ignore)
    except Exception as e:
        raise click.UsageError(
            f"Download failed: {e}\n(Gated repo? Accept the license on "
            "huggingface.co and run `hf auth login` or set $HF_TOKEN.)") from e
    _write_source(target_dir, repo_id, [s.rfilename for s in wanted], fmt="hf")

    click.echo(f"\nFiles now in {target_dir}:")
    for p in sorted(target_dir.iterdir()):
        if p.is_file() and p.name != _SOURCE_FILE:
            click.echo(f"  {p.name}  ({_fmt_size(p.stat().st_size)})")
    if not backends.is_hf_model_dir(target_dir):
        click.echo("warning: folder doesn't look like a servable checkpoint "
                   "(no config.json + weights).", err=True)


def _run_download(repo_id: str, filename: str | None, no_mmproj: bool,
                  no_draft: bool, subdir: str | None, *, revalidate: bool = False,
                  target_dir: Path | None = None, hf: bool = False) -> None:
    """Shared core for `download` (revalidate=False) and `update` (=True).

    Repos holding GGUFs go through the quant picker; anything else (or
    ``hf=True``) is snapshotted whole as a vllm checkpoint.

    With ``revalidate`` the "already present" short-circuit is skipped and every
    selected file is re-checked against the remote (``hf_hub_download`` only
    re-pulls blobs whose commit/etag changed).
    """
    try:
        from huggingface_hub import HfApi, hf_hub_download
    except ImportError as e:
        raise click.UsageError(
            "huggingface_hub not installed. Re-run ./install.sh to pick up new deps."
        ) from e

    api = HfApi()
    try:
        info = api.model_info(repo_id, files_metadata=True)
    except Exception as e:
        raise click.UsageError(f"Failed to fetch repo info for '{repo_id}': {e}") from e

    siblings = {s.rfilename: s for s in info.siblings if s.rfilename.endswith(".gguf")}
    if hf or not siblings:
        if filename:
            raise click.UsageError("--file only applies to GGUF repos; a Hugging "
                                   "Face checkpoint is downloaded whole.")
        if not siblings:
            click.echo(f"No .gguf files in {repo_id} — treating it as a Hugging "
                       "Face checkpoint for vllm.")
        _snapshot_hf(repo_id, info,
                     target_dir or MODELS_DIR / (subdir or _derive_subdir(repo_id)),
                     revalidate=revalidate)
        return

    mmproj_names = sorted(f for f in siblings if "mmproj" in f.lower())
    # Drafters may live in an MTP/ subdir (rfilename keeps the path), so match
    # on the basename stem, not the full rfilename.
    draft_names = sorted(
        f for f in siblings
        if f not in set(mmproj_names) and registry._looks_like_draft(Path(f).stem)
    )
    companions = set(mmproj_names) | set(draft_names)
    quant_names = sorted(f for f in siblings if f not in companions)
    if not quant_names:
        raise click.UsageError(f"No quant .gguf files (non-mmproj) found in {repo_id}.")

    if filename is None:
        try:
            import questionary
        except ImportError as e:
            raise click.UsageError(
                "questionary not installed. Re-run ./install.sh, or pass --file <name>."
            ) from e
        choices = [
            questionary.Choice(
                title=f"{_fmt_size(getattr(siblings[name], 'size', None))}  {name}",
                value=name,
            )
            for name in quant_names
        ]
        selected = questionary.select(
            f"Select a quant from {repo_id}:",
            choices=choices,
            use_shortcuts=False,
        ).ask()
        if selected is None:
            raise click.Abort()
    else:
        if filename not in siblings:
            raise click.UsageError(
                f"'{filename}' not found in {repo_id}. Available .gguf files:\n  "
                + "\n  ".join(sorted(siblings))
            )
        selected = filename

    # `--file <mmproj> --subdir <folder>` fetches only a vision projector into
    # an existing model folder (e.g. a base model's mmproj for a fine-tune whose
    # GGUF repo ships none): no companion prompts, and the folder keeps
    # recording the repo its quant came from.
    companion_only = selected in mmproj_names
    if companion_only:
        no_mmproj = no_draft = True

    chosen_mmproj: str | None = None
    if mmproj_names and not no_mmproj:
        if len(mmproj_names) == 1:
            chosen_mmproj = mmproj_names[0]
        else:
            try:
                import questionary
                mm_choices = [
                    questionary.Choice(
                        title=f"{_fmt_size(getattr(siblings[name], 'size', None))}  {name}",
                        value=name,
                    )
                    for name in mmproj_names
                ] + [questionary.Choice(title="(skip)", value=None)]
                chosen_mmproj = questionary.select(
                    "Select vision projector (mmproj):",
                    choices=mm_choices,
                    use_shortcuts=False,
                ).ask()
            except ImportError:
                chosen_mmproj = mmproj_names[0]

    chosen_draft: str | None = None
    if draft_names and not no_draft:
        if len(draft_names) == 1:
            chosen_draft = draft_names[0]
        else:
            try:
                import questionary
                d_choices = [
                    questionary.Choice(
                        title=f"{_fmt_size(getattr(siblings[name], 'size', None))}  {name}",
                        value=name,
                    )
                    for name in draft_names
                ] + [questionary.Choice(title="(skip)", value=None)]
                chosen_draft = questionary.select(
                    "Select speculative/MTP drafter:",
                    choices=d_choices,
                    use_shortcuts=False,
                ).ask()
            except ImportError:
                chosen_draft = draft_names[0]

    if target_dir is None:
        target_dir = MODELS_DIR / (subdir or _derive_subdir(repo_id))
    target_dir.mkdir(parents=True, exist_ok=True)
    click.echo(f"Target: {target_dir}")

    to_download = [selected]
    to_download += [chosen_mmproj] if chosen_mmproj else []
    to_download += [chosen_draft] if chosen_draft else []
    for f in to_download:
        # Flatten any repo subdir (e.g. MTP/foo.gguf) so companions land beside
        # the base model where discovery's sibling-attach logic finds them.
        dest = target_dir / Path(f).name
        before = dest.stat().st_size if dest.exists() else None
        if before and not revalidate:
            click.echo(f"  ✓ already present: {dest.name}")
            continue
        verb = "checking" if revalidate else "downloading"
        click.echo(f"  ↓ {verb} {f} ({_fmt_size(getattr(siblings[f], 'size', None))}) ...")
        local = Path(hf_hub_download(
            repo_id=repo_id,
            filename=f,
            local_dir=str(target_dir),
        ))
        if local != dest:
            local.replace(dest)
            with contextlib.suppress(OSError):
                local.parent.rmdir()  # drop the now-empty subdir if any
        # Size is a cheap staleness proxy: a requant almost always changes it.
        after = dest.stat().st_size
        status = ("downloaded" if before is None
                  else "updated" if after != before else "up to date")
        click.echo(f"    {status}: {dest.name}")

    if not (companion_only and _read_source(target_dir).get("repo_id") not in (None, repo_id)):
        _write_source(target_dir, repo_id, to_download)

    click.echo(f"\nFiles now in {target_dir}:")
    for p in sorted(target_dir.iterdir()):
        if p.is_file() and p.name != _SOURCE_FILE:
            click.echo(f"  {p.name}  ({_fmt_size(p.stat().st_size)})")


@cli.command()
@click.argument("repo_id")
@click.option("--file", "filename", default=None,
              help="Skip the interactive picker and download this exact file.")
@click.option("--no-mmproj", is_flag=True,
              help="Don't download the vision projector (mmproj) alongside the quant.")
@click.option("--no-draft", is_flag=True,
              help="Don't download the speculative/MTP drafter alongside the quant.")
@click.option("--subdir", default=None,
              help=f"Override target subdir under {MODELS_DIR} (default: derived from repo name).")
@click.option("--hf", is_flag=True,
              help="Fetch the repo as a Hugging Face checkpoint for vllm "
                   "(safetensors + config) even if it also has GGUFs. "
                   "Repos without GGUFs are snapshotted this way automatically.")
def download(repo_id: str, filename: str | None, no_mmproj: bool, no_draft: bool,
             subdir: str | None, hf: bool) -> None:
    """Download a model from a Hugging Face REPO_ID into MODELS_DIR.

    GGUF repos: pick a quant (served by llama-server).
    Checkpoint repos (safetensors): snapshot the whole repo (served by vllm).

    Examples:

      llmctl download unsloth/gemma-4-E4B-it-GGUF

      llmctl download Qwen/Qwen3-8B            # → vllm

      # a base model's vision projector into a fine-tune's folder
      llmctl download unsloth/Qwen3.5-4B-GGUF --file mmproj-F16.gguf --subdir h2o-lightning-4b

    Records the source repo so `llmctl update` can re-check it later.
    """
    _require_online("download")
    _run_download(repo_id, filename, no_mmproj, no_draft, subdir, hf=hf)


def _resolve_update_target(target: str | None) -> tuple[str, Path, str | None]:
    """Resolve an `update` TARGET to (repo_id, model_folder, known_filename).

    TARGET may be a model key (repo read from its provenance sidecar), a repo id
    (owner/name), or None (pick from models that have a recorded source).
    """
    specs = discover_models()
    updatable: dict[str, tuple[str, Path, str | None]] = {}
    installed: set[str] = set()
    for key, spec in specs.items():
        # GGUF: the folder holding the file. vllm: the checkpoint dir itself,
        # which is re-snapshotted whole (no single file to name).
        folder = spec.model_path if spec.backend == VLLM else spec.model_path.parent
        if _read_source(folder).get("source") == "install":
            installed.add(key)
            continue
        repo = _read_source(folder).get("repo_id")
        if repo:
            known = None if spec.backend == VLLM else spec.model_path.name
            updatable[key] = (repo, folder, known)

    if target is None:
        if not updatable:
            raise click.UsageError(
                "No downloaded model has a recorded source repo (provenance is "
                "written on new downloads). Run `llmctl update <repo_id>` instead.")
        key = _pick_model_interactively({k: specs[k] for k in updatable})
        return updatable[key]
    if target in updatable:
        return updatable[target]
    if target in installed:
        raise click.UsageError(
            f"'{target}' was installed from a bundle; there's no repo to re-check. "
            "Install the new version side by side (`llmctl install <bundle> --name …`).")
    if target in specs:
        raise click.UsageError(
            f"No source repo recorded for '{target}' (downloaded before provenance "
            f"tracking, or placed manually). Run `llmctl update <repo_id>` to point "
            f"at its Hugging Face repo.")
    if "/" in target:
        return target, MODELS_DIR / _derive_subdir(target), None
    raise click.BadArgumentUsage(
        f"'{target}' is not a known model key or a repo id (owner/name).")


@cli.command()
@click.argument("target", required=False)
@click.option("--file", "filename", default=None,
              help="Revalidate this exact file instead of the picker / recorded quant.")
@click.option("--no-mmproj", is_flag=True, help="Skip the vision projector (mmproj).")
@click.option("--no-draft", is_flag=True, help="Skip the speculative/MTP drafter.")
def update(target: str | None, filename: str | None, no_mmproj: bool,
           no_draft: bool) -> None:
    """Re-check a downloaded model's HF repo and pull any changed/added files.

    TARGET is a model key (uses the repo recorded at download time) or a repo id
    (owner/name). With no TARGET, pick from models that have a recorded source.
    Re-validates against the remote and re-downloads only what changed — handy
    when a repo fixes a quant or adds an `mtp-` drafter after you first pulled.
    """
    _require_online("update")
    repo_id, target_dir, known_file = _resolve_update_target(target)
    if filename is None:
        filename = known_file  # may be None for a repo-id target → show picker
    click.echo(f"Updating from {repo_id}")
    hf = _read_source(target_dir).get("format") == "hf"
    _run_download(repo_id, filename, no_mmproj, no_draft, subdir=None,
                  revalidate=True, target_dir=target_dir, hf=hf)


# ------------------------------------------------------------- apply --

def _model_key(model: str, specs: dict) -> str | None:
    """A serving file's ``model``: a model key, or a folder under MODELS_DIR."""
    if model in specs:
        return model
    hits = [k for k, s in specs.items()
            if (s.model_path if s.model_path.is_dir() else s.model_path.parent).name == model]
    return hits[0] if len(hits) == 1 else None


def _same_runtime(recorded: dict | None, desired: dict) -> bool:
    host = {"kind": containers.HOST, "gpus": "all"}
    return (proc.chosen_runtime(recorded) or host) == (proc.chosen_runtime(desired) or host)


def _plan_apply(entries: list, prune: bool) -> tuple[list, list[str]]:
    """What `apply` would do: one Action per entry (plus stops with --prune)."""
    from llm_runner import apply as apply_mod
    specs = discover_models()
    problems: list[str] = []
    running = {p.name: p for p in proc.list_running() if p.backend in backends.BACKENDS}
    autostart = {s.name: s for s in proc.autostart_list()}
    wanted = {e.name for e in entries}
    actions: list = []
    stopping = set(running) - wanted if prune else set()
    for e in entries:
        key = _model_key(e.model, specs)
        if key is None:
            problems.append(f"{e.where}: models.{e.name}.model {e.model!r} is not a model "
                            f"key or folder under {MODELS_DIR}")
            continue
        argv = e.run_argv(key)
        try:
            ctx = run.make_context("run", list(argv))
            backend = ctx.params["backend"]
            if backend == "auto":
                backend = specs[key].backend
            desired_rt = _resolve_runtime(backend, ctx.params["runtime_kind"],
                                          ctx.params["image"], ctx.params["allow_tag"],
                                          ctx.params["gpu"], ctx.params["dns"])
        except click.ClickException as err:
            problems.append(f"{e.where}: models.{e.name}: {err.format_message()}")
            continue
        desired = [key, *_run_option_args(ctx)]
        rec, spec = running.get(e.name), autostart.get(e.name)
        if rec is None:
            action, why = "start", "not running"
        elif spec is None:
            action, why = "restart", "running without a launch record"
        elif spec.port != e.port or rec.port != e.port:
            action, why = "restart", f"port {rec.port} -> {e.port}"
        elif spec.run_args != desired:
            action, why = "restart", apply_mod.diff_args(spec.run_args, desired)
        elif not _same_runtime(rec.runtime, desired_rt):
            action, why = "restart", "runtime or image changed"
        else:
            action, why = "unchanged", ""
        if action != "unchanged":
            other = next((p for p in running.values() if p.port == e.port
                          and p.name != e.name and p.name not in stopping), None)
            if other is not None:
                problems.append(f"{e.where}: port {e.port} is in use by running model "
                                f"{other.name!r} (stop it, or apply with --prune)")
                continue
        actions.append(apply_mod.Action(e.name, action, e.port, why, argv))
    for name in sorted(stopping):
        actions.append(apply_mod.Action(name, "stop", running[name].port,
                                        "not in the serving files (--prune)"))
    return actions, problems


@cli.command("apply")
@click.argument("files", nargs=-1, required=True,
                type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option("--dry-run", is_flag=True, default=False,
              help="Show what would start, restart and stop; change nothing.")
@click.option("--prune", is_flag=True, default=False,
              help="Also stop running models that no file lists.")
@click.option("--wait/--no-wait", "wait", default=True, show_default=True,
              help="Wait for each started model to answer /health before the next.")
@click.option("--json", "as_json", is_flag=True, default=False,
              help="Print one JSON result on stdout; progress goes to stderr.")
def apply_cmd(files: tuple[Path, ...], dry_run: bool, prune: bool, wait: bool,
              as_json: bool) -> None:
    """Make the running models match serving FILES (YAML).

    Each entry under `models:` is started as `llmctl run` would start it. An
    entry already running with the same arguments is left alone; a changed one
    is restarted; a missing one is started. Nothing else is stopped unless
    --prune. Several files merge (a bundle's fragment plus a host file). See
    the README for the keys.

    \b
    Exit codes: 0 done (or nothing to do), 3 invalid files, 1 a start failed.
    """
    from llm_runner import apply as apply_mod
    try:
        entries = apply_mod.load(list(files))
        actions, problems = _plan_apply(entries, prune)
        if problems:
            raise apply_mod.ApplyError(problems)
    except apply_mod.ApplyError as e:
        if as_json:
            click.echo(json.dumps({"ok": False, "error": "invalid", "problems": e.problems},
                                  indent=2))
        else:
            for p in e.problems:
                click.echo(f"error: {p}", err=True)
        sys.exit(e.exit_code)

    say = (lambda m: click.echo(m, err=True)) if as_json else click.echo
    for a in actions:
        say(f"{a.action:<10} {a.name:<28} port {a.port}" + (f"  ({a.reason})" if a.reason else ""))
    failed = False
    if not dry_run:
        out = sys.stderr if as_json else sys.stdout
        for a in [a for a in actions if a.action in ("stop", "restart")]:
            # A restart is planned: held, so callers get 503 (not 404) until
            # the new instance is up, and kept in autostart meanwhile.
            planned = a.action == "restart"
            if planned and (old := proc.find(a.name)) is not None:
                proc.hold(a.name, f"llmctl apply: {a.reason}", group=old.group)
            rec = proc.stop(a.name, keep_autostart=planned)
            if rec is not None:
                proc.record_event(a.name, "stopped", f"by apply, {a.action}: {a.reason}; "
                                  f"pid {rec.pid}", pid=rec.pid, port=rec.port, by="apply",
                                  reason=f"{a.action}: {a.reason}")
        for a in [a for a in actions if a.action in ("start", "restart")]:
            say(f"\n==> llmctl run {shlex.join(a.argv)}")
            try:
                with contextlib.redirect_stdout(out):
                    run.main(args=[*a.argv, "--wait" if wait else "--no-wait"],
                             prog_name="llmctl run", standalone_mode=False,
                             obj={"by": "apply", "reason": f"{a.action}: {a.reason}"})
            except SystemExit as ex:
                if ex.code:
                    a.action, a.reason, failed = "error", "did not become healthy", True
            except click.ClickException as ex:
                a.action, a.reason, failed = "error", ex.format_message(), True
            except Exception as ex:          # one bad entry never stops the rest
                a.action, a.reason, failed = "error", str(ex), True
            finally:
                proc.release(a.name)
            if a.action == "error":
                say(f"error: {a.name}: {a.reason}")
    if as_json:
        click.echo(json.dumps({"ok": not failed, "dry_run": dry_run,
                               "actions": [a.to_json() for a in actions]}, indent=2))
    elif not dry_run:
        changed = sum(a.action in ("start", "restart", "stop") for a in actions)
        click.echo(f"\n{changed} change(s), {sum(a.action == 'unchanged' for a in actions)} "
                   "unchanged" + (", with errors" if failed else "") + ".")
    if failed:
        sys.exit(1)


# ------------------------------------------------------------ bundles --

def _bundle_fail(e: bundle.BundleError, as_json: bool) -> None:
    """Report a bundle error and exit with its documented code."""
    if as_json:
        click.echo(json.dumps(e.to_json(), indent=2))
    else:
        click.echo(f"error: {e}", err=True)
        for m in e.details.get("mismatches", []):
            what = ("missing" if m.get("missing") else
                    f"size {m['actual_size']} != {m['expected_size']}" if "actual_size" in m
                    else f"sha256 {m.get('actual')} != {m['expected']}")
            click.echo(f"  {m['path']}: {what}", err=True)
    sys.exit(e.exit_code)


def _progress(as_json: bool):
    """Progress on stderr: big files by name, the rest every few seconds."""
    last = [0.0]

    def show(i: int, n: int, f) -> None:
        big = getattr(f, "size", 0) >= 256 << 20
        if big or time.time() - last[0] >= 5 or i == n - 1:
            last[0] = time.time()
            click.echo(f"  [{i + 1}/{n}] {getattr(f, 'path', f)}", err=True)
    return show if sys.stderr.isatty() or as_json else None


def _expected_cache_rule(caches: list[dict]) -> str:
    """The rule a vllm start with today's config would match, for the report."""
    rt = containers.runtime_from_config(VLLM, config_mod.load())
    docker = rt["kind"] == containers.DOCKER
    digest = containers.pinned_digest(rt.get("image") or "") if docker else None
    ver = None if docker else containers.engine_version(VLLM)
    _, rule = containers.match_cache(caches, kind="vllm", digest=digest, engine_ver=ver,
                                     arch=containers.gpu_arch(0))
    return rule


@cli.command("install")
@click.argument("source")
@click.option("--name", default=None,
              help="Folder under MODELS_DIR (default: the manifest's name). Install "
                   "versions side by side, e.g. --name qwen3.5-9b-2026.10.1.")
@click.option("--replace", is_flag=True, default=False,
              help="Replace an existing folder of that name (refused while it's served).")
@click.option("--plain-http", is_flag=True, default=False,
              help="oci:// only: talk HTTP to the registry (a local zot without TLS).")
@click.option("--json", "as_json", is_flag=True, default=False,
              help="Print one JSON result on stdout; messages go to stderr.")
def install_cmd(source: str, name: str | None, replace: bool, plain_http: bool,
                as_json: bool) -> None:
    """Install a model bundle into MODELS_DIR, verifying every file.

    SOURCE is a bundle directory (holding llmctl-manifest.json) or
    oci://registry/repo@sha256:… (pulled with the `oras` CLI). Every listed
    file's size and sha256 must match the manifest, or nothing is installed.
    Works offline.

    \b
    Exit codes: 0 installed, 3 bad manifest, 4 checksum mismatch,
    5 name conflict, 6 source unreachable, 1 anything else.
    """
    served = {p.model_path for p in proc.list_running()}
    tmp: Path | None = None
    try:
        if source.startswith("oci://"):
            MODELS_DIR.mkdir(parents=True, exist_ok=True)
            tmp = Path(tempfile.mkdtemp(prefix=".pull-", dir=MODELS_DIR))
            if not as_json:
                click.echo(f"Pulling {source} ...", err=True)
            src_dir = bundle.pull_oci(source.removeprefix("oci://"), tmp,
                                      plain_http=plain_http)
            move = True
        else:
            src_dir = Path(source).expanduser()
            if not src_dir.is_dir():
                raise bundle.BadManifest(f"{source} is not a directory or an oci:// ref")
            move = False
        result = bundle.install(src_dir, MODELS_DIR, name=name, replace=replace,
                                served=served, move=move, progress=_progress(as_json))
    except bundle.BundleError as e:
        _bundle_fail(e, as_json)
        return
    finally:
        if tmp is not None:
            shutil.rmtree(tmp, ignore_errors=True)
    result["cache"] = ({"shipped": True, "seeded": False,
                        "expected_rule": _expected_cache_rule(result["caches"]),
                        "note": "seeded into the model's writable cache on its first start"}
                       if result["caches"] else {"shipped": False, "seeded": False})
    if as_json:
        click.echo(json.dumps(result, indent=2))
        return
    m = result["model"]
    click.echo(f"Installed {result['name']}: {m['id']} @ {m['revision']} ({m['license']})")
    click.echo(f"  {result['files_verified']} files verified, "
               f"{_fmt_size(result['bytes']).strip()}, manifest sha256 "
               f"{result['manifest_sha256']}")
    click.echo(f"  {result['path']}")
    if result["caches"]:
        click.echo(f"  ships a compile cache (with today's config: "
                   f"{result['cache']['expected_rule']} match)")


def _model_folder(target: str) -> Path:
    p = Path(target).expanduser()
    if p.is_dir():
        return p.resolve()
    specs = discover_models()
    if target in specs:
        mp = specs[target].model_path
        return mp if mp.is_dir() else mp.parent
    if (MODELS_DIR / target).is_dir():
        return MODELS_DIR / target
    raise click.BadArgumentUsage(f"'{target}' is not a model key or folder.")


@cli.command("verify")
@click.argument("model")
@click.option("--json", "as_json", is_flag=True, default=False)
def verify_cmd(model: str, as_json: bool) -> None:
    """Re-hash an installed model against the checksums recorded at install.

    Exit codes: 0 all match, 3 not installed from a bundle, 4 mismatch."""
    folder = _model_folder(model)
    try:
        result = bundle.verify(folder)
    except bundle.BundleError as e:
        _bundle_fail(e, as_json)
        return
    if as_json:
        click.echo(json.dumps(result, indent=2))
    else:
        click.echo(f"OK: {result['files_verified']} files in {folder} match "
                   f"(manifest sha256 {result['manifest_sha256']}).")


@cli.command("manifest")
@click.argument("folder", type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.option("--id", "model_id", default=None,
              help="Upstream model id (default: the repo recorded at download).")
@click.option("--revision", default=None,
              help="Upstream revision, e.g. the commit (default: recorded, else 'unknown').")
@click.option("--license", "license_", required=True, help="SPDX id, or 'other'.")
@click.option("--license-file", default=None, help="The licence file in FOLDER, if any.")
@click.option("--name", default=None, help="Default install name (default: FOLDER's name).")
@click.option("--output", "-o", type=click.Path(path_type=Path), default=None,
              help="Where to write it (default: FOLDER/llmctl-manifest.json).")
def manifest_cmd(folder: Path, model_id: str | None, revision: str | None,
                 license_: str, license_file: str | None, name: str | None,
                 output: Path | None) -> None:
    """Write an llmctl-manifest.json for FOLDER, making it a bundle.

    Hashes every file (llmctl's own files and hidden top-level entries
    excepted). Cache subtrees written by `llmctl cache export` are listed
    under `caches`."""
    src = _read_source(folder)
    model_id = model_id or src.get("repo_id")
    if not model_id:
        raise click.UsageError("No recorded source repo; pass --id.")
    fmt = "hf" if backends.is_hf_model_dir(folder) else "gguf"
    try:
        data = bundle.build_manifest(
            folder, name=name or folder.name, model_id=model_id,
            revision=revision or src.get("revision") or "unknown", license=license_,
            fmt=fmt, license_file=license_file, progress=_progress(False))
    except bundle.BundleError as e:
        _bundle_fail(e, False)
        return
    out = output or folder / bundle.MANIFEST_FILE
    out.write_text(json.dumps(data, indent=2) + "\n")
    click.echo(f"Wrote {out}: {len(data['files'])} files"
               + (f", {len(data['caches'])} cache(s)" if data.get("caches") else ""))


@cli.group("cache")
def cache_group() -> None:
    """Per-model compile caches (vLLM torch.compile, Triton, FlashInfer)."""


@cache_group.command("export")
@click.argument("name")
@click.argument("bundle_dir", type=click.Path(file_okay=False, path_type=Path))
@click.option("--path", "path_in_bundle", default="cache/vllm", show_default=True,
              help="Where in BUNDLE_DIR the cache goes (its manifest `caches.path`).")
@click.option("--from", "from_dir", type=click.Path(exists=True, file_okay=False,
                                                     path_type=Path), default=None,
              help="The cache dir to export, when NAME has several.")
@click.option("--json", "as_json", is_flag=True, default=False)
def cache_export(name: str, bundle_dir: Path, path_in_bundle: str,
                 from_dir: Path | None, as_json: bool) -> None:
    """Copy model NAME's warm compile cache into BUNDLE_DIR.

    Start NAME once (in the image the bundle targets) so it compiles, then
    export. `llmctl manifest` lists the exported subtree under `caches`.

    \b
    Exit codes: 0 exported, 3 bad --path, 1 no cache / not empty."""
    try:
        path_in_bundle = bundle.safe_relpath(path_in_bundle, "--path")
    except bundle.BundleError as e:
        _bundle_fail(e, as_json)
        return
    if from_dir is None:
        running = proc.find(name)
        cache = (running.runtime or {}).get("cache_dir") if running else None
        if cache:
            from_dir = Path(cache)
        else:
            dirs = containers.find_cache_dirs(proc.STATE_DIR, name)
            if len(dirs) != 1:
                msg = (f"no compile cache for '{name}'" if not dirs else
                       f"'{name}' has {len(dirs)} caches; pick one with --from: "
                       + ", ".join(map(str, dirs)))
                _bundle_fail(bundle.BundleError(msg), as_json)
                return
            from_dir = dirs[0]
    out = bundle_dir / path_in_bundle
    try:
        entry = containers.export_cache(from_dir, out, path_in_bundle=path_in_bundle)
    except containers.ContainerError as e:
        _bundle_fail(bundle.BundleError(str(e)), as_json)
        return
    files = sum(1 for p in out.rglob("*") if p.is_file())
    size = sum(p.stat().st_size for p in out.rglob("*") if p.is_file())
    result = {"ok": True, "name": name, "from": str(from_dir), "path": str(out),
              "files": files, "bytes": size, "cache": entry}
    if as_json:
        click.echo(json.dumps(result, indent=2))
    else:
        click.echo(f"Exported {files} files ({_fmt_size(size).strip()}) from {from_dir}\n"
                   f"  to {out}\n  manifest entry: {json.dumps(entry)}")


# Non-model leftovers a download leaves in a model folder: the provenance
# sidecar and huggingface_hub's local_dir cache. A folder holding nothing else
# once its models are gone is removed too.
_FOLDER_LEFTOVERS = {_SOURCE_FILE, bundle.INSTALLED_MANIFEST, ".cache"}


def _removal_plan(spec: ModelSpec) -> list[Path]:
    """Paths to delete for ``spec``: the checkpoint dir (vllm), or the GGUF
    plus companions (mmproj / drafter) when no other base GGUF in its folder
    still uses them — and the folder itself once nothing else is left."""
    if spec.model_path.is_dir():
        return [spec.model_path]
    folder = spec.model_path.parent
    paths = [spec.model_path]
    siblings = [s for s in discover_models().values()
                if s.model_path.parent == folder and s.model_path != spec.model_path
                and not s.model_path.is_dir()]
    if not siblings:
        paths += [p for p in (spec.mmproj_path, spec.draft_path) if p]
        rest = {e.name for e in folder.iterdir()} - {p.name for p in paths}
        if folder != MODELS_DIR and rest <= _FOLDER_LEFTOVERS:
            return [folder]
    return paths


@cli.command("rm")
@click.argument("model", required=False)
@click.option("--yes", "-y", is_flag=True, help="Don't ask for confirmation.")
def rm(model: str | None, yes: bool) -> None:
    """Delete a downloaded MODEL from disk.

    GGUF: removes the file, plus its mmproj / drafter (and the folder) when no
    other quant in the folder still uses them. vllm: removes the checkpoint
    directory. Refuses while the model is running, and drops it from the
    autostart manifest. With no MODEL, pick one interactively.
    """
    specs = discover_models()
    if not specs:
        raise click.UsageError("No models found on disk.")
    if not model:
        model = _pick_model_interactively(specs, "Select a model to delete:")
    if model not in specs:
        available = "\n  ".join(specs)
        raise click.BadArgumentUsage(
            f"Unknown model '{model}'. Available:\n  {available}")
    spec = specs[model]

    running = [p for p in proc.list_running()
               if Path(p.model_path).resolve() == spec.model_path.resolve()]
    if running:
        names = ", ".join(f"{p.name} (pid {p.pid})" for p in running)
        raise click.UsageError(f"'{model}' is running: {names}. `llmctl stop` it first.")

    paths = _removal_plan(spec)
    base = MODELS_DIR.resolve()
    for p in paths:
        # Never reach outside the models dir (e.g. through a symlinked folder).
        if p.resolve() == base or not p.resolve().is_relative_to(base):
            raise click.ClickException(f"Refusing to delete {p}: outside {base}.")

    total = 0
    for p in paths:
        size = (backends.hf_dir_size(p) if p.is_dir() else p.stat().st_size)
        total += size
        click.echo(f"  {_fmt_size(size).strip():>8}  {p}{'/' if p.is_dir() else ''}")
    if not yes and not click.confirm(
            f"Delete {model} ({_fmt_size(total).strip()})?", default=False):
        raise click.Abort()

    for p in paths:
        if p.is_dir():
            shutil.rmtree(p)
        else:
            p.unlink()
    for s in proc.autostart_list():
        if Path(s.model_path).resolve() == spec.model_path.resolve():
            proc.autostart_remove(s.name)
            click.echo(f"Removed {s.name} from the autostart manifest.")
    click.echo(f"Deleted {model}.")


if __name__ == "__main__":
    cli()

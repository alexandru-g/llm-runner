"""CLI for managing background model servers (llama-server / vllm)."""
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
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

import click

from llm_runner import __version__
from llm_runner import backends
from llm_runner import config as config_mod
from llm_runner import process as proc
from llm_runner import registry
from llm_runner.backends import DECIDER, GATEWAY, LLAMA, VLLM
from llm_runner.registry import MODELS_DIR, ModelSpec, discover_models


_port_in_use = proc.port_in_use


def _next_free_port(start: int = 8080) -> int:
    p = start
    while _port_in_use(p):
        p += 1
    return p


# Substrings that mark a GGUF as an embedding model, so `run` can default to
# --embeddings without an explicit flag. Matched case-insensitively against the
# model's filename stem. Override either way with --embeddings/--no-embeddings.
_EMBEDDING_NAME_HINTS = (
    "bge", "e5", "gte", "nomic-embed", "mxbai-embed", "arctic-embed",
    "minilm", "embed",
)


def _looks_like_embedding_model(name: str) -> bool:
    low = name.lower()
    return any(hint in low for hint in _EMBEDDING_NAME_HINTS)


def _build_llama_cmd(model_path: Path, mmproj: Path | None, port: int, ctx_size: int,
                     reasoning: bool, *, kv_quant: str = "f16",
                     n_gpu_layers: int = 999, embeddings: bool = False,
                     pooling: str = "cls", batch_size: int | None = None,
                     ubatch_size: int | None = None, draft_path: Path | None = None,
                     spec_type: str = "draft-mtp", spec_n_max: int = 3,
                     n_parallel: int | None = None,
                     cache_ram: int = 0,
                     kv_unified: bool | None = None,
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
            "--host", "0.0.0.0",
            "--embeddings",
            "--pooling", pooling,
            *(extra or []),
        ]
    cmd = [
        "llama-server",
        "--model", str(model_path),
        "--n-gpu-layers", str(n_gpu_layers),
        "--ctx-size", str(ctx_size),
        "--flash-attn", "on",
        "--cache-reuse", "256",
        # Host-RAM prompt cache. Off by default: on models with sliding-window
        # layers (Gemma 4) every saved prompt carries its SWA context
        # checkpoints (~1.7 GB for a 900-token prompt), so the save costs more
        # than the prefill it spares unless prompts are long AND share a long
        # prefix. Measured on the enersec v2 pipeline: 330 ms/call of cache
        # memcpy vs 225 ms/call of inference. See --cache-ram.
        "--cache-ram", str(cache_ram),
        "--threads", "8",
        "--port", str(port),
        "--host", "0.0.0.0",
    ]
    if n_parallel is not None:
        cmd.extend(["--parallel", str(n_parallel)])
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
    ``--speculative-config`` model).

    ``draft_spec`` (from ``--draft``) may be a path (GGUF file, or checkpoint
    dir for vllm), a registry key, or — vllm only — a Hugging Face repo id
    that vllm fetches itself. When it is omitted, fall back to the drafter
    auto-attached to ``auto_spec`` (a sibling ``*-MTP`` / ``*-assistant``
    GGUF), if any — llama only, since that file is a GGUF.
    """
    if draft_spec:
        p = Path(draft_spec).expanduser()
        if p.is_file() or (backend == VLLM and p.is_dir()):
            return p.resolve()
        specs = discover_models()
        if draft_spec in specs:
            cand = specs[draft_spec]
            if cand.backend != backend:
                raise click.BadArgumentUsage(
                    f"--draft '{draft_spec}' is a {cand.backend} model; the "
                    f"drafter must match the {backend} backend.")
            return cand.model_path
        if backend == VLLM and re.fullmatch(r"[\w.-]+/[\w.-]+", draft_spec):
            return Path(draft_spec)  # HF repo id; vllm downloads it
        raise click.BadArgumentUsage(
            f"--draft '{draft_spec}' is neither an existing model file/dir nor a "
            "known model key."
        )
    if backend == VLLM:
        return None
    return auto_spec.draft_path if auto_spec else None


def _run_option_args(ctx: click.Context) -> list[str]:
    """Rebuild the option part of the `llmctl run` argv from the params the
    user passed on the command line (defaults are omitted so re-detection —
    e.g. embeddings-by-filename — still applies on a re-run). ``--port`` is
    left out: the manifest pins it separately. So are ``--wait`` and
    ``--menu``: they're about this invocation, not the launch."""
    src = click.core.ParameterSource.COMMANDLINE
    args: list[str] = []
    for p in ctx.command.params:
        if not isinstance(p, click.Option) or p.name in ("port", "wait", "menu"):
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


def _require_backend(backend: str) -> str:
    """Return the server binary for ``backend`` or fail with an install hint."""
    binary = backends.find(backend)
    if binary is None:
        raise click.UsageError(backends.missing_hint(backend))
    return binary


# `run` options that only mean something to llama-server. Passing one with
# --backend vllm is almost certainly a mistake (or a stale `restart`
# override), so warn instead of silently dropping it.
_LLAMA_ONLY_PARAMS = {
    "reasoning": "--reasoning", "n_gpu_layers": "--n-gpu-layers",
    "no_vision": "--no-vision", "kv_unified": "--kv-unified/--no-kv-unified",
    "cache_ram": "--cache-ram", "ctx_checkpoints": "--ctx-checkpoints",
    "pooling": "--pooling",
    "batch_size": "--batch-size", "ubatch_size": "--ubatch-size",
}
_VLLM_ONLY_PARAMS = {
    "gpu_mem": "--gpu-mem", "tool_parser": "--tool-parser",
    "reasoning_parser": "--reasoning-parser",
    "trust_remote_code": "--trust-remote-code",
}


def _warn_foreign_flags(ctx: click.Context, backend: str) -> None:
    src = click.core.ParameterSource.COMMANDLINE
    if backend == DECIDER:
        # decider takes only --port / --n-parallel / -X / -E; everything
        # engine-specific is meaningless there.
        groups = {LLAMA: _LLAMA_ONLY_PARAMS, VLLM: _VLLM_ONLY_PARAMS}
    else:
        groups = {VLLM: _VLLM_ONLY_PARAMS} if backend == LLAMA else {LLAMA: _LLAMA_ONLY_PARAMS}
    for other, foreign in groups.items():
        used = [flag for name, flag in foreign.items()
                if ctx.get_parameter_source(name) == src]
        if used:
            click.echo(f"warning: {other}-only option(s) ignored for {backend}: "
                       f"{', '.join(used)}", err=True)


def _wait_healthy(rec: proc.RunningProcess, timeout: float) -> bool:
    """Poll ``/health`` until the server answers 200, the process dies, or
    ``timeout`` elapses. Both llama-server and vllm expose it."""
    url = f"http://127.0.0.1:{rec.port}/health"
    deadline = time.time() + timeout
    spinner = "|/-\\"
    i = 0
    while time.time() < deadline:
        if not proc._alive(rec.pid):
            click.echo(f"\rServer exited before becoming healthy — see "
                       f"`llmctl logs {rec.name}`.", err=True)
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
    if os.environ.get("LLM_RUNNER_VLLM_BIN"):
        click.echo("  (vllm overridden by $LLM_RUNNER_VLLM_BIN)")
    if os.environ.get("LLM_RUNNER_DECIDER_DIR"):
        click.echo("  (decider overridden by $LLM_RUNNER_DECIDER_DIR)")
    gw_port = config_mod.get_gateway_port()
    click.echo(f"\nGateway: {'off' if gw_port is None else f'port {gw_port}'}")


@config.command("vllm-bin")
@click.argument("path", required=False,
                type=click.Path(dir_okay=False, path_type=Path))
@click.option("--clear", is_flag=True, help="Forget the stored path.")
def config_vllm_bin(path: Path | None, clear: bool) -> None:
    """Persist the path to the `vllm` CLI, or print it when PATH is omitted.

    Only needed when vllm lives somewhere llmctl doesn't look: PATH, a
    conda/venv env named `vllm`, or any conda env. Example:

      llmctl config vllm-bin ~/miniconda3/envs/vllm/bin/vllm
    """
    if clear:
        config_mod.set_vllm_bin(None)
        click.echo("vllm_bin cleared")
        return
    if path is None:
        configured = config_mod.get_vllm_bin()
        if configured:
            click.echo(str(configured))
        else:
            found = backends.find_vllm()
            click.echo(f"(not set; {'auto-detected ' + found if found else 'vllm not found'})")
        return
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        click.echo(f"warning: {resolved} does not exist", err=True)
    saved = config_mod.set_vllm_bin(resolved)
    click.echo(f"vllm_bin set to {saved}")
    if os.environ.get("LLM_RUNNER_VLLM_BIN"):
        click.echo("note: $LLM_RUNNER_VLLM_BIN is set and overrides this.", err=True)


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
    click.echo("")
    click.echo(f"{'KEY':<{key_w}}  {'BACKEND':<7}  {'SIZE':>8}  {'VISION':<6}  {'DRAFT':<6}  {'DECISION':<8}  PATH")
    click.echo("-" * (key_w + 67))
    for key, spec in specs.items():
        size = _fmt_size(spec.size).strip()
        vision = "yes" if spec.has_vision else ""
        draft = "yes" if spec.has_draft else ""
        decision = "yes" if spec.decision else ""
        rel = spec.model_path.relative_to(MODELS_DIR) if spec.model_path.is_relative_to(MODELS_DIR) else spec.model_path
        click.echo(f"{key:<{key_w}}  {spec.backend:<7}  {size:>8}  {vision:<6}  {draft:<6}  {decision:<8}  {rel}")


@cli.command()
@click.argument("model", required=False)
@click.option("--backend", type=click.Choice(["auto", LLAMA, VLLM, DECIDER]), default="auto",
              show_default=True,
              help="Serving engine. auto = llama-server for a GGUF, vllm for a "
                   "Hugging Face checkpoint directory.")
@click.option("--port", "-p", type=int, default=None,
              help="Listen port. Auto-selected starting at 8080 if omitted or taken.")
@click.option("--ctx-size", "-c", type=int, default=None,
              help="Context size (default: 32768 for chat, 8192 for --embeddings). "
                   "vllm: --max-model-len.")
@click.option("--ft", "ft_path",
              type=click.Path(exists=True, path_type=Path),
              help="Path to a finetuned GGUF file, or an HF checkpoint directory "
                   "for vllm (bypasses the registry).")
@click.option("--name", "ft_name", default=None,
              help="Override the display name (useful with --ft). For vllm this "
                   "is also the model id clients send (--served-model-name).")
@click.option("--group", "group_name", default=None,
              help="Second name the gateway routes by, shared by interchangeable "
                   "models (e.g. every quant of one model, across machines). Default: "
                   "the model's folder, e.g. gemma-4-26b-a4b-it. 'none' for no group.")
@click.option("--wait/--no-wait", "wait", default=False,
              help="Block until the server answers /health (or exits). Handy for "
                   "vllm, which takes a minute or more to load and compile.")
@click.option("--menu/--no-menu", "-i", "menu", default=None,
              help="Show the launch-options menu (vision, drafter, reasoning, "
                   "context, KV cache, slots) before starting. On by default when "
                   "MODEL is picked interactively; choices are remembered per model.")
@click.option("--extra", "-X", "extra", multiple=True,
              help="Extra argument(s) appended verbatim to the server command; "
                   "repeatable, each value is shell-split. e.g. "
                   "-X '--max-num-batched-tokens 8192'.")
@click.option("--env", "-E", "env_pairs", multiple=True, metavar="KEY=VALUE",
              help="Environment variable for the server process; repeatable. "
                   "Remembered for restart/restore. e.g. "
                   "-E VLLM_USE_FLASHINFER_SAMPLER=0.")
@click.option("--reasoning", is_flag=True, default=False,
              help="Enable reasoning. Disabled by default (--reasoning-budget 0). "
                   "llama only; for vllm see --reasoning-parser.")
@click.option("--kv-quant", type=click.Choice(["f16", "q8_0", "q4_0"]),
              default="f16",
              help="Quantize the KV cache to save VRAM. q8_0 ≈ half the f16 size "
                   "and is near-lossless; q4_0 halves it again with some quality "
                   "cost. Great for fitting big models / long context in VRAM. "
                   "vllm: q8_0 → --kv-cache-dtype fp8 (no 4-bit KV in vllm).")
@click.option("--gpu-mem", "gpu_mem", type=click.FloatRange(0.05, 1.0), default=None,
              help="vllm only: fraction of *total* GPU memory vllm may claim "
                   "(--gpu-memory-utilization). vllm refuses to start if that "
                   "much isn't free, so the default is computed from the VRAM "
                   "currently free (minus headroom, capped at vllm's own 0.9) — "
                   "pass it explicitly to pin a value.")
@click.option("--tool-parser", "tool_parser", default=None,
              help="vllm only: enable OpenAI tool calling with this parser "
                   "(--enable-auto-tool-choice --tool-call-parser), e.g. hermes, "
                   "llama3_json, mistral, pythonic, qwen3_coder.")
@click.option("--reasoning-parser", "reasoning_parser", default=None,
              help="vllm only: split reasoning into reasoning_content with this "
                   "parser (--reasoning-parser), e.g. deepseek_r1, qwen3.")
@click.option("--trust-remote-code", is_flag=True, default=False,
              help="vllm only: allow the checkpoint's custom modeling code.")
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
                   "llama-server auto-selects. vllm: --max-num-seqs (default 256).")
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
                   "0 disables it. Only worth enabling for long prompts that "
                   "share a long prefix (e.g. 16384 for a multi-KB system prompt "
                   "reused across calls); for many short, mostly-distinct "
                   "prompts the per-call state save costs more than it saves.")
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
              help="Logical batch size for --embeddings mode "
                   "(default: the context size). Ignored without --embeddings.")
@click.option("--ubatch-size", "-ub", "ubatch_size", type=int, default=None,
              help="Physical batch size for --embeddings mode "
                   "(default: the context size). Ignored without --embeddings.")
@click.pass_context
def run(ctx: click.Context, model: str | None, backend: str, port: int | None,
        ctx_size: int | None, ft_path: Path | None, ft_name: str | None,
        group_name: str | None, wait: bool, menu: bool | None, extra: tuple[str, ...], env_pairs: tuple[str, ...],
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
    on vllm, or on decider.serve when they carry a decider_config.json (see
    --backend). Use --ft <path> to bypass the registry.
    """
    spec: ModelSpec | None = None
    if ft_path:
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
        if (menu or (menu is None and picked)) and sys.stdin.isatty():
            from llm_runner import launch
            target = launch.Target(spec, spec.backend if backend == "auto" else backend)
            if launch.configure(ctx, target):
                # Start over with the chosen values, as if they had been typed.
                return ctx.invoke(run.callback, **{**ctx.params, "model": model, "menu": False})
        model_path = spec.model_path
        mmproj = None if no_vision else spec.mmproj_path
        name = ft_name or spec.key
        detected = spec.backend

    # Backend: what the model format implies unless the user forces it. A
    # GGUF *can* be forced onto vllm (experimental there, needs --tokenizer via
    # -X) but an HF directory can never run on llama-server.
    if backend == "auto":
        backend = detected
    elif backend == LLAMA and model_path.is_dir():
        raise click.UsageError(
            f"'{model_path.name}' is a Hugging Face checkpoint directory; "
            "llama-server only loads GGUF files. Use --backend vllm.")
    elif backend == DECIDER and not backends.is_decider_model_dir(model_path):
        raise click.UsageError(
            f"'{model_path.name}' is not a decider checkpoint (no decider_config.json).")
    binary = _require_backend(backend)
    _warn_foreign_flags(ctx, backend)
    if backend == VLLM and kv_quant == "q4_0":
        raise click.UsageError("vllm has no 4-bit KV cache; use --kv-quant q8_0 "
                               "(→ fp8) or f16.")
    # decider.serve speaks the decision protocol itself; on llama/vllm a
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
        embeddings = _looks_like_embedding_model(stem)
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
            n_parallel=n_parallel, extra=extra_args)
        env = {**launch_env, **env}      # -E overrides the profile defaults
    elif backend == VLLM:
        if gpu_mem is None:
            picked = backends.default_gpu_mem()
            if picked:
                gpu_mem, why = picked
                click.echo(f"Using --gpu-mem {gpu_mem:g} ({why}); pass --gpu-mem "
                           "to override.")
        cmd = backends.build_vllm_cmd(
            binary, model_path, name=name, port=chosen_port, ctx_size=ctx_size,
            kv_quant=kv_quant, gpu_mem=gpu_mem, n_parallel=n_parallel,
            embeddings=embeddings, draft_path=draft_path, spec_type=spec_type,
            spec_n_max=spec_n_max, tool_parser=tool_parser,
            reasoning_parser=reasoning_parser,
            trust_remote_code=trust_remote_code, extra=extra_args)
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

        cmd = _build_llama_cmd(model_path, mmproj, chosen_port, ctx_size, reasoning,
                               kv_quant=kv_quant, n_gpu_layers=n_gpu_layers,
                               embeddings=embeddings, pooling=pooling,
                               batch_size=batch_size, ubatch_size=ubatch_size,
                               draft_path=draft_path, spec_type=spec_type,
                               spec_n_max=spec_n_max, n_parallel=n_parallel,
                               cache_ram=cache_ram, kv_unified=kv_unified,
                               extra=extra_args)
        # Report the llmctl name as the model id (as vLLM's --served-model-name
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
                     backend=backend, env=env, group=group)
    click.echo(f"Started {rec.name}  pid={rec.pid}  port={rec.port}  backend={backend}"
               + (f"  group={group}" if group else ""))
    _ensure_gateway()
    if env:
        click.echo("Env:     " + " ".join(f"{k}={v}" for k, v in env.items()))
    if draft_path:
        if backend == VLLM:
            click.echo(f"Speculative drafter: {draft_path.name}  "
                       f"(num_speculative_tokens={spec_n_max})")
        else:
            click.echo(f"Speculative drafter: {draft_path.name}  "
                       f"(--spec-type {spec_type} --spec-draft-n-max {spec_n_max})")
    click.echo(f"Command: {shlex.join(cmd)}")
    click.echo(f"Logs:  {rec.log_file}")
    click.echo(f"Tail:  llmctl logs {rec.name}")
    if wait:
        # vllm loads weights, profiles memory and captures CUDA graphs before
        # it listens; llama-server is usually up in seconds.
        ok = _wait_healthy(rec, timeout=600 if backend in (VLLM, DECIDER) else 120)
        if not ok:
            raise SystemExit(1)
    elif backend == VLLM:
        click.echo("Note: vllm takes a minute or more to become ready; use "
                   "--wait to block on /health.")
    elif backend == DECIDER:
        click.echo("Note: decider loads the weights and captures its CUDA graphs "
                   "before it listens (~30 s); use --wait to block on /health.")


@cli.command("ps")
@click.option("--all", "-a", "show_all", is_flag=True,
              help="Every model in the cluster, with the node calls go to (needs the gateway).")
def ps_cmd(show_all: bool) -> None:
    """List running models."""
    if show_all:
        return _ps_cluster()
    procs = proc.list_running()
    if not procs:
        click.echo("(no models running)")
        return
    click.echo(f"{'PID':<8} {'PORT':<6} {'BACKEND':<8} {'STARTED':<20} NAME")
    click.echo("-" * 78)
    for p in procs:
        started = dt.datetime.fromtimestamp(p.started_at).strftime("%Y-%m-%d %H:%M:%S")
        click.echo(f"{p.pid:<8} {p.port:<6} {p.backend:<8} {started:<20} {p.name}")
    gw = _gateway_record()
    if gw:
        click.echo(f"\nAll models: {_gateway_url(gw.port)}  (GET /v1/models lists them)")


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
        with urllib.request.urlopen(f"http://127.0.0.1:{port}{path}", timeout=5) as r:
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
    # A decision endpoint is useless without its model: take it down too.
    if rec.backend != backends.DECIDE:
        for p in _proxy_attached_to(rec.port):
            proc.stop(str(p.pid))
            click.echo(f"Stopped {p.name}  pid={p.pid}  port={p.port}")
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
    otherwise show an interactive picker.
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
            raise click.UsageError(f"No running model matches '{identifier}'.")
    log_path = Path(target.log_file)
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

    conda_exe = os.environ.get("CONDA_EXE")
    if conda_exe:
        # Source conda + activate the env so PATH and LD_LIBRARY_PATH (CUDA
        # libs, llama-server) are exactly what works in the interactive shell.
        profile = Path(conda_exe).resolve().parents[1] / "etc" / "profile.d" / "conda.sh"
        env_name = os.environ.get("CONDA_DEFAULT_ENV", "base")
        exec_start = (
            f"/bin/bash -lc 'source \"{profile}\" && conda activate {env_name} "
            f"&& exec \"{llmctl}\" restore'"
        )
    else:
        # No conda: bake the current PATH/LD_LIBRARY_PATH so llama-server's
        # shared libs still resolve under systemd's otherwise-bare environment.
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
    # Only demand the binaries the manifest actually needs. A vllm entry
    # records an absolute binary path, so it doesn't depend on PATH.
    wanted = {s.backend for s in proc.autostart_list()
              if not names or s.name in names}
    for backend in sorted(wanted):
        if backends.find(backend) is None:
            click.echo(f"warning: {backends.missing_hint(backend)}", err=True)
    results = proc.restore(list(names) or None)
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
    """
    if identifier is None:
        running = _models_running()
        if not running:
            raise click.UsageError(
                "No models running. Pass a name from `llmctl autostart list`.")
        identifier = _pick_instance_interactively(running).name
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
        # Legacy manifest entry: relaunch the resolved server command as-is.
        rec = proc.start(name=spec.name, cmd=spec.cmd, port=spec.port,
                         model_path=spec.model_path, backend=spec.backend)
        click.echo(f"Started {rec.name}  pid={rec.pid}  port={rec.port}")
        return

    # Re-issue `llmctl run` with the remembered args; overrides come last so
    # they win, and `run` re-records the merged set in the manifest.
    args = spec.run_cmd()[1:] + list(overrides)
    click.echo(f"Re-running: llmctl run {shlex.join(args)}")
    sub = run.make_context("llmctl run", args, parent=ctx)
    with sub:
        run.invoke(sub)


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
    needle = f"http://127.0.0.1:{port}"
    return [p for p in proc.list_running() if p.backend == backends.DECIDE and needle in p.cmd]


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
    cmd = [binary, "serve", "--backend", f"http://127.0.0.1:{target.port}",
           "--port", str(chosen_port), "--profile", profile,
           "--concurrency", str(concurrency), "--cache", cache,
           *[tok for chunk in extra for tok in shlex.split(chunk)]]
    rec = proc.start(name=name, cmd=cmd, port=chosen_port, model_path=target.model_path,
                     backend=backends.DECIDE,
                     group=f"{target.group}.decide" if target.group else "")
    click.echo(f"Started {rec.name}  pid={rec.pid}  port={rec.port}  profile={profile}")
    click.echo(f"Endpoint: POST http://127.0.0.1:{rec.port}/v1/systemone  "
               f"(→ llama-server :{target.port})")
    click.echo(f"Contract: llmctl decide spec {target.name}   "
               f"or GET http://127.0.0.1:{rec.port}/v1/spec")
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
    return f"http://{socket.gethostname()}:{port}/v1"


def _start_gateway(port: int) -> proc.RunningProcess:
    cmd = [sys.executable, "-m", "llm_runner", "gateway", "serve", "--port", str(port)]
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
    if gw:
        proc.stop(str(gw.pid))
        if not wanted:
            click.echo(f"Stopped gateway  pid={gw.pid}  port={gw.port}")
            return
        for _ in range(20):
            if not _port_in_use(gw.port):
                break
            time.sleep(0.25)
        rec = _start_gateway(gw.port)
        click.echo(f"Restarted gateway  pid={rec.pid}  port={rec.port}")
    elif wanted:
        _ensure_gateway()


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


@node.command("set")
@click.option("--name", default=None, help="Node name (default: the hostname).")
@click.option("--priority", type=int, default=None,
              help="Lower is preferred: e.g. 0 for the fast workstation, 1 for the Orin.")
@click.option("--advertise", default=None,
              help="URL other nodes use to reach this gateway, e.g. http://orin.ts.net:9000 "
                   "(default: the address the registry sees).")
@click.option("--no-advertise", is_flag=True, help="Forget --advertise.")
def node_set(name: str | None, priority: int | None, advertise: str | None,
             no_advertise: bool) -> None:
    """Set this machine's name, priority or advertised URL."""
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
    if not values:
        raise click.UsageError("Nothing to set (--name, --priority, --advertise).")
    config_mod.update(**values)
    c = config_mod.get_cluster()
    click.echo(f"node {c.node_name}  priority={c.node_priority}"
               + (f"  advertise={c.node_advertise}" if c.node_advertise else ""))
    if _gateway_record():
        _restart_gateway()


def _require_gateway_on() -> None:
    if config_mod.get_gateway_port() is None:
        raise click.UsageError("The gateway is turned off; a cluster needs it "
                               "(`llmctl config gateway 9000`).")


def _normalize_url(url: str) -> str:
    if "://" not in url:
        url = "http://" + url
    parts = urllib.parse.urlsplit(url)
    if parts.port is None:
        url = f"{parts.scheme}://{parts.hostname}:{config_mod.DEFAULT_GATEWAY_PORT}"
    return url.rstrip("/")


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
    """Report to the registry at URL (host, host:port or http://host:port)."""
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
    click.echo(f"Wrote {unit_path}")
    if os.environ.get("CONDA_EXE"):
        click.echo(f"Conda env baked in: {os.environ.get('CONDA_DEFAULT_ENV', 'base')}")
    else:
        click.echo("warning: no conda env detected — baked current PATH/"
                   "LD_LIBRARY_PATH instead.", err=True)

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
    _run_download(repo_id, filename, no_mmproj, no_draft, subdir, hf=hf)


def _resolve_update_target(target: str | None) -> tuple[str, Path, str | None]:
    """Resolve an `update` TARGET to (repo_id, model_folder, known_filename).

    TARGET may be a model key (repo read from its provenance sidecar), a repo id
    (owner/name), or None (pick from models that have a recorded source).
    """
    specs = discover_models()
    updatable: dict[str, tuple[str, Path, str | None]] = {}
    for key, spec in specs.items():
        # GGUF: the folder holding the file. vllm: the checkpoint dir itself,
        # which is re-snapshotted whole (no single file to name).
        folder = spec.model_path if spec.backend == VLLM else spec.model_path.parent
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
    repo_id, target_dir, known_file = _resolve_update_target(target)
    if filename is None:
        filename = known_file  # may be None for a repo-id target → show picker
    click.echo(f"Updating from {repo_id}")
    hf = _read_source(target_dir).get("format") == "hf"
    _run_download(repo_id, filename, no_mmproj, no_draft, subdir=None,
                  revalidate=True, target_dir=target_dir, hf=hf)


# Non-model leftovers a download leaves in a model folder: the provenance
# sidecar and huggingface_hub's local_dir cache. A folder holding nothing else
# once its models are gone is removed too.
_FOLDER_LEFTOVERS = {_SOURCE_FILE, ".cache"}


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

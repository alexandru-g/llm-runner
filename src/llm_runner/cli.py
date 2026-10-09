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
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path

import click

from llm_runner import __version__
from llm_runner import backends
from llm_runner import config as config_mod
from llm_runner import process as proc
from llm_runner import registry
from llm_runner.backends import DECIDER, LLAMA, VLLM
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
    left out: the manifest pins it separately. So is ``--wait``: it's about
    this invocation, not the launch."""
    src = click.core.ParameterSource.COMMANDLINE
    args: list[str] = []
    for p in ctx.command.params:
        if not isinstance(p, click.Option) or p.name in ("port", "wait"):
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
    "cache_ram": "--cache-ram", "pooling": "--pooling",
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
    click.echo(f"{'KEY':<{key_w}}  {'BACKEND':<7}  {'SIZE':>8}  {'VISION':<6}  {'DRAFT':<6}  PATH")
    click.echo("-" * (key_w + 57))
    for key, spec in specs.items():
        size = _fmt_size(spec.size).strip()
        vision = "yes" if spec.has_vision else ""
        draft = "yes" if spec.has_draft else ""
        rel = spec.model_path.relative_to(MODELS_DIR) if spec.model_path.is_relative_to(MODELS_DIR) else spec.model_path
        click.echo(f"{key:<{key_w}}  {spec.backend:<7}  {size:>8}  {vision:<6}  {draft:<6}  {rel}")


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
@click.option("--wait/--no-wait", "wait", default=False,
              help="Block until the server answers /health (or exits). Handy for "
                   "vllm, which takes a minute or more to load and compile.")
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
        wait: bool, extra: tuple[str, ...], env_pairs: tuple[str, ...],
        reasoning: bool,
        kv_quant: str, gpu_mem: float | None, tool_parser: str | None,
        reasoning_parser: str | None, trust_remote_code: bool,
        n_gpu_layers: int, no_vision: bool,
        embeddings: bool | None, pooling: str, batch_size: int | None,
        ubatch_size: int | None, draft_spec: str | None, no_draft: bool,
        spec_type: str, spec_n_max: int, n_parallel: int | None,
        cache_ram: int, kv_unified: bool | None) -> None:
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
        if not model:
            model = _pick_model_interactively(specs)
        if model not in specs:
            available = "\n  ".join(specs)
            raise click.BadArgumentUsage(
                f"Unknown model '{model}'. Available:\n  {available}"
            )
        spec = specs[model]
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
    # Remember the llmctl-level invocation (with the interactively-picked model
    # made explicit) so `restart`/`autostart list` can show and re-issue it.
    run_args = ([] if ft_path else [model]) + _run_option_args(ctx)
    rec = proc.start(name=name, cmd=cmd, port=chosen_port,
                     model_path=str(model_path), run_args=run_args,
                     backend=backend, env=env)
    click.echo(f"Started {rec.name}  pid={rec.pid}  port={rec.port}  backend={backend}")
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
def ps_cmd() -> None:
    """List running models."""
    procs = proc.list_running()
    if not procs:
        click.echo("(no models running)")
        return
    click.echo(f"{'PID':<8} {'PORT':<6} {'BACKEND':<8} {'STARTED':<20} NAME")
    click.echo("-" * 78)
    for p in procs:
        started = dt.datetime.fromtimestamp(p.started_at).strftime("%Y-%m-%d %H:%M:%S")
        click.echo(f"{p.pid:<8} {p.port:<6} {p.backend:<8} {started:<20} {p.name}")


@cli.command()
@click.argument("identifier", required=False)
def stop(identifier: str | None) -> None:
    """Stop a running model by NAME or PID.

    With no IDENTIFIER: if exactly one model is running, stop it;
    otherwise show an interactive picker.
    """
    if identifier is None:
        running = proc.list_running()
        if not running:
            raise click.UsageError("No running models.")
        target = running[0] if len(running) == 1 else _pick_instance_interactively(running)
        identifier = str(target.pid)
    rec = proc.stop(identifier)
    if rec is None:
        raise click.UsageError(f"No running model matches '{identifier}'.")
    click.echo(f"Stopped {rec.name}  pid={rec.pid}  port={rec.port}")


def _resolve_running_target(identifier: str | None) -> proc.RunningProcess:
    """Find a running instance by NAME/PID, or pick interactively if omitted."""
    if identifier is None:
        running = proc.list_running()
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
        running = proc.list_running()
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
        running = proc.list_running()
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


@cli.group()
def autostart() -> None:
    """Restart the last-running models on boot via a systemd user service."""


def _fmt_run_line(s: proc.LaunchSpec) -> str:
    """The `llmctl run …` line that reproduces a manifest entry, or a hint for
    entries recorded before run args were tracked."""
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


def _pick_model_interactively(specs: dict[str, ModelSpec]) -> str:
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
                  f"{'[mtp]' if spec.has_draft else '     '}",
            value=key,
        )
        for key, spec in specs.items()
    ]
    selected = questionary.select(
        "Select a model to run:",
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


if __name__ == "__main__":
    cli()

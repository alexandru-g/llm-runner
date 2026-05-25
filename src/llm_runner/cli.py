"""CLI for managing background llama-server instances."""
from __future__ import annotations

import datetime as dt
import re
import shutil
import socket
from pathlib import Path

import click

from llm_runner import process as proc
from llm_runner.registry import MODELS_DIR, ModelSpec, discover_models


def _port_in_use(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.2)
        return s.connect_ex(("127.0.0.1", port)) == 0


def _next_free_port(start: int = 8080) -> int:
    p = start
    while _port_in_use(p):
        p += 1
    return p


def _build_cmd(model_path: Path, mmproj: Path | None, port: int, ctx_size: int,
               reasoning: bool) -> list[str]:
    cmd = [
        "llama-server",
        "--model", str(model_path),
        "--n-gpu-layers", "999",
        "--ctx-size", str(ctx_size),
        "--flash-attn", "on",
        "--cache-reuse", "256",
        "--cache-ram", "16384",
        "--threads", "8",
        "--port", str(port),
        "--host", "0.0.0.0",
    ]
    if not reasoning:
        cmd.extend(["--reasoning-budget", "0"])
    if mmproj and mmproj.exists():
        cmd.extend(["--mmproj", str(mmproj)])
    return cmd


def _require_llama_server() -> None:
    if shutil.which("llama-server") is None:
        raise click.UsageError(
            "`llama-server` binary not found on PATH. Install llama.cpp "
            "(https://github.com/ggml-org/llama.cpp) and ensure llama-server "
            "is on PATH."
        )


@click.group()
def cli() -> None:
    """Manage local llama-server instances."""


@cli.command("list")
def list_cmd() -> None:
    """List GGUF models discovered on disk under MODELS_DIR."""
    click.echo(f"Models dir: {MODELS_DIR}")
    specs = discover_models()
    if not specs:
        click.echo("\n(no .gguf models found — use `llmctl download <repo>` to fetch one)")
        return
    key_w = max(20, max(len(k) for k in specs))
    click.echo("")
    click.echo(f"{'KEY':<{key_w}}  {'SIZE':>8}  {'VISION':<6}  PATH")
    click.echo("-" * (key_w + 40))
    for key, spec in specs.items():
        size = _fmt_size(spec.size).strip()
        vision = "yes" if spec.has_vision else ""
        rel = spec.model_path.relative_to(MODELS_DIR) if spec.model_path.is_relative_to(MODELS_DIR) else spec.model_path
        click.echo(f"{key:<{key_w}}  {size:>8}  {vision:<6}  {rel}")


@cli.command()
@click.argument("model", required=False)
@click.option("--port", "-p", type=int, default=None,
              help="Listen port. Auto-selected starting at 8080 if omitted or taken.")
@click.option("--ctx-size", "-c", type=int, default=32768, help="Context size.")
@click.option("--ft", "ft_path",
              type=click.Path(exists=True, dir_okay=False, path_type=Path),
              help="Path to a finetuned GGUF file (bypasses the registry).")
@click.option("--name", "ft_name", default=None,
              help="Override the display name (useful with --ft).")
@click.option("--reasoning", is_flag=True, default=False,
              help="Enable reasoning. Disabled by default (--reasoning-budget 0).")
def run(model: str | None, port: int | None, ctx_size: int,
        ft_path: Path | None, ft_name: str | None, reasoning: bool) -> None:
    """Start MODEL in the background. Use --ft <path> to run a finetune GGUF."""
    _require_llama_server()

    if ft_path:
        name = ft_name or f"ft:{ft_path.stem}"
        model_path = ft_path
        mmproj: Path | None = None
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
        spec: ModelSpec = specs[model]
        model_path = spec.model_path
        mmproj = spec.mmproj_path
        name = ft_name or spec.key

    if port is not None and _port_in_use(port):
        raise click.UsageError(f"Port {port} is already in use.")
    chosen_port = port or _next_free_port()

    cmd = _build_cmd(model_path, mmproj, chosen_port, ctx_size, reasoning)
    rec = proc.start(name=name, cmd=cmd, port=chosen_port, model_path=str(model_path))
    click.echo(f"Started {rec.name}  pid={rec.pid}  port={rec.port}")
    click.echo(f"Logs:  {rec.log_file}")
    click.echo(f"Tail:  llmctl logs {rec.name}")


@cli.command("ps")
def ps_cmd() -> None:
    """List running models."""
    procs = proc.list_running()
    if not procs:
        click.echo("(no models running)")
        return
    click.echo(f"{'PID':<8} {'PORT':<6} {'STARTED':<20} NAME")
    click.echo("-" * 78)
    for p in procs:
        started = dt.datetime.fromtimestamp(p.started_at).strftime("%Y-%m-%d %H:%M:%S")
        click.echo(f"{p.pid:<8} {p.port:<6} {started:<20} {p.name}")


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


@cli.command()
@click.argument("identifier", required=False)
@click.option("--lines", "-n", type=int, default=80, help="Tail length.")
def logs(identifier: str | None, lines: int) -> None:
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
    content = log_path.read_text(errors="replace").splitlines()
    click.echo("\n".join(content[-lines:]))


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
            title=f"{key:<{key_w}}  {_fmt_size(spec.size).strip():>8}  "
                  f"{'[vision]' if spec.has_vision else '        '}",
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


@cli.command()
@click.argument("repo_id")
@click.option("--file", "filename", default=None,
              help="Skip the interactive picker and download this exact file.")
@click.option("--no-mmproj", is_flag=True,
              help="Don't download the vision projector (mmproj) alongside the quant.")
@click.option("--subdir", default=None,
              help=f"Override target subdir under {MODELS_DIR} (default: derived from repo name).")
def download(repo_id: str, filename: str | None, no_mmproj: bool, subdir: str | None) -> None:
    """Download a GGUF quant from a Hugging Face REPO_ID into MODELS_DIR.

    Example: llmctl download unsloth/gemma-4-E4B-it-GGUF
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
    if not siblings:
        raise click.UsageError(f"No .gguf files found in {repo_id}.")

    mmproj_names = sorted(f for f in siblings if "mmproj" in f.lower())
    quant_names = sorted(f for f in siblings if f not in set(mmproj_names))
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

    target_dir = MODELS_DIR / (subdir or _derive_subdir(repo_id))
    target_dir.mkdir(parents=True, exist_ok=True)
    click.echo(f"Target: {target_dir}")

    to_download = [selected] + ([chosen_mmproj] if chosen_mmproj else [])
    for f in to_download:
        dest = target_dir / f
        if dest.exists() and dest.stat().st_size > 0:
            click.echo(f"  ✓ already present: {f}")
            continue
        click.echo(f"  ↓ downloading {f} ({_fmt_size(getattr(siblings[f], 'size', None))}) ...")
        hf_hub_download(
            repo_id=repo_id,
            filename=f,
            local_dir=str(target_dir),
        )
        click.echo(f"    done: {dest}")

    click.echo(f"\nFiles now in {target_dir}:")
    for p in sorted(target_dir.iterdir()):
        if p.is_file():
            click.echo(f"  {p.name}  ({_fmt_size(p.stat().st_size)})")


if __name__ == "__main__":
    cli()

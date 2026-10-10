"""The launch-options menu of a bare ``llmctl run``.

After the model is picked, a keyboard menu shows the settings that apply to
it (vision, drafter, reasoning, context, ...) with Start on top, so Enter
launches straight away. Picking a setting toggles it (two values) or opens
its value list. Choices are remembered per model and offered next time.

Each setting is a row in :data:`SETTINGS` naming a ``run`` option: the menu
writes the chosen values into the click context as if typed on the command
line, so ``restart`` and autostart reproduce them. Adding a setting is one
entry here; nothing in ``run`` changes.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Callable

import click

from llm_runner import process as proc
from llm_runner.backends import DECIDER, LLAMA, VLLM
from llm_runner.registry import ModelSpec

CHOICES_FILE = proc.STATE_DIR / "launch.json"


@dataclass(frozen=True)
class Target:
    spec: ModelSpec
    backend: str            # resolved: llama | vllm | decider


@dataclass(frozen=True)
class Setting:
    param: str                                          # `run` parameter name
    label: str
    values: Callable[[Target], list[tuple[Any, str]]]   # (value, title), default first
    applies: Callable[[Target], bool]


def _ctx_values(t: Target) -> list[tuple[Any, str]]:
    sizes = [4096, 8192, 16384, 32768, 65536, 131072]
    return [(None, "default (32768)")] + [(n, f"{n:,}") for n in sizes]


def _kv_values(t: Target) -> list[tuple[Any, str]]:
    vals = [("f16", "f16"), ("q8_0", "q8_0 (half the memory)")]
    if t.backend == LLAMA:
        vals.append(("q4_0", "q4_0 (quarter, some quality cost)"))
    return vals


def _draft_title(t: Target) -> str:
    return f"on ({t.spec.draft_path.name})" if t.spec.draft_path else "on"


SETTINGS: list[Setting] = [
    Setting("no_vision", "Vision",
            lambda t: [(False, "on"), (True, "off (saves ~1 GB)")],
            lambda t: t.backend == LLAMA and t.spec.mmproj_path is not None),
    Setting("no_draft", "MTP drafter",
            lambda t: [(False, _draft_title(t)), (True, "off")],
            lambda t: t.backend in (LLAMA, VLLM) and t.spec.has_draft),
    Setting("reasoning", "Reasoning",
            lambda t: [(False, "off"), (True, "on")],
            lambda t: t.backend == LLAMA),
    Setting("ctx_size", "Context", _ctx_values,
            lambda t: t.backend in (LLAMA, VLLM)),
    Setting("kv_quant", "KV cache", _kv_values,
            lambda t: t.backend in (LLAMA, VLLM)),
    Setting("n_parallel", "Parallel slots",
            lambda t: [(None, "auto"), (1, "1"), (2, "2"), (4, "4"), (8, "8")],
            lambda t: True),
]


def _load_choices() -> dict:
    try:
        data = json.loads(CHOICES_FILE.read_text())
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def _save_choices(key: str, chosen: dict) -> None:
    data = _load_choices()
    data[key] = chosen
    try:
        CHOICES_FILE.parent.mkdir(parents=True, exist_ok=True)
        CHOICES_FILE.write_text(json.dumps(data, indent=2) + "\n")
    except OSError:
        pass  # remembering is a convenience


def _title(values: list[tuple[Any, str]], value: Any) -> str:
    for v, title in values:
        if v == value:
            return title
    return str(value)


def configure(ctx: click.Context, target: Target) -> bool:
    """Show the menu and write the chosen values into ``ctx.params``. Returns
    False when no setting applies (nothing shown)."""
    import questionary

    src = click.core.ParameterSource
    params = {p.name: p for p in ctx.command.params}
    settings = [s for s in SETTINGS if s.applies(target) and s.param in params]
    if not settings:
        return False

    defaults = {s.param: params[s.param].default for s in settings}
    remembered = _load_choices().get(target.spec.key, {})
    current: dict[str, Any] = {}
    for s in settings:
        valid = [v for v, _ in s.values(target)]
        if ctx.get_parameter_source(s.param) == src.COMMANDLINE:
            current[s.param] = ctx.params[s.param]       # typed: wins
        elif s.param in remembered and remembered[s.param] in valid:
            current[s.param] = remembered[s.param]
        else:
            current[s.param] = defaults[s.param]

    width = max(len(s.label) for s in settings)
    cursor: Any = "start"
    while True:
        changed = [s for s in settings if current[s.param] != defaults[s.param]]
        choices = [questionary.Choice("▶ Start", value="start")]
        for s in settings:
            mark = "*" if current[s.param] != defaults[s.param] else " "
            choices.append(questionary.Choice(
                f"{mark} {s.label:<{width}}  {_title(s.values(target), current[s.param])}",
                value=s.param))
        if changed:
            choices.append(questionary.Choice("↺ Reset to defaults", value="reset"))
        picked = questionary.select(
            f"Launch {target.spec.key} on {target.backend}  (Enter starts; * = not default)",
            choices=choices, default=next((c for c in choices if c.value == cursor), choices[0]),
            use_shortcuts=False, instruction=" ").ask()
        if picked is None:
            raise click.Abort()
        if picked == "start":
            break
        if picked == "reset":
            current = dict(defaults)
            cursor = "start"
            continue
        s = next(x for x in settings if x.param == picked)
        values = s.values(target)
        if len(values) == 2:
            current[s.param] = values[1][0] if current[s.param] == values[0][0] else values[0][0]
        else:
            sub = [questionary.Choice(title, value=i) for i, (_, title) in enumerate(values)]
            at = next((i for i, (v, _) in enumerate(values) if v == current[s.param]), 0)
            i = questionary.select(f"{s.label}:", choices=sub, default=sub[at],
                                   use_shortcuts=False, instruction=" ").ask()
            if i is None:
                raise click.Abort()
            current[s.param] = values[i][0]
        cursor = s.param

    for name, value in current.items():
        ctx.params[name] = value
        # As if typed, so `restart` / autostart re-issue it; a default stays
        # unrecorded so run's own defaulting still applies on re-runs.
        ctx.set_parameter_source(name, src.COMMANDLINE if value != defaults[name] else src.DEFAULT)
    _save_choices(target.spec.key, {k: v for k, v in current.items() if v != defaults[k]})
    return True

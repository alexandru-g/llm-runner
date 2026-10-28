"""The launch-options menu of a bare ``llmctl run``.

After the model is picked, a keyboard menu shows the settings that apply to
it (vision, drafter, reasoning, context, ...) with Start on top, so Enter
launches straight away. Picking a setting toggles it (two values) or opens
its value list. Choices are remembered per model and offered next time.

A *profile* fills every row at once: ``last`` (this model's previous run),
``default``, the built-in :data:`PROFILES` (fast, long, multi, serve, lean) and the
user's own, saved from the menu. Only profiles that fit the model are
offered: ``fast`` needs a drafter, and decision and embedding models get no
profiles at all (their defaults are their profile).

Each setting is a row in :data:`SETTINGS` naming a ``run`` option: the menu
writes the chosen values into the click context as if typed on the command
line, so ``restart`` and autostart reproduce them. Adding a setting is one
entry here; nothing in ``run`` changes.

The Backend row (checkpoint dirs, when more than one engine is installed) is
the one row the others depend on: changing it rebuilds the menu for the new
engine, keeping every value that still fits.
"""
from __future__ import annotations

import functools
import json
import re
import urllib.request
from dataclasses import dataclass
from typing import Any, Callable

import click

from llm_runner import backends
from llm_runner import config as config_mod
from llm_runner import process as proc
from llm_runner.backends import DECIDER, GATEWAY, HF_ENGINES, LLAMA
from llm_runner.registry import ModelSpec, default_group, looks_like_embedding_model

CHOICES_FILE = proc.STATE_DIR / "launch.json"
PROFILES_FILE = proc.STATE_DIR / "profiles.json"
_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*")


@dataclass(frozen=True)
class Target:
    spec: ModelSpec
    backend: str            # resolved: llama | vllm | sglang | decider

    @property
    def decision(self) -> bool:
        return self.backend != DECIDER and self.spec.decision


def engine_for(spec: ModelSpec, backend: str) -> str:
    """What ``run --backend BACKEND`` resolves to for ``spec``."""
    return spec.backend if backend == "auto" else backend


@functools.cache
def _installed(engine: str) -> bool:
    return backends.find(engine) is not None


def _backend_values(t: Target) -> list[tuple[Any, str]]:
    """``auto`` (the engine the format implies) first, then every other
    installed engine that serves the same checkpoint."""
    if t.spec.backend not in HF_ENGINES:
        return []
    return [("auto", t.spec.backend)] + [
        (e, e) for e in HF_ENGINES if e != t.spec.backend and _installed(e)]


@dataclass(frozen=True)
class Setting:
    param: str                                          # `run` parameter name
    label: str
    values: Callable[[Target], list[tuple[Any, str]]]   # (value, title), default first
    applies: Callable[[Target], bool]
    custom: str | None = None       # offer "other…": a typed value, with this prompt


def _ctx_values(t: Target) -> list[tuple[Any, str]]:
    sizes = [4096, 8192, 16384, 32768, 65536, 131072]
    return [(None, "default (32768)")] + [(n, f"{n:,}") for n in sizes]


def _kv_values(t: Target) -> list[tuple[Any, str]]:
    vals = [("f16", "f16"), ("q8_0", "q8_0 (half the memory)")]
    if t.backend == LLAMA:
        vals.append(("q4_0", "q4_0 (quarter, some quality cost)"))
    return vals


def _checkpoint_values(t: Target) -> list[tuple[Any, str]]:
    auto = "auto (off: decision model)" if t.decision else "auto (llama-server: 32)"
    return [(None, auto), (0, "off"), (4, "4"), (32, "32")]


def _cluster_groups() -> list[str]:
    """Groups running anywhere the local gateway can see (best effort)."""
    gw = next((p for p in proc.list_running() if p.backend == GATEWAY), None)
    if gw is None:
        return []
    try:
        with urllib.request.urlopen(config_mod.local_url(gw.port, "/v1/models"), timeout=1) as r:
            data = json.loads(r.read())["data"]
    except (OSError, ValueError, KeyError):
        return []
    return sorted({g for m in data if "members" not in m.get("llmctl", {"members": 1})
                   for g in m["llmctl"].get("groups") or [m["llmctl"].get("group")] if g})


def _group_values(t: Target) -> list[tuple[Any, str]]:
    auto = default_group(t.spec.model_path)
    vals: list[tuple[Any, str]] = [(None, f"auto ({auto})" if auto else "auto (none)"),
                                   ("none", "none")]
    for g in _cluster_groups():
        if g != auto:
            vals.append((g, f"{g}  (running)"))
    return vals


def _draft_title(t: Target) -> str:
    return f"on ({t.spec.draft_path.name})" if t.spec.draft_path else "on"


SETTINGS: list[Setting] = [
    Setting("backend", "Backend", _backend_values,
            lambda t: len(_backend_values(t)) > 1),
    Setting("no_vision", "Vision",
            lambda t: [(False, "on"), (True, "off (saves ~1 GB)")],
            lambda t: t.backend == LLAMA and t.spec.mmproj_path is not None),
    Setting("no_draft", "MTP drafter",
            lambda t: [(False, _draft_title(t)), (True, "off")],
            lambda t: t.backend in (LLAMA, *HF_ENGINES) and t.spec.has_draft),
    Setting("reasoning", "Reasoning",
            lambda t: [(False, "off"), (True, "on")],
            lambda t: t.backend == LLAMA),
    Setting("ctx_size", "Context", _ctx_values,
            lambda t: t.backend in (LLAMA, *HF_ENGINES)),
    Setting("kv_quant", "KV cache", _kv_values,
            lambda t: t.backend in (LLAMA, *HF_ENGINES)),
    Setting("ctx_checkpoints", "Checkpoints", _checkpoint_values,
            lambda t: t.backend == LLAMA),
    Setting("cache_ram", "Prompt cache",
            lambda t: [(0, "off"), (2048, "2 GB of RAM"), (8192, "8 GB of RAM"),
                       (16384, "16 GB of RAM")],
            lambda t: t.backend == LLAMA),
    Setting("n_parallel", "Parallel slots",
            lambda t: [(None, "auto"), (1, "1"), (2, "2"), (4, "4"), (8, "8")],
            lambda t: True),
    Setting("group_name", "Group", _group_values, lambda t: True,
            custom="Group name (letters, digits, . _ -):"),
]


@dataclass(frozen=True)
class Profile:
    name: str
    blurb: str
    values: dict[str, Any]      # the rows it sets; every other row goes to its default
    applies: Callable[[Target], bool] = lambda t: True
    user: bool = False
    audience: str = "chat"          # for `llmctl profile list`


def _chat(t: Target) -> bool:
    return t.backend in (LLAMA, *HF_ENGINES)


# Quantized KV zeroes MTP acceptance in llama.cpp, so the profiles that
# quantize it also drop the drafter (and free its memory).
PROFILES: list[Profile] = [
    Profile("fast", "single-user speed",
            {"no_draft": False, "kv_quant": "f16", "n_parallel": 1},
            lambda t: _chat(t) and t.spec.has_draft, audience="chat, drafter"),
    Profile("long", "long documents and code",
            {"ctx_size": 131072, "kv_quant": "q8_0", "no_draft": True, "n_parallel": 1}, _chat),
    # Gemma's sliding-window attention can't resume a conversation another
    # one has replaced in the slot; the RAM cache brings it back (on the Orin,
    # a 9K-token switch went from 16 s to 1.3 s; short prompts pay ~8%).
    Profile("multi", "several long conversations in turn",
            {"cache_ram": 8192, "n_parallel": 1}, lambda t: t.backend == LLAMA,
            audience="chat on llama"),
    Profile("serve", "many clients at once",
            {"ctx_size": 65536, "kv_quant": "q8_0", "no_draft": True, "n_parallel": 4,
             "ctx_checkpoints": 4}, _chat),
    Profile("lean", "least memory",
            {"no_vision": True, "ctx_size": 8192, "kv_quant": "q8_0", "no_draft": True,
             "ctx_checkpoints": 0, "n_parallel": 1}, _chat),
]
RESERVED = {"last", "default"} | {p.name for p in PROFILES}
# Who the model is and what serves it, not how it runs: profiles neither set
# nor reset these (``last`` restores them).
KEEP = {"group_name", "backend"}


def _load(path) -> dict:
    try:
        data = json.loads(path.read_text())
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def _store(path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n")


def _load_choices() -> dict:
    return _load(CHOICES_FILE)


def _save_choices(key: str, chosen: dict) -> None:
    data = _load_choices()
    data[key] = chosen
    try:
        _store(CHOICES_FILE, data)
    except OSError:
        pass  # remembering is a convenience


def user_profiles() -> dict[str, dict]:
    """``{name: {"backend": ..., "values": {...}}}`` from profiles.json."""
    return {k: v for k, v in _load(PROFILES_FILE).items()
            if isinstance(v, dict) and isinstance(v.get("values"), dict)}


def save_profile(name: str, backend: str, values: dict) -> None:
    data = _load(PROFILES_FILE)
    data[name] = {"backend": backend, "values": values}
    _store(PROFILES_FILE, data)


def delete_profile(name: str) -> bool:
    data = _load(PROFILES_FILE)
    if data.pop(name, None) is None:
        return False
    _store(PROFILES_FILE, data)
    return True


def _title(values: list[tuple[Any, str]], value: Any) -> str:
    for v, title in values:
        if v == value:
            return title
    return str(value)


class _Menu:
    """The settings that apply to one launch, their defaults, and the
    profiles that fit."""

    def __init__(self, ctx: click.Context, target: Target):
        params = {p.name: p for p in ctx.command.params}
        self.ctx, self.target = ctx, target
        self.settings = [s for s in SETTINGS if s.applies(target) and s.param in params]
        self.defaults = {s.param: params[s.param].default for s in self.settings}
        self.remembered = _load_choices().get(target.spec.key)

    def typed(self, param: str) -> bool:
        return self.ctx.get_parameter_source(param) == click.core.ParameterSource.COMMANDLINE

    @property
    def tunable(self) -> bool:
        """Whether profiles apply: chat models on llama, vLLM or SGLang."""
        emb = self.ctx.params.get("embeddings")
        if emb is None:
            path = self.target.spec.model_path
            emb = looks_like_embedding_model(path.stem if path.is_file() else path.name, path)
        return _chat(self.target) and not self.target.decision and not emb

    def _valid(self, s: Setting, value: Any) -> bool:
        return (s.custom is not None and isinstance(value, str)) or \
            value in [v for v, _ in s.values(self.target)]

    def _fits(self, values: dict) -> bool:
        by_param = {s.param: s for s in self.settings}
        return all(k in by_param and self._valid(by_param[k], v) for k, v in values.items())

    def profiles(self) -> list[Profile]:
        out = []
        if self.remembered is not None:
            # The one profile that also restores the group.
            out.append(Profile("last", "this model's last run", dict(self.remembered)))
        out.append(Profile("default", "llmctl's defaults", {}))
        if not self.tunable:
            return out
        out += [p for p in PROFILES if p.applies(self.target)]
        for name, rec in sorted(user_profiles().items()):
            values = {k: v for k, v in rec["values"].items() if k not in KEEP}
            if name not in RESERVED and rec.get("backend") == self.target.backend \
                    and self._fits(values):
                out.append(Profile(name, "yours", values, user=True))
        return out

    def find(self, name: str) -> Profile:
        found = {p.name: p for p in self.profiles()}
        if name not in found:
            raise click.BadParameter(
                f"no profile '{name}' for {self.target.spec.key} on {self.target.backend}; "
                f"choose from: {', '.join(found)}", param_hint="'--profile'")
        return found[name]

    def resolve(self, profile: Profile, current: dict) -> dict:
        """Every row as ``profile`` sets it: its values, defaults elsewhere
        (rows in :data:`KEEP` stay as they are unless it names them)."""
        out = {}
        for s in self.settings:
            if s.param in profile.values and self._valid(s, profile.values[s.param]):
                out[s.param] = profile.values[s.param]
            else:
                out[s.param] = current[s.param] if s.param in KEEP else self.defaults[s.param]
        return out

    def summary(self, values: dict) -> str:
        parts = [f"{s.label} {_title(s.values(self.target), values[s.param])}"
                 for s in self.settings
                 if s.param not in KEEP and values[s.param] != self.defaults[s.param]]
        return " · ".join(parts) or "all defaults"

    def warning(self, current: dict) -> str | None:
        if self.target.backend == LLAMA and self.target.spec.has_draft \
                and current.get("no_draft") is False and current.get("kv_quant", "f16") != "f16":
            return ("quantized KV drops MTP acceptance to ~0%: "
                    "turn the drafter off, or use f16")
        return None

    def commit(self, current: dict) -> None:
        src = click.core.ParameterSource
        for name, value in current.items():
            self.ctx.params[name] = value
            # As if typed, so `restart` / autostart re-issue it; a default stays
            # unrecorded so run's own defaulting still applies on re-runs.
            self.ctx.set_parameter_source(
                name, src.COMMANDLINE if value != self.defaults[name] else src.DEFAULT)
        _save_choices(self.target.spec.key,
                      {k: v for k, v in current.items() if v != self.defaults[k]})


def apply_profile(ctx: click.Context, target: Target, name: str) -> None:
    """``run --profile NAME`` without the menu: set its rows in ``ctx.params``
    (flags typed alongside it win)."""
    m = _Menu(ctx, target)
    current = m.resolve(m.find(name), {s.param: ctx.params[s.param] for s in m.settings})
    for s in m.settings:
        if m.typed(s.param):
            current[s.param] = ctx.params[s.param]
    m.commit(current)


def _retarget(m: _Menu, current: dict) -> tuple[_Menu, dict]:
    """The menu for the engine ``current["backend"]`` picks. On a change, every
    row keeps its value where the new engine offers it (or it was typed) and
    falls back to the default where it doesn't."""
    backend = engine_for(m.target.spec, current.get("backend", "auto"))
    if backend == m.target.backend:
        return m, current
    new = _Menu(m.ctx, Target(m.target.spec, backend))
    return new, {s.param: current[s.param]
                 if s.param in current and (new._valid(s, current[s.param]) or new.typed(s.param))
                 else new.defaults[s.param]
                 for s in new.settings}


def _ask(question):
    answer = question.ask()
    if answer is None:
        raise click.Abort()
    return answer


def configure(ctx: click.Context, target: Target, profile: str | None = None) -> bool:
    """Show the menu and write the chosen values into ``ctx.params``. Returns
    False when no setting applies (nothing shown). ``profile`` preselects one;
    otherwise the menu opens on ``last`` (or ``default`` the first time)."""
    import questionary

    m = _Menu(ctx, target)
    if not m.settings:
        return False

    start = {s.param: ctx.params[s.param] for s in m.settings}
    if start.get("backend") == target.spec.backend:
        start["backend"] = "auto"          # `--backend vllm` on a checkpoint is auto
    chosen = m.find(profile) if profile else m.profiles()[0]
    current = m.resolve(chosen, start)
    for s in m.settings:
        if m.typed(s.param):
            current[s.param] = start[s.param]            # typed: wins

    cursor: Any = "start"
    while True:
        m, current = _retarget(m, current)
        settings, defaults, target = m.settings, m.defaults, m.target
        width = max([len("Profile")] + [len(s.label) for s in settings])
        profiles = m.profiles()
        matching = [p.name for p in profiles if m.resolve(p, current) == current]
        if chosen.name not in matching and matching:
            chosen = next(p for p in profiles if p.name == matching[0])
        shown = chosen.name if chosen.name in matching else f"{chosen.name} (modified)"
        choices = [questionary.Choice("▶ Start", value="start")]
        if len(profiles) > 1:
            choices.append(questionary.Choice(
                f"  {'Profile':<{width}}  {shown}  ▸", value="profile"))
        for s in settings:
            mark = "*" if current[s.param] != defaults[s.param] else " "
            choices.append(questionary.Choice(
                f"{mark} {s.label:<{width}}  {_title(s.values(target), current[s.param])}",
                value=s.param))
        warning = m.warning(current)
        if warning:
            choices.append(questionary.Separator(f"  ⚠ {warning}"))
        if m.tunable:
            choices.append(questionary.Choice("＋ Save as profile…", value="save"))
        if any(current[s.param] != defaults[s.param] for s in settings):
            choices.append(questionary.Choice("↺ Reset to defaults", value="reset"))
        picked = _ask(questionary.select(
            f"Launch {target.spec.key} on {target.backend}  (Enter starts; * = not default)",
            choices=choices, default=next((c for c in choices if c.value == cursor), choices[0]),
            use_shortcuts=False, instruction=" "))
        if picked == "start":
            break
        if picked == "reset":
            current = dict(defaults)
            cursor = "start"
            continue
        if picked == "profile":
            pw = max(len(p.name) for p in profiles)
            bw = max(len(p.blurb) for p in profiles)
            sub = [questionary.Choice(
                f"{p.name:<{pw}}  {p.blurb:<{bw}}  [{m.summary(m.resolve(p, current))}]", value=i)
                for i, p in enumerate(profiles)]
            at = next((i for i, p in enumerate(profiles) if p.name == chosen.name), 0)
            i = _ask(questionary.select("Profile:", choices=sub, default=sub[at],
                                        use_shortcuts=False, instruction=" "))
            chosen = profiles[i]
            current = m.resolve(chosen, current)
            cursor = "start"        # a profile is usually the last thing to pick
            continue
        if picked == "save":
            name = _ask(questionary.text(
                "Profile name:", validate=lambda x: (
                    "letters, digits, '.', '_' and '-'" if not _NAME.fullmatch(x.strip())
                    else f"'{x.strip()}' is built in; pick another name"
                    if x.strip() in RESERVED else True))).strip()
            if name in user_profiles() and not _ask(questionary.confirm(
                    f"Replace your profile '{name}'?", default=False)):
                continue
            save_profile(name, target.backend,
                         {k: v for k, v in current.items()
                          if k not in KEEP and v != defaults[k]})
            chosen = next(p for p in m.profiles() if p.name == name)
            cursor = "start"
            continue
        s = next(x for x in settings if x.param == picked)
        values = s.values(target)
        if current[s.param] not in [v for v, _ in values]:
            values.append((current[s.param], str(current[s.param])))   # a typed one
        if len(values) == 2 and not s.custom:
            current[s.param] = values[1][0] if current[s.param] == values[0][0] else values[0][0]
        else:
            sub = [questionary.Choice(title, value=i) for i, (_, title) in enumerate(values)]
            if s.custom:
                sub.append(questionary.Choice("other…", value="custom"))
            at = next((i for i, (v, _) in enumerate(values) if v == current[s.param]), 0)
            i = _ask(questionary.select(f"{s.label}:", choices=sub, default=sub[at],
                                        use_shortcuts=False, instruction=" "))
            if i == "custom":
                current[s.param] = _ask(questionary.text(
                    s.custom, validate=lambda x: bool(_NAME.fullmatch(x.strip())) or
                    "letters, digits, '.', '_' and '-'")).strip()
            else:
                current[s.param] = values[i][0]
        cursor = s.param

    m.commit(current)
    return True

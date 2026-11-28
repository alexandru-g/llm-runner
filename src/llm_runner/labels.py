"""Model labels: what a model is or can do (chat, vision, image, ...).

A label describes a model; a group (``run --group``) names interchangeable
copies of one. Both route: a request for ``"chat"`` goes to a running model
labelled chat when no model or group has that name (``cluster.pick``), best
first by the label's preferred order (``llmctl label prefer``), then as a
group's members are.

Detected labels come from the files each time they're asked for, so they
follow the model (a projector added later makes it ``vision``). What you
change is kept next to the model, in ``.llmctl-labels.json`` in its folder
(``{"add": [...], "remove": [...]}``), so every quant in the folder shares it
and it travels with copies. Labels you create (``llmctl label create``) and
the preferred orders live in this machine's config.
"""
from __future__ import annotations

import json
import os
import re
from pathlib import Path

from llm_runner import backends
from llm_runner import config as config_mod

FILE = ".llmctl-labels.json"
CACHE = "label-cache.json"
NAME = re.compile(r"[a-z0-9][a-z0-9_.-]{0,31}")

# The labels llmctl detects, in the order they're shown.
BUILTIN: dict[str, str] = {
    "chat": "generates text in a conversation",
    "vision": "reads images",
    "reasoning": "can think before it answers (its chat template)",
    "tools": "can call tools (its chat template)",
    "ocr": "reads the text in images (its name)",
    "embed": "turns text into embeddings",
    "rerank": "scores documents against a query (its name)",
    "image": "generates images",
    "decide": "a calibrated decision model",
    "cv": "a computer-vision model (detection, classification)",
}

# Chat-template text that marks a reasoning model: Qwen3/GLM's enable_thinking,
# the reasoning_content field (Gemma 4, Inkling), gpt-oss's reasoning_effort.
# A bare <think> doesn't count: Qwen3Guard's template inserts an empty one.
_REASONING = re.compile(r"enable_thinking|reasoning_content|reasoning_effort")
_TOOLS = re.compile(r"\btools\b")


def _tokens(name: str) -> list[str]:
    return [t for t in re.split(r"[-_.\s/]+", name.lower()) if t]


# ------------------------------------------------------------- templates --

_template_cache: dict[str, tuple[int, int, dict]] = {}


def _template_text(path: Path) -> str:
    if path.suffix == ".gguf":
        meta = backends.gguf_metadata(path, lambda k: k.startswith("tokenizer.chat_template"))
        return "\n".join(str(v) for v in meta.values())
    if path.is_dir():
        text = ""
        for name in ("chat_template.jinja", "chat_template.json", "tokenizer_config.json"):
            p = path / name
            if p.is_file():
                text += p.read_text(errors="replace")
        return text
    return ""


def _stamp(path: Path) -> tuple[int, int]:
    """What changes when the template can: the GGUF, or a checkpoint's
    tokenizer files."""
    if path.is_dir():
        size = mtime = 0
        for name in ("chat_template.jinja", "chat_template.json", "tokenizer_config.json"):
            try:
                st = (path / name).stat()
            except OSError:
                continue
            size, mtime = size + st.st_size, max(mtime, st.st_mtime_ns)
        return size, mtime
    st = path.stat()
    return st.st_size, st.st_mtime_ns


def template_features(path: Path) -> dict:
    """``{"reasoning": bool, "tools": bool}`` from a model's chat template.
    Reading a GGUF's header takes 0.1-0.2 s, so results are kept (in memory
    and in the state dir) until the file changes."""
    from llm_runner import process as proc
    key = str(path)
    try:
        size, mtime = _stamp(path)
    except OSError:
        return {"reasoning": False, "tools": False}
    hit = _template_cache.get(key)
    if hit and hit[:2] == (size, mtime):
        return hit[2]
    cache_file = proc.STATE_DIR / CACHE
    try:
        disk = json.loads(cache_file.read_text())
    except (OSError, ValueError):
        disk = {}
    if not isinstance(disk, dict):
        disk = {}
    rec = disk.get(key)
    if isinstance(rec, dict) and rec.get("size") == size and rec.get("mtime") == mtime:
        feats = {"reasoning": bool(rec.get("reasoning")), "tools": bool(rec.get("tools"))}
    else:
        try:
            text = _template_text(path)
        except OSError:
            text = ""
        feats = {"reasoning": bool(_REASONING.search(text)), "tools": bool(_TOOLS.search(text))}
        disk[key] = {"size": size, "mtime": mtime, **feats}
        try:
            cache_file.parent.mkdir(parents=True, exist_ok=True)
            tmp = cache_file.with_name(f".{CACHE}.{os.getpid()}")
            tmp.write_text(json.dumps(disk))
            tmp.replace(cache_file)
        except OSError:
            pass
    _template_cache[key] = (size, mtime, feats)
    return feats


# --------------------------------------------------------------- detect --

def detect(path: Path, backend: str, name: str, *, vision: bool, decision: bool,
           embeddings: bool | None = None) -> list[str]:
    """The labels a model gets from its files. ``embeddings`` is what its
    launch says (None: guess from the name, as `run` does)."""
    from llm_runner.registry import looks_like_embedding_model
    out: list[str] = []
    toks = _tokens(name) + _tokens(path.name)
    if backend == backends.OMNI:
        out.append("image")
    elif backend in backends.CV_BACKENDS:
        out.append("cv")
    elif decision or backend in (backends.DECIDER, backends.DECIDE) or (
            path.is_dir() and backends._DECISION_FILENAMES.match(path.name)):
        out.append("decide")
    elif embeddings or (embeddings is None and looks_like_embedding_model(
            name, path if path.is_dir() else None)):
        out.append("embed")
    elif any(t.startswith("rerank") for t in toks):
        out.append("rerank")
    else:
        out.append("chat")
    if vision and backend not in (backends.OMNI, *backends.CV_BACKENDS):
        out.append("vision")
    if "chat" in out:
        feats = template_features(path)
        out += [f for f in ("reasoning", "tools") if feats[f]]
        if "ocr" in toks:
            out.append("ocr")
    return sorted(out, key=_rank)


def _rank(label: str) -> tuple[int, str]:
    order = list(BUILTIN)
    return (order.index(label), "") if label in order else (len(order), label)


def detect_spec(spec) -> list[str]:
    return detect(spec.model_path, spec.backend, spec.key, vision=spec.has_vision,
                  decision=spec.decision)


# -------------------------------------------------------------- sidecar --

def sidecar(model_path: Path) -> Path:
    """Where a model's label changes are kept: its folder's file (shared by
    the quants in it), or one beside a GGUF that sits right in MODELS_DIR."""
    from llm_runner import registry
    path = Path(model_path)
    if path.is_dir():
        return path / FILE
    folder = path.parent
    try:
        at_root = folder.resolve() == registry.MODELS_DIR.resolve()
    except OSError:
        at_root = False
    return folder / (f".{path.name}{FILE}" if at_root else FILE)


_edits_cache: dict[str, tuple[int, dict | None]] = {}


def edits(model_path: Path) -> dict | None:
    """``{"add": [...], "remove": [...]}``, or None when the labels were never
    changed or reviewed."""
    p = sidecar(model_path)
    try:
        mtime = p.stat().st_mtime_ns
    except OSError:
        return None
    hit = _edits_cache.get(str(p))
    if hit and hit[0] == mtime:
        return hit[1]
    try:
        data = json.loads(p.read_text())
    except (OSError, ValueError):
        data = None
    if isinstance(data, dict):
        data = {k: [str(x) for x in data.get(k) or [] if isinstance(x, str)]
                for k in ("add", "remove")}
    else:
        data = None
    _edits_cache[str(p)] = (mtime, data)
    return data


def save_edits(model_path: Path, add: list[str], remove: list[str]) -> Path:
    p = sidecar(model_path)
    p.write_text(json.dumps({"add": sorted(set(add), key=_rank),
                             "remove": sorted(set(remove), key=_rank)}, indent=2) + "\n")
    return p


def apply_edits(detected: list[str], changed: dict | None) -> list[str]:
    if not changed:
        return list(detected)
    gone = set(changed.get("remove") or [])
    out = [label for label in detected if label not in gone]
    out += [label for label in changed.get("add") or [] if label not in out]
    return sorted(dict.fromkeys(out), key=_rank)


def of_spec(spec) -> list[str]:
    """A model's labels: detected, with your changes."""
    return apply_edits(detect_spec(spec), edits(spec.model_path))


def set_labels(spec, wanted: list[str]) -> Path:
    """Keep ``wanted`` as the model's labels (stored as changes to what's
    detected, so detection keeps working for the rest)."""
    detected = detect_spec(spec)
    return save_edits(spec.model_path, [x for x in wanted if x not in detected],
                      [x for x in detected if x not in wanted])


def of_instance(name: str, backend: str, model_path: str, *, vision: bool,
                decision: bool, embeddings: bool | None) -> list[str]:
    """A running model's labels. ``vision`` is whether this launch reads
    images (a model started without its projector isn't ``vision``)."""
    if not model_path:
        return []
    path = Path(model_path)
    try:
        found = detect(path, backend, name, vision=vision, decision=decision,
                       embeddings=embeddings)
        out = apply_edits(found, edits(path))
    except OSError:
        return []
    # What the launch says beats what the file says.
    if vision and "vision" not in out:
        out.append("vision")
    if not vision and "vision" in out:
        out.remove("vision")
    return sorted(out, key=_rank)


# -------------------------------------------------- definitions, orders --

def created() -> dict[str, str]:
    """The labels you created on this machine: ``{name: description}``."""
    data = config_mod.load().get("labels") or {}
    return {str(k): str(v or "") for k, v in data.items()} if isinstance(data, dict) else {}


def create(name: str, about: str = "") -> None:
    cfg = created()
    cfg[name] = about
    config_mod.update(labels=dict(sorted(cfg.items())))


def delete(name: str) -> bool:
    cfg = created()
    if name not in cfg:
        return False
    del cfg[name]
    config_mod.update(labels=dict(sorted(cfg.items())) or None)
    return True


def preferences() -> dict[str, list[str]]:
    """``{label: [model or group, ...]}``: which model a request for the
    label goes to first."""
    data = config_mod.load().get("label_prefer") or {}
    if not isinstance(data, dict):
        return {}
    return {str(k): [str(x) for x in v] for k, v in data.items() if isinstance(v, list)}


def set_preference(label: str, names: list[str]) -> None:
    prefs = preferences()
    if names:
        prefs[label] = names
    else:
        prefs.pop(label, None)
    config_mod.update(label_prefer=dict(sorted(prefs.items())) or None)


def known(in_use: list[str] = ()) -> list[str]:
    """Every label to offer: detected ones, yours, and any in use."""
    out = list(BUILTIN) + [x for x in created() if x not in BUILTIN]
    out += sorted(x for x in set(in_use) if x not in out)
    return out


def about(label: str) -> str:
    return BUILTIN.get(label) or created().get(label, "")


def check_name(label: str) -> str:
    import click
    label = label.strip().lower()
    if not NAME.fullmatch(label):
        raise click.BadParameter(
            f"'{label}': a label is up to 32 lower-case letters, digits, '.', '_' and '-', "
            "starting with a letter or digit.")
    return label


def of_launch(name: str, backend: str, model_path: str | None, cmd: list[str] | tuple) -> list[str]:
    """A launch's labels from its command, as the gateway reports them
    (``ps --json``, without asking the gateway)."""
    if not model_path:
        return []
    path = Path(model_path)
    cmd = list(cmd or ())
    try:
        if backend in (backends.VLLM, backends.SGLANG):
            vision = backends.hf_has_vision(path) and not any(
                a.partition("=")[0] in ("--language-model-only", "--language-only") for a in cmd)
        else:
            vision = "--mmproj" in cmd
        decision = backend in (backends.DECIDE, backends.DECIDER) or (
            backend == backends.LLAMA and backends.is_decision_model(path))
    except OSError:
        return []
    embeddings = any(a in cmd for a in ("--embeddings", "pooling", "--is-embedding")) \
        if backend in (backends.LLAMA, backends.VLLM, backends.SGLANG) else None
    return of_instance(name, backend, model_path, vision=vision, decision=decision,
                       embeddings=embeddings)

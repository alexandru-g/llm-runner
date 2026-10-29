"""Serving files: ``llmctl apply`` brings the running models in line with YAML.

A serving file says which models run on this host and how::

    models:
      qwen35-9b:                          # the name clients call (--name)
        model: qwen3.5-9b-2026.10.1       # a model key or folder under MODELS_DIR
        group: qwen3.5-9b                 # a stable route name across versions
                                          # (or a list: [assistant, qwen3.5-9b])
        backend: vllm
        runtime: docker
        port: 8101
        gpu: 0
        ctx: 32768
        n_parallel: 8
        kv_cache_mem: 6G
        args: ["--max-num-batched-tokens", "8192"]
        env: {VLLM_LOGGING_LEVEL: INFO}
      h2o:
        model: h2o-lightning-4b-q8_0      # a llama-server decision model
        port: 8102
        group: decider
        decide:                           # its decision endpoint (`llmctl decide start`)
          port: 8192
          profile: h2o-lightning-4b       # default: detected from the GGUF

Several files merge (a bundle ships a fragment per model, a host file lists
the rest); the same model name in two of them is an error. Each entry becomes
the ``llmctl run`` command that would start it, so a model started by
``apply`` is recorded, restarted and restored like any other. ``apply`` reads
local files only and never touches the network.

This module parses and validates (with the file and line of every error)
and turns entries into ``run`` arguments; ``cli.apply_cmd`` plans and acts.
"""
from __future__ import annotations

import re
import shlex
from dataclasses import dataclass, field
from pathlib import Path

import yaml

NAME_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:-]*")


class ApplyError(Exception):
    """One or more problems in serving files, each with its file and line."""
    exit_code = 3

    def __init__(self, problems: list[str]):
        super().__init__("\n".join(problems))
        self.problems = problems


# key: (`run` option, kind). kind: str, int, float, bool, gpu (str or int),
# dns (a list of addresses, or one "IP[,IP]" string).
OPTIONS: dict[str, tuple[str, str]] = {
    "backend": ("--backend", "str"),
    "group": ("--group", "str"),
    "runtime": ("--runtime", "str"),
    "image": ("--image", "str"),
    "allow_tag": ("--allow-tag", "bool"),
    "dns": ("--dns", "dns"),
    "gpu": ("--gpu", "gpu"),
    "ctx": ("--ctx-size", "int"),
    "n_parallel": ("--n-parallel", "int"),
    "gpu_mem": ("--gpu-mem", "float"),
    "kv_cache_mem": ("--kv-cache-mem", "str"),
    "kv_quant": ("--kv-quant", "str"),
    "tool_parser": ("--tool-parser", "str"),
    "reasoning_parser": ("--reasoning-parser", "str"),
    "trust_remote_code": ("--trust-remote-code", "bool"),
    "reasoning": ("--reasoning", "bool"),
    "embeddings": ("--embeddings/--no-embeddings", "bool"),
    "no_vision": ("--no-vision", "bool"),
    "draft": ("--draft", "str"),
    "no_draft": ("--no-draft", "bool"),
    "spec_type": ("--spec-type", "str"),
    "spec_n_max": ("--spec-draft-n-max", "int"),
    "n_gpu_layers": ("--n-gpu-layers", "int"),
    "kv_unified": ("--kv-unified/--no-kv-unified", "bool"),
    "ctx_checkpoints": ("--ctx-checkpoints", "int"),
    "batch_size": ("--batch-size", "int"),
    "ubatch_size": ("--ubatch-size", "int"),
    "cache_ram": ("--cache-ram", "int"),
    "restart": ("--restart", "str"),
    "restart_max": ("--restart-max", "int"),
    "owner": ("--owner", "str"),
}
SPECIAL = ("model", "port", "args", "env", "decide")
# `decide:` keys: (`decide start` option, kind). `port` is required (pinned).
DECIDE_OPTIONS: dict[str, tuple[str, str]] = {
    "port": ("--port", "int"),
    "profile": ("--profile", "str"),
    "cache": ("--cache", "str"),
    "concurrency": ("--concurrency", "int"),
}
DECIDE_KEYS = (*DECIDE_OPTIONS, "args")
KEYS = (*SPECIAL, *OPTIONS)


@dataclass
class Entry:
    name: str
    model: str
    port: int
    options: dict[str, object] = field(default_factory=dict)
    args: list[str] = field(default_factory=list)
    env: dict[str, str] = field(default_factory=dict)
    where: str = ""                 # "file:line" of the entry
    # The decision endpoint: DECIDE_OPTIONS values plus "args" (a list), or None.
    decide: dict | None = None

    @property
    def decide_name(self) -> str:
        return f"{self.name}.decide"

    def decide_argv(self) -> list[str]:
        """The ``llmctl decide start`` arguments for this entry's endpoint."""
        d = self.decide or {}
        argv = [self.name]
        for key, (opt, _) in DECIDE_OPTIONS.items():
            if d.get(key) is not None:
                argv += [opt, str(d[key])]
        if d.get("args"):
            argv += ["--extra", shlex.join(d["args"])]
        return argv

    def run_argv(self, model_key: str) -> list[str]:
        """The ``llmctl run`` arguments that start this entry (port included)."""
        argv = [model_key, "--name", self.name, "--port", str(self.port)]
        for key, value in self.options.items():
            opt, kind = OPTIONS[key]
            if kind == "bool":
                on, _, off = opt.partition("/")
                if value:
                    argv.append(on)
                elif off:
                    argv.append(off)
            elif key == "group" and value is None:
                argv += [opt, "none"]
            else:
                argv += [opt, str(value)]
        if self.args:
            argv += ["--extra", shlex.join(self.args)]
        for k, v in self.env.items():
            argv += ["--env", f"{k}={v}"]
        return argv


# ----------------------------------------------------------------- parsing --

def _at(path: Path, node: yaml.Node) -> str:
    return f"{path}:{node.start_mark.line + 1}"


def _scalar(path: Path, node: yaml.Node, kind: str, what: str, problems: list[str]):
    if not isinstance(node, yaml.ScalarNode):
        problems.append(f"{_at(path, node)}: {what} must be a single value")
        return None
    value = yaml.safe_load(yaml.serialize(node))
    if kind == "gpu":
        if isinstance(value, int) and not isinstance(value, bool):
            return str(value)
        kind = "str"
    ok = {"str": isinstance(value, str),
          "int": isinstance(value, int) and not isinstance(value, bool),
          "float": isinstance(value, (int, float)) and not isinstance(value, bool),
          "bool": isinstance(value, bool)}[kind]
    if what.endswith(".group") and value is None:
        return None
    if not ok:
        want = {"str": "a string", "int": "an integer", "float": "a number",
                "bool": "true or false"}[kind]
        problems.append(f"{_at(path, node)}: {what} must be {want}, not {value!r}")
        return None
    return value


def _mapping(path: Path, node: yaml.Node, what: str,
             problems: list[str]) -> list[tuple[yaml.ScalarNode, yaml.Node]] | None:
    if not isinstance(node, yaml.MappingNode):
        problems.append(f"{_at(path, node)}: {what} must be a mapping")
        return None
    seen: dict[str, yaml.Node] = {}
    pairs = []
    for k, v in node.value:
        if not isinstance(k, yaml.ScalarNode):
            problems.append(f"{_at(path, k)}: {what} keys must be plain names")
            continue
        if k.value in seen:
            problems.append(f"{_at(path, k)}: {what}.{k.value} is set twice "
                            f"(first at line {seen[k.value].start_mark.line + 1})")
            continue
        seen[k.value] = k
        pairs.append((k, v))
    return pairs


def _entry(path: Path, name_node: yaml.ScalarNode, node: yaml.Node,
           problems: list[str]) -> Entry | None:
    name = name_node.value
    what = f"models.{name}"
    if not NAME_RE.fullmatch(name):
        problems.append(f"{_at(path, name_node)}: model name {name!r} must be letters, "
                        "digits, '.', '_', ':' and '-'")
    pairs = _mapping(path, node, what, problems)
    if pairs is None:
        return None
    before = len(problems)
    entry = Entry(name=name, model="", port=0, where=_at(path, name_node))
    for k, v in pairs:
        key = k.value
        if key not in KEYS:
            problems.append(f"{_at(path, k)}: {what}.{key} is not a known setting "
                            f"(known: {', '.join(KEYS)})")
        elif key == "model":
            entry.model = _scalar(path, v, "str", f"{what}.model", problems) or ""
        elif key == "port":
            port = _scalar(path, v, "int", f"{what}.port", problems)
            if port is not None and not 1 <= port <= 65535:
                problems.append(f"{_at(path, v)}: {what}.port must be 1-65535")
            entry.port = port or 0
        elif key == "args":
            if not isinstance(v, yaml.SequenceNode):
                problems.append(f"{_at(path, v)}: {what}.args must be a list of strings")
                continue
            for item in v.value:
                val = (yaml.safe_load(yaml.serialize(item))
                       if isinstance(item, yaml.ScalarNode) else None)
                if isinstance(val, bool) or not isinstance(val, (str, int, float)):
                    problems.append(f"{_at(path, item)}: {what}.args items must be "
                                    "strings or numbers")
                    continue
                entry.args.append(str(val))
        elif key == "group" and isinstance(v, yaml.SequenceNode):
            names = [yaml.safe_load(yaml.serialize(i)) if isinstance(i, yaml.ScalarNode)
                     else None for i in v.value]
            if not names or not all(isinstance(n, str) and re.fullmatch(
                    r"[A-Za-z0-9][A-Za-z0-9_.-]*", n) for n in names):
                problems.append(f"{_at(path, v)}: {what}.group must be a name or a list of "
                                "names (letters, digits, '.', '_' and '-')")
                continue
            entry.options["group"] = ",".join(dict.fromkeys(names))
        elif key == "dns":
            from llm_runner.containers import parse_dns
            if isinstance(v, yaml.SequenceNode):
                raw = [yaml.safe_load(yaml.serialize(i)) if isinstance(i, yaml.ScalarNode)
                       else None for i in v.value]
            else:
                raw = _scalar(path, v, "str", f"{what}.dns", problems)
            if raw is None:
                continue                      # _scalar said why
            if isinstance(raw, list) and None in raw:
                problems.append(f"{_at(path, v)}: {what}.dns must be a list of IP addresses")
                continue
            try:
                entry.options["dns"] = ",".join(parse_dns(raw))
            except ValueError as err:
                problems.append(f"{_at(path, v)}: {what}.dns: {err}")
        elif key == "decide":
            entry.decide = _decide(path, v, f"{what}.decide", problems)
        elif key == "env":
            env_pairs = _mapping(path, v, f"{what}.env", problems) or []
            for ek, ev in env_pairs:
                if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", ek.value):
                    problems.append(f"{_at(path, ek)}: {what}.env key {ek.value!r} is not "
                                    "a variable name")
                    continue
                val = yaml.safe_load(yaml.serialize(ev)) if isinstance(ev, yaml.ScalarNode) else None
                if val is None or isinstance(val, (dict, list)):
                    problems.append(f"{_at(path, ev)}: {what}.env.{ek.value} must be a value")
                    continue
                entry.env[ek.value] = str(val).lower() if isinstance(val, bool) else str(val)
        else:
            _, kind = OPTIONS[key]
            val = _scalar(path, v, kind, f"{what}.{key}", problems)
            if val is not None or key == "group":
                entry.options[key] = val
    if not entry.model:
        problems.append(f"{entry.where}: {what}.model is required")
    if not entry.port:
        problems.append(f"{entry.where}: {what}.port is required (ports are pinned)")
    return entry if len(problems) == before else None


def _decide(path: Path, node: yaml.Node, what: str, problems: list[str]) -> dict | None:
    pairs = _mapping(path, node, what, problems)
    if pairs is None:
        return None
    out: dict = {}
    for k, v in pairs:
        if k.value not in DECIDE_KEYS:
            problems.append(f"{_at(path, k)}: {what}.{k.value} is not a known setting "
                            f"(known: {', '.join(DECIDE_KEYS)})")
        elif k.value == "args":
            items = [yaml.safe_load(yaml.serialize(i)) if isinstance(i, yaml.ScalarNode)
                     else None for i in getattr(v, "value", [])]
            if not isinstance(v, yaml.SequenceNode) or not all(
                    isinstance(i, (str, int, float)) and not isinstance(i, bool) for i in items):
                problems.append(f"{_at(path, v)}: {what}.args must be a list of strings")
                continue
            out["args"] = [str(i) for i in items]
        else:
            val = _scalar(path, v, DECIDE_OPTIONS[k.value][1], f"{what}.{k.value}", problems)
            if val is None:
                continue
            if k.value == "port" and not 1 <= val <= 65535:
                problems.append(f"{_at(path, v)}: {what}.port must be 1-65535")
            elif k.value == "cache" and val not in ("off", "request"):
                problems.append(f"{_at(path, v)}: {what}.cache must be off or request")
            elif k.value == "concurrency" and val < 1:
                problems.append(f"{_at(path, v)}: {what}.concurrency must be 1 or more")
            else:
                out[k.value] = val
    if "port" not in out:
        problems.append(f"{_at(path, node)}: {what}.port is required (ports are pinned)")
    return out


def decide_differs(d: dict, cmd: list[str], model_port: int) -> str:
    """Why a running endpoint's ``systemone-proxy serve`` argv isn't what
    ``decide:`` asks for ("" when it is). Unset keys accept whatever runs."""
    def val(flag: str) -> str | None:
        return cmd[cmd.index(flag) + 1] if flag in cmd[:-1] else None
    backend = val("--backend") or ""
    if not backend.rstrip("/").endswith(f":{model_port}"):
        return f"backend {backend} -> port {model_port}"
    for key in ("port", "profile", "concurrency"):
        if d.get(key) is not None and val(f"--{key}") != str(d[key]):
            return f"{key} {val(f'--{key}')} -> {d[key]}"
    if (val("--cache") or "off") != (d.get("cache") or "off"):
        return f"cache {val('--cache')} -> {d.get('cache') or 'off'}"
    extra = cmd[cmd.index("--cache") + 2:] if "--cache" in cmd[:-1] else []
    if extra != list(d.get("args") or []):
        return f"args {shlex.join(extra) or '(none)'} -> {shlex.join(d.get('args') or []) or '(none)'}"
    return ""


def load(paths: list[Path]) -> list[Entry]:
    """Parse and merge serving files. Raises ApplyError naming every problem."""
    problems: list[str] = []
    entries: dict[str, Entry] = {}
    ports: dict[int, Entry] = {}
    for path in paths:
        try:
            text = path.read_text()
        except OSError as e:
            problems.append(f"{path}: {e.strerror or e}")
            continue
        try:
            root = yaml.compose(text, Loader=yaml.SafeLoader)
        except yaml.YAMLError as e:
            mark = getattr(e, "problem_mark", None)
            line = f":{mark.line + 1}" if mark else ""
            problems.append(f"{path}{line}: not valid YAML: {getattr(e, 'problem', e)}")
            continue
        if root is None:
            continue                                   # an empty file
        top = _mapping(path, root, "the file", problems)
        if top is None:
            continue
        for k, v in top:
            if k.value == "version":
                if _scalar(path, v, "int", "version", problems) not in (1, None):
                    problems.append(f"{_at(path, v)}: only version 1 is understood")
            elif k.value != "models":
                problems.append(f"{_at(path, k)}: unknown top-level key {k.value!r} "
                                "(expected models, version)")
        models = next((v for k, v in top if k.value == "models"), None)
        if models is None or (isinstance(models, yaml.ScalarNode) and models.value in ("", "~", "null")):
            continue
        for name_node, node in _mapping(path, models, "models", problems) or []:
            entry = _entry(path, name_node, node, problems)
            if entry is None:
                continue
            if entry.name in entries:
                problems.append(f"{entry.where}: model {entry.name!r} is also defined at "
                                f"{entries[entry.name].where}")
                continue
            dport = (entry.decide or {}).get("port")
            clash = next((p for p in (entry.port, dport) if p in ports), None)
            if clash is not None or (dport is not None and dport == entry.port):
                other = ports.get(clash) or entry
                problems.append(f"{entry.where}: port {clash or dport} is also used by "
                                f"{other.name!r} ({other.where})")
                continue
            entries[entry.name] = entry
            ports[entry.port] = entry
            if dport is not None:
                ports[dport] = entry
    if problems:
        raise ApplyError(problems)
    return list(entries.values())


# ---------------------------------------------------------------- planning --

@dataclass
class Action:
    name: str
    action: str                     # start, restart, stop, unchanged, error
    port: int | None = None
    reason: str = ""
    argv: list[str] = field(default_factory=list)
    kind: str = "run"               # run, or decide (argv is `decide start`'s)

    @property
    def command(self) -> list[str]:
        return ["llmctl", *(["decide", "start"] if self.kind == "decide" else ["run"]),
                *self.argv]

    def to_json(self) -> dict:
        out = {"name": self.name, "action": self.action, "port": self.port}
        if self.reason:
            out["reason"] = self.reason
        if self.argv:
            out["run"] = self.command
        return out


def without_owner(args: list[str]) -> list[str]:
    """``args`` without ``--owner TAG``: a label, so a different one (another
    session's $LLMCTL_OWNER) is no reason to restart a model."""
    out, i = [], 0
    while i < len(args):
        if args[i] == "--owner" and i + 1 < len(args):
            i += 2
            continue
        out.append(args[i])
        i += 1
    return out


def diff_args(old: list[str], new: list[str]) -> str:
    """A short human diff of two `run` argument lists."""
    o, n = set(_pairs(old)), set(_pairs(new))
    gone = [p for p in _pairs(old) if p not in n]
    added = [p for p in _pairs(new) if p not in o]
    parts = [f"-{' '.join(p)}" for p in gone] + [f"+{' '.join(p)}" for p in added]
    return ", ".join(parts) or "same arguments"


def _pairs(args: list[str]) -> list[tuple[str, ...]]:
    out: list[tuple[str, ...]] = []
    i = 0
    while i < len(args):
        a = args[i]
        if a.startswith("-") and i + 1 < len(args) and not args[i + 1].startswith("--"):
            out.append((a, args[i + 1]))
            i += 2
        else:
            out.append((a,))
            i += 1
    return out

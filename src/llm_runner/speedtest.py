"""``llmctl speedtest``: measure a running model's speed on its own API.

The test talks to the instance's port directly (not through the gateway), so
it measures the server exactly as launched: KV cache type, drafter, slots,
container or host. What is measured depends on what the instance serves:

``text``    chat and completion models (llama, vllm, sglang). Prefill speed
            and time to first token per prompt size, then decode speed over a
            fixed number of generated tokens (EOS ignored). llama-server's own
            ``timings`` are used when it sends them; otherwise the stream is
            timed here. Drafter acceptance comes from llama-server's timings
            or the difference in vLLM's spec-decode counters.
``embed``   embedding servers: inputs and tokens per second.
``decide``  decision endpoints (decider, systemone-proxy): decisions per
            second and latency, for a state of a given size.
``image``   vLLM-Omni pipelines: seconds per image and steps per second.

Every prompt is fresh random text, so a prompt cache can't make prefill look
free; the length of that text is calibrated against the server's own token
counts. The options are rows of a menu like ``run``'s, filled from profiles:
``last`` (this instance's previous test), ``default``, the built-in
:data:`PROFILES` and the user's own, saved from the menu.
"""
from __future__ import annotations

import json
import random
import re
import statistics
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable
from urllib.parse import urlsplit

import click

from llm_runner import backends
from llm_runner import config as config_mod
from llm_runner import process as proc
from llm_runner.backends import DECIDE, DECIDER, LLAMA, OMNI, SGLANG, VLLM

STATE_FILE = proc.STATE_DIR / "speedtest.json"
_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*")

TEXT, EMBED, DECISION, IMAGE = "text", "embed", "decide", "image"
KINDS = (TEXT, EMBED, DECISION, IMAGE)
_KIND_TITLE = {TEXT: "text generation", EMBED: "embeddings", DECISION: "decisions",
               IMAGE: "image generation"}


# ------------------------------------------------------------------ target --

@dataclass(frozen=True)
class Target:
    name: str
    backend: str
    port: int
    kind: str
    completion: bool = False    # raw /v1/completions (decision models on llama)
    cmd: tuple[str, ...] = ()

    @property
    def drafter(self) -> bool:
        return any(f in self.cmd for f in ("--model-draft", "--speculative-config",
                                           "--speculative-algorithm"))


def kind_of(backend: str, cmd: list[str]) -> str:
    if backend in (DECIDER, DECIDE):
        return DECISION
    if backend == OMNI:
        return IMAGE
    if any(a in cmd for a in ("--embeddings", "--embedding", "--is-embedding")):
        return EMBED
    if "--runner" in cmd and cmd[cmd.index("--runner") + 1:][:1] == ["pooling"]:
        return EMBED
    return TEXT


def target_of(p: proc.RunningProcess) -> Target:
    kind = kind_of(p.backend, p.cmd)
    completion = False
    if kind == TEXT and p.backend == LLAMA:
        try:
            completion = backends.is_decision_model(Path(p.model_path))
        except OSError:
            completion = False
    return Target(p.name, p.backend, p.port, kind, completion, tuple(p.cmd))


# ---------------------------------------------------------------- settings --

def parse_sizes(text: str) -> list[int]:
    """``"512,4096"`` -> ``[512, 4096]``; raises ValueError on anything else."""
    out = [int(x) for x in str(text).replace(" ", "").split(",") if x]
    if not out or any(n < 1 or n > 1_048_576 for n in out):
        raise ValueError(text)
    return out


def parse_image_size(text: str) -> tuple[int, int]:
    m = re.fullmatch(r"(\d{2,5})[x×](\d{2,5})", str(text).strip())
    if not m:
        raise ValueError(text)
    return int(m[1]), int(m[2])


def _ok(parse: Callable[[str], Any]) -> Callable[[Any], bool]:
    def check(v: Any) -> bool:
        try:
            parse(v)
            return True
        except (ValueError, TypeError):
            return False
    return check


@dataclass(frozen=True)
class Setting:
    param: str
    labels: dict[str, str]                  # kind -> row label; the kinds it applies to
    values: list[tuple[Any, str]]           # (value, title)
    defaults: dict[str, Any]                # kind -> default
    custom: str | None = None               # "other…" prompt for a typed value
    valid: Callable[[Any], bool] | None = None   # what a typed value must pass

    def applies(self, kind: str) -> bool:
        return kind in self.labels


_PROMPTS = [("128", "128"), ("512", "512"), ("1024", "1,024"), ("512,4096", "512 and 4,096"),
            ("4096", "4,096"), ("16384", "16,384"), ("16384,32768", "16,384 and 32,768"),
            ("32768", "32,768")]

SETTINGS: list[Setting] = [
    Setting("prompt", {TEXT: "Prompt tokens", EMBED: "Tokens per input",
                       DECISION: "State tokens"},
            _PROMPTS, {TEXT: "512,4096", EMBED: "512", DECISION: "1024"},
            custom="Token counts, comma-separated (e.g. 2048,8192):",
            valid=_ok(parse_sizes)),
    Setting("gen", {TEXT: "Generate"},
            [(64, "64 tokens"), (128, "128 tokens"), (256, "256 tokens"),
             (512, "512 tokens"), (1024, "1,024 tokens")], {TEXT: 256}),
    Setting("inputs", {EMBED: "Inputs per request"},
            [(1, "1"), (8, "8"), (32, "32"), (128, "128")], {EMBED: 8}),
    Setting("questions", {DECISION: "Questions"},
            [(1, "1"), (8, "8"), (32, "32"), (64, "64")], {DECISION: 8}),
    Setting("size", {IMAGE: "Image size"},
            [("512x512", "512×512"), ("768x768", "768×768"), ("1024x1024", "1024×1024")],
            {IMAGE: "1024x1024"}, custom="Size as WxH (e.g. 1328x1328):",
            valid=_ok(parse_image_size)),
    Setting("steps", {IMAGE: "Steps"},
            [(None, "the model's default"), (4, "4"), (8, "8"), (20, "20"), (28, "28"),
             (50, "50")], {IMAGE: None}),
    Setting("concurrency", {TEXT: "Concurrent requests", EMBED: "Concurrent requests",
                            DECISION: "Concurrent requests"},
            [(1, "1 (single stream)"), (2, "2"), (4, "4"), (8, "8"), (16, "16")],
            {TEXT: 1, EMBED: 1, DECISION: 1}),
    Setting("runs", {k: "Runs" for k in KINDS},
            [(1, "1"), (2, "2"), (3, "3"), (5, "5"), (10, "10")],
            {TEXT: 3, EMBED: 3, DECISION: 3, IMAGE: 2}),
]
PARAMS = [s.param for s in SETTINGS]


@dataclass(frozen=True)
class Profile:
    name: str
    blurb: str
    values: dict[str, Any]
    kinds: tuple[str, ...] = KINDS
    user: bool = False


PROFILES: list[Profile] = [
    Profile("quick", "one short pass", {"prompt": "512", "gen": 128, "runs": 1,
                                        "size": "512x512", "inputs": 8, "questions": 8}),
    Profile("long", "long prompts: prefill speed",
            {"prompt": "16384,32768", "gen": 128, "runs": 2}, (TEXT, DECISION)),
    Profile("load", "4 clients at once",
            {"prompt": "1024", "concurrency": 4, "runs": 3}, (TEXT, EMBED, DECISION)),
]
RESERVED = {"last", "default"} | {p.name for p in PROFILES}


def _title(s: Setting, value: Any) -> str:
    for v, t in s.values:
        if v == value:
            return t
    return str(value)


def settings_for(kind: str) -> list[Setting]:
    return [s for s in SETTINGS if s.applies(kind)]


def defaults_for(kind: str) -> dict[str, Any]:
    return {s.param: s.defaults[kind] for s in settings_for(kind)}


def _valid(s: Setting, value: Any) -> bool:
    return value in [v for v, _ in s.values] or (s.valid is not None and s.valid(value))


# ------------------------------------------------------------------- state --

def _load() -> dict:
    try:
        data = json.loads(STATE_FILE.read_text())
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def _store(data: dict) -> None:
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    STATE_FILE.write_text(json.dumps(data, indent=2) + "\n")


def user_profiles() -> dict[str, dict]:
    raw = _load().get("profiles")
    return {k: v for k, v in (raw or {}).items()
            if isinstance(v, dict) and isinstance(v.get("values"), dict)}


def save_profile(name: str, kind: str, values: dict) -> None:
    data = _load()
    data.setdefault("profiles", {})[name] = {"kind": kind, "values": values}
    _store(data)


def delete_profile(name: str) -> bool:
    data = _load()
    if (data.get("profiles") or {}).pop(name, None) is None:
        return False
    _store(data)
    return True


def _group(ts: Target | list[Target]) -> list[Target]:
    """One target or several of one kind, as a non-empty list."""
    group = [ts] if isinstance(ts, Target) else list(ts)
    if not group or len({t.kind for t in group}) != 1:
        raise ValueError("a speedtest group is one or more targets of one kind")
    return group


def _last_key(group: list[Target]) -> str:
    """Where ``last`` lives: the model's name, or ``*<kind>`` for a comparison."""
    return group[0].name if len(group) == 1 else f"*{group[0].kind}"


def remembered(name: str) -> dict | None:
    last = (_load().get("last") or {}).get(name)
    return last if isinstance(last, dict) else None


def remember(ts: Target | list[Target] | str, values: dict) -> None:
    """Remember ``values`` as ``last`` for each model tested, and for
    comparisons of their kind when there are several."""
    keys = [ts] if isinstance(ts, str) else [t.name for t in _group(ts)]
    if not isinstance(ts, str) and len(_group(ts)) > 1:
        keys.append(_last_key(_group(ts)))
    data = _load()
    for k in keys:
        data.setdefault("last", {})[k] = values
    try:
        _store(data)
    except OSError:
        pass        # remembering is a convenience


def profiles_for(ts: Target | list[Target]) -> list[Profile]:
    group = _group(ts)
    kind = group[0].kind
    out = []
    last = remembered(_last_key(group))
    if last is not None:
        out.append(Profile("last", "this model's last test" if len(group) == 1
                           else "the last comparison", last))
    out.append(Profile("default", "llmctl's defaults", {}))
    out += [p for p in PROFILES if kind in p.kinds]
    for name, rec in sorted(user_profiles().items()):
        if name not in RESERVED and rec.get("kind") == kind:
            out.append(Profile(name, "yours", rec["values"], user=True))
    return out


def find_profile(ts: Target | list[Target], name: str) -> Profile:
    kind = _group(ts)[0].kind
    found = {p.name: p for p in profiles_for(ts)}
    if name not in found:
        raise click.BadParameter(
            f"no profile '{name}' for {_KIND_TITLE[kind]}; choose from: "
            + ", ".join(found), param_hint="'--profile'")
    return found[name]


def resolve(kind: str, profile: Profile) -> dict[str, Any]:
    """Every row of ``kind`` as ``profile`` sets it, defaults elsewhere."""
    out = {}
    for s in settings_for(kind):
        v = profile.values.get(s.param, s.defaults[kind])
        out[s.param] = v if s.param in profile.values and _valid(s, v) else s.defaults[kind]
    return out


def summary(kind: str, values: dict) -> str:
    d = defaults_for(kind)
    parts = [f"{s.labels[kind]} {_title(s, values[s.param])}" for s in settings_for(kind)
             if values[s.param] != d[s.param]]
    return " · ".join(parts) or "all defaults"


def _ask(question):
    answer = question.ask()
    if answer is None:
        raise click.Abort()
    return answer


def configure(ts: Target | list[Target], chosen: Profile,
              typed: dict[str, Any]) -> dict[str, Any]:
    """The options menu: Run on top, a Profile row, one row per setting,
    Save as profile, Reset. Returns the chosen values. Several targets (of
    one kind) share one menu: they are tested with the same options."""
    import questionary

    group = _group(ts)
    t = group[0]
    settings, defaults = settings_for(t.kind), defaults_for(t.kind)
    current = {**resolve(t.kind, chosen), **typed}
    width = max(len("Profile"), *(len(s.labels[t.kind]) for s in settings))
    cursor: Any = "run"
    while True:
        profiles = profiles_for(group)
        matching = [p.name for p in profiles if resolve(t.kind, p) == current]
        if chosen.name not in matching and matching:
            chosen = next(p for p in profiles if p.name == matching[0])
        shown = chosen.name if chosen.name in matching else f"{chosen.name} (modified)"
        choices = [questionary.Choice("▶ Run", value="run"),
                   questionary.Choice(f"  {'Profile':<{width}}  {shown}  ▸", value="profile")]
        for s in settings:
            mark = "*" if current[s.param] != defaults[s.param] else " "
            choices.append(questionary.Choice(
                f"{mark} {s.labels[t.kind]:<{width}}  {_title(s, current[s.param])}",
                value=s.param))
        warning = next((w for w in (_warning(x, current) for x in group) if w), None)
        if warning:
            choices.append(questionary.Separator(f"  ⚠ {warning}"))
        choices.append(questionary.Choice("＋ Save as profile…", value="save"))
        if current != defaults:
            choices.append(questionary.Choice("↺ Reset to defaults", value="reset"))
        picked = _ask(questionary.select(
            (f"Speedtest {t.name} ({_KIND_TITLE[t.kind]} on {t.backend})  " if len(group) == 1
             else f"Speedtest {len(group)} models ({_KIND_TITLE[t.kind]})  ")
            + "(Enter runs; * = not default)",
            choices=choices, default=next((c for c in choices
                                           if getattr(c, "value", None) == cursor), choices[0]),
            use_shortcuts=False, instruction=" "))
        if picked == "run":
            return current
        if picked == "reset":
            current, cursor = dict(defaults), "run"
            continue
        if picked == "profile":
            pw = max(len(p.name) for p in profiles)
            bw = max(len(p.blurb) for p in profiles)
            sub = [questionary.Choice(
                f"{p.name:<{pw}}  {p.blurb:<{bw}}  [{summary(t.kind, resolve(t.kind, p))}]",
                value=i) for i, p in enumerate(profiles)]
            at = next((i for i, p in enumerate(profiles) if p.name == chosen.name), 0)
            i = _ask(questionary.select("Profile:", choices=sub, default=sub[at],
                                        use_shortcuts=False, instruction=" "))
            chosen = profiles[i]
            current, cursor = resolve(t.kind, chosen), "run"
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
            save_profile(name, t.kind, {k: v for k, v in current.items() if v != defaults[k]})
            chosen = next(p for p in profiles_for(group) if p.name == name)
            cursor = "run"
            continue
        s = next(x for x in settings if x.param == picked)
        values = list(s.values)
        if current[s.param] not in [v for v, _ in values]:
            values.append((current[s.param], str(current[s.param])))     # a typed one
        sub = [questionary.Choice(title, value=i) for i, (_, title) in enumerate(values)]
        if s.custom:
            sub.append(questionary.Choice("other…", value="custom"))
        at = next((i for i, (v, _) in enumerate(values) if v == current[s.param]), 0)
        i = _ask(questionary.select(f"{s.labels[t.kind]}:", choices=sub, default=sub[at],
                                    use_shortcuts=False, instruction=" "))
        if i == "custom":
            check = s.valid or (lambda _v: True)
            current[s.param] = _ask(questionary.text(
                s.custom, validate=lambda x: check(x.strip()) or "not a valid value")).strip()
        else:
            current[s.param] = values[i][0]
        cursor = s.param


def _warning(t: Target, values: dict) -> str | None:
    if t.kind == TEXT and values.get("concurrency", 1) > 1 and t.backend == LLAMA:
        slots = _flag(list(t.cmd), "--parallel", "-np")
        if slots and slots.isdigit() and int(slots) < values["concurrency"]:
            return (f"{t.name} has {slots} slot(s): requests beyond that queue, "
                    "so per-stream numbers include waiting")
    return None


def _flag(cmd: list[str], *names: str) -> str | None:
    for i, a in enumerate(cmd):
        if a in names and i + 1 < len(cmd):
            return cmd[i + 1]
        for n in names:
            if a.startswith(n + "="):
                return a.split("=", 1)[1]
    return None


# -------------------------------------------------------------------- http --

class HTTPFail(Exception):
    def __init__(self, status: int | None, message: str):
        super().__init__(message)
        self.status = status


def _url(t: Target, path: str) -> str:
    return config_mod.local_url(t.port, path)


def _get_json(t: Target, path: str, timeout: float = 5.0) -> Any:
    with urllib.request.urlopen(_url(t, path), timeout=timeout) as r:
        return json.loads(r.read())


def _get_text(t: Target, path: str, timeout: float = 5.0) -> str:
    with urllib.request.urlopen(_url(t, path), timeout=timeout) as r:
        return r.read().decode(errors="replace")


def _request(t: Target, path: str, body: dict, timeout: float = 900.0):
    req = urllib.request.Request(_url(t, path), data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"},
                                 method="POST")
    try:
        return urllib.request.urlopen(req, timeout=timeout)
    except urllib.error.HTTPError as e:
        detail = e.read().decode(errors="replace")[:400]
        try:
            err = json.loads(detail)
            detail = (err.get("error", {}).get("message") if isinstance(err.get("error"), dict)
                      else err.get("error") or err.get("detail") or err.get("message")) or detail
        except ValueError:
            pass
        raise HTTPFail(e.code, f"HTTP {e.code}: {str(detail)[:300]}") from None
    except (urllib.error.URLError, OSError) as e:
        raise HTTPFail(None, f"no answer from port {t.port}: {e}") from None


def _post_json(t: Target, path: str, body: dict, timeout: float = 900.0) -> dict:
    with _request(t, path, body, timeout) as r:
        return json.loads(r.read())


def _sse(resp):
    """The JSON events of a server-sent-events reply, with arrival times."""
    for raw in resp:
        line = raw.decode(errors="replace").strip()
        if not line.startswith("data:"):
            continue
        data = line[5:].strip()
        if data == "[DONE]":
            return
        try:
            yield time.perf_counter(), json.loads(data)
        except ValueError:
            continue


# ---------------------------------------------------------------- probing --

def health(t: Target) -> str | None:
    """None when the instance answers /health as ready, else why not."""
    try:
        with urllib.request.urlopen(_url(t, "/health"), timeout=5) as r:
            return None if r.status == 200 else f"/health answered {r.status}"
    except urllib.error.HTTPError as e:
        if e.code == 404:
            return None             # no /health: the test itself will tell
        return f"not ready (/health answered {e.code})"
    except (urllib.error.URLError, OSError) as e:
        return f"not answering on port {t.port} ({e})"


def _metric_sum(text: str, name: str) -> float | None:
    total = None
    for line in text.splitlines():
        if line.startswith(name) and (len(line) == len(name) or line[len(name)] in " {"):
            try:
                total = (total or 0.0) + float(line.rsplit(" ", 1)[1])
            except ValueError:
                pass
    return total


_ACTIVE_METRICS = {VLLM: ("vllm:num_requests_running", "vllm:num_requests_waiting"),
                   SGLANG: ("sglang:num_running_reqs", "sglang:num_queue_reqs")}


def active_requests(t: Target) -> int | None:
    """Requests the instance is working on or holding now; None = unknown.
    A decision endpoint in front of a llama-server is as busy as that server."""
    if t.backend == DECIDE:
        port = urlsplit(_flag(list(t.cmd), "--backend") or "").port
        if port is None:
            return None
        t = Target(t.name, LLAMA, port, TEXT)
    try:
        if t.backend == LLAMA:
            slots = _get_json(t, "/slots", timeout=2)
            if isinstance(slots, list):
                return sum(1 for s in slots if isinstance(s, dict) and s.get("is_processing"))
            return None
        if t.backend in _ACTIVE_METRICS:
            text = _get_text(t, "/metrics", timeout=2)
            vals = [_metric_sum(text, n) for n in _ACTIVE_METRICS[t.backend]]
            return None if all(v is None for v in vals) else int(sum(v or 0 for v in vals))
    except (OSError, ValueError):
        return None
    return None


def served_model(t: Target) -> str:
    """The id the server answers to (``model`` in requests)."""
    try:
        data = _get_json(t, "/v1/models")["data"]
        if data and data[0].get("id"):
            return data[0]["id"]
    except (OSError, ValueError, KeyError, IndexError, TypeError):
        pass
    return t.name


def context_size(t: Target) -> int | None:
    """Tokens one request may hold (prompt + output), when the server says."""
    try:
        if t.backend == LLAMA:
            props = _get_json(t, "/props")
            n = (props.get("default_generation_settings") or {}).get("n_ctx") or props.get("n_ctx")
            return int(n) if n else None
        if t.backend in (VLLM, SGLANG):
            data = _get_json(t, "/v1/models")["data"]
            n = data[0].get("max_model_len") if data else None
            return int(n) if n else None
    except (OSError, ValueError, KeyError, IndexError, TypeError):
        return None
    return None


_SPEC_METRICS = ("vllm:spec_decode_num_accepted_tokens", "vllm:spec_decode_num_draft_tokens")


def _spec_counters(t: Target) -> tuple[float, float] | None:
    if t.backend != VLLM:
        return None
    try:
        text = _get_text(t, "/metrics", timeout=2)
    except OSError:
        return None
    acc, drafted = (_metric_sum(text, n + "_total") for n in _SPEC_METRICS)
    return None if acc is None or drafted is None else (acc, drafted)


# ------------------------------------------------------------------ prompts --

_WORDS = ("time year people way day thing world life hand part child eye woman place "
          "work week case point number group problem fact water room money story month "
          "book job word business issue side kind head house service friend father power "
          "hour game line end member law car city community name team minute idea body "
          "information back parent face level office door health person art war history "
          "party result change morning reason research girl moment air teacher force "
          "education river mountain garden window bridge market island forest letter "
          "music paper station winter summer road ship train cloud stone field kitchen "
          "lamp coat glass bread table horse bird street village signal engine harbor "
          "valley cable mirror ladder candle tower desert planet orchard meadow canyon "
          "slow bright quiet heavy narrow ancient silver golden early late brief open "
          "carries follows builds watches remembers opens crosses finds keeps holds").split()


def filler(words: int, rng: random.Random) -> str:
    """Random sentences of ``words`` words: text no cache has seen."""
    out, sentence = [], []
    for _ in range(words):
        sentence.append(rng.choice(_WORDS))
        if len(sentence) >= rng.randint(8, 16):
            out.append(" ".join(sentence).capitalize() + ".")
            sentence = []
    if sentence:
        out.append(" ".join(sentence).capitalize() + ".")
    return " ".join(out)


_TASK = ("\n\nThe notes above are scattered and can be ignored. Write a long, detailed "
         "story about a lighthouse keeper who finds a map in a bottle. Keep writing "
         "until you are stopped; do not summarise or end the story early.")
_IMAGE_PROMPT = ("A red fox reading a newspaper on a snowy park bench at dawn, soft light, "
                 "detailed fur, shallow depth of field")


@dataclass
class Calibration:
    """Words of :func:`filler` per token, and the fixed tokens around them."""
    overhead: float = 0.0
    per_word: float = 1.3

    def words(self, tokens: int) -> int:
        return max(1, round((tokens - self.overhead) / self.per_word))

    def learn_overhead(self, words: int, tokens: int | None) -> None:
        if tokens:
            self.overhead = max(0.0, tokens - words * self.per_word)

    def learn(self, words: int, tokens: int | None) -> None:
        if tokens and tokens > self.overhead and words > 0:
            self.per_word = (tokens - self.overhead) / words


# ------------------------------------------------------------- workloads --

@dataclass
class Sample:
    ok: bool = True
    error: str | None = None
    prompt_tokens: int | None = None
    output_tokens: int | None = None
    ttft: float | None = None           # s
    prefill_tps: float | None = None
    decode_tps: float | None = None
    elapsed: float | None = None        # s, request start to end
    start: float = 0.0
    end: float = 0.0
    draft: tuple[int, int] | None = None    # (accepted, drafted)
    server_timed: bool = False
    extra: dict = field(default_factory=dict)


def _text_body(t: Target, model: str, words: int, gen: int, rng: random.Random,
               stream: bool) -> tuple[str, dict]:
    text = filler(words, rng)
    body: dict[str, Any] = {"model": model, "max_tokens": gen, "stream": stream}
    if t.completion:
        path, body["prompt"] = "/v1/completions", text
    else:
        path = "/v1/chat/completions"
        body["messages"] = [{"role": "user", "content": text + _TASK}]
    if gen > 1:
        body["ignore_eos"] = True
    if stream:
        body["stream_options"] = {"include_usage": True}
    if t.backend == LLAMA:
        body["cache_prompt"] = False
        if stream:
            body["timings_per_token"] = False
    return path, body


def text_tokens(t: Target, model: str, words: int, rng: random.Random) -> int | None:
    path, body = _text_body(t, model, words, 1, rng, stream=False)
    usage = _post_json(t, path, body).get("usage") or {}
    return usage.get("prompt_tokens")


def text_once(t: Target, model: str, words: int, gen: int, rng: random.Random) -> Sample:
    path, body = _text_body(t, model, words, gen, rng, stream=True)
    s = Sample(start=time.perf_counter())
    first = last = None
    chunks = 0
    usage: dict = {}
    timings: dict = {}
    with _request(t, path, body) as resp:
        for when, ev in _sse(resp):
            if isinstance(ev.get("usage"), dict):
                usage = ev["usage"]
            if isinstance(ev.get("timings"), dict):
                timings = ev["timings"]
            for ch in ev.get("choices") or []:
                delta = ch.get("delta") or {}
                piece = (ch.get("text") or delta.get("content")
                         or delta.get("reasoning_content") or delta.get("reasoning"))
                if piece:
                    chunks += 1
                    first = first or when
                    last = when
    s.end = time.perf_counter()
    s.elapsed = s.end - s.start
    s.prompt_tokens = usage.get("prompt_tokens") or timings.get("prompt_n")
    s.output_tokens = usage.get("completion_tokens") or timings.get("predicted_n") or chunks
    if first is not None:
        s.ttft = first - s.start
        if s.prompt_tokens and s.ttft > 0:
            s.prefill_tps = s.prompt_tokens / s.ttft
        if last > first and s.output_tokens and s.output_tokens > 1:
            s.decode_tps = (s.output_tokens - 1) / (last - first)
    if timings.get("prompt_per_second") and timings.get("predicted_per_second"):
        s.prefill_tps = float(timings["prompt_per_second"])
        s.decode_tps = float(timings["predicted_per_second"])
        s.server_timed = True
    if timings.get("draft_n"):
        s.draft = (int(timings.get("draft_n_accepted") or 0), int(timings["draft_n"]))
    if not s.output_tokens:
        s.ok, s.error = False, "the server sent no tokens"
    return s


def embed_tokens(t: Target, model: str, words: int, rng: random.Random) -> int | None:
    usage = _post_json(t, "/v1/embeddings", {"model": model, "input": [filler(words, rng)]}
                       ).get("usage") or {}
    return usage.get("prompt_tokens") or usage.get("total_tokens")


def embed_once(t: Target, model: str, words: int, inputs: int, rng: random.Random) -> Sample:
    texts = [filler(words, rng) for _ in range(inputs)]
    s = Sample(start=time.perf_counter())
    reply = _post_json(t, "/v1/embeddings", {"model": model, "input": texts})
    s.end = time.perf_counter()
    s.elapsed = s.end - s.start
    usage = reply.get("usage") or {}
    s.prompt_tokens = usage.get("prompt_tokens") or usage.get("total_tokens")
    got = len(reply.get("data") or [])
    if got != inputs:
        s.ok, s.error = False, f"asked for {inputs} embeddings, got {got}"
    s.extra["inputs"] = inputs
    return s


_STATEMENTS = ("The text mentions a river.", "The text is about a city.",
               "The text describes the weather.", "The text mentions a person by name.",
               "The text is written in English.", "The text talks about money.",
               "The text mentions an animal.", "The text describes a journey.")


def _decide_body(words: int, questions: int, rng: random.Random) -> dict:
    return {"state": filler(words, rng),
            "questions": {f"q{i + 1}": {"type": "noul",
                                        "instructions": _STATEMENTS[i % len(_STATEMENTS)]}
                          for i in range(questions)}}


def decide_tokens(t: Target, words: int, rng: random.Random) -> int | None:
    usage = _post_json(t, "/v1/systemone", _decide_body(words, 1, rng)).get("usage") or {}
    return usage.get("input_tokens")


def decide_once(t: Target, words: int, questions: int, rng: random.Random) -> Sample:
    s = Sample(start=time.perf_counter())
    reply = _post_json(t, "/v1/systemone", _decide_body(words, questions, rng))
    s.end = time.perf_counter()
    s.elapsed = s.end - s.start
    usage = reply.get("usage") or {}
    s.prompt_tokens = usage.get("input_tokens")
    answers = reply.get("answers") or {}
    if len(answers) != questions:
        s.ok, s.error = False, f"asked {questions} questions, got {len(answers)} answers"
    if usage.get("input_tokens") and s.elapsed:
        s.prefill_tps = usage["input_tokens"] / s.elapsed
    s.extra["questions"] = questions
    s.extra["truncated"] = any(isinstance(a, dict) and a.get("truncated")
                               for a in answers.values())
    return s


def image_once(t: Target, model: str, size: str, steps: int | None,
               rng: random.Random) -> Sample:
    body: dict[str, Any] = {"model": model, "prompt": _IMAGE_PROMPT, "size": size, "n": 1,
                            "seed": rng.randrange(2**31), "response_format": "b64_json"}
    if steps:
        body["num_inference_steps"] = steps
    s = Sample(start=time.perf_counter())
    reply = _post_json(t, "/v1/images/generations", body)
    s.end = time.perf_counter()
    s.elapsed = s.end - s.start
    if not reply.get("data"):
        s.ok, s.error = False, "the server returned no image"
    return s


def _together(n: int, job: Callable[[int], Sample]) -> list[Sample]:
    """``n`` copies of ``job`` started together; failures become samples."""
    out: list[Sample | None] = [None] * n
    gate = threading.Barrier(n)

    def work(i: int) -> None:
        try:
            gate.wait()
        except threading.BrokenBarrierError:
            pass
        try:
            out[i] = job(i)
        except HTTPFail as e:
            out[i] = Sample(ok=False, error=str(e), start=time.perf_counter(),
                            end=time.perf_counter())
        except Exception as e:      # noqa: BLE001 - a stray reply shape: report, don't crash
            out[i] = Sample(ok=False, error=f"{type(e).__name__}: {e}",
                            start=time.perf_counter(), end=time.perf_counter())

    if n == 1:
        work(0)
    else:
        threads = [threading.Thread(target=work, args=(i,), daemon=True) for i in range(n)]
        for th in threads:
            th.start()
        for th in threads:
            th.join()
    return [s for s in out if s is not None]


# ------------------------------------------------------------------- report --

def _median(xs: list[float | None]) -> float | None:
    xs = [x for x in xs if x is not None]
    return statistics.median(xs) if xs else None


def _span(xs: list[float | None]) -> tuple[float, float] | None:
    xs = [x for x in xs if x is not None]
    return (min(xs), max(xs)) if len(xs) > 1 else None


@dataclass
class Row:
    """One measured configuration (a prompt size, or the image size)."""
    label: str
    samples: list[list[Sample]] = field(default_factory=list)   # per run: one per request
    note: str | None = None             # why it was skipped, or a caveat
    spec: tuple[float, float] | None = None   # vLLM counters delta (accepted, drafted)

    @property
    def flat(self) -> list[Sample]:
        return [s for run in self.samples for s in run if s.ok]

    @property
    def errors(self) -> list[str]:
        return sorted({s.error for run in self.samples for s in run if not s.ok and s.error})

    def stats(self, kind: str) -> dict[str, Any]:
        good = self.flat
        out: dict[str, Any] = {"label": self.label, "ok": len(good),
                               "failed": sum(1 for r in self.samples for s in r if not s.ok)}
        if self.note:
            out["note"] = self.note
        if self.errors:
            out["errors"] = self.errors
        if not good:
            return out
        out["prompt_tokens"] = _median([s.prompt_tokens for s in good])
        walls = [max(s.end for s in r) - min(s.start for s in r)
                 for r in self.samples if r and all(s.ok for s in r)]
        if kind == TEXT:
            out["ttft_s"] = _median([s.ttft for s in good])
            out["prefill_tps"] = _median([s.prefill_tps for s in good])
            out["decode_tps"] = _median([s.decode_tps for s in good])
            out["decode_span"] = _span([s.decode_tps for s in good])
            out["output_tokens"] = _median([s.output_tokens for s in good])
            out["server_timed"] = all(s.server_timed for s in good)
            totals = [sum(s.output_tokens or 0 for s in r) / w
                      for r, w in zip([r for r in self.samples if r and all(s.ok for s in r)],
                                      walls) if w > 0]
            out["total_tps"] = _median(totals)
            drafts = [s.draft for s in good if s.draft]
            if drafts:
                out["draft_accept"] = (sum(a for a, _ in drafts)
                                       / max(1, sum(d for _, d in drafts)))
            elif self.spec and self.spec[1] > 0:
                out["draft_accept"] = self.spec[0] / self.spec[1]
        elif kind == EMBED:
            out["prompt_tokens"] = _median([s.prompt_tokens / s.extra["inputs"] for s in good
                                            if s.prompt_tokens and s.extra.get("inputs")])
            out["latency_s"] = _median([s.elapsed for s in good])
            ok_runs = [r for r in self.samples if r and all(s.ok for s in r)]
            out["inputs_per_s"] = _median([sum(s.extra.get("inputs", 0) for s in r) / w
                                           for r, w in zip(ok_runs, walls) if w > 0])
            out["tokens_per_s"] = _median([sum(s.prompt_tokens or 0 for s in r) / w
                                           for r, w in zip(ok_runs, walls) if w > 0])
        elif kind == DECISION:
            # input_tokens counts the state once per question (each is its own prompt).
            out["prompt_tokens"] = _median([s.prompt_tokens / s.extra["questions"] for s in good
                                            if s.prompt_tokens and s.extra.get("questions")])
            out["latency_s"] = _median([s.elapsed for s in good])
            ok_runs = [r for r in self.samples if r and all(s.ok for s in r)]
            out["decisions_per_s"] = _median([sum(s.extra.get("questions", 0) for s in r) / w
                                              for r, w in zip(ok_runs, walls) if w > 0])
            out["prefill_tps"] = _median([sum(s.prompt_tokens or 0 for s in r) / w
                                          for r, w in zip(ok_runs, walls) if w > 0])
            out["truncated"] = any(s.extra.get("truncated") for s in good)
        else:
            out["seconds_per_image"] = _median([s.elapsed for s in good])
            out["span"] = _span([s.elapsed for s in good])
        return out


def _n(x: float | None, digits: int = 1) -> str:
    if x is None:
        return "—"
    return f"{x:,.{digits}f}" if digits else f"{x:,.0f}"


def _setup(t: Target, header: dict) -> str:
    bits = [t.backend, f"port {t.port}"]
    if header.get("ctx"):
        bits.append(f"ctx {header['ctx']:,}")
    if t.kind == TEXT:
        bits.append("drafter on" if t.drafter else "no drafter")
        if t.completion:
            bits.append("raw completions")
    return " · ".join(bits)


def _what(kind: str, values: dict) -> str:
    conc = values.get("concurrency", 1)
    what = {TEXT: f"{values.get('gen')} tokens generated per request",
            EMBED: f"{values.get('inputs')} inputs per request",
            DECISION: f"{values.get('questions')} questions per request",
            IMAGE: f"{values.get('size')}, "
                   + (f"{values['steps']} steps" if values.get("steps") else "default steps")}[kind]
    return f"{values['runs']} run(s), {conc} concurrent request(s), {what}"


def headline(kind: str, values: dict) -> tuple[str, str, bool]:
    """``(stats key, column title, higher is better)``: what a comparison
    ranks the models by."""
    if kind == TEXT:
        return (("total_tps", "total t/s", True) if values.get("concurrency", 1) > 1
                else ("decode_tps", "decode t/s", True))
    return {EMBED: ("tokens_per_s", "tokens/s", True),
            DECISION: ("decisions_per_s", "decisions/s", True),
            IMAGE: ("seconds_per_image", "s/image", False)}[kind]


def render_compare(kind: str, values: dict,
                   results: list[tuple[Target, list[dict], dict]]) -> str:
    """Several models tested with the same options: one table per prompt size
    (or image size), a row per model, and how each compares with the best."""
    key, key_title, higher = headline(kind, values)
    conc = values.get("concurrency", 1)
    lines = [f"speedtest  {len(results)} models · {_KIND_TITLE[kind]}",
             f"  {_what(kind, values)}",
             f"  vs best: {key_title}, {'higher' if higher else 'lower'} is better", ""]
    nw = max(len(t.name) for t, _, _ in results)
    for t, _, header in results:
        lines.append(f"  {t.name:<{nw}}  {_setup(t, header)}")

    labels: list[str] = []
    for _, rows, _ in results:
        for r in rows:
            if r["label"] not in labels:
                labels.append(r["label"])
    notes: list[str] = []
    for label in labels:
        entries = [(t, next((r for r in rows if r["label"] == label), None))
                   for t, rows, _ in results]
        scores = [r.get(key) for _, r in entries if r and r.get("ok") and r.get(key)]
        best = (max(scores) if higher else min(scores)) if scores else None
        if kind == TEXT:
            cols = ["model", "prompt", "ttft", "prefill t/s", "decode t/s"]
            if conc > 1:
                cols.append("total t/s")
            if any(r and "draft_accept" in r for _, r in entries):
                cols.append("draft")
        elif kind == EMBED:
            cols = ["model", "tokens/input", "latency", "inputs/s", "tokens/s"]
        elif kind == DECISION:
            cols = ["model", "state", "latency", "decisions/s", "prefill t/s"]
        else:
            cols = ["model", "s/image", "steps/s"]
        cols.append("vs best")
        table = [cols]
        for t, r in entries:
            if r is None or not r.get("ok"):
                why = (r or {}).get("note") or "; ".join((r or {}).get("errors") or []) \
                    or "not measured"
                table.append([t.name] + ["—"] * (len(cols) - 2) + ["skipped"])
                notes.append(f"{t.name} · {label}: {why}")
                continue
            p = r.get("prompt_tokens")
            if kind == TEXT:
                row = [t.name, _n(p, 0), f"{_n(r.get('ttft_s'), 2)} s",
                       _n(r.get("prefill_tps"), 0), _n(r.get("decode_tps"))]
                if conc > 1:
                    row.append(_n(r.get("total_tps")))
                if "draft" in cols:
                    row.append(f"{r['draft_accept'] * 100:.0f}%" if "draft_accept" in r else "—")
            elif kind == EMBED:
                row = [t.name, _n(p, 0), f"{_n(r.get('latency_s'), 2)} s",
                       _n(r.get("inputs_per_s")), _n(r.get("tokens_per_s"), 0)]
            elif kind == DECISION:
                row = [t.name, _n(p, 0), f"{_n(r.get('latency_s'), 2)} s",
                       _n(r.get("decisions_per_s")), _n(r.get("prefill_tps"), 0)]
            else:
                spi, steps = r.get("seconds_per_image"), values.get("steps")
                row = [t.name, f"{_n(spi, 2)} s", _n(steps / spi, 2) if steps and spi else "—"]
            score = r.get(key)
            if best and score:
                rel = score / best if higher else best / score
                row.append("best" if rel >= 0.9995 else "<1%" if rel < 0.005
                           else f"{rel * 100:.0f}%")
            else:
                row.append("—")
            table.append(row)
            if r.get("failed"):
                notes.append(f"{t.name} · {label}: {r['failed']} request(s) failed: "
                             + "; ".join(r.get("errors") or []))
            if r.get("note"):
                notes.append(f"{t.name} · {label}: {r['note']}")
            if r.get("truncated"):
                notes.append(f"{t.name} · {label}: the endpoint truncated the state")
        title = f"  {label}" if kind != IMAGE else f"  {label} image"
        if len(table) - 1 == sum(1 for row in table[1:] if row[-1] == "skipped"):
            lines += ["", f"{title}: no model measured (see the notes)"]
            continue
        lines += ["", title]
        w = [max(len(str(row[i])) for row in table) for i in range(len(cols))]
        for row in table:
            lines.append("    " + "   ".join(
                str(c).ljust(w[i]) if i == 0 else str(c).rjust(w[i])
                for i, c in enumerate(row)).rstrip())
    if notes:
        lines.append("")
        lines += [f"  note: {n}" for n in notes]
    return "\n".join(lines)


def render(t: Target, values: dict, rows: list[dict], header: dict) -> str:
    kind = t.kind
    lines = []
    bits = [t.backend, f"port {t.port}"]
    if header.get("ctx"):
        bits.append(f"ctx {header['ctx']:,}")
    if kind == TEXT:
        bits.append("drafter on" if t.drafter else "no drafter")
        if t.completion:
            bits.append("raw completions")
    lines.append(f"speedtest  {t.name}  ({' · '.join(bits)})")
    conc = values.get("concurrency", 1)
    what = {TEXT: f"{values.get('gen')} tokens generated per request",
            EMBED: f"{values.get('inputs')} inputs per request",
            DECISION: f"{values.get('questions')} questions per request",
            IMAGE: f"{values.get('size')}, "
                   + (f"{values['steps']} steps" if values.get("steps") else "default steps")}[kind]
    lines.append(f"  {values['runs']} run(s), {conc} concurrent request(s), {what}"
                 + (" · server timings" if any(r.get("server_timed") for r in rows) else ""))
    lines.append("")

    if kind == TEXT:
        cols = ["prompt", "ttft", "prefill t/s", "decode t/s"]
        if conc > 1:
            cols.append("total t/s")
        if any("draft_accept" in r for r in rows):
            cols.append("draft accepted")
    elif kind == EMBED:
        cols = ["tokens/input", "latency", "inputs/s", "tokens/s"]
    elif kind == DECISION:
        cols = ["state", "latency", "decisions/s", "prefill t/s"]
    else:
        cols = ["size", "s/image", "steps/s"]
    table = [cols]
    notes = []
    for r in rows:
        if not r.get("ok"):
            why = r.get("note") or "; ".join(r.get("errors") or []) or "failed"
            notes.append(f"{r['label']}: {why}")
            continue
        p = r.get("prompt_tokens")
        first = _n(p, 0) if p is not None and kind != IMAGE else r["label"]
        if kind == TEXT:
            dec = _n(r.get("decode_tps"))
            span = r.get("decode_span")
            if span and span[1] - span[0] >= 0.1:
                dec += f" ({span[0]:.1f}–{span[1]:.1f})"
            row = [first, f"{_n(r.get('ttft_s'), 2)} s", _n(r.get("prefill_tps"), 0), dec]
            if conc > 1:
                row.append(_n(r.get("total_tps")))
            if "draft accepted" in cols:
                row.append(f"{r['draft_accept'] * 100:.0f}%" if "draft_accept" in r else "—")
        elif kind == EMBED:
            row = [first, f"{_n(r.get('latency_s'), 2)} s", _n(r.get("inputs_per_s")),
                   _n(r.get("tokens_per_s"), 0)]
        elif kind == DECISION:
            row = [first, f"{_n(r.get('latency_s'), 2)} s", _n(r.get("decisions_per_s")),
                   _n(r.get("prefill_tps"), 0)]
            if r.get("truncated"):
                notes.append(f"{r['label']}: the endpoint truncated the state")
        else:
            spi = r.get("seconds_per_image")
            steps = values.get("steps")
            row = [first, f"{_n(spi, 2)} s",
                   _n(steps / spi, 2) if steps and spi else "—"]
        table.append(row)
        if r.get("failed"):
            notes.append(f"{r['label']}: {r['failed']} request(s) failed: "
                         + "; ".join(r.get("errors") or []))
        if r.get("note"):
            notes.append(f"{r['label']}: {r['note']}")
    if len(table) > 1:
        w = [max(len(str(row[i])) for row in table) for i in range(len(cols))]
        for row in table:
            lines.append("  " + "   ".join(
                str(c).ljust(w[i]) if i == 0 else str(c).rjust(w[i])
                for i, c in enumerate(row)).rstrip())
    for n in notes:
        lines.append(f"  note: {n}")
    if kind == TEXT and conc == 1 and rows and any(r.get("ok") for r in rows):
        lines.append("")
        lines.append("  ttft and prefill: one fresh prompt each (no cache hit); decode: "
                     "median per request" + (", range in brackets" if values["runs"] > 1 else ""))
    return "\n".join(lines)


# --------------------------------------------------------------------- run --

def run(t: Target, values: dict, *, echo: Callable[[str], None] = lambda _m: None,
        seed: int | None = None) -> tuple[list[dict], dict]:
    """Measure; returns ``(rows, header)`` (rows as :meth:`Row.stats`)."""
    rng = random.Random(seed)
    model = served_model(t) if t.kind != DECISION else t.name
    header: dict[str, Any] = {"model": model}
    runs, conc = values["runs"], values.get("concurrency", 1)
    rows: list[Row] = []

    if t.kind == IMAGE:
        row = Row(values["size"])
        echo("warming up (one image)…")
        warm = _together(1, lambda i: image_once(t, model, values["size"], values["steps"], rng))
        if not warm[0].ok:
            row.samples.append(warm)
            return [row.stats(t.kind)], header
        for k in range(runs):
            echo(f"image {k + 1}/{runs}…")
            row.samples.append(_together(
                1, lambda i: image_once(t, model, values["size"], values["steps"], rng)))
        return [row.stats(t.kind)], header

    ctx = context_size(t) if t.kind in (TEXT, EMBED) else None
    header["ctx"] = ctx
    cal = Calibration()
    measure: Callable[[int], int | None] = {
        TEXT: lambda w: text_tokens(t, model, w, rng),
        EMBED: lambda w: embed_tokens(t, model, w, rng),
        DECISION: lambda w: decide_tokens(t, w, rng),
    }[t.kind]
    echo("warming up…")
    try:
        cal.learn_overhead(1, measure(1))
    except HTTPFail as e:
        return [Row("warm-up", [[Sample(ok=False, error=str(e))]]).stats(t.kind)], header

    room = values.get("gen", 0) + 16
    for size in parse_sizes(values["prompt"]):
        row = Row(f"{size:,} tokens")
        rows.append(row)
        if ctx and size + room > ctx:
            row.note = f"doesn't fit the context ({ctx:,} tokens per request)"
            continue
        words = cal.words(size)
        echo(f"{size:,} tokens: calibrating…")
        try:
            cal.learn(words, measure(words))
        except HTTPFail as e:
            row.samples.append([Sample(ok=False, error=str(e))])
            continue
        words = cal.words(size)
        before = _spec_counters(t)
        for k in range(runs):
            echo(f"{size:,} tokens: run {k + 1}/{runs}…")
            if t.kind == TEXT:
                job = lambda i: text_once(t, model, words, values["gen"], rng)  # noqa: E731
            elif t.kind == EMBED:
                job = lambda i: embed_once(t, model, words, values["inputs"], rng)  # noqa: E731
            else:
                job = lambda i: decide_once(t, words, values["questions"], rng)  # noqa: E731
            row.samples.append(_together(conc, job))
            if all(not s.ok for s in row.samples[-1]):
                break           # the same error again on every run says nothing new
        after = _spec_counters(t)
        if before and after:
            row.spec = (after[0] - before[0], after[1] - before[1])
        flat = row.flat
        if t.kind == TEXT and flat and any((s.output_tokens or 0) < values["gen"] for s in flat):
            row.note = (f"some replies stopped before {values['gen']} tokens "
                        "(the server ignored ignore_eos)")
    return [r.stats(t.kind) for r in rows], header

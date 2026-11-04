"""``llmctl chat``: talk to a running model from the terminal.

It goes through this machine's gateway, like any client, so it reaches every
model in the cluster: ``GET /v1/models`` says what is running where, and a
request names a model, or ``name@node`` to keep it to one node. Files land on
this machine wherever the model runs. What a message does depends on what the
model serves:

``text``    chat models (llama, vllm, sglang). The reply is streamed and the
            conversation is kept, so each message sends the whole history.
``image``   vLLM-Omni pipelines. Each message is a fresh prompt; the images go
            to files, and open in the desktop's viewer when run from a terminal.
            With reference images it is an edit instead.

Everything else answers in another shape and is refused (:func:`refusal`):
decision endpoints and models, embedding servers, CV models.

The gateway's traces and metrics count these calls like any other. Traces carry
metadata only, never a prompt or reply.

The reply goes to stdout and everything about it (reasoning, numbers, notes)
to stderr, so ``llmctl chat m -m "..." > out.txt`` keeps just the answer.
"""
from __future__ import annotations

import base64
import http.client
import json
import mimetypes
import os
import random
import re
import shutil
import subprocess
import sys
import time
import urllib.request
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import click

from llm_runner import speedtest as st
from llm_runner.backends import GATEWAY
from llm_runner.cluster import IMAGES, KSERVE, OPENAI, SYSTEMONE, split_pin

# A reply is one long read: prefill sends nothing until the first token, and a
# long history or a big image can take many minutes on the Orin.
TIMEOUT = 3600.0

# Fields chat itself fills in; setting them would break the exchange.
RESERVED = {"model", "messages", "stream", "stream_options", "prompt", "response_format"}
_IMAGE_ALIASES = {"steps": "num_inference_steps", "negative": "negative_prompt",
                  "guidance": "guidance_scale", "cfg": "true_cfg_scale"}
_TEXT_ALIASES = {"temp": "temperature"}
_LOST = (OSError, ValueError, http.client.HTTPException)


def refusal(entry: dict) -> str | None:
    """Why this ``/v1/models`` entry can't chat; None when it can."""
    info = entry.get("llmctl") or {}
    api = info.get("api")
    if api == KSERVE:
        return "a computer-vision model: it takes frames, not text"
    if api == SYSTEMONE:
        return "a decision endpoint: it answers typed questions in JSON (`llmctl decide spec`)"
    if info.get("decision"):
        return ("a decision model: it answers typed questions, not chat "
                "(`llmctl decide` serves it)")
    if info.get("embeddings"):
        return "an embedding model: it returns vectors, not text"
    if api not in (OPENAI, IMAGES):
        return f"serving the {api} API, which chat doesn't speak"
    return None


@dataclass(frozen=True)
class Model:
    """A model to chat with, as the gateway knows it."""
    ref: str                            # what requests name: id, group or name@node
    id: str
    node: str
    backend: str
    kind: str
    status: str = "ready"
    context: int | None = None
    local: bool = False
    owner: str | None = None

    def target(self, port: int) -> st.Target:
        """Requests go to the gateway; the ``model`` they name does the routing."""
        return st.Target(self.ref, self.backend, port, self.kind)


def models_on(port: int | None) -> list[dict]:
    """The gateway's ``/v1/models`` entries: every model in the cluster."""
    if port is None:
        raise click.UsageError("`llmctl chat` goes through the gateway, which is turned off: "
                               "`llmctl config gateway 9000` turns it on.")
    try:
        data = st._get_json(st.Target("gateway", GATEWAY, port, st.TEXT), "/v1/models")["data"]
    except (OSError, ValueError, KeyError, TypeError) as e:
        raise click.UsageError(f"`llmctl chat` goes through the gateway, and it doesn't answer "
                               f"on port {port} ({e}): `llmctl gateway start`.") from None
    return data if isinstance(data, list) else []


def _copies(entry: dict) -> list[dict]:
    """Where an entry runs, best first."""
    info = entry.get("llmctl") or {}
    return info.get("nodes") or [{"node": info.get("node") or "?", "status": info.get("status"),
                                  "context": info.get("context"), "local": info.get("local")}]


def _model(entry: dict, copy: dict, ref: str) -> Model:
    info = entry.get("llmctl") or {}
    return Model(ref, entry["id"], copy.get("node") or "?", info.get("backend") or "?",
                 st.IMAGE if info.get("api") == IMAGES else st.TEXT,
                 copy.get("status") or "unknown", copy.get("context"), bool(copy.get("local")),
                 info.get("owner") if copy.get("node") == info.get("node") else None)


def choices(entries: list[dict]) -> list[Model]:
    """What the picker offers: one row per copy of each model that can chat, this
    machine's first. A model on several nodes is pinned to the row's node, since
    picking a node is the point; one on a single node may move if it restarts."""
    out = []
    for e in entries:
        if (e.get("llmctl") or {}).get("members") or refusal(e):
            continue
        copies = _copies(e)
        out += [_model(e, c, f"{e['id']}@{c.get('node')}" if len(copies) > 1 else e["id"])
                for c in copies]
    return sorted(out, key=lambda m: (not m.local, m.id, m.node))


def resolve(entries: list[dict], identifier: str) -> Model:
    """``identifier`` (a model, a group, or either as ``name@node``) as a Model."""
    name, node = split_pin(identifier)
    entry = next((e for e in entries if e.get("id") == name), None) or next(
        (e for e in entries if name in ((e.get("llmctl") or {}).get("groups")
                                        or [(e.get("llmctl") or {}).get("group")])), None)
    if entry is None:
        raise click.UsageError(f"No running model matches '{name}'. Running: "
                               + (", ".join(sorted(e["id"] for e in entries)) or "(none)") + ".")
    why = refusal(entry)
    if why:
        raise click.UsageError(f"Can't chat with {entry['id']}: it is {why}.")
    copies = _copies(entry)
    copy = next((c for c in copies if c.get("node") == node), None) if node else copies[0]
    if copy is None:
        raise click.UsageError(f"{entry['id']} isn't running on node '{node}'. It is on: "
                               + ", ".join(str(c.get("node")) for c in copies) + ".")
    return _model(entry, copy, identifier)


def parse_value(text: str) -> Any:
    """A number, list, ... as JSON; anything else stays the string it is."""
    try:
        return json.loads(text)
    except ValueError:
        return text


def opener() -> str | None:
    """The command that shows a file in the desktop's default app; None when there's
    no desktop to show it on (plain SSH, a server)."""
    if sys.platform == "darwin":
        return shutil.which("open")
    if not (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")):
        return None
    return shutil.which("xdg-open")


def _note(message: str) -> None:
    click.echo(click.style(message, dim=True), err=True)


# -------------------------------------------------------------------- text --

@dataclass
class Reply:
    text: str = ""
    finish: str | None = None
    stopped: bool = False               # Ctrl-C while it was answering
    prompt_tokens: int | None = None
    output_tokens: int | None = None
    ttft: float | None = None           # s
    tps: float | None = None


def _print_piece(kind: str, piece: str) -> None:
    if kind == "reasoning":
        click.echo(click.style(piece, dim=True), nl=False, err=True)
    else:
        click.echo(piece, nl=False)


def stream_reply(t: st.Target, model: str, messages: list[dict], params: dict,
                 emit: Callable[[str, str], None] = _print_piece) -> Reply:
    """One chat completion, its pieces handed to ``emit`` as they arrive.
    Ctrl-C ends it (closing the connection cancels it on the server) and
    returns what came so far."""
    body = {**params, "model": model, "messages": messages, "stream": True,
            "stream_options": {"include_usage": True}}
    reply = Reply()
    start = time.perf_counter()
    first = last = None
    pieces = 0
    usage: dict = {}
    timings: dict = {}
    thinking = False
    try:
        with st._request(t, "/v1/chat/completions", body, TIMEOUT) as resp:
            for when, ev in st._sse(resp):
                if ev.get("error"):
                    err = ev["error"]
                    raise st.HTTPFail(None, str(err.get("message") if isinstance(err, dict)
                                                else err))
                if isinstance(ev.get("usage"), dict):
                    usage = ev["usage"]
                if isinstance(ev.get("timings"), dict):
                    timings = ev["timings"]
                for ch in ev.get("choices") or []:
                    reply.finish = ch.get("finish_reason") or reply.finish
                    delta = ch.get("delta") or {}
                    reasoning = delta.get("reasoning_content") or delta.get("reasoning")
                    content = delta.get("content")
                    if reasoning:
                        thinking = True
                        emit("reasoning", reasoning)
                    if content:
                        if thinking:
                            thinking = False
                            emit("reasoning", "\n\n")
                        reply.text += content
                        emit("text", content)
                    if reasoning or content:
                        pieces += 1
                        first = first or when
                        last = when
    except KeyboardInterrupt:
        reply.stopped = True
    if thinking:
        emit("reasoning", "\n")
    reply.prompt_tokens = usage.get("prompt_tokens") or timings.get("prompt_n")
    reply.output_tokens = usage.get("completion_tokens") or timings.get("predicted_n") or pieces
    if first is not None:
        reply.ttft = first - start
        if last > first and reply.output_tokens > 1:
            reply.tps = (reply.output_tokens - 1) / (last - first)
    if timings.get("predicted_per_second"):
        reply.tps = float(timings["predicted_per_second"])
    return reply


def _numbers(reply: Reply, ctx: int | None) -> str:
    bits = []
    if reply.prompt_tokens:
        bits.append(f"prompt {reply.prompt_tokens:,}" + (f"/{ctx:,}" if ctx else ""))
    if reply.output_tokens:
        bits.append(f"{reply.output_tokens:,} tokens")
    if reply.tps:
        bits.append(f"{reply.tps:.1f} t/s")
    if reply.ttft is not None:
        bits.append(f"{reply.ttft:.2f} s to first token")
    return " · ".join(bits)


# ---------------------------------------------------------------- sessions --

class Session:
    """What one instance is told; the REPL and the one-shot both drive it."""

    kind = ""
    aliases: dict[str, str] = {}
    help_rows: list[tuple[str, str]] = []

    def __init__(self, m: Model, port: int, params: dict | None = None):
        self.m = m
        self.t = m.target(port)
        self.params: dict[str, Any] = {}
        for key, value in (params or {}).items():
            self.set_param(key, value)

    def set_param(self, key: str, value: Any) -> None:
        key = key.replace("-", "_")
        key = self.aliases.get(key, key)
        if key in RESERVED:
            raise ValueError(f"chat sets '{key}' itself")
        self.params[key] = value

    def send(self, text: str) -> bool:
        """Send one message; False when it got no complete answer."""
        raise NotImplementedError

    def banner(self) -> str:
        return f"{self.m.id} on {self.m.node}  ({self.m.backend})"

    def command(self, line: str) -> bool:
        """Run ``/name arg``; False to leave the REPL."""
        name, _, arg = line[1:].partition(" ")
        arg = arg.strip()
        if name in ("exit", "quit", "q"):
            return False
        if name == "help":
            rows = [*self.help_rows,
                    ("/set KEY VALUE", "send KEY with every request (no value: show them all)"),
                    ("/unset KEY", "stop sending KEY"),
                    ("/exit", "leave (or Ctrl-D)")]
            w = max(len(u) for u, _ in rows)
            for usage, what in rows:
                click.echo(f"  {usage:<{w}}  {what}", err=True)
            _note("  Enter sends, Alt+Enter starts a new line, // sends a leading /.")
        elif name == "set":
            key, _, value = arg.partition(" ")
            if not key:
                _note(json.dumps(self.params) if self.params else "nothing set")
            elif not value.strip():
                click.echo("usage: /set KEY VALUE", err=True)
            else:
                try:
                    self.set_param(key, parse_value(value.strip()))
                except ValueError as e:
                    click.echo(str(e), err=True)
        elif name == "unset":
            key = arg.replace("-", "_")
            self.params.pop(self.aliases.get(key, key), None)
        elif not self.own_command(name, arg):
            click.echo(f"Unknown command /{name} (/help lists them; // sends a leading /).",
                       err=True)
        return True

    def own_command(self, name: str, arg: str) -> bool:
        return False


class TextSession(Session):
    kind = st.TEXT
    aliases = _TEXT_ALIASES
    help_rows = [("/clear", "forget the conversation"),
                 ("/retry", "answer the last message again"),
                 ("/system [TEXT]", "show or replace the system prompt")]

    def __init__(self, m: Model, port: int, params: dict | None = None,
                 system: str | None = None):
        super().__init__(m, port, params)
        self.system = system
        self.history: list[dict] = []
        self.unanswered: str | None = None      # what /retry resends after a failure
        self.ctx = m.context

    def banner(self) -> str:
        ctx = f" · ctx {self.ctx:,}" if self.ctx else ""
        return f"{self.m.id} on {self.m.node}  ({self.m.backend}{ctx})"

    def send(self, text: str) -> bool:
        self.history.append({"role": "user", "content": text})
        self.unanswered = None
        messages = ([{"role": "system", "content": self.system}] if self.system else []
                    ) + self.history
        failure = None
        printed = False

        def emit(kind: str, piece: str) -> None:
            nonlocal printed
            printed = printed or kind == "text"
            _print_piece(kind, piece)

        try:
            reply = stream_reply(self.t, self.m.ref, messages, self.params, emit)
        except st.HTTPFail as e:
            failure = str(e)
            if (e.status in (400, 413, 422) and "context" in failure.lower()
                    and len(self.history) > 1):
                failure += "\nThe conversation no longer fits: /clear starts over."
        except _LOST as e:
            failure = f"Lost {self.t.name} while it was answering: {e}"
        if failure is not None:
            if printed:
                click.echo()
            return self._forget(text, failure)
        if reply.text:
            click.echo()
        if not reply.text and not reply.stopped:
            why = f" (finish_reason: {reply.finish})" if reply.finish else ""
            return self._forget(text, f"{self.t.name} sent no reply{why}.")
        if not reply.text:
            self.history.pop()
            self.unanswered = text
            _note("[stopped]")
            return False
        self.history.append({"role": "assistant", "content": reply.text})
        notes = [_numbers(reply, self.ctx)]
        if reply.stopped:
            notes.insert(0, "[stopped]")
        elif reply.finish == "length":
            notes.insert(0, "[cut off: max_tokens or the context ran out]")
        elif reply.finish is None:
            notes.insert(0, "[the reply ended without a finish reason: is the model still up?]")
        _note("  ".join(n for n in notes if n))
        return not reply.stopped

    def _forget(self, text: str, why: str) -> bool:
        self.history.pop()
        self.unanswered = text
        click.echo(f"{why}", err=True)
        return False

    def own_command(self, name: str, arg: str) -> bool:
        if name == "clear":
            self.history.clear()
            self.unanswered = None
            _note("Conversation cleared.")
        elif name == "retry":
            text = self.unanswered
            if text is None and self.history:
                if self.history[-1]["role"] == "assistant":
                    self.history.pop()
                text = self.history.pop()["content"]
            if text is None:
                click.echo("Nothing to retry.", err=True)
            else:
                self.send(text)
        elif name == "system":
            if arg:
                self.system = arg
                _note("System prompt set; it applies to the whole conversation.")
            else:
                _note(self.system or "No system prompt.")
        else:
            return False
        return True


class ImageSession(Session):
    kind = st.IMAGE
    aliases = _IMAGE_ALIASES
    help_rows = [("/image FILE", "add a reference image: prompts now edit (repeat for more)"),
                 ("/image", "drop the references: prompts generate again"),
                 ("/retry", "the last prompt again (a new seed, unless one is set)"),
                 ("/set size 768x768", "also steps, seed, negative, guidance, cfg, n")]

    def __init__(self, m: Model, port: int, params: dict | None = None,
                 out: Path | None = None, references: list[Path] | None = None,
                 preview: bool = False):
        super().__init__(m, port, params)
        self.out = out or Path(".")
        self.preview = preview
        self.references = list(references or [])
        self.last_prompt: str | None = None

    def banner(self) -> str:
        return f"{self.m.id} on {self.m.node}  ({self.m.backend} · images go to {self.out})"

    def _generate(self, prompt: str, seed: int) -> dict:
        body = {**self.params, "model": self.m.ref, "prompt": prompt, "seed": seed,
                "response_format": "b64_json"}
        return st._post_json(self.t, "/v1/images/generations", body, TIMEOUT)

    def _edit(self, prompt: str, seed: int) -> dict:
        # The gateway routes a multipart upload by X-Model and leaves the form alone,
        # and vLLM-Omni wants its exact id in the form's model field, or nothing.
        fields = {**self.params, "prompt": prompt, "seed": seed, "response_format": "b64_json"}
        boundary = uuid.uuid4().hex
        parts: list[bytes] = []
        for key, value in fields.items():
            text = value if isinstance(value, str) else json.dumps(value)
            parts.append(f'--{boundary}\r\nContent-Disposition: form-data; name="{key}"\r\n\r\n'
                         f"{text}\r\n".encode())
        for path in self.references:
            ctype = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
            parts.append(f'--{boundary}\r\nContent-Disposition: form-data; name="image"; '
                         f'filename="{path.name}"\r\nContent-Type: {ctype}\r\n\r\n'.encode()
                         + path.read_bytes() + b"\r\n")
        parts.append(f"--{boundary}--\r\n".encode())
        req = urllib.request.Request(
            st._url(self.t, "/v1/images/edits"), data=b"".join(parts), method="POST",
            headers={"Content-Type": f"multipart/form-data; boundary={boundary}",
                     "X-Model": self.m.ref})
        with st._open(self.t, req, TIMEOUT) as r:
            return json.loads(r.read())

    def _save(self, data: list[dict], seed: int) -> list[Path]:
        self.out.mkdir(parents=True, exist_ok=True)
        stem = re.sub(r"[^A-Za-z0-9_.-]+", "-", self.m.id)
        stamp = time.strftime("%Y%m%d-%H%M%S")
        saved = []
        for i, item in enumerate(data):
            blob = base64.b64decode(item["b64_json"])
            ext = ("png" if blob[:4] == b"\x89PNG" else "jpg" if blob[:2] == b"\xff\xd8"
                   else "webp" if blob[8:12] == b"WEBP" else "bin")
            suffix = f"-{i + 1}" if len(data) > 1 else ""
            path = self.out / f"{stem}-{stamp}-{seed}{suffix}.{ext}"
            path.write_bytes(blob)
            saved.append(path)
        return saved

    def _show(self, path: Path) -> None:
        """Open ``path`` in the default viewer, detached: the chat doesn't wait for
        the viewer, and a closed terminal doesn't take it down."""
        command = opener()
        if command is None:
            return
        try:
            subprocess.Popen([command, str(path.resolve())], stdin=subprocess.DEVNULL,
                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                             start_new_session=True)
        except OSError as e:
            self.preview = False
            _note(f"Can't open images with {command}: {e}. They are still saved.")

    def send(self, text: str) -> bool:
        self.last_prompt = text
        seed = self.params.get("seed", random.randrange(2**31))
        started = time.perf_counter()
        _note("editing…" if self.references else "generating…")
        try:
            reply = (self._edit if self.references else self._generate)(text, seed)
            data = [d for d in reply.get("data") or [] if d.get("b64_json")]
            if not data:
                raise ValueError("the server returned no image")
            saved = self._save(data, seed)
        except KeyboardInterrupt:
            _note("[stopped waiting; the server may still finish the image]")
            return False
        except st.HTTPFail as e:
            click.echo(str(e), err=True)
            return False
        except _LOST as e:
            click.echo(f"No image from {self.t.name}: {e}", err=True)
            return False
        for path in saved:
            click.echo(str(path))
            if self.preview:
                self._show(path)
        _note(f"seed {seed} · {time.perf_counter() - started:.1f} s")
        return True

    def own_command(self, name: str, arg: str) -> bool:
        if name == "image":
            if not arg:
                self.references.clear()
                _note("No reference images: prompts generate.")
            elif not Path(arg).expanduser().is_file():
                click.echo(f"No such file: {arg}", err=True)
            else:
                self.references.append(Path(arg).expanduser())
                _note(f"{len(self.references)} reference image(s): prompts edit.")
        elif name == "retry":
            if self.last_prompt is None:
                click.echo("Nothing to retry.", err=True)
            else:
                self.send(self.last_prompt)
        elif name == "clear":
            self.last_prompt = None
        else:
            return False
        return True


# -------------------------------------------------------------------- repl --

def _reader() -> Callable[[], str]:
    """Reads one message: Enter sends, Alt+Enter adds a line. Ctrl-J stays an Enter:
    that's what an Enter typed ahead during a reply arrives as, the terminal not
    being in raw mode then."""
    from prompt_toolkit import PromptSession
    from prompt_toolkit.key_binding import KeyBindings

    keys = KeyBindings()

    @keys.add("enter")
    def _send(event) -> None:
        event.current_buffer.validate_and_handle()

    @keys.add("escape", "enter")
    def _newline(event) -> None:
        event.current_buffer.insert_text("\n")

    session: PromptSession = PromptSession(key_bindings=keys, multiline=True,
                                           prompt_continuation="  ")
    return lambda: session.prompt("> ")


def repl(session: Session, read: Callable[[], str] | None = None) -> None:
    """Read messages until Ctrl-D or /exit. Ctrl-C drops the line being typed."""
    read = read or _reader()
    click.echo(f"Chatting with {session.banner()}", err=True)
    _note("/help lists the commands; Ctrl-C stops a reply; Ctrl-D leaves.")
    while True:
        try:
            line = read()
        except KeyboardInterrupt:
            continue
        except EOFError:
            return
        text = line.strip()
        if not text:
            continue
        if text.startswith("/") and not text.startswith("//"):
            if not session.command(text):
                return
            continue
        session.send(text[1:] if text.startswith("//") else text)

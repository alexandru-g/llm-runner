"""Remote control: run llmctl commands on another node through its gateway.

``llmctl run --node orin`` (or ``llmctl --node orin run``) asks orin's
gateway to run ``llmctl run`` there, in a pseudo-terminal when this side has
one, and relays the keystrokes and output both ways. Every picker, launch menu
and prompt works as it does on orin itself, with orin's models, engines and
remembered choices; scripts get stdout and stderr apart and the exit code.

The request is ``POST /control/v1/exec`` with a JSON body naming the command;
the gateway answers ``101 Switching Protocols`` and the connection then
carries frames (:func:`frame`): a type byte and a length, then the bytes.

Who may: nothing, until ``llmctl gateway control on`` on the node. Then:

* this machine (loopback);
* another node's gateway or CLI, with the cluster token, from the address
  the registry has for it, and named in ``llmctl gateway control allow``;
* a client with an API key made with ``llmctl gateway key add NAME --control``.

The LAN exemption of model calls never applies here.

What: the commands in :data:`ALLOWED` (models, not this machine's set-up), and
none of the options in :data:`FORBIDDEN` (they would run any program here).
``apply`` sends its serving files along; their entries may not carry
``args``, ``env`` or an ``image``.
"""
from __future__ import annotations

import errno
import fcntl
import json
import os
import pty
import secrets
import select
import shutil
import signal
import socket
import struct
import subprocess
import sys
import termios
import tty
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

from llm_runner import config as config_mod
from llm_runner import process as proc

EXEC_PATH = "/control/v1/exec"
STATUS_PATH = "/control/v1/status"
NODE_HEADER = "X-Llmctl-Control-Node"
UPGRADE = "llmctl-exec"
SINCE = "1.17"
# Env var for a control key, when the caller isn't a cluster node.
KEY_ENV = "LLMCTL_CONTROL_KEY"

# Commands a remote caller may run: managing models, not this machine.
ALLOWED = {
    ("apply",), ("autostart", "list"), ("autostart", "status"), ("chat",),
    ("decide", "profiles"), ("decide", "spec"), ("decide", "start"),
    ("download",), ("events",), ("label",), ("label", "add"), ("label", "create"),
    ("label", "delete"), ("label", "edit"), ("label", "list"), ("label", "prefer"),
    ("label", "reset"), ("label", "rm"), ("label", "show"),
    ("list",), ("logs",), ("pause",), ("profile",),
    ("profile", "list"), ("profile", "rm"), ("ps",), ("registry", "status"),
    ("restart",), ("restore",), ("resume",), ("rm",), ("run",), ("show",),
    ("speedtest",), ("stop",), ("update",), ("verify",), ("version",),
}
# Options that would run any program, or write anywhere, on the node.
FORBIDDEN = {"extra": "-X/--extra", "env_pairs": "-E/--env", "ft_path": "--ft",
             "image": "--image", "allow_tag": "--allow-tag", "out": "--out"}
# A serving file's keys that do the same.
FORBIDDEN_KEYS = ("args", "env", "image", "allow_tag")

# Frame types. Client to node: input, resize (cols, rows), end of input.
# Node to client: stdout, stderr, exit code.
IN, RESIZE, EOF, OUT, ERR, EXIT = b"i", b"r", b"e", b"o", b"E", b"x"
_HEAD = struct.Struct(">cI")
MAX_FRAME = 1 << 20


class Refused(Exception):
    def __init__(self, status: int, code: str, message: str):
        super().__init__(message)
        self.status, self.code, self.message = status, code, message


# ---------------------------------------------------------------- settings --

def settings(cfg: dict | None = None) -> dict:
    cfg = config_mod.load() if cfg is None else cfg
    nodes = cfg.get("remote_control_from")
    keys = cfg.get("gateway_control_keys")
    return {"on": cfg.get("remote_control") is True,
            "nodes": sorted(n for n in nodes if isinstance(n, str)) if isinstance(nodes, list)
            else [],
            "keys": sorted(k for k in keys if isinstance(k, str)) if isinstance(keys, list)
            else []}


# -------------------------------------------------------------------- who --

def _addresses(url: str) -> set[str]:
    host = urllib.parse.urlsplit(url).hostname or ""
    try:
        return {a[4][0] for a in socket.getaddrinfo(host, None)}
    except OSError:
        return {host}


def authorize(headers, peer: str, *, token: str | None, policy, nodes: list[dict],
              node: str) -> str:
    """Who is asking (``local``, ``node rtx``, ``key ops``), or :class:`Refused`."""
    from llm_runner.cluster import TOKEN_HEADER, token_ok
    s = settings()
    if not s["on"]:
        raise Refused(403, "control_off", f"Remote control is off on {node} "
                      f"(`llmctl gateway control on` there).")
    import ipaddress
    try:
        loopback = ipaddress.ip_address(peer).is_loopback
    except ValueError:
        loopback = False
    if loopback and not policy.via_proxy(peer, headers):
        return "local"
    claimed = headers.get(NODE_HEADER)
    if claimed:
        if not token_ok(token, headers.get(TOKEN_HEADER)):
            raise Refused(401, "unauthorized", "Missing or wrong cluster token."
                          if token else f"{node} is not in a cluster.")
        if claimed not in s["nodes"]:
            raise Refused(403, "node_not_allowed",
                          f"{claimed} may not control {node} "
                          f"(`llmctl gateway control allow {claimed}` there).")
        urls = [n.get("url") or "" for n in nodes if n.get("node") == claimed]
        if not any(peer in _addresses(u) for u in urls):
            raise Refused(403, "wrong_address",
                          f"The request names {claimed}, but comes from {peer}, not the "
                          "address the registry has for it.")
        return f"node {claimed}"
    key = policy.key_name(headers)
    if key and key in s["keys"]:
        return f"key {key}"
    if key:
        raise Refused(403, "no_control_scope",
                      f"Key {key} has no control scope (`llmctl gateway key add NAME --control`).")
    raise Refused(401, "control_key_needed",
                  "Remote control needs a cluster node (`llmctl gateway control allow`) or a "
                  "control key (Authorization: Bearer <key>).")


# ------------------------------------------------------------------- what --

def check_argv(argv: list[str]) -> tuple[str, ...]:
    """The command path ``argv`` runs, if a remote caller may run it."""
    import click
    from llm_runner.cli import cli
    if not argv or not all(isinstance(a, str) for a in argv):
        raise Refused(400, "invalid_command", "Name an llmctl command to run.")
    if argv[0] in ("--version", "-V", "--help", "-h"):
        return ()
    cmd: click.Command = cli
    path: list[str] = []
    rest = list(argv)
    ctx = click.Context(cli, info_name="llmctl", resilient_parsing=True)
    while isinstance(cmd, click.Group) and rest and not rest[0].startswith("-"):
        if path == ["decide"] and rest[0] not in cmd.commands:
            rest = ["start", *rest]             # `decide MODEL` is `decide start MODEL`
        word = rest[0]
        try:
            name, sub, rest = cmd.resolve_command(ctx, rest)
        except click.UsageError as e:
            raise Refused(400, "invalid_command", e.format_message()) from None
        if sub is None:
            raise Refused(400, "invalid_command", f"No command {word!r}.")
        path.append(sub.name or name or "")
        cmd = sub
        ctx = click.Context(cmd, info_name=name, parent=ctx, resilient_parsing=True)
    if tuple(path) not in ALLOWED:
        raise Refused(403, "command_not_allowed",
                      f"`llmctl {' '.join(path) or argv[0]}` can't be run remotely: only "
                      "commands that manage models can.")
    try:
        sub_ctx = cmd.make_context(path[-1], list(rest), resilient_parsing=True)
    except click.ClickException:
        return tuple(path)                      # let the command itself say what's wrong
    typed = click.core.ParameterSource.COMMANDLINE
    bad = [flag for p, flag in FORBIDDEN.items()
           if p in sub_ctx.params and sub_ctx.get_parameter_source(p) == typed]
    if bad:
        raise Refused(403, "option_not_allowed",
                      f"{', '.join(bad)} can't be used remotely.")
    return tuple(path)


def stage_files(path: tuple[str, ...], argv: list[str], files: dict, workdir: Path) -> list[str]:
    """Write ``apply``'s serving files to ``workdir`` and point ``argv`` at them."""
    if not files:
        return argv
    if path != ("apply",):
        raise Refused(400, "files_not_allowed", "Only `apply` takes files.")
    workdir.mkdir(parents=True, exist_ok=True)
    out = list(argv)
    staged = []
    for i, (name, text) in enumerate(files.items()):
        if not isinstance(text, str) or not isinstance(name, str):
            raise Refused(400, "invalid_files", "Files are {name: text}.")
        dest = workdir / f"{i}-{Path(name).name}"
        dest.write_text(text)
        staged.append(dest)
        out = [str(dest) if a == name else a for a in out]
    _check_serving_files(staged)
    return out


def _check_serving_files(paths: list[Path]) -> None:
    import yaml
    for p in paths:
        try:
            doc = yaml.safe_load(p.read_text()) or {}
        except yaml.YAMLError as e:
            raise Refused(400, "invalid_files", f"{p.name.split('-', 1)[-1]}: {e}") from None
        models = doc.get("models") if isinstance(doc, dict) else None
        for name, entry in (models or {}).items() if isinstance(models, dict) else ():
            if not isinstance(entry, dict):
                continue
            bad = [k for k in FORBIDDEN_KEYS if k in entry]
            if isinstance(entry.get("decide"), dict) and "args" in entry["decide"]:
                bad.append("decide.args")
            if bad:
                raise Refused(403, "option_not_allowed",
                              f"models.{name}: {', '.join(bad)} can't be applied remotely.")


# ------------------------------------------------------------------ frames --

def frame(kind: bytes, data: bytes = b"") -> bytes:
    return _HEAD.pack(kind, len(data)) + data


class Reader:
    """Splits a byte stream into frames."""

    def __init__(self) -> None:
        self.buf = bytearray()

    def feed(self, data: bytes) -> list[tuple[bytes, bytes]]:
        self.buf += data
        out = []
        while len(self.buf) >= _HEAD.size:
            kind, n = _HEAD.unpack_from(self.buf)
            if n > MAX_FRAME:
                raise ValueError("frame too large")
            if len(self.buf) < _HEAD.size + n:
                break
            out.append((kind, bytes(self.buf[_HEAD.size:_HEAD.size + n])))
            del self.buf[:_HEAD.size + n]
        return out


def _send_all(sock: socket.socket, data: bytes) -> None:
    sock.sendall(data)


# ------------------------------------------------------------------ server --

def llmctl_argv() -> list[str]:
    return [sys.executable, "-m", "llm_runner"]


def run_exec(sock: socket.socket, argv: list[str], *, who: str, use_tty: bool,
             size: tuple[int, int] | None, term: str | None) -> int:
    """Run ``llmctl ARGV`` and relay it over ``sock`` (already upgraded)."""
    env = {**os.environ, "LLMCTL_REMOTE": who, "PYTHONUNBUFFERED": "1"}
    if term:
        env["TERM"] = term
    master = None
    if use_tty:
        master, slave = pty.openpty()
        if size:
            _set_size(master, size)
        child = subprocess.Popen([*llmctl_argv(), *argv], stdin=slave, stdout=slave,
                                 stderr=slave, env=env, start_new_session=True, close_fds=True)
        os.close(slave)
        outs = {master: OUT}
        stdin_fd = master
    else:
        child = subprocess.Popen([*llmctl_argv(), *argv], stdin=subprocess.PIPE,
                                 stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env,
                                 start_new_session=True, close_fds=True)
        outs = {child.stdout.fileno(): OUT, child.stderr.fileno(): ERR}
        stdin_fd = child.stdin.fileno()
    reader = Reader()
    gone = False
    try:
        while outs:
            ready, _, _ = select.select([*outs, *([] if gone else [sock])], [], [], 1.0)
            for fd in ready:
                if fd is sock:
                    try:
                        data = sock.recv(65536)
                    except OSError:
                        data = b""
                    if not data:
                        gone = True             # the client left: hang up on the command
                        _signal(child, signal.SIGHUP)
                        continue
                    for kind, payload in reader.feed(data):
                        if kind == IN and stdin_fd is not None:
                            try:
                                os.write(stdin_fd, payload)
                            except OSError:
                                pass
                        elif kind == RESIZE and master is not None and len(payload) == 4:
                            _set_size(master, struct.unpack(">HH", payload))
                        elif kind == EOF:
                            if master is not None:
                                os.write(master, b"\x04")
                            elif child.stdin:
                                child.stdin.close()
                                stdin_fd = None
                    continue
                try:
                    data = os.read(fd, 65536)
                except OSError as e:            # a pty reads EIO once the child has gone
                    if e.errno != errno.EIO:
                        raise
                    data = b""
                if not data:
                    del outs[fd]
                    continue
                if not gone:
                    try:
                        _send_all(sock, frame(outs[fd], data))
                    except OSError:
                        gone = True
                        _signal(child, signal.SIGHUP)
            if master is not None and child.poll() is not None and master in outs:
                # Drain what's left, then stop: the pty may never report EOF.
                r, _, _ = select.select([master], [], [], 0.1)
                if not r:
                    del outs[master]
        code = child.wait()
        if not gone:
            try:
                _send_all(sock, frame(EXIT, struct.pack(">i", code)))
            except OSError:
                pass
        return code
    finally:
        if master is not None:
            os.close(master)
        if child.poll() is None:
            _signal(child, signal.SIGTERM)


def _signal(child: subprocess.Popen, sig: int) -> None:
    try:
        os.killpg(child.pid, sig)
    except (ProcessLookupError, PermissionError):
        pass


def _set_size(fd: int, size) -> None:
    cols, rows = size
    fcntl.ioctl(fd, termios.TIOCSWINSZ, struct.pack("HHHH", rows, cols, 0, 0))


def workdir() -> Path:
    return proc.STATE_DIR / "control" / secrets.token_hex(6)


# ------------------------------------------------------------------ client --

def split_node(argv: list[str]) -> tuple[str | None, list[str]]:
    """``--node NAME`` (anywhere before ``--``) and the argv without it."""
    for i, a in enumerate(argv):
        if a == "--":
            break
        if a == "--node" and i + 1 < len(argv):
            return argv[i + 1], argv[:i] + argv[i + 2:]
        if a.startswith("--node="):
            return a.split("=", 1)[1], argv[:i] + argv[i + 1:]
    return None, argv


def is_self(node: str) -> bool:
    return node in (config_mod.get_cluster().node_name, "local", "localhost")


def node_url(node: str) -> tuple[str, str | None]:
    """The gateway URL of ``node`` (a URL is taken as is) and its llmctl version."""
    if node.startswith(("http://", "https://")):
        return node.rstrip("/"), None
    from llm_runner.cluster import TOKEN_HEADER
    cl = config_mod.get_cluster()
    sources = []
    port = config_mod.get_gateway_port() or 9000
    sources.append((config_mod.local_url(port, "/cluster/v1/status"), {}, "nodes"))
    if cl.registry_url:
        sources.append((cl.registry_url.rstrip("/") + "/registry/v1/nodes",
                        {TOKEN_HEADER: cl.token or ""}, "nodes"))
    for url, headers, key in sources:
        try:
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=5) as r:
                data = json.loads(r.read())
        except (OSError, ValueError, urllib.error.URLError):
            continue
        for n in data.get(key) or []:
            if (n.get("node") or n.get("name")) == node and n.get("url"):
                return n["url"].rstrip("/"), n.get("version")
    raise LookupError(f"No node {node!r} in the cluster (see `llmctl registry status`).")


def _collect_files(argv: list[str]) -> dict[str, str]:
    """``apply``'s local serving files, sent along to the node."""
    if not argv or argv[0] != "apply":
        return {}
    out = {}
    for a in argv[1:]:
        if not a.startswith("-") and os.path.isfile(a):
            out[a] = Path(a).read_text()
    return out


def exec_remote(node: str, argv: list[str]) -> int:
    """Run ``llmctl ARGV`` on ``node`` here in this terminal; its exit code."""
    from llm_runner.cluster import TOKEN_HEADER
    try:
        url, version = node_url(node)
    except LookupError as e:
        print(f"llmctl: {e}", file=sys.stderr)
        return 1
    use_tty = sys.stdin.isatty() and sys.stdout.isatty()
    size = shutil.get_terminal_size() if use_tty else None
    body = json.dumps({"argv": argv, "tty": use_tty, "files": _collect_files(argv),
                       "size": [size.columns, size.lines] if size else None,
                       "term": os.environ.get("TERM")}).encode()
    cl = config_mod.get_cluster()
    headers = {"Content-Type": "application/json", "Content-Length": str(len(body)),
               "Upgrade": UPGRADE, "Connection": "Upgrade"}
    key = os.environ.get(KEY_ENV)
    if key:
        headers["Authorization"] = f"Bearer {key}"
    elif cl.token:
        headers[NODE_HEADER] = cl.node_name
        headers[TOKEN_HEADER] = cl.token
    parts = urllib.parse.urlsplit(url)
    try:
        sock = socket.create_connection((parts.hostname, parts.port or 80), timeout=10)
    except OSError as e:
        print(f"llmctl: can't reach {node} at {url}: {e}", file=sys.stderr)
        return 1
    head = f"POST {EXEC_PATH} HTTP/1.1\r\nHost: {parts.netloc}\r\n" + "".join(
        f"{k}: {v}\r\n" for k, v in headers.items()) + "\r\n"
    sock.sendall(head.encode() + body)
    status, rest = _read_head(sock)
    if status != 101:
        print(f"llmctl: {node}: {_refusal(sock, status, rest, version)}", file=sys.stderr)
        return 1
    sock.settimeout(None)
    return _pump(sock, rest, use_tty)


def _read_head(sock: socket.socket) -> tuple[int, bytes]:
    buf = b""
    while b"\r\n\r\n" not in buf:
        chunk = sock.recv(65536)
        if not chunk:
            break
        buf += chunk
    head, _, rest = buf.partition(b"\r\n\r\n")
    try:
        status = int(head.split(b"\r\n", 1)[0].split()[1])
    except (IndexError, ValueError):
        status = 0
    rest_head = head.decode("latin-1")
    length = next((int(line.split(":", 1)[1]) for line in rest_head.split("\r\n")
                   if line.lower().startswith("content-length:")), None)
    if status != 101 and length is not None:
        while len(rest) < length:
            chunk = sock.recv(65536)
            if not chunk:
                break
            rest += chunk
    return status, rest


def _refusal(sock, status: int, body: bytes, version: str | None) -> str:
    try:
        msg = json.loads(body)["error"]["message"]
    except (ValueError, KeyError, TypeError):
        msg = None
    if status == 404 and msg is None or (status == 404 and "No route" in (msg or "")):
        return (f"its gateway has no remote control (needs llmctl {SINCE} or later"
                + (f"; it runs {version})" if version else ")"))
    return msg or f"HTTP {status}"


def _pump(sock: socket.socket, first: bytes, use_tty: bool) -> int:
    reader = Reader()
    code = 1
    try:
        stdin = sys.stdin.fileno()
    except (AttributeError, OSError, ValueError):
        stdin = None                            # no stdin to forward (a pseudofile)
    saved = termios.tcgetattr(stdin) if use_tty else None
    out, err = sys.stdout.buffer, sys.stderr.buffer

    def resize(*_):
        s = shutil.get_terminal_size()
        try:
            sock.sendall(frame(RESIZE, struct.pack(">HH", s.columns, s.lines)))
        except OSError:
            pass
    old_winch = signal.signal(signal.SIGWINCH, resize) if use_tty else None
    if use_tty:
        tty.setraw(stdin)
    watch_stdin = stdin is not None

    def handle(frames) -> bool:
        nonlocal code
        for kind, payload in frames:
            if kind == OUT:
                out.write(payload)
                out.flush()
            elif kind == ERR:
                err.write(payload)
                err.flush()
            elif kind == EXIT:
                code = struct.unpack(">i", payload)[0]
                return True
        return False
    try:
        if stdin is None:
            sock.sendall(frame(EOF))
        if first and handle(reader.feed(first)):
            return code
        while True:
            fds = [sock] + ([stdin] if watch_stdin else [])
            ready, _, _ = select.select(fds, [], [])
            if sock in ready:
                data = sock.recv(65536)
                if not data:
                    return code
                if handle(reader.feed(data)):
                    return code
            if watch_stdin and stdin in ready:
                data = os.read(stdin, 65536)
                if data:
                    sock.sendall(frame(IN, data))
                else:
                    sock.sendall(frame(EOF))
                    watch_stdin = False
    except KeyboardInterrupt:
        return 130
    finally:
        if saved is not None:
            termios.tcsetattr(stdin, termios.TCSADRAIN, saved)
        if old_winch is not None:
            signal.signal(signal.SIGWINCH, old_winch)
        sock.close()

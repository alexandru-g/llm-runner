"""The gateway's log lines: plain text (the default) or JSON lines.

``llmctl config gateway-log json`` switches the format; the gateway reads
it when it starts. A JSON line always has ``ts``, ``level``, ``node``,
``event`` and ``msg``, plus the event's own fields. Request lines
(``event: "request"``) carry ids, model, status and timings, never prompt
or completion text: there is no mode that logs content.
"""
from __future__ import annotations

import json
import sys
import time

TEXT, JSON = "text", "json"
FORMATS = (TEXT, JSON)

_format = TEXT
_node = ""


def configure(fmt: str | None, node: str) -> None:
    global _format, _node
    _format = fmt if fmt in FORMATS else TEXT
    _node = node


def is_json() -> bool:
    return _format == JSON


def _ts() -> str:
    t = time.time()
    return time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(t)) + f".{int(t % 1 * 1000):03d}Z"


def emit(msg: str, *, event: str = "log", level: str = "info", **fields) -> None:
    """One line on stdout (the gateway's log file)."""
    if _format == JSON:
        line = {"ts": _ts(), "level": level, "node": _node, "event": event, "msg": msg}
        line.update({k: v for k, v in fields.items() if v is not None})
        print(json.dumps(line, default=str), flush=True)
    else:
        print(msg, flush=True)


def request(fields: dict, text: str) -> None:
    """The line for one call: ``text`` in text mode, ``fields`` as JSON."""
    if _format == JSON:
        emit(text, event="request", **fields)
    else:
        print(text, file=sys.stdout, flush=True)

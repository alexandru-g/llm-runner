"""W3C trace context through the gateway, and optional OTLP span export.

Every call gets a trace: the client's ``traceparent`` when it sends a valid
one, else a new trace id. The gateway adds a server span for the call and
a client span per upstream attempt (so failovers show), and sends the
backend a ``traceparent`` naming the attempt span as parent; ``tracestate``
passes through untouched.

Export is off by default (no egress). With ``llmctl config gateway-otlp
http://collector:4318`` (or ``LLM_RUNNER_GATEWAY_OTLP``), sampled spans
are batched on a background thread and POSTed to ``<endpoint>/v1/traces``
as OTLP/HTTP JSON, or protobuf (``--protocol protobuf``, which Phoenix
needs; encoded here, see :func:`to_otlp_proto`); when the queue is full or
the collector is down, spans are dropped rather than slowing calls. A span
with a ``project`` goes out under a resource with
``openinference.project.name`` (Phoenix's project). Standard library only.

Spans carry metadata only (names, ids, status, timings, token counts),
never a request or reply body: see ``gateway._export_span``.
"""
from __future__ import annotations

import json
import queue
import re
import secrets
import struct
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field

from llm_runner import logs

_TRACEPARENT = re.compile(r"00-([0-9a-f]{32})-([0-9a-f]{16})-([0-9a-f]{2})")
SERVER, CLIENT = 2, 3              # OTLP SpanKind
STATUS_ERROR = 2
PROJECT_ATTR = "openinference.project.name"
SCOPE = "llmctl-gateway"
JSON, PROTOBUF = "json", "protobuf"


@dataclass
class Span:
    trace_id: str
    span_id: str
    parent_id: str | None
    name: str
    kind: int
    start_ns: int = field(default_factory=time.time_ns)
    end_ns: int | None = None
    attrs: dict = field(default_factory=dict)
    error: bool = False
    project: str | None = None     # the resource's openinference.project.name

    def end(self, **attrs) -> None:
        self.attrs.update({k: v for k, v in attrs.items() if v is not None})
        self.end_ns = time.time_ns()


@dataclass
class Context:
    trace_id: str
    parent_id: str | None          # the client's span, if it sent one
    sampled: bool
    project: str | None = None     # given to every span of the call; "" = not exported

    @property
    def exported(self) -> bool:
        return self.sampled and self.project != ""

    @classmethod
    def from_headers(cls, headers) -> "Context":
        m = _TRACEPARENT.fullmatch((headers.get("traceparent") or "").strip().lower())
        if m and m.group(1) != "0" * 32 and m.group(2) != "0" * 16:
            return cls(m.group(1), m.group(2), bool(int(m.group(3), 16) & 1))
        return cls(secrets.token_hex(16), None, True)

    def span(self, name: str, kind: int, parent: str | None = None, **attrs) -> Span:
        return Span(self.trace_id, secrets.token_hex(8), parent or self.parent_id, name,
                    kind, attrs={k: v for k, v in attrs.items() if v is not None},
                    project=self.project)

    def traceparent(self, span: Span) -> str:
        return f"00-{self.trace_id}-{span.span_id}-{'01' if self.sampled else '00'}"


def _value(v) -> dict:
    if isinstance(v, bool):
        return {"boolValue": v}
    if isinstance(v, int):
        return {"intValue": str(v)}
    if isinstance(v, float):
        return {"doubleValue": v}
    return {"stringValue": str(v)}


def _by_resource(spans: list[Span], resource: dict) -> list[tuple[dict, list[Span]]]:
    """One resource per project, in the order projects first appear."""
    groups: dict[str | None, list[Span]] = {}
    for s in spans:
        groups.setdefault(s.project, []).append(s)
    return [({**resource, PROJECT_ATTR: project} if project else resource, group)
            for project, group in groups.items()]


def to_otlp(spans: list[Span], resource: dict) -> dict:
    return {"resourceSpans": [{
        "resource": {"attributes": [{"key": k, "value": _value(v)} for k, v in res.items()]},
        "scopeSpans": [{"scope": {"name": SCOPE}, "spans": [
            {"traceId": s.trace_id, "spanId": s.span_id,
             **({"parentSpanId": s.parent_id} if s.parent_id else {}),
             "name": s.name, "kind": s.kind,
             "startTimeUnixNano": str(s.start_ns), "endTimeUnixNano": str(s.end_ns or s.start_ns),
             "attributes": [{"key": k, "value": _value(v)} for k, v in s.attrs.items()],
             "status": {"code": STATUS_ERROR} if s.error else {}}
            for s in group]}]} for res, group in _by_resource(spans, resource)]}


# --- OTLP protobuf (opentelemetry/proto/collector/trace/v1, trace/v1, common/v1).
# Only what the gateway sends: field numbers as in those .proto files.

def _varint(n: int) -> bytes:
    out = bytearray()
    n &= (1 << 64) - 1                 # int64: negatives as two's complement
    while True:
        b, n = n & 0x7F, n >> 7
        out.append(b | (0x80 if n else 0))
        if not n:
            return bytes(out)


def _len(num: int, payload: bytes) -> bytes:          # wire type 2
    return _varint(num << 3 | 2) + _varint(len(payload)) + payload


def _str(num: int, text: str) -> bytes:
    return _len(num, text.encode())


def _any_value(v) -> bytes:                           # common.v1.AnyValue
    if isinstance(v, bool):
        return _varint(2 << 3) + _varint(int(v))
    if isinstance(v, int):
        return _varint(3 << 3) + _varint(v)
    if isinstance(v, float):
        return _varint(4 << 3 | 1) + struct.pack("<d", v)
    return _str(1, str(v))


def _kv(k: str, v) -> bytes:                          # common.v1.KeyValue
    return _str(1, k) + _len(2, _any_value(v))


def _span_proto(s: Span) -> bytes:                    # trace.v1.Span
    out = (_len(1, bytes.fromhex(s.trace_id)) + _len(2, bytes.fromhex(s.span_id))
           + (_len(4, bytes.fromhex(s.parent_id)) if s.parent_id else b"")
           + _str(5, s.name) + _varint(6 << 3) + _varint(s.kind)
           + _varint(7 << 3 | 1) + struct.pack("<Q", s.start_ns)
           + _varint(8 << 3 | 1) + struct.pack("<Q", s.end_ns or s.start_ns)
           + b"".join(_len(9, _kv(k, v)) for k, v in s.attrs.items()))
    if s.error:                                       # Status.code = 3
        out += _len(15, _varint(3 << 3) + _varint(STATUS_ERROR))
    return out


def to_otlp_proto(spans: list[Span], resource: dict) -> bytes:
    """An ExportTraceServiceRequest, the body of OTLP/HTTP protobuf."""
    out = bytearray()
    for res, group in _by_resource(spans, resource):
        resource_msg = b"".join(_len(1, _kv(k, v)) for k, v in res.items())
        scope_spans = _len(1, _str(1, SCOPE)) + b"".join(_len(2, _span_proto(s)) for s in group)
        out += _len(1, _len(1, resource_msg) + _len(2, scope_spans))
    return bytes(out)


class Exporter:
    """Batches spans to an OTLP/HTTP collector. ``submit`` never blocks."""

    def __init__(self, endpoint: str, resource: dict, *, protocol: str = JSON,
                 headers: dict[str, str] | None = None, max_queue: int = 4096,
                 batch: int = 256, every: float = 2.0, timeout: float = 3.0):
        self.url = endpoint.rstrip("/") + ("" if endpoint.rstrip("/").endswith("/v1/traces")
                                           else "/v1/traces")
        self.resource, self.batch, self.every, self.timeout = resource, batch, every, timeout
        self.protocol = protocol
        self.headers = dict(headers or {})
        self.q: queue.Queue[Span] = queue.Queue(max_queue)
        self.dropped = 0
        self.sent = 0
        self._failing = False
        self._stop = threading.Event()
        threading.Thread(target=self._loop, daemon=True, name="otlp").start()

    def submit(self, span: Span) -> None:
        try:
            self.q.put_nowait(span)
        except queue.Full:
            self.dropped += 1

    def _loop(self) -> None:
        while not self._stop.is_set():
            spans = self._drain(wait=self.every)
            if spans:
                self._post(spans)

    def _drain(self, wait: float) -> list[Span]:
        out: list[Span] = []
        try:
            out.append(self.q.get(timeout=wait))
        except queue.Empty:
            return out
        while len(out) < self.batch:
            try:
                out.append(self.q.get_nowait())
            except queue.Empty:
                break
        return out

    def _post(self, spans: list[Span]) -> None:
        if self.protocol == PROTOBUF:
            body, ctype = to_otlp_proto(spans, self.resource), "application/x-protobuf"
        else:
            body, ctype = json.dumps(to_otlp(spans, self.resource)).encode(), "application/json"
        req = urllib.request.Request(self.url, data=body, method="POST",
                                     headers={**self.headers, "Content-Type": ctype})
        try:
            with urllib.request.urlopen(req, timeout=self.timeout):
                pass
            self.sent += len(spans)
            if self._failing:
                self._failing = False
                logs.emit(f"otlp: {self.url} accepts spans again", event="otlp")
        except (urllib.error.URLError, OSError, ValueError) as e:
            self.dropped += len(spans)
            if not self._failing:          # say it once, not every batch
                self._failing = True
                logs.emit(f"otlp: dropping spans, {self.url} failed: {e}", event="otlp",
                          level="warn")

    def flush(self) -> None:
        while spans := self._drain(wait=0):
            self._post(spans)

    def close(self) -> None:
        self._stop.set()
        self.flush()

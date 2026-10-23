"""Prometheus metrics for the gateway (text exposition format 0.0.4).

No client library: counters and histograms live in this process, behind
one lock, and :func:`Registry.render` writes them out on each scrape.
Gauges that describe state elsewhere (in-flight calls aside) are read at
scrape time through collector callbacks.

Labels are bounded by the running models and the cluster's nodes.
"""
from __future__ import annotations

import math
import re
import threading
from collections.abc import Callable, Iterable

# Time to first token: fine enough for a p95 from ~50 ms to ~30 s.
TTFT_BUCKETS = (0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 1, 1.5, 2, 3, 5, 7.5, 10, 15, 20, 30, 60)
DURATION_BUCKETS = (0.1, 0.25, 0.5, 1, 2, 5, 10, 20, 30, 60, 120, 300, 600)

Labels = tuple[tuple[str, str], ...]
Sample = tuple[dict[str, str], float]


def _esc(v: str) -> str:
    return str(v).replace("\\", r"\\").replace("\n", r"\n").replace('"', r'\"')


def fmt_labels(labels: Iterable[tuple[str, str]]) -> str:
    inner = ",".join(f'{k}="{_esc(v)}"' for k, v in labels)
    return "{" + inner + "}" if inner else ""


def fmt_value(v: float) -> str:
    if math.isinf(v):
        return "+Inf" if v > 0 else "-Inf"
    if v == int(v) and abs(v) < 1e15:
        return str(int(v))
    return repr(float(v))


class _Metric:
    kind = ""

    def __init__(self, name: str, help_: str, labelnames: tuple[str, ...]):
        self.name, self.help, self.labelnames = name, help_, labelnames

    def key(self, labels: dict[str, str]) -> Labels:
        return tuple((n, str(labels.get(n, ""))) for n in self.labelnames)

    def header(self) -> list[str]:
        return [f"# HELP {self.name} {self.help}", f"# TYPE {self.name} {self.kind}"]


class Counter(_Metric):
    kind = "counter"

    def __init__(self, *a, lock: threading.Lock):
        super().__init__(*a)
        self._lock, self._values = lock, {}

    def inc(self, n: float = 1, **labels: str) -> None:
        k = self.key(labels)
        with self._lock:
            self._values[k] = self._values.get(k, 0.0) + n

    def lines(self) -> list[str]:
        return [f"{self.name}{fmt_labels(k)} {fmt_value(v)}"
                for k, v in sorted(self._values.items())]


class Gauge(_Metric):
    """Set with ``add`` (in-flight counts), or read from ``collect`` at
    scrape time."""
    kind = "gauge"

    def __init__(self, *a, lock: threading.Lock,
                 collect: Callable[[], Iterable[Sample]] | None = None):
        super().__init__(*a)
        self._lock, self._values, self.collect = lock, {}, collect

    def add(self, n: float, **labels: str) -> None:
        k = self.key(labels)
        with self._lock:
            self._values[k] = self._values.get(k, 0.0) + n

    def lines(self) -> list[str]:
        values = dict(self._values)
        if self.collect is not None:
            for labels, v in self.collect():
                values[self.key(labels)] = v
        return [f"{self.name}{fmt_labels(k)} {fmt_value(v)}" for k, v in sorted(values.items())]


class Histogram(_Metric):
    kind = "histogram"

    def __init__(self, *a, lock: threading.Lock, buckets: tuple[float, ...]):
        super().__init__(*a)
        self._lock, self.buckets = lock, tuple(sorted(buckets))
        self._values: dict[Labels, list] = {}      # key -> [counts…, sum, count]

    def observe(self, v: float, **labels: str) -> None:
        k = self.key(labels)
        with self._lock:
            row = self._values.setdefault(k, [0] * len(self.buckets) + [0.0, 0])
            for i, b in enumerate(self.buckets):
                if v <= b:
                    row[i] += 1
            row[-2] += v
            row[-1] += 1

    def lines(self) -> list[str]:
        out = []
        for k, row in sorted(self._values.items()):
            for b, n in zip(self.buckets, row):
                out.append(f"{self.name}_bucket{fmt_labels(k + (('le', fmt_value(b)),))} {n}")
            out.append(f"{self.name}_bucket{fmt_labels(k + (('le', '+Inf'),))} {row[-1]}")
            out.append(f"{self.name}_sum{fmt_labels(k)} {fmt_value(row[-2])}")
            out.append(f"{self.name}_count{fmt_labels(k)} {row[-1]}")
        return out


class Registry:
    def __init__(self):
        self._lock = threading.Lock()
        self._metrics: list[_Metric] = []

    def counter(self, name, help_, labelnames=()) -> Counter:
        return self._add(Counter(name, help_, tuple(labelnames), lock=self._lock))

    def gauge(self, name, help_, labelnames=(), collect=None) -> Gauge:
        return self._add(Gauge(name, help_, tuple(labelnames), lock=self._lock, collect=collect))

    def histogram(self, name, help_, labelnames=(), buckets=DURATION_BUCKETS) -> Histogram:
        return self._add(Histogram(name, help_, tuple(labelnames), lock=self._lock,
                                   buckets=buckets))

    def _add(self, m):
        self._metrics.append(m)
        return m

    def render(self) -> str:
        out: list[str] = []
        for m in self._metrics:
            try:
                if m.kind == "gauge":
                    lines = m.lines()            # collectors run outside the lock
                else:
                    with self._lock:
                        lines = m.lines()
            except Exception as e:  # a broken collector mustn't break the scrape
                out.append(f"# {m.name}: collect failed: {type(e).__name__}")
                continue
            out += m.header() + lines
        return "\n".join(out) + "\n"


# ------------------------------------------------- re-exported backends --

_SAMPLE = re.compile(r"^([a-zA-Z_:][a-zA-Z0-9_:]*)(\{(.*?)\})?(\s.*)$")
_LABEL_NAME = re.compile(r'([a-zA-Z_][a-zA-Z0-9_]*)\s*=\s*"')


def add_labels(text: str, labels: dict[str, str]) -> str:
    """Add ``labels`` to every sample of a Prometheus text exposition.
    A label the sample already has keeps its own value. Comments and blank
    lines pass through."""
    out = []
    for line in text.splitlines():
        if not line or line.startswith("#"):
            out.append(line)
            continue
        m = _SAMPLE.match(line)
        if m is None:
            continue                       # not a sample; drop rather than corrupt
        name, _, inner, rest = m.groups()
        have = set(_LABEL_NAME.findall(inner or ""))
        extra = ",".join(f'{k}="{_esc(v)}"' for k, v in labels.items() if k not in have)
        parts = [p for p in ((inner or "").strip().rstrip(","), extra) if p]
        out.append(f"{name}{{{','.join(parts)}}}{rest}" if parts else line)
    return "\n".join(out) + "\n"

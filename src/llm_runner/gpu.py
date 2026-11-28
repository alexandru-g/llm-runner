"""This machine's GPU memory, for `ps --json` and `run`'s fit check.

Discrete GPUs: nvidia-smi's totals and free memory, and each process's use.
A Jetson's GPU shares the system's memory, and its nvidia-smi reports
neither, so there the GPU is the machine's memory: total ``MemTotal``, free
``MemAvailable`` (page cache gives way to CUDA allocations), with no use per
process. Everything here is best effort: ``None`` when it can't tell.
"""
from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass
from pathlib import Path

_TIMEOUT = 3


@dataclass(frozen=True)
class GPU:
    index: int
    name: str
    total_mib: int | None
    free_mib: int | None
    unified: bool = False           # shares the system's memory (Jetson)

    def as_dict(self) -> dict:
        return {"index": self.index, "name": self.name, "total_mib": self.total_mib,
                "free_mib": self.free_mib, "unified": self.unified}


def is_jetson() -> bool:
    return Path("/etc/nv_tegra_release").exists()


def _smi(*query: str) -> list[list[str]] | None:
    try:
        out = subprocess.run(["nvidia-smi", *query, "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=_TIMEOUT)
    except (OSError, subprocess.TimeoutExpired):
        return None
    if out.returncode != 0:
        return None
    return [[c.strip() for c in line.split(",")] for line in out.stdout.splitlines() if line]


def _int(s: str) -> int | None:
    try:
        return int(float(s))
    except ValueError:
        return None


def _meminfo() -> dict[str, int]:
    out = {}
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            key, _, rest = line.partition(":")
            out[key] = int(rest.split()[0]) // 1024        # kB -> MiB
    except (OSError, ValueError, IndexError):
        pass
    return out


def gpus() -> list[GPU]:
    return read_gpus()


def read_gpus() -> list[GPU]:
    if is_jetson():
        m = _meminfo()
        try:
            name = Path("/proc/device-tree/model").read_text().strip("\x00 \n") or "Jetson"
        except OSError:
            name = "Jetson"
        return [GPU(0, name, m.get("MemTotal"), m.get("MemAvailable"), unified=True)]
    rows = _smi("--query-gpu=index,name,memory.total,memory.free")
    return [GPU(_int(r[0]) or 0, r[1], _int(r[2]), _int(r[3])) for r in rows or [] if len(r) >= 4]


def visible(all_gpus: list[GPU], spec: str | None) -> list[GPU]:
    """The GPUs a launch's ``--gpu`` spec ("all", "0", "0,1") gives it."""
    if not spec or spec == "all":
        return all_gpus
    want = {_int(s) for s in spec.split(",")}
    return [g for g in all_gpus if g.index in want]


def process_mib() -> dict[int, int] | None:
    """{pid: MiB of GPU memory} from nvidia-smi; None where it can't say (a
    Jetson, no nvidia-smi)."""
    if is_jetson():
        return None
    rows = _smi("--query-compute-apps=pid,used_memory")
    if rows is None:
        return None
    out: dict[int, int] = {}
    for r in rows:
        if len(r) >= 2 and _int(r[0]) is not None and _int(r[1]) is not None:
            out[_int(r[0])] = out.get(_int(r[0]), 0) + _int(r[1])
    return out


def _children() -> dict[int, list[int]]:
    kids: dict[int, list[int]] = {}
    for d in os.listdir("/proc"):
        if not d.isdigit():
            continue
        try:
            stat = Path(f"/proc/{d}/stat").read_text()
            ppid = int(stat.rsplit(")", 1)[1].split()[1])
        except (OSError, ValueError, IndexError):
            continue
        kids.setdefault(ppid, []).append(int(d))
    return kids


def tree_mib(pids: list[int], per_pid: dict[int, int] | None) -> dict[int, int | None]:
    """GPU MiB of each pid with its descendants (vLLM's workers are children)."""
    if per_pid is None:
        return {p: None for p in pids}
    kids = _children()
    out: dict[int, int | None] = {}
    for root in pids:
        total, stack, seen = 0, [root], set()
        while stack:
            p = stack.pop()
            if p in seen:
                continue
            seen.add(p)
            total += per_pid.get(p, 0)
            stack.extend(kids.get(p, ()))
        out[root] = total
    return out

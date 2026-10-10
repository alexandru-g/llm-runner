"""Several machines, one model list: the registry, node views and routing.

Every machine runs an llmctl gateway (a *node*, with a name and a priority).
One gateway, usually on an always-on box, also acts as the *registry*. Each
node sends it a heartbeat every few seconds with its running models; the
reply is the rest of the cluster. A node whose heartbeats stop is dropped, so
the heartbeat is the health check.

A gateway can then route a request to any instance of a model in the
cluster: ready instances first, then the lowest priority number, then the
local copy. ``model@node`` pins a node. Requests between gateways carry the
cluster token and a hop header, and a hop is only ever routed locally.

This module holds the logic (no sockets); ``gateway.py`` wires it to HTTP.
"""
from __future__ import annotations

import hmac
import re
import threading
import time
from dataclasses import dataclass, field

PROTOCOL = 1
HEARTBEAT_EVERY = 5.0        # s between heartbeats
NODE_TIMEOUT = 15.0          # the registry drops a node silent this long
VIEW_TTL = 120.0             # a node forgets remote models after this long without the registry

TOKEN_HEADER = "X-Llmctl-Token"
HOP_HEADER = "X-Llmctl-Hop"
# On a hop: "the caller has other instances to try; if all your slots are
# busy, say so (503 + BUSY_HEADER) instead of queueing the request".
SPILL_HEADER = "X-Llmctl-Spill"
BUSY_HEADER = "X-Llmctl-Busy"

OPENAI = "openai"
SYSTEMONE = "systemone"

_STATUS_RANK = {"ready": 0, "loading": 1}


class Unroutable(Exception):
    def __init__(self, status: int, code: str, message: str):
        super().__init__(message)
        self.status, self.code, self.message = status, code, message


def token_ok(expected: str | None, given: str | None) -> bool:
    return bool(expected) and given is not None and hmac.compare_digest(expected, given)


# ------------------------------------------------------------------ routing --

@dataclass(frozen=True)
class Instance:
    """One running copy of a model, on this machine or another node."""
    id: str
    node: str
    priority: int
    local: bool
    info: dict                      # the `llmctl` block of /v1/models
    url: str | None = None          # the remote node's gateway; None when local
    created: int = 0

    @property
    def api(self) -> str:
        return self.info.get("api", OPENAI)

    @property
    def status(self) -> str:
        return self.info.get("status", "unknown")

    @property
    def group(self) -> str:
        return self.info.get("group") or ""


def order(cands: list[Instance]) -> list[Instance]:
    """Best first: ready, then lowest priority number, then local, then name."""
    return sorted(cands, key=lambda i: (_STATUS_RANK.get(i.status, 2), i.priority,
                                        not i.local, i.node))


def split_pin(requested: str) -> tuple[str, str | None]:
    name, at, node = requested.rpartition("@")
    return (name, node) if at and name and node else (requested, None)


def pick(instances: list[Instance], requested: object, api: str,
         needs_vision: bool = False) -> list[Instance]:
    """The instances a request may go to, best first. Raises :class:`Unroutable`.

    ``requested`` is a model id, a group (any of its members will do), a
    unique prefix of either, or empty when only one model serves ``api``;
    ``name@node`` keeps it to one node."""
    pool = [i for i in instances if i.api == api]
    kind = "decision endpoint" if api == SYSTEMONE else "model"
    ids = sorted({i.id for i in pool})
    groups = {i.group: [] for i in pool if i.group and i.group not in ids}
    for i in pool:
        if i.group in groups:
            groups[i.group].append(i.id)
    listing = ", ".join(ids) or "(none running)"
    if requested is not None and not isinstance(requested, str):
        raise Unroutable(400, "invalid_model", "\"model\" must be a string.")
    name, node = split_pin(requested) if requested else ("", None)

    if not name:
        if len(ids) == 1:
            chosen = ids[0]
        elif not ids:
            raise Unroutable(503, "no_models", f"No {kind} is running in the cluster.")
        else:
            raise Unroutable(400, "model_required",
                             f"Several {kind}s are running; set \"model\" to one of: {listing}")
    elif name in ids or name in groups:
        chosen = name
    else:
        prefixed = [i for i in ids if i.startswith(name)]
        owners = {i.group for i in pool if i.id in prefixed}
        named = [g for g in groups if g.startswith(name)]
        if len(prefixed) == 1:
            chosen = prefixed[0]
        elif prefixed and len(owners) == 1 and "" not in owners:
            chosen = owners.pop()               # several quants of one model
        elif not prefixed and len(named) == 1:
            chosen = named[0]
        elif prefixed or named:
            raise Unroutable(400, "model_ambiguous",
                             f"\"{name}\" matches several {kind}s: "
                             + ", ".join(sorted(set(prefixed) | set(named))))
        else:
            raise Unroutable(404, "model_not_found",
                             f"No running {kind} named \"{name}\". Running: {listing}")

    cands = [i for i in pool if chosen in (i.id, i.group)]
    if node is not None:
        where = ", ".join(sorted({i.node for i in cands}))
        cands = [i for i in cands if i.node == node]
        if not cands:
            raise Unroutable(404, "model_not_found",
                             f"{chosen} is not running on node \"{node}\" (it runs on: {where}).")
    if needs_vision:
        seeing = [i for i in cands if i.info.get("vision")]
        if not seeing:
            raise Unroutable(400, "vision_unavailable",
                             f"The request has images, but no instance of {chosen} reads them "
                             f"(running on: {', '.join(sorted({i.node for i in cands}))}; "
                             "start one with its mmproj).")
        cands = seeing
    return order(cands)


_IMAGE_PART_TYPES = {"image_url", "input_image", "image"}


def has_images(payload: object, depth: int = 0) -> bool:
    """Whether a JSON request carries images: OpenAI content parts, the
    SystemOne ``images`` field, or llama-server's ``multimodal_data``."""
    if depth > 8:
        return False
    if isinstance(payload, dict):
        if payload.get("type") in _IMAGE_PART_TYPES:
            return True
        for key in ("images", "multimodal_data"):
            if payload.get(key):
                return True
        return any(has_images(v, depth + 1) for v in payload.values()
                   if isinstance(v, (dict, list)))
    if isinstance(payload, list):
        return any(has_images(v, depth + 1) for v in payload if isinstance(v, (dict, list)))
    return False


_CONTEXT_ERROR = re.compile(
    rb"exceed[^\"]{0,40}context|context (size|length|window)|maximum context|input_too_long", re.I)


def is_context_error(status: int, body: bytes) -> bool:
    """An instance refused the prompt as too long (another may have more context)."""
    return status in (400, 413, 422) and bool(_CONTEXT_ERROR.search(body))


# ----------------------------------------------------------------- registry --

@dataclass
class NodeRecord:
    name: str
    url: str
    priority: int
    version: str
    models: list[dict]
    last_seen: float = field(default_factory=time.monotonic)

    def public(self, now: float) -> dict:
        return {"node": self.name, "url": self.url, "priority": self.priority,
                "version": self.version, "age": round(now - self.last_seen, 1),
                "models": self.models}


class RegistryTable:
    """The registry's view of the cluster: one record per node, in memory.
    Nodes repopulate it within one heartbeat after a restart."""

    def __init__(self, version: str):
        self.version = version
        self._nodes: dict[str, NodeRecord] = {}
        self._lock = threading.Lock()

    def _expire(self, now: float) -> None:
        for name in [n for n, r in self._nodes.items() if now - r.last_seen > NODE_TIMEOUT]:
            del self._nodes[name]

    def heartbeat(self, payload: dict, peer_ip: str) -> tuple[int, dict]:
        if payload.get("protocol") != PROTOCOL:
            return 409, {"error": {"code": "protocol_mismatch", "message":
                         f"registry speaks cluster protocol {PROTOCOL} (llmctl {self.version}), "
                         f"node sent {payload.get('protocol')}; redeploy llmctl on both."}}
        try:
            name = str(payload["node"])
            port = int(payload["port"])
            priority = int(payload.get("priority", 0))
            models = list(payload.get("models", []))
        except (KeyError, TypeError, ValueError):
            return 400, {"error": {"code": "invalid_heartbeat",
                                   "message": "node, port, priority and models are required"}}
        url = payload.get("advertise") or f"http://{peer_ip}:{port}"
        now = time.monotonic()
        with self._lock:
            self._expire(now)
            old = self._nodes.get(name)
            if old is not None and old.url != url:
                return 409, {"error": {"code": "node_name_taken", "message":
                             f"node name \"{name}\" is in use by {old.url}; "
                             "give this machine another name (`llmctl node set --name`)."}}
            node_version = str(payload.get("version", "?"))
            self._nodes[name] = NodeRecord(name, url, priority, node_version, models, now)
            others = [r.public(now) for r in self._nodes.values() if r.name != name]
            versions = {r.name: r.version for r in self._nodes.values()}
        warnings = []
        if len(set(versions.values()) | {self.version}) > 1:
            warnings.append("llmctl versions differ: registry " + self.version + ", "
                            + ", ".join(f"{n} {v}" for n, v in sorted(versions.items())))
        return 200, {"protocol": PROTOCOL, "registry_version": self.version,
                     "node_url": url, "nodes": others, "warnings": warnings}

    def goodbye(self, name: str) -> None:
        with self._lock:
            self._nodes.pop(name, None)

    def snapshot(self) -> list[dict]:
        now = time.monotonic()
        with self._lock:
            self._expire(now)
            return [r.public(now) for r in sorted(self._nodes.values(),
                                                  key=lambda r: (r.priority, r.name))]


# ---------------------------------------------------------------- node view --

class ClusterView:
    """What a node knows about the other nodes, from its last heartbeat."""

    def __init__(self) -> None:
        self._nodes: list[dict] = []
        self._ok_at: float | None = None
        self.error: str | None = None
        self.warnings: list[str] = []
        self.node_url: str | None = None
        self._lock = threading.Lock()

    def update(self, reply: dict) -> None:
        with self._lock:
            self._nodes = list(reply.get("nodes", []))
            self._ok_at = time.monotonic()
            self.error = None
            self.warnings = list(reply.get("warnings", []))
            self.node_url = reply.get("node_url")

    def fail(self, error: str) -> None:
        with self._lock:
            self.error = error

    @property
    def age(self) -> float | None:
        return None if self._ok_at is None else time.monotonic() - self._ok_at

    def nodes(self) -> list[dict]:
        with self._lock:
            if self._ok_at is None or time.monotonic() - self._ok_at > VIEW_TTL:
                return []
            return list(self._nodes)

    def instances(self) -> list[Instance]:
        out = []
        for n in self.nodes():
            for m in n.get("models", []):
                info = dict(m.get("llmctl", {}))
                out.append(Instance(id=m["id"], node=n["node"], priority=int(n["priority"]),
                                    local=False, info=info, url=n["url"],
                                    created=int(m.get("created", 0))))
        return out

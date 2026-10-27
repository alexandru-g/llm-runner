"""API keys on the gateway, required of callers from outside the local
networks. Loopback is always local (the CLI calls its own gateway), and so
is the bind address when it is a single one (``llmctl config bind
172.19.0.1``): the CLI's calls then come from it, and the kernel drops packets
from outside that claim it. To want keys from the LAN too, narrow the local
networks to loopback.

Who the caller is: the peer address, except that a request from a reverse
proxy is judged by the address the proxy forwarded (the last entry of
``X-Forwarded-For``, else ``X-Real-IP``). A proxy is any peer listed with
``llmctl gateway auth --proxy``, or any peer on a local network that sends
the header. A listed proxy that forwards no address counts as outside, so a
misconfigured proxy fails closed. A peer outside the local networks is
judged by its own address, whatever headers it sends.

Keys are stored as SHA-256 hashes in the config file and checked on every
request; the file is re-read when it changes, so `llmctl gateway key add`
and `rm` need no restart.
"""
from __future__ import annotations

import hashlib
import hmac
import ipaddress
import secrets
import threading
from dataclasses import dataclass, field

from llm_runner import config as config_mod

OFF, ON = "off", "on"

# Loopback, private and link-local ranges, plus 100.64.0.0/10 (Tailscale).
DEFAULT_LOCAL = ("127.0.0.0/8", "10.0.0.0/8", "172.16.0.0/12", "192.168.0.0/16",
                 "100.64.0.0/10", "169.254.0.0/16", "::1/128", "fc00::/7", "fe80::/10")

Network = ipaddress.IPv4Network | ipaddress.IPv6Network


def parse_network(text: str) -> Network:
    """A network or a single address (``172.20.0.45`` is a /32)."""
    return ipaddress.ip_network(text.strip(), strict=False)


def _ip(text: str | None) -> ipaddress.IPv4Address | ipaddress.IPv6Address | None:
    if not text:
        return None
    text = text.strip()
    if text.startswith("[") and "]" in text:          # [v6]:port
        text = text[1:text.index("]")]
    elif text.count(":") == 1:                         # v4:port
        text = text.split(":")[0]
    try:
        ip = ipaddress.ip_address(text.split("%")[0])
    except ValueError:
        return None
    return ip.ipv4_mapped or ip if isinstance(ip, ipaddress.IPv6Address) else ip


def hash_key(key: str) -> str:
    return hashlib.sha256(key.encode()).hexdigest()


def new_key() -> str:
    return "llmctl-" + secrets.token_urlsafe(32)


@dataclass(frozen=True)
class Policy:
    mode: str = OFF
    keys: dict[str, str] = field(default_factory=dict)        # name -> sha256
    local: tuple[Network, ...] = tuple(parse_network(n) for n in DEFAULT_LOCAL)
    proxies: tuple[Network, ...] = ()
    # sha256 of the scrape-only token for /metrics*, or None: then scrapes
    # need an API key like any other call.
    metrics_token: str | None = None
    # The configured bind address when it is one address: this machine's own.
    own: tuple = ()

    def caller(self, peer: str, headers) -> ipaddress.IPv4Address | ipaddress.IPv6Address | None:
        """The address a request is judged by; None means "unknown", i.e.
        outside."""
        ip = _ip(peer)
        if ip is None:
            return None
        forwarded = (headers.get("X-Forwarded-For") or "").split(",")[-1].strip() \
            or (headers.get("X-Real-IP") or "").strip()
        if any(ip in n for n in self.proxies):
            return _ip(forwarded)
        if forwarded and self.is_local(ip):
            return _ip(forwarded)
        return ip

    def via_proxy(self, peer: str, headers) -> bool:
        """The request came through a reverse proxy (the same test as
        ``caller``), so the peer address is the proxy's."""
        ip = _ip(peer)
        if ip is None:
            return False
        if any(ip in n for n in self.proxies):
            return True
        forwarded = headers.get("X-Forwarded-For") or headers.get("X-Real-IP")
        return bool(forwarded) and self.is_local(ip)

    def is_local(self, ip) -> bool:
        return ip is not None and (ip.is_loopback or ip in self.own or any(
            ip.version == n.version and ip in n for n in self.local))

    def needs_key(self, peer: str, headers) -> bool:
        return self.mode == ON and not self.is_local(self.caller(peer, headers))

    def is_metrics_token(self, headers) -> bool:
        key = _sent_key(headers)
        return bool(key and self.metrics_token
                    and hmac.compare_digest(hash_key(key), self.metrics_token))

    def key_name(self, headers) -> str | None:
        """The name of the key a request carries, or None."""
        key = _sent_key(headers)
        if not key:
            return None
        digest = hash_key(key)
        found = None
        for name, stored in self.keys.items():
            if hmac.compare_digest(digest, stored):
                found = name
        return found


def _sent_key(headers) -> str:
    auth = headers.get("Authorization") or ""
    return auth[7:].strip() if auth[:7].lower() == "bearer " else \
        (headers.get("X-Api-Key") or "").strip()


def policy_from(cfg: dict) -> Policy:
    mode = ON if cfg.get("gateway_auth") == ON else OFF
    keys = cfg.get("gateway_keys") if isinstance(cfg.get("gateway_keys"), dict) else {}
    mtok = cfg.get("gateway_metrics_token") if isinstance(
        cfg.get("gateway_metrics_token"), str) else None
    try:
        local = tuple(parse_network(n) for n in cfg.get("gateway_local_networks")
                      or DEFAULT_LOCAL)
        proxies = tuple(parse_network(n) for n in cfg.get("gateway_proxies") or ())
    except (ValueError, TypeError) as e:
        # Hand-edited into something unreadable (the CLI checks what it saves):
        # keep the mode and keys, use the defaults, and say so.
        print(f"auth: ignoring the networks in the config ({e}); using the defaults",
              flush=True)
        return Policy(mode=mode, keys=dict(keys), metrics_token=mtok, own=_own(cfg))
    return Policy(mode=mode, keys=dict(keys), local=local, proxies=proxies, metrics_token=mtok,
                  own=_own(cfg))


def _own(cfg: dict) -> tuple:
    """The bind address, when it is one address rather than every interface."""
    ip = _ip(cfg.get("bind_host") if isinstance(cfg.get("bind_host"), str) else None)
    return () if ip is None or ip.is_unspecified else (ip,)


class ConfigPolicy:
    """The policy in the config file, re-read when the file changes."""

    def __init__(self):
        self._lock = threading.Lock()
        self._stamp: tuple | None = None
        self._policy = Policy()

    def __call__(self) -> Policy:
        try:
            st = config_mod.CONFIG_FILE.stat()
            stamp = (st.st_mtime_ns, st.st_size)
        except OSError:
            stamp = None
        with self._lock:
            if stamp != self._stamp:
                self._policy = policy_from(config_mod.load())
                self._stamp = stamp
            return self._policy

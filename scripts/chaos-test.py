#!/usr/bin/env python3
"""Kill a served model mid-stream and time what clients see.

    scripts/chaos-test.py MODEL [--gateway http://127.0.0.1:9000] [--key KEY]

1. Opens a streamed chat completion on MODEL through the gateway.
2. Once tokens flow, kills the engine with SIGKILL (`docker kill` for a
   container model).
3. Measures how long until the open stream ends with an error event, and
   until a new call gets an explicit error (503 `model_unavailable`).
4. Polls until a call succeeds again (the supervisor restarted it).

Passes when the error arrives within 10 s and the model serves again
within 120 s (override with --error-within / --recover-within). Prints one
JSON line with the timings. Run it on the machine that serves MODEL.
"""
from __future__ import annotations

import argparse
import http.client
import json
import os
import signal
import subprocess
import sys
import threading
import time
from urllib.parse import urlsplit


def ps(model: str) -> dict:
    out = subprocess.run(["llmctl", "ps", "--json"], check=True, capture_output=True,
                         text=True).stdout
    for m in json.loads(out)["models"]:
        if m["name"] == model and m["pid"]:
            return m
    sys.exit(f"{model} is not running here (llmctl ps)")


def conn(gw: str) -> http.client.HTTPConnection:
    u = urlsplit(gw)
    cls = http.client.HTTPSConnection if u.scheme == "https" else http.client.HTTPConnection
    return cls(u.hostname, u.port or (443 if u.scheme == "https" else 80), timeout=180)


def chat(gw, key, model, stream, max_tokens=8):
    c = conn(gw)
    headers = {"Content-Type": "application/json"}
    if key:
        headers["Authorization"] = f"Bearer {key}"
    c.request("POST", "/v1/chat/completions", headers=headers, body=json.dumps({
        "model": model, "stream": stream, "max_tokens": max_tokens,
        "messages": [{"role": "user", "content": "Count from 1 to 1000, one per line."}]}))
    return c, c.getresponse()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("model")
    ap.add_argument("--gateway", default="http://127.0.0.1:9000")
    ap.add_argument("--key", default=os.environ.get("LLMCTL_KEY"))
    ap.add_argument("--error-within", type=float, default=10.0)
    ap.add_argument("--recover-within", type=float, default=120.0)
    a = ap.parse_args()

    target = ps(a.model)
    c, resp = chat(a.gateway, a.key, a.model, stream=True, max_tokens=2000)
    if resp.status != 200:
        sys.exit(f"the stream didn't start: HTTP {resp.status} {resp.read()[:300]!r}")
    first = resp.read1(4096)
    if not first:
        sys.exit("the stream sent nothing")
    stream_end: dict = {}

    def drain():
        tail = first
        try:
            while chunk := resp.read1(65536):
                tail = (tail + chunk)[-4096:]
        except (OSError, http.client.HTTPException) as e:
            stream_end["exception"] = repr(e)
        stream_end["t"] = time.monotonic()
        stream_end["tail"] = tail.decode(errors="replace")
    reader = threading.Thread(target=drain, daemon=True)
    reader.start()
    time.sleep(0.5)

    container = (target.get("runtime") or {}).get("container")
    t0 = time.monotonic()
    if container:
        subprocess.run(["docker", "kill", container], check=False, capture_output=True)
    else:
        try:
            os.killpg(os.getpgid(target["pid"]), signal.SIGKILL)
        except ProcessLookupError:
            pass
    reader.join(a.error_within + 5)
    stream_error = "model_unavailable" in stream_end.get("tail", "")
    stream_s = stream_end["t"] - t0 if "t" in stream_end else None

    call_status, call_s, recovered_s = None, None, None
    while time.monotonic() - t0 < a.recover_within + 60:
        try:
            c2, r2 = chat(a.gateway, a.key, a.model, stream=False)
            body = r2.read()
            c2.close()
        except OSError:
            time.sleep(0.5)
            continue
        if call_status is None and r2.status != 200:
            call_status, call_s = r2.status, time.monotonic() - t0
            try:
                call_code = json.loads(body)["error"]["code"]
            except (ValueError, KeyError, TypeError):
                call_code = None
        if r2.status == 200 and call_status is not None:
            recovered_s = time.monotonic() - t0
            break
        time.sleep(1)

    ok = (stream_error and stream_s is not None and stream_s <= a.error_within
          and call_status == 503 and call_s is not None and call_s <= a.error_within
          and recovered_s is not None and recovered_s <= a.recover_within)
    print(json.dumps({"ok": ok, "model": a.model, "killed": container or target["pid"],
                      "stream_error_event": stream_error,
                      "stream_ended_s": None if stream_s is None else round(stream_s, 2),
                      "first_error_status": call_status,
                      "first_error_code": locals().get("call_code"),
                      "first_error_s": None if call_s is None else round(call_s, 2),
                      "recovered_s": None if recovered_s is None else round(recovered_s, 1)}))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()

"""The HTTP layer alone, with a stub service."""
from __future__ import annotations

import http.client
import threading
import time
from http.server import ThreadingHTTPServer
from types import SimpleNamespace

from systemone_proxy.server import make_handler


def test_keep_alive_replies_are_not_held_by_nagle():
    # With Nagle on, each reply's body waits ~40 ms for the client's delayed ACK.
    service = SimpleNamespace(profile=SimpleNamespace(id="stub"))
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(service, {}))
    httpd.daemon_threads = True
    threading.Thread(target=httpd.serve_forever, args=(0.05,), daemon=True).start()
    try:
        conn = http.client.HTTPConnection("127.0.0.1", httpd.server_address[1], timeout=5)
        times = []
        for _ in range(6):
            t0 = time.perf_counter()
            conn.request("GET", "/health")
            conn.getresponse().read()
            times.append(time.perf_counter() - t0)
        conn.close()
    finally:
        httpd.shutdown()
    assert sorted(times[1:])[2] < 0.02, times

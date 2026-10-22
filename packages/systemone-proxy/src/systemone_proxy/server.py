"""Threaded stdlib HTTP server exposing the DecisionService."""
from __future__ import annotations

import json
import sys
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from systemone_proxy import images as imagelib
from systemone_proxy.errors import ApiError
from systemone_proxy.service import DecisionService

# room for a few 20 MB images as base64
MAX_BODY = 128 * 1024 * 1024


def make_handler(service: DecisionService, spec: dict):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"
        server_version = "systemone-proxy"
        # Headers and body are separate small writes; with Nagle on, the body
        # waits for the client's delayed ACK (~40 ms) on every keep-alive reply.
        disable_nagle_algorithm = True

        def _send(self, status: int, payload: dict) -> None:
            body = json.dumps(payload, ensure_ascii=False).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _error(self, err: ApiError) -> None:
            self._send(err.status, err.to_json())

        def do_GET(self) -> None:  # noqa: N802
            path = self.path.split("?", 1)[0]
            if path == "/health":
                self._send(200, {"status": "ok", "model": service.profile.id})
            elif path == "/v1/spec":
                self._send(200, spec)
            elif path == "/v1/models":
                self._send(200, {"object": "list", "data": [
                    {"id": service.profile.id, "object": "model", "owned_by": "systemone-proxy"}]})
            else:
                self._error(ApiError(404, "not_found", f"no route for GET {path}"))

        def do_POST(self) -> None:  # noqa: N802
            path = self.path.split("?", 1)[0]
            if path not in ("/v1/systemone", "/v1/decisions"):
                self._error(ApiError(404, "not_found", f"no route for POST {path}"))
                return
            try:
                length = int(self.headers.get("Content-Length") or 0)
            except ValueError:
                length = -1
            if length < 0:
                self._error(ApiError(400, "bad_request", "invalid Content-Length"))
                return
            if length > MAX_BODY:
                self._error(ApiError(413, "body_too_large", f"body over {MAX_BODY} bytes"))
                self.close_connection = True
                return
            raw = self.rfile.read(length)
            ctype = self.headers.get("Content-Type") or ""
            try:
                if ctype.lower().startswith("multipart/form-data"):
                    body = imagelib.parse_multipart(ctype, raw)
                else:
                    body = json.loads(raw)
            except ApiError as e:
                self._error(e)
                return
            except (json.JSONDecodeError, UnicodeDecodeError) as e:
                self._error(ApiError(400, "bad_request", f"body is not valid JSON: {e}"))
                return
            try:
                self._send(200, service.decide(body))
            except ApiError as e:
                self._error(e)
            except Exception as e:  # noqa: BLE001 — report, keep serving
                traceback.print_exc(file=sys.stderr)
                self._error(ApiError(500, "internal_error", f"{type(e).__name__}: {e}"))

        def log_message(self, fmt: str, *args) -> None:
            sys.stderr.write(f"{self.address_string()} {fmt % args}\n")

    return Handler


def serve(service: DecisionService, spec: dict, host: str, port: int) -> None:
    httpd = ThreadingHTTPServer((host, port), make_handler(service, spec))
    httpd.daemon_threads = True
    print(f"systemone-proxy: {service.profile.id} on http://{host}:{port}/v1/systemone "
          f"→ {service.backend.url}", flush=True)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        httpd.server_close()

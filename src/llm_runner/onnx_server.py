"""ONNX Runtime behind Triton's HTTP API: the ``onnx`` backend's server.

Run by llmctl with the Python of an ONNX Runtime venv (``~/.venvs/onnxruntime``),
so it imports nothing from llmctl; only ``onnxruntime`` and ``numpy``::

    python -I onnx_server.py --model model.onnx --name NAME --port 8100 \\
        [--host 0.0.0.0] [--config config.json] [--device auto|cuda|cpu]

It answers the KServe v2 calls a Triton client makes, so one client works
against this and the ``triton`` backend:

- ``GET /v2/health/live``, ``/v2/health/ready`` (and ``/health`` for llmctl)
- ``GET /v2/models/<m>``: metadata; ``/v2/models/<m>/ready``
- ``GET /v2/models/<m>/config``: Triton's JSON model config, from ``--config``
- ``POST /v2/models/<m>/infer``: tensors as JSON ``data``, or with the
  binary tensor extension (``Inference-Header-Content-Length``, raw
  little-endian bytes after the JSON header), both ways

``/versions/<v>`` in a model path is accepted and ignored. Requests run
concurrently on one session (ONNX Runtime's ``run`` is thread-safe); there
is no dynamic batching, which is Triton's job.
"""
from __future__ import annotations

import argparse
import json
import math
import re
import sys
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np
import onnxruntime as ort

_NP = {"BOOL": np.bool_, "UINT8": np.uint8, "UINT16": np.uint16, "UINT32": np.uint32,
       "UINT64": np.uint64, "INT8": np.int8, "INT16": np.int16, "INT32": np.int32,
       "INT64": np.int64, "FP16": np.float16, "FP32": np.float32, "FP64": np.float64}
# ONNX Runtime's type strings -> KServe v2 datatypes.
_ORT = {"tensor(bool)": "BOOL", "tensor(uint8)": "UINT8", "tensor(uint16)": "UINT16",
        "tensor(uint32)": "UINT32", "tensor(uint64)": "UINT64", "tensor(int8)": "INT8",
        "tensor(int16)": "INT16", "tensor(int32)": "INT32", "tensor(int64)": "INT64",
        "tensor(float16)": "FP16", "tensor(float)": "FP32", "tensor(double)": "FP64"}
_MODEL_PATH = re.compile(r"^/v2/models/([^/]+)(?:/versions/[^/]+)?(/[a-z]*)?$")


class BadRequest(Exception):
    pass


def _dims(shape) -> list[int]:
    return [d if isinstance(d, int) and d >= 0 else -1 for d in shape]


class Model:
    def __init__(self, path: str, name: str, device: str, config: dict | None,
                 threads: int | None):
        providers = ["CPUExecutionProvider"]
        if device in ("auto", "cuda") and "CUDAExecutionProvider" in ort.get_available_providers():
            providers.insert(0, "CUDAExecutionProvider")
        elif device == "cuda":
            raise SystemExit("--device cuda, but this onnxruntime has no CUDAExecutionProvider "
                             f"(it has {ort.get_available_providers()}); install onnxruntime-gpu")
        opts = ort.SessionOptions()
        if threads:
            opts.intra_op_num_threads = threads
        self.session = ort.InferenceSession(path, sess_options=opts, providers=providers)
        self.name = name
        self.providers = self.session.get_providers()
        self.inputs = {i.name: i for i in self.session.get_inputs()}
        self.outputs = [o.name for o in self.session.get_outputs()]
        self.config = config or {"name": name, "platform": "onnxruntime_onnx",
                                 "backend": "onnxruntime", "max_batch_size": 0}
        self.config["name"] = name

    def metadata(self) -> dict:
        def tensor(t) -> dict:
            return {"name": t.name, "datatype": _ORT.get(t.type, t.type),
                    "shape": _dims(t.shape)}
        return {"name": self.name, "versions": ["1"], "platform": "onnxruntime_onnx",
                "inputs": [tensor(t) for t in self.session.get_inputs()],
                "outputs": [tensor(t) for t in self.session.get_outputs()]}

    def infer(self, header: dict, blob: memoryview) -> tuple[dict, list[bytes]]:
        feeds, offset = {}, 0
        for inp in header.get("inputs") or []:
            name, dtype = inp.get("name"), inp.get("datatype")
            if name not in self.inputs:
                raise BadRequest(f"unexpected input {name!r}; the model takes "
                                 f"{sorted(self.inputs)}")
            if dtype not in _NP:
                raise BadRequest(f"input {name}: datatype {dtype!r} is not supported")
            shape = [int(d) for d in inp.get("shape") or []]
            size = (inp.get("parameters") or {}).get("binary_data_size")
            if size is not None:
                raw = bytes(blob[offset:offset + int(size)])
                offset += int(size)
                arr = np.frombuffer(raw, dtype=np.dtype(_NP[dtype]).newbyteorder("<"))
            else:
                arr = np.asarray(inp.get("data"), dtype=_NP[dtype])
            if arr.size != math.prod(shape):
                raise BadRequest(f"input {name}: {arr.size} values for shape {shape}")
            feeds[name] = arr.reshape(shape)
        missing = set(self.inputs) - set(feeds)
        if missing:
            raise BadRequest(f"missing input(s) {sorted(missing)}")
        binary_all = bool((header.get("parameters") or {}).get("binary_data_output"))
        wanted = header.get("outputs") or [{"name": n} for n in self.outputs]
        names = [o.get("name") for o in wanted]
        unknown = [n for n in names if n not in self.outputs]
        if unknown:
            raise BadRequest(f"unknown output(s) {unknown}; the model has {self.outputs}")
        results = self.session.run(names, feeds)
        outs, blobs = [], []
        for spec, arr in zip(wanted, results):
            arr = np.asarray(arr)
            dtype = next((k for k, v in _NP.items() if np.dtype(v) == arr.dtype), None)
            if dtype is None:
                raise BadRequest(f"output {spec['name']}: dtype {arr.dtype} has no v2 type")
            out = {"name": spec["name"], "datatype": dtype, "shape": list(arr.shape)}
            binary = (spec.get("parameters") or {}).get("binary_data", binary_all)
            if binary:
                data = np.ascontiguousarray(arr, dtype=arr.dtype.newbyteorder("<")).tobytes()
                out["parameters"] = {"binary_data_size": len(data)}
                blobs.append(data)
            else:
                out["data"] = arr.reshape(-1).tolist()
            outs.append(out)
        reply = {"model_name": self.name, "model_version": "1", "outputs": outs}
        if header.get("id") is not None:
            reply["id"] = header["id"]
        return reply, blobs


def make_handler(model: Model):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"
        # Small replies otherwise wait on Nagle (40 ms per request).
        disable_nagle_algorithm = True

        def log_message(self, fmt, *args):          # one line per error only
            pass

        def _send(self, status: int, body: bytes, ctype: str = "application/json",
                  extra: dict | None = None) -> None:
            self.send_response(status)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            for k, v in (extra or {}).items():
                self.send_header(k, v)
            self.end_headers()
            if self.command != "HEAD":
                self.wfile.write(body)

        def _json(self, status: int, obj) -> None:
            self._send(status, json.dumps(obj).encode())

        def _error(self, status: int, msg: str) -> None:
            self._json(status, {"error": msg})

        def _model(self, path: str) -> tuple[str | None, str]:
            m = _MODEL_PATH.match(path)
            if not m:
                return None, ""
            if m.group(1) != model.name:
                self._error(404, f"Request for unknown model: '{m.group(1)}' is not found")
                return "", ""
            return m.group(1), m.group(2) or ""

        def do_GET(self):
            path = self.path.split("?", 1)[0]
            if path in ("/health", "/v2/health/live", "/v2/health/ready"):
                return self._send(200, b"")
            if path == "/v2":
                return self._json(200, {"name": "llmctl-onnx", "version": ort.__version__,
                                        "extensions": ["binary_tensor_data", "model_configuration"]})
            name, rest = self._model(path)
            if name is None:
                return self._error(404, f"Not found: {path}")
            if not name:
                return
            if rest == "":
                return self._json(200, model.metadata())
            if rest == "/ready":
                return self._send(200, b"")
            if rest == "/config":
                return self._json(200, model.config)
            return self._error(404, f"Not found: {path}")

        do_HEAD = do_GET

        def do_POST(self):
            path = self.path.split("?", 1)[0]
            name, rest = self._model(path)
            if name is None or (name and rest != "/infer"):
                return self._error(404, f"Not found: {path}")
            if not name:
                return
            n = int(self.headers.get("Content-Length") or 0)
            body = memoryview(self.rfile.read(n))
            hlen = self.headers.get("Inference-Header-Content-Length")
            try:
                split = int(hlen) if hlen is not None else len(body)
                header = json.loads(bytes(body[:split]))
                if not isinstance(header, dict):
                    raise BadRequest("the request header is not a JSON object")
                reply, blobs = model.infer(header, body[split:])
            except (BadRequest, ValueError) as e:
                return self._error(400, str(e))
            except Exception as e:          # ONNX Runtime's own errors
                print(f"infer failed: {type(e).__name__}: {e}", file=sys.stderr, flush=True)
                return self._error(400, f"{type(e).__name__}: {e}")
            head = json.dumps(reply).encode()
            if not blobs:
                return self._send(200, head)
            self._send(200, head + b"".join(blobs), "application/octet-stream",
                       {"Inference-Header-Content-Length": str(len(head))})

    return Handler


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("--model", required=True, help="the .onnx file")
    ap.add_argument("--name", required=True, help="the name clients call it by")
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--config", help="Triton JSON model config to answer /config with")
    ap.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    ap.add_argument("--threads", type=int, default=None, help="intra-op threads (CPU)")
    args = ap.parse_args(argv)
    config = None
    if args.config:
        with open(args.config) as f:
            config = json.load(f)
    t0 = time.time()
    model = Model(args.model, args.name, args.device, config, args.threads)
    server = ThreadingHTTPServer((args.host, args.port), make_handler(model))
    server.daemon_threads = True
    print(f"onnx_server: {args.name} ({args.model}) on {args.host}:{args.port}, "
          f"providers {model.providers}, onnxruntime {ort.__version__}, "
          f"loaded in {time.time() - t0:.1f} s", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()

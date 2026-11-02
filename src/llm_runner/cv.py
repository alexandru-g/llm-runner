"""Computer-vision models: ONNX graphs served by the ``onnx`` and ``triton`` backends.

A CV model is a folder in Triton's model layout, so the same files serve both
backends and a Triton deployment elsewhere::

    <key>/
      1/model.onnx      the graph (``1/model.plan``: a prebuilt TensorRT engine)
      config.pbtxt      optional Triton model config; generated when missing
      spec.toml         optional detector spec (M0's ModelSpec: family, input
                        size and layout, output names, labels), read as is
      model.toml        optional llmctl fields: task, version, license, source,
                        and which ONNX file to serve when there are several

The version folder is always ``1/``: a release version lives in
``model.toml`` and the Triton config's ``model_version`` parameter, never in
the folder name. A folder holding a single ``*.onnx`` (at the top, in ``1/``
or in ``onnx/``) is a model too; with several, ``model.toml`` says which.

Both backends speak Triton's HTTP API (KServe v2 with the binary tensor
extension, plus ``/v2/models/<m>/config``), so one client works against
either. ``triton`` runs NVIDIA's Triton server in a container: TensorRT on a
GPU, with the engine built on the host and cached per ONNX file, TensorRT
version, GPU and build shapes; ONNX Runtime on a host without one. ``onnx``
runs ONNX Runtime natively (``onnx_server.py``), CUDA or CPU, without Docker.

Nothing here imports onnx or numpy: the ONNX file's inputs and outputs are
read straight from its protobuf, so ``llmctl list`` and config generation
work on any host.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
import time
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

VERSION_DIR = "1"
MODEL_TOML = "model.toml"
SPEC_TOML = "spec.toml"
CONFIG_PBTXT = "config.pbtxt"
BUILD_JSON = "build.json"
# A3's defaults (m0/services/detector/build-engine.sh): 16 frames per batch,
# frames of any size up to 1080p, RGB or RGBA.
DEFAULT_MAX_BATCH = 16
FRAMES_SHAPES = ("1x64x64x3", "{b}x720x1280x4", "{b}x1080x1920x4")
# Built strongly typed, as M0's build-engine.sh does: precision comes from the
# graph (FP16 from ModelOpt). TensorRT 11 does that anyway; TensorRT 10
# (Jetson images) otherwise builds an FP16 graph as FP32, 2.2x slower on orin.
STRONGLY_TYPED = "--stronglyTyped"
_PREFERRED_BATCHES = (4, 8, 16)
_QUEUE_DELAY_US = 2000


class CVError(Exception):
    """A CV model folder llmctl can't serve as it is."""


# ------------------------------------------------------------ ONNX graph I/O --

# onnx.TensorProto.DataType -> KServe v2 datatype.
_ONNX_TYPES = {1: "FP32", 2: "UINT8", 3: "INT8", 4: "UINT16", 5: "INT16", 6: "INT32",
               7: "INT64", 8: "BYTES", 9: "BOOL", 10: "FP16", 11: "FP64", 12: "UINT32",
               13: "UINT64", 16: "BF16"}


@dataclass(frozen=True)
class Tensor:
    name: str
    datatype: str            # KServe v2: FP32, UINT8, ...
    shape: tuple             # ints; -1 for a dynamic dimension

    @property
    def triton_type(self) -> str:
        return "TYPE_STRING" if self.datatype == "BYTES" else f"TYPE_{self.datatype}"


def _varint(buf, i: int) -> tuple[int, int]:
    shift = out = 0
    while True:
        b = buf[i]
        i += 1
        out |= (b & 0x7F) << shift
        if b < 0x80:
            return out, i
        shift += 7


def _fields(buf, start: int = 0, end: int | None = None):
    """``(field number, wire type, value)`` for each field of one protobuf
    message; a length-delimited value is a memoryview (no copy)."""
    end = len(buf) if end is None else end
    i = start
    while i < end:
        key, i = _varint(buf, i)
        num, wt = key >> 3, key & 7
        if wt == 0:
            val, i = _varint(buf, i)
        elif wt == 1:
            val, i = buf[i:i + 8], i + 8
        elif wt == 2:
            n, i = _varint(buf, i)
            val, i = buf[i:i + n], i + n
        elif wt == 5:
            val, i = buf[i:i + 4], i + 4
        else:
            raise CVError(f"unsupported protobuf wire type {wt}")
        yield num, wt, val


def _value_info(buf) -> Tensor:
    name, dtype, dims = "", 0, []
    for num, _, val in _fields(buf):
        if num == 1:                                  # ValueInfoProto.name
            name = bytes(val).decode()
        elif num == 2:                                # .type (TypeProto)
            for tnum, _, tval in _fields(val):
                if tnum != 1:                         # TypeProto.tensor_type
                    continue
                for snum, _, sval in _fields(tval):
                    if snum == 1:                     # Tensor.elem_type
                        dtype = sval
                    elif snum == 2:                   # Tensor.shape
                        for dnum, _, dval in _fields(sval):
                            if dnum != 1:             # TensorShapeProto.dim
                                continue
                            dim = -1
                            for vnum, _, v in _fields(dval):
                                if vnum == 1:         # Dimension.dim_value
                                    dim = v
                            dims.append(dim)
    return Tensor(name, _ONNX_TYPES.get(dtype, f"ONNX_{dtype}"), tuple(dims))


def onnx_io(path: Path) -> tuple[list[Tensor], list[Tensor]]:
    """The graph's inputs (initializers excluded, as older exporters list
    them among the inputs) and outputs, read from the file's protobuf."""
    try:
        data = memoryview(path.read_bytes())
    except OSError as e:
        raise CVError(f"can't read {path}: {e}") from None
    graph = next((val for num, wt, val in _fields(data) if num == 7 and wt == 2), None)
    if graph is None:
        raise CVError(f"{path} is not an ONNX model (no graph)")
    inits: set[str] = set()
    ins, outs = [], []
    for num, wt, val in _fields(graph):
        if wt != 2:
            continue
        if num == 5:                                  # GraphProto.initializer
            for tnum, twt, tval in _fields(val):
                if tnum == 8 and twt == 2:            # TensorProto.name
                    inits.add(bytes(tval).decode())
        elif num == 11:
            ins.append(_value_info(val))
        elif num == 12:
            outs.append(_value_info(val))
    return [t for t in ins if t.name not in inits], outs


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(1 << 20):
            h.update(chunk)
    return h.hexdigest()


# ------------------------------------------------------------ model folders --

def _onnx_candidates(folder: Path) -> list[Path]:
    """Graphs where llmctl looks: ``1/model.onnx`` first (Triton's layout),
    then any ``*.onnx`` in ``1/``, the folder itself or ``onnx/``."""
    first = folder / VERSION_DIR / "model.onnx"
    if first.is_file():
        return [first]
    found: list[Path] = []
    for sub in (folder / VERSION_DIR, folder, folder / "onnx"):
        if sub.is_dir():
            found += sorted(p for p in sub.glob("*.onnx") if p.is_file())
    return found


def _read_toml(path: Path) -> dict:
    try:
        return tomllib.loads(path.read_text())
    except FileNotFoundError:
        return {}
    except (OSError, tomllib.TOMLDecodeError) as e:
        raise CVError(f"{path}: {e}") from None


def is_cv_dir(path: Path) -> bool:
    """A folder llmctl serves as a CV model: ``model.toml``, ``spec.toml``
    or ``config.pbtxt``, or exactly one ONNX graph where llmctl looks.
    An HF checkpoint's ``onnx/`` exports don't count (the folder has its
    own weights), nor do several graphs nobody chose between."""
    if not path.is_dir():
        return False
    if any((path / f).is_file() for f in (MODEL_TOML, SPEC_TOML, CONFIG_PBTXT)):
        return True
    if (path / VERSION_DIR / "model.plan").is_file():
        return True
    if any(path.glob("*.safetensors")) or any(path.glob("*.gguf")):
        return False
    return len(_onnx_candidates(path)) == 1


@dataclass
class CVModel:
    """One CV model folder, resolved."""
    folder: Path
    onnx: Path | None
    plan: Path | None                     # a shipped engine (1/model.plan)
    meta: dict = field(default_factory=dict)      # model.toml
    spec: dict = field(default_factory=dict)      # spec.toml, untouched
    config: str | None = None                     # shipped config.pbtxt text

    @property
    def version(self) -> str:
        return str(self.meta.get("version") or "0.0.0")

    @property
    def task(self) -> str | None:
        return self.meta.get("task") or ("detection" if self.spec.get("family") else None)

    @property
    def license(self) -> str | None:
        return self.meta.get("license")

    def io(self) -> tuple[list[Tensor], list[Tensor]]:
        if self.onnx is None:
            raise CVError(f"{self.folder} has no ONNX graph")
        return onnx_io(self.onnx)

    def input_name(self, inputs: list[Tensor]) -> str:
        return self.spec.get("input_name") or inputs[0].name

    def max_batch(self, inputs: list[Tensor]) -> int:
        """Triton's ``max_batch_size``: 0 for a graph with a fixed batch
        (spec ``batch``, or a static first dimension), else model.toml's
        ``max_batch`` (16 by default)."""
        if self.spec.get("batch") or not inputs or not inputs[0].shape \
                or inputs[0].shape[0] != -1:
            return 0
        return int(self.meta.get("max_batch") or DEFAULT_MAX_BATCH)

    def detector_spec(self) -> str | None:
        """The spec as compact JSON (label keys as strings), the way M0's
        client reads it from the ``detector_spec`` config parameter."""
        if not self.spec:
            return None
        spec = dict(self.spec)
        if isinstance(spec.get("labels"), dict):
            spec["labels"] = {str(k): v for k, v in spec["labels"].items()}
        return json.dumps(spec, separators=(",", ":"))


def load(folder: Path) -> CVModel:
    """Resolve a CV model folder (or a bare ``.onnx`` file)."""
    if folder.is_file():
        if folder.suffix != ".onnx":
            raise CVError(f"{folder} is not an ONNX file")
        spec = _read_toml(folder.parent / SPEC_TOML)
        return CVModel(folder.parent, folder, None, {}, spec)
    meta = _read_toml(folder / MODEL_TOML)
    spec_file = folder / str(meta.get("spec") or SPEC_TOML)
    spec = _read_toml(spec_file)
    if meta.get("onnx"):
        onnx = folder / str(meta["onnx"])
        if not onnx.is_file():
            raise CVError(f"{folder / MODEL_TOML} names {meta['onnx']}, which isn't there")
    else:
        cands = _onnx_candidates(folder)
        if len(cands) > 1:
            raise CVError(
                f"{folder} has {len(cands)} ONNX graphs; say which one in {MODEL_TOML} "
                f"(onnx = \"{cands[0].relative_to(folder)}\").")
        onnx = cands[0] if cands else None
    plan = folder / VERSION_DIR / "model.plan"
    plan = plan if plan.is_file() else None
    if onnx is None and plan is None:
        raise CVError(f"{folder} has no ONNX graph or TensorRT engine")
    cfg = folder / CONFIG_PBTXT
    return CVModel(folder, onnx, plan, meta, spec,
                   cfg.read_text() if cfg.is_file() else None)


# ---------------------------------------------------------- Triton configs --

def _pbtxt_str(s: str) -> str:
    return '"' + s.replace("\\", "\\\\").replace('"', '\\"') + '"'


def _with_parameters(text: str, params: dict[str, str]) -> str:
    for key, value in params.items():
        if re.search(rf'key:\s*"{re.escape(key)}"', text):
            continue
        text = text.rstrip("\n") + (f"\nparameters {{ key: {_pbtxt_str(key)} value: "
                                    f"{{ string_value: {_pbtxt_str(value)} }} }}\n")
    return text


def triton_config(m: CVModel, name: str, *, engine: str, gpu: bool) -> str:
    """``config.pbtxt`` for serving ``m`` as ``name``.

    A shipped config is kept, renamed to ``name`` (Triton wants the folder's
    name), moved to ONNX Runtime on CPU when this host has no GPU, and given
    the ``detector_spec`` and ``model_version`` parameters M0's client reads
    when it lacks them. Otherwise one in M0's shape is written: dynamic
    batching up to ``max_batch``, one instance."""
    params = {k: v for k, v in (("detector_spec", m.detector_spec()),
                                ("model_version", m.version)) if v is not None}
    if m.config is not None:
        text = m.config
        if re.search(r'^\s*name:\s*".*"', text, re.M):
            text = re.sub(r'^(\s*name:\s*)".*"', lambda mt: mt.group(1) + _pbtxt_str(name),
                          text, count=1, flags=re.M)
        else:
            text = f"name: {_pbtxt_str(name)}\n" + text
        if engine == "onnxruntime":
            text = re.sub(r'^\s*platform:\s*"tensorrt_plan"', 'backend: "onnxruntime"',
                          text, flags=re.M)
        if not gpu:
            text = text.replace("KIND_GPU", "KIND_CPU")
        return _with_parameters(text, params)
    inputs, _ = m.io() if m.onnx else ([], [])
    mb = m.max_batch(inputs) if inputs else int(m.meta.get("max_batch") or 0)
    lines = [f"name: {_pbtxt_str(name)}",
             'platform: "tensorrt_plan"' if engine == "tensorrt" else 'backend: "onnxruntime"',
             f"max_batch_size: {mb}"]
    if mb:
        pref = [b for b in _PREFERRED_BATCHES if b <= mb]
        pref_txt = f"preferred_batch_size: [{', '.join(map(str, pref))}] " if pref else ""
        lines.append(f"dynamic_batching {{ {pref_txt}max_queue_delay_microseconds: "
                     f"{_QUEUE_DELAY_US} }}")
    lines.append(f"instance_group [{{ kind: {'KIND_GPU' if gpu else 'KIND_CPU'}, count: 1 }}]")
    return _with_parameters("\n".join(lines) + "\n", params)


def server_config(m: CVModel, name: str, *, gpu: bool) -> dict:
    """The model config the ``onnx`` backend answers ``GET
    /v2/models/<m>/config`` with: Triton's JSON form of :func:`triton_config`,
    with the inputs and outputs Triton would fill in from the graph."""
    inputs, outputs = m.io()
    mb = m.max_batch(inputs)

    def tensors(ts: list[Tensor]) -> list[dict]:
        return [{"name": t.name, "data_type": t.triton_type,
                 "dims": list(t.shape[1:] if mb else t.shape)} for t in ts]
    cfg: dict = {"name": name, "platform": "onnxruntime_onnx", "backend": "onnxruntime",
                 "max_batch_size": mb, "input": tensors(inputs), "output": tensors(outputs),
                 "instance_group": [{"kind": "KIND_GPU" if gpu else "KIND_CPU", "count": 1}],
                 "parameters": {}}
    if m.detector_spec():
        cfg["parameters"]["detector_spec"] = {"string_value": m.detector_spec()}
    cfg["parameters"]["model_version"] = {"string_value": m.version}
    return cfg


# ----------------------------------------------------------------- engines --

@dataclass(frozen=True)
class GPU:
    name: str
    cc: str                  # compute capability, "8.9"

    @property
    def tag(self) -> str:
        return "sm" + self.cc.replace(".", "")


def gpu(index: int = 0) -> GPU | None:
    """This host's GPU from nvidia-smi (Jetson's too), or None."""
    smi = shutil.which("nvidia-smi")
    if smi is None:
        return None
    try:
        out = subprocess.run([smi, f"--id={index}", "--query-gpu=name,compute_cap",
                              "--format=csv,noheader"], capture_output=True, text=True,
                             timeout=10)
    except (OSError, subprocess.TimeoutExpired):
        return None
    line = out.stdout.strip().splitlines()[:1]
    if out.returncode != 0 or not line or "," not in line[0]:
        return None
    name, cc = (s.strip() for s in line[0].rsplit(",", 1))
    return GPU(name, cc) if re.fullmatch(r"\d+\.\d+", cc) else None


def build_shapes(m: CVModel, inputs: list[Tensor], max_batch: int) -> dict[str, str] | None:
    """trtexec's ``--minShapes/--optShapes/--maxShapes`` for the input, or
    None for a graph without dynamic dimensions. ``model.toml``'s
    ``[tensorrt]`` min/opt/max win; then the spec's layout (M0's defaults:
    frames of any size up to 1080p, or NCHW at the spec's size); then the
    graph, when only the batch is dynamic."""
    name = m.input_name(inputs)
    inp = next((t for t in inputs if t.name == name), None)
    if inp is None:
        raise CVError(f"the graph has no input named {name!r}")
    trt = m.meta.get("tensorrt") or {}
    b = max_batch or 1
    if all(trt.get(k) for k in ("min", "opt", "max")):
        return {k: str(trt[k]) for k in ("min", "opt", "max")}
    if -1 not in inp.shape:
        return None
    layout = m.spec.get("input_layout")
    if layout == "frames":
        lo, opt, hi = FRAMES_SHAPES
        return {"min": lo, "opt": opt.format(b=b), "max": hi.format(b=b)}
    if layout == "nchw" and m.spec.get("height") and m.spec.get("width"):
        hw = f"3x{m.spec['height']}x{m.spec['width']}"
        return {"min": f"1x{hw}", "opt": f"{b}x{hw}", "max": f"{b}x{hw}"}
    if inp.shape[0] == -1 and -1 not in inp.shape[1:]:
        rest = "x".join(str(d) for d in inp.shape[1:])
        return {"min": f"1x{rest}", "opt": f"{b}x{rest}", "max": f"{b}x{rest}"}
    raise CVError(f"input {name} has dynamic dimensions {list(inp.shape)}; set "
                  f"[tensorrt] min, opt and max shapes in {MODEL_TOML}")


@dataclass(frozen=True)
class EngineBuild:
    """What one TensorRT build takes; :attr:`key` names its cache folder."""
    onnx_sha256: str
    tensorrt: str
    gpu: GPU
    input: str
    shapes: dict | None
    max_batch: int
    flags: tuple = ()

    @property
    def key(self) -> str:
        # The opt shape picks the kernels, so the shapes are part of the key.
        h = hashlib.sha256(json.dumps({"shapes": self.shapes, "flags": list(self.flags)},
                                      sort_keys=True).encode()).hexdigest()[:10]
        return f"trt{self.tensorrt}-{self.gpu.tag}-{h}"

    def trtexec_args(self, onnx: str, plan: str) -> list[str]:
        args = [f"--onnx={onnx}", f"--saveEngine={plan}"]
        if self.shapes:
            args += [f"--{k}Shapes={self.input}:{self.shapes[k]}" for k in ("min", "opt", "max")]
        return args + list(self.flags)


def trtexec_flags(m: CVModel) -> tuple[str, ...]:
    """``--stronglyTyped`` unless model.toml's ``[tensorrt]`` says
    ``strongly_typed = false``, then its ``flags``."""
    trt = m.meta.get("tensorrt") or {}
    flags = [] if trt.get("strongly_typed") is False else [STRONGLY_TYPED]
    flags += [str(f) for f in trt.get("flags") or () if str(f) not in flags]
    return tuple(flags)


def _log_precision(log: Path) -> str | None:
    """trtexec's ``Precision:`` build option ("Strongly Typed", "FP32", ...)."""
    m = re.search(r"\]\s*Precision:\s*(.+)", log.read_text(errors="replace"))
    return m.group(1).strip() if m else None


def engine_dir(cache_root: Path, build: EngineBuild) -> Path:
    return cache_root / build.onnx_sha256[:16] / build.key


def image_tensorrt_version(image: str) -> str | None:
    """TensorRT's version in a Triton image (its ``TRT_VERSION``)."""
    d = shutil.which("docker")
    if d is None:
        return None
    r = subprocess.run([d, "image", "inspect", image, "--format", "{{json .Config.Env}}"],
                       capture_output=True, text=True)
    try:
        env = json.loads(r.stdout) if r.returncode == 0 else []
    except json.JSONDecodeError:
        return None
    for e in env or []:
        k, _, v = e.partition("=")
        if k in ("TRT_VERSION", "TENSORRT_VERSION") and v:
            return v
    return None


def build_engine(m: CVModel, build: EngineBuild, out: Path, *, image: str,
                 gpu_args: list[str], served_image: str, log) -> Path:
    """Build ``m``'s TensorRT engine with trtexec inside the Triton image
    (no network), into ``out``, with a ``build.json`` in M0's shape and the
    trtexec log beside it. Returns the plan."""
    assert m.onnx is not None
    plan = out / "model.plan"
    if plan.is_file() and (out / BUILD_JSON).is_file():
        return plan
    d = shutil.which("docker") or "docker"
    tmp = out.with_name(out.name + ".building")
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)
    # iGPU (Jetson) images keep trtexec off PATH.
    script = ('t=$(command -v trtexec || echo /usr/src/tensorrt/bin/trtexec); '
              'exec "$t" "$@"')
    argv = [d, "run", "--rm", "--pull", "never", "--network", "none",
            "--user", f"{os.getuid()}:{os.getgid()}", *gpu_args,
            "-v", f"{m.onnx.parent}:/in:ro", "-v", f"{tmp}:/out",
            "--entrypoint", "/bin/sh", image, "-c", script, "trtexec",
            *build.trtexec_args(f"/in/{m.onnx.name}", "/out/model.plan")]
    log(f"Building the TensorRT engine ({build.key}); this takes minutes. Log: "
        f"{out / 'trtexec.log'}")
    t0 = time.time()
    with (tmp / "trtexec.log").open("w") as f:
        f.write("$ " + " ".join(argv) + "\n")
        f.flush()
        rc = subprocess.run(argv, stdout=f, stderr=subprocess.STDOUT).returncode
    precision = _log_precision(tmp / "trtexec.log")
    if rc == 0 and STRONGLY_TYPED in build.flags and precision != "Strongly Typed":
        rc = -1                     # M0 refuses such an engine too
        with (tmp / "trtexec.log").open("a") as f:
            f.write(f"llmctl: asked for {STRONGLY_TYPED}, built with precision "
                    f"{precision!r}; refused\n")
    if rc != 0 or not (tmp / "model.plan").is_file():
        out.parent.mkdir(parents=True, exist_ok=True)
        failed = out.with_name(out.name + ".failed")
        shutil.rmtree(failed, ignore_errors=True)
        os.replace(tmp, failed)
        tail = (failed / "trtexec.log").read_text(errors="replace").splitlines()[-15:]
        raise CVError("trtexec failed (exit %d); log %s:\n  %s"
                      % (rc, failed / "trtexec.log", "\n  ".join(tail)))
    inputs, _ = m.io()
    record = {
        "model": m.folder.name, "model_version": m.version, "onnx": m.onnx.name,
        "onnx_sha256": build.onnx_sha256, "served": "tensorrt",
        "model_sha256": sha256(tmp / "model.plan"), "image": served_image,
        "tensorrt": build.tensorrt, "precision": precision,
        "gpu": build.gpu.name, "compute_capability": build.gpu.cc,
        "input": build.input, "layout": m.spec.get("input_layout"),
        "model_size": [m.spec.get("height"), m.spec.get("width")]
        if m.spec.get("height") else None,
        "shapes": build.shapes, "max_batch": build.max_batch,
        "flags": list(build.flags), "build_seconds": round(time.time() - t0, 1),
        "built_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    (tmp / BUILD_JSON).write_text(json.dumps(record, indent=2) + "\n")
    shutil.rmtree(out, ignore_errors=True)
    out.parent.mkdir(parents=True, exist_ok=True)
    os.replace(tmp, out)
    log(f"Engine built in {record['build_seconds']:.0f} s: {out / 'model.plan'}")
    return out / "model.plan"


def import_files(source: Path, onnx: Path | None) -> dict[str, Path]:
    """``{path in the model folder: source file}`` to import a CV model
    from ``source`` (a folder, or one ``.onnx``) into llmctl's layout: the
    graph as ``1/model.onnx`` with its external-data files beside it, a
    shipped engine as ``1/model.plan`` (with its ``build.json``), and
    ``spec.toml``, ``config.pbtxt`` and a model card kept as they are."""
    folder = source if source.is_dir() else source.parent
    if onnx is None and source.is_file():
        onnx = source
    m = load(folder) if onnx is None else CVModel(folder, onnx, None)
    files: dict[str, Path] = {}
    if m.onnx is not None:
        files[f"{VERSION_DIR}/model.onnx"] = m.onnx
        # External data (`model.onnx.data`, `model.onnx_data`): the graph
        # names it relative to itself, so it keeps its name.
        for p in sorted(m.onnx.parent.iterdir()):
            if p.is_file() and p != m.onnx and p.name.startswith(m.onnx.name) and \
                    p.suffix != ".onnx":
                files[f"{VERSION_DIR}/{p.name}"] = p
    plan = folder / VERSION_DIR / "model.plan"
    if plan.is_file():
        files[f"{VERSION_DIR}/model.plan"] = plan
        if (folder / VERSION_DIR / BUILD_JSON).is_file():
            files[f"{VERSION_DIR}/{BUILD_JSON}"] = folder / VERSION_DIR / BUILD_JSON
    if not any(k.endswith((".onnx", ".plan")) for k in files):
        raise CVError(f"{source} has no ONNX graph to import")
    for name in (SPEC_TOML, CONFIG_PBTXT):
        for where in (folder, m.onnx.parent if m.onnx else folder):
            if (where / name).is_file():
                files.setdefault(name, where / name)
    return files


def _place(src: Path, dst: Path) -> None:
    """``dst`` as a hard link to ``src``, or a copy across filesystems."""
    if dst.exists() and dst.stat().st_ino == src.stat().st_ino:
        return
    dst.unlink(missing_ok=True)
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def stage_repository(root: Path, name: str, config: str, model_file: Path) -> Path:
    """A one-model Triton repository under ``root``: ``<name>/config.pbtxt``
    and ``<name>/1/model.{plan,onnx}``. Real files, not symlinks, since the
    folder is mounted into the container. Returns ``root``."""
    entry = root / name
    shutil.rmtree(entry, ignore_errors=True)
    (entry / VERSION_DIR).mkdir(parents=True)
    (entry / CONFIG_PBTXT).write_text(config)
    _place(model_file, entry / VERSION_DIR / ("model.plan" if model_file.suffix == ".plan"
                                              else "model.onnx"))
    return root

"""CV models: ONNX graphs on the onnx (ONNX Runtime) and triton backends —
the graph reader, folder detection, Triton configs, engine builds and their
cache, run, the KServe v2 server, gateway routing, import and download."""
import http.client
import json
import os
import socket
import struct
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pytest
from click.testing import CliRunner

from llm_runner import backends, cli, containers, cv, process as proc
from llm_runner.cluster import KSERVE, OPENAI, Instance, Unroutable, pick
from llm_runner.registry import discover_models

from test_cluster import _call, _node, _serve


# ------------------------------------------------- a hand-encoded ONNX graph --

def _varint(n: int) -> bytes:
    out = b""
    while True:
        b, n = n & 0x7F, n >> 7
        out += bytes([b | (0x80 if n else 0)])
        if not n:
            return out


def _msg(num: int, payload: bytes) -> bytes:
    return _varint(num << 3 | 2) + _varint(len(payload)) + payload


def _int(num: int, value: int) -> bytes:
    return _varint(num << 3) + _varint(value)


def _value_info(name: str, elem: int, dims) -> bytes:
    shape = b"".join(_msg(1, _msg(2, d.encode()) if isinstance(d, str) else _int(1, d))
                     for d in dims)
    return _msg(1, name.encode()) + _msg(2, _msg(1, _int(1, elem) + _msg(2, shape)))


def onnx_add(path: Path, *, dims=("N", 3), elem=1) -> Path:
    """``y = x + w`` with ``w = [1, 2, 3]``; ``w`` is also listed among the
    inputs, as older exporters did with initializers."""
    w = _int(1, 3) + _int(2, 1) + _msg(8, b"w") + _msg(9, struct.pack("<3f", 1, 2, 3))
    node = _msg(1, b"x") + _msg(1, b"w") + _msg(2, b"y") + _msg(4, b"Add")
    graph = (_msg(1, node) + _msg(2, b"g") + _msg(5, w)
             + _msg(11, _value_info("x", elem, dims)) + _msg(11, _value_info("w", 1, [3]))
             + _msg(12, _value_info("y", elem, dims)))
    model = _int(1, 8) + _msg(8, _msg(1, b"") + _int(2, 17)) + _msg(7, graph)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(model)
    return path


SPEC = """family = "detr"
height = 640
width = 640
input_name = "frames"
input_layout = "frames"
outputs = { boxes = "dets", logits = "labels" }

[labels]
0 = "person"
1 = "van"
"""


def _cv_folder(root: Path, name="det", *, spec=True, layout=True) -> Path:
    d = root / name
    onnx_add(d / "1" / "model.onnx" if layout else d / "export.onnx")
    if spec:
        (d / "spec.toml").write_text(SPEC)
    return d


# ------------------------------------------------------------------ reading --

def test_the_reader_skips_initializers_and_marks_dynamic_dims(tmp_path):
    ins, outs = cv.onnx_io(onnx_add(tmp_path / "m.onnx"))
    assert ins == [cv.Tensor("x", "FP32", (-1, 3))]
    assert outs == [cv.Tensor("y", "FP32", (-1, 3))]
    ins, _ = cv.onnx_io(onnx_add(tmp_path / "u8.onnx", dims=(1, "h", "w", 3), elem=2))
    assert ins[0].datatype == "UINT8" and ins[0].shape == (1, -1, -1, 3)
    (tmp_path / "junk.onnx").write_bytes(b"\x08\x07")
    with pytest.raises(cv.CVError, match="not an ONNX model"):
        cv.onnx_io(tmp_path / "junk.onnx")


def test_folders_that_are_cv_models(tmp_path):
    det = _cv_folder(tmp_path)
    bare = tmp_path / "bare"
    onnx_add(bare / "export.onnx")
    many = tmp_path / "many"
    onnx_add(many / "onnx" / "model.onnx")
    onnx_add(many / "onnx" / "model_fp16.onnx")
    hf = tmp_path / "Qwen3-8B"
    onnx_add(hf / "onnx" / "model.onnx")
    (hf / "config.json").write_text("{}")
    (hf / "model.safetensors").write_text("")
    assert cv.is_cv_dir(det) and cv.is_cv_dir(bare)
    assert not cv.is_cv_dir(many) and not cv.is_cv_dir(hf)
    (many / "model.toml").write_text('onnx = "onnx/model_fp16.onnx"\n')
    assert cv.is_cv_dir(many) and cv.load(many).onnx == many / "onnx" / "model_fp16.onnx"
    specs = discover_models(tmp_path)
    assert {k: s.backend for k, s in specs.items()} == {
        "det": "onnx", "bare": "onnx", "many": "onnx", "qwen3-8b": "vllm"}
    assert specs["det"].model_path == det                      # the folder, not 1/
    assert backends.infer_backend(det / "1" / "model.onnx") == backends.ONNX


def test_several_graphs_need_a_choice(tmp_path):
    d = tmp_path / "two"
    onnx_add(d / "a.onnx")
    onnx_add(d / "b.onnx")
    (d / "spec.toml").write_text(SPEC)
    with pytest.raises(cv.CVError, match='onnx = "a.onnx"'):
        cv.load(d)


# ------------------------------------------------------------ Triton configs --

def test_a_generated_config_has_m0s_shape(tmp_path):
    m = cv.load(_cv_folder(tmp_path))
    m.meta["version"] = "1.2.0"
    text = cv.triton_config(m, "det-a", engine="tensorrt", gpu=True)
    assert text.splitlines()[:5] == [
        'name: "det-a"', 'platform: "tensorrt_plan"', "max_batch_size: 16",
        "dynamic_batching { preferred_batch_size: [4, 8, 16] max_queue_delay_microseconds: 2000 }",
        "instance_group [{ kind: KIND_GPU, count: 1 }]"]
    spec_line = next(line for line in text.splitlines() if '"detector_spec"' in line)
    raw = spec_line.split("string_value: ", 1)[1].rsplit(" } }", 1)[0]
    spec = json.loads(json.loads(raw))                      # pbtxt string, then JSON
    assert spec["labels"] == {"0": "person", "1": "van"} and spec["family"] == "detr"
    assert 'key: "model_version" value: { string_value: "1.2.0" }' in text
    cpu = cv.triton_config(m, "det-a", engine="onnxruntime", gpu=False)
    assert 'backend: "onnxruntime"' in cpu and "KIND_CPU" in cpu


def test_a_shipped_config_is_kept_renamed_and_completed(tmp_path):
    d = _cv_folder(tmp_path)
    (d / "config.pbtxt").write_text('name: "rfdetr-medium-trt"\nplatform: "tensorrt_plan"\n'
                                    "max_batch_size: 8\ndynamic_batching {}\n"
                                    "instance_group [{ kind: KIND_GPU, count: 1 }]\n"
                                    'parameters { key: "model_version" value: '
                                    '{ string_value: "9.9.9" } }\n')
    m = cv.load(d)
    text = cv.triton_config(m, "det", engine="tensorrt", gpu=True)
    assert text.startswith('name: "det"\nplatform: "tensorrt_plan"\nmax_batch_size: 8\n')
    assert text.count("model_version") == 1 and "9.9.9" in text       # theirs wins
    assert '"detector_spec"' in text                                    # added
    cpu = cv.triton_config(m, "det", engine="onnxruntime", gpu=False)
    assert 'backend: "onnxruntime"' in cpu and "tensorrt_plan" not in cpu
    assert "KIND_CPU" in cpu and "KIND_GPU" not in cpu


def test_the_onnx_servers_config_is_tritons_json(tmp_path):
    m = cv.load(_cv_folder(tmp_path))
    cfg = cv.server_config(m, "det", gpu=False)
    assert cfg["max_batch_size"] == 16
    assert cfg["input"] == [{"name": "x", "data_type": "TYPE_FP32", "dims": [3]}]
    assert json.loads(cfg["parameters"]["detector_spec"]["string_value"])["height"] == 640
    assert cfg["parameters"]["model_version"] == {"string_value": "0.0.0"}
    static = tmp_path / "static"
    onnx_add(static / "1" / "model.onnx", dims=(8, 3))
    assert cv.server_config(cv.load(static), "s", gpu=True)["input"][0]["dims"] == [8, 3]


# --------------------------------------------------------------- engines --

def test_build_shapes(tmp_path):
    frames = cv.load(_cv_folder(tmp_path))
    ins = [cv.Tensor("frames", "UINT8", (-1, -1, -1, -1))]
    assert cv.build_shapes(frames, ins, 16) == {
        "min": "1x64x64x3", "opt": "16x720x1280x4", "max": "16x1080x1920x4"}
    frames.spec["input_layout"] = "nchw"
    nchw = [cv.Tensor("frames", "FP32", (-1, 3, -1, -1))]
    assert cv.build_shapes(frames, nchw, 16) == {
        "min": "1x3x640x640", "opt": "16x3x640x640", "max": "16x3x640x640"}
    frames.meta["tensorrt"] = {"min": "1x3x320x320", "opt": "4x3x320x320",
                               "max": "8x3x320x320"}
    assert cv.build_shapes(frames, nchw, 16)["opt"] == "4x3x320x320"
    bare = cv.load(onnx_add(tmp_path / "b" / "m.onnx"))
    assert cv.build_shapes(bare, [cv.Tensor("x", "FP32", (-1, 3))], 16) == {
        "min": "1x3", "opt": "16x3", "max": "16x3"}
    assert cv.build_shapes(bare, [cv.Tensor("x", "FP32", (8, 3))], 0) is None
    with pytest.raises(cv.CVError, match=r"\[tensorrt\]"):
        cv.build_shapes(bare, [cv.Tensor("x", "FP32", (-1, 3, -1, -1))], 16)


def test_the_engine_key_covers_what_changes_kernels():
    g = cv.GPU("NVIDIA GeForce RTX 4090", "8.9")
    shapes = {"min": "1x3", "opt": "16x3", "max": "16x3"}
    a = cv.EngineBuild("ab" * 32, "11.3.0.99", g, "x", shapes, 16)
    assert a.key.startswith("trt11.3.0.99-sm89-")
    assert cv.EngineBuild("ab" * 32, "11.3.0.99", g, "x", {**shapes, "opt": "8x3"}, 16).key \
        != a.key
    assert cv.EngineBuild("ab" * 32, "10.11.0.33+jp6", cv.GPU("Orin (nvgpu)", "8.7"), "x",
                          shapes, 16).key.startswith("trt10.11.0.33+jp6-sm87-")
    assert a.trtexec_args("/in/m.onnx", "/out/p") == [
        "--onnx=/in/m.onnx", "--saveEngine=/out/p", "--minShapes=x:1x3",
        "--optShapes=x:16x3", "--maxShapes=x:16x3"]


def test_build_engine_records_the_build(tmp_path, monkeypatch):
    m = cv.load(_cv_folder(tmp_path / "models"))
    m.spec = {}
    calls = []

    def run(argv, **kw):
        calls.append(argv)
        out = Path(next(a.split(":")[0] for a in argv if a.endswith(":/out")))
        (out / "model.plan").write_bytes(b"plan")
        kw["stdout"].write("[11/02/2026-17:19:56] [I] Precision: Strongly Typed\n")
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr(cv.subprocess, "run", run)
    build = cv.EngineBuild(cv.sha256(m.onnx), "11.3.0.99", cv.GPU("RTX", "8.9"), "x",
                           {"min": "1x3", "opt": "16x3", "max": "16x3"}, 16,
                           cv.trtexec_flags(m))
    out = cv.engine_dir(tmp_path / "engines", build)
    plan = cv.build_engine(m, build, out, image="img@sha256:1", gpu_args=["--gpus", "all"],
                           served_image="img@sha256:1", log=lambda s: None)
    assert plan == out / "model.plan" and plan.read_bytes() == b"plan"
    argv = calls[0]
    assert argv[argv.index("--network") + 1] == "none" and "--gpus" in argv
    assert "--optShapes=x:16x3" in argv and "/usr/src/tensorrt/bin/trtexec" in " ".join(argv)
    assert argv[-1] == "--stronglyTyped"
    rec = json.loads((out / "build.json").read_text())
    assert rec["tensorrt"] == "11.3.0.99" and rec["max_batch"] == 16
    assert rec["precision"] == "Strongly Typed" and rec["flags"] == ["--stronglyTyped"]
    assert rec["onnx_sha256"] == build.onnx_sha256 and rec["served"] == "tensorrt"
    assert (out / "trtexec.log").is_file()
    cv.build_engine(m, build, out, image="x", gpu_args=[], served_image="x",
                    log=lambda s: None)
    assert len(calls) == 1                                      # cached


def test_strongly_typed_is_the_default_and_checked(tmp_path, monkeypatch):
    m = cv.load(_cv_folder(tmp_path))
    assert cv.trtexec_flags(m) == ("--stronglyTyped",)
    m.meta["tensorrt"] = {"flags": ["--builderOptimizationLevel=5", "--stronglyTyped"]}
    assert cv.trtexec_flags(m) == ("--stronglyTyped", "--builderOptimizationLevel=5")
    m.meta["tensorrt"] = {"strongly_typed": False, "flags": ["--fp16"]}
    assert cv.trtexec_flags(m) == ("--fp16",)

    def run(argv, **kw):          # TensorRT ignored the flag: FP32
        out = Path(next(a.split(":")[0] for a in argv if a.endswith(":/out")))
        (out / "model.plan").write_bytes(b"plan")
        kw["stdout"].write("[I] Precision: FP32\n")
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr(cv.subprocess, "run", run)
    build = cv.EngineBuild("ef" * 32, "10.11", cv.GPU("Orin", "8.7"), "x", None, 0,
                           ("--stronglyTyped",))
    out = tmp_path / "e" / build.key
    with pytest.raises(cv.CVError, match="trtexec failed"):
        cv.build_engine(m, build, out, image="i", gpu_args=[], served_image="i",
                        log=lambda s: None)
    log = (out.with_name(out.name + ".failed") / "trtexec.log").read_text()
    assert "precision 'FP32'; refused" in log and not out.exists()


def test_a_failed_build_keeps_its_log(tmp_path, monkeypatch):
    m = cv.load(_cv_folder(tmp_path))
    monkeypatch.setattr(cv.subprocess, "run", lambda argv, **kw: SimpleNamespace(returncode=1))
    build = cv.EngineBuild("cd" * 32, "11", cv.GPU("RTX", "8.9"), "x", None, 0)
    out = tmp_path / "e" / build.key
    with pytest.raises(cv.CVError, match="trtexec failed"):
        cv.build_engine(m, build, out, image="i", gpu_args=[], served_image="i",
                        log=lambda s: None)
    assert not out.exists() and (out.with_name(out.name + ".failed") / "trtexec.log").is_file()


# ------------------------------------------------------------------- run --

@pytest.fixture
def launched(tmp_path, monkeypatch):
    monkeypatch.setattr(backends, "find", lambda b: f"/bin/{b}")
    monkeypatch.setattr(cli, "_port_in_use", lambda p: False)
    monkeypatch.setattr(cli, "_ensure_gateway", lambda: None)
    monkeypatch.setattr(proc, "STATE_DIR", tmp_path / "state")
    monkeypatch.setattr(containers, "docker_bin", lambda: "/usr/bin/docker")
    monkeypatch.setattr(containers, "resolve_digest", lambda ref, state=None: "sha256:" + "1" * 64)
    monkeypatch.setattr(cli.images, "local_image", lambda ref, state: None)
    monkeypatch.setattr(cv, "gpu", lambda index=0: None)
    started = {}
    monkeypatch.setattr(proc, "start", lambda **kw: started.update(kw) or proc.RunningProcess(
        name=kw["name"], pid=1, port=kw["port"], started_at=0, model_path=kw["model_path"],
        log_file="/dev/null", cmd=kw["cmd"], backend=kw["backend"]))
    return started


def _arg(cmd, flag):
    return cmd[cmd.index(flag) + 1]


def test_run_a_cv_model_on_onnx(tmp_path, launched):
    d = _cv_folder(tmp_path / "models")
    out = CliRunner().invoke(cli.cli, ["run", "--ft", str(d), "--port", "8100"])
    assert out.exit_code == 0, out.output
    cmd = launched["cmd"]
    assert launched["backend"] == "onnx" and launched["name"] == "det"
    assert cmd[:3] == ["/bin/onnx", "-I", backends.onnx_server_script()]
    assert _arg(cmd, "--model") == str(d / "1" / "model.onnx") and _arg(cmd, "--name") == "det"
    cfg = json.loads(Path(_arg(cmd, "--config")).read_text())
    assert cfg["name"] == "det" and "detector_spec" in cfg["parameters"]
    assert "/v2/models/det/infer" in out.output


def test_run_a_cv_model_on_triton_without_a_gpu(tmp_path, launched):
    d = _cv_folder(tmp_path / "models")
    out = CliRunner().invoke(cli.cli, ["run", "--ft", str(d), "--backend", "triton",
                                       "--port", "8101"])
    assert out.exit_code == 0, out.output
    cmd, rt = launched["cmd"], launched["runtime"]
    assert rt["kind"] == "docker" and rt["gpus"] == containers.NO_GPU
    assert rt["image"] == containers.default_image("triton")
    repo = Path(cmd[1].split("=", 1)[1])
    entry = repo / "det"
    assert (entry / "1" / "model.onnx").read_bytes() == (d / "1" / "model.onnx").read_bytes()
    assert 'backend: "onnxruntime"' in (entry / "config.pbtxt").read_text()
    assert _arg(cmd, "--http-port") == "8101" and "--allow-grpc=false" in cmd
    argv = containers.docker_argv(name="det", cmd=cmd, port=8101, backend="triton",
                                  model_path=str(d), env={}, runtime={**rt, "container": "c"},
                                  models_dir=tmp_path / "models", cache_dir=None)
    assert f"{repo}:/repo:ro" in argv and f"stack={containers.TRITON_STACK}" in argv
    assert "--gpus" not in argv and "--device" not in argv
    inner = argv[argv.index("tritonserver") + 1 if "tritonserver" in argv else 0:]
    assert "--model-repository=/repo" in inner and _arg(inner, "--http-address") == "0.0.0.0"


def test_run_on_triton_with_a_gpu_builds_then_reuses_the_engine(tmp_path, launched, monkeypatch):
    monkeypatch.setattr(cv, "gpu", lambda index=0: cv.GPU("NVIDIA GeForce RTX 4090", "8.9"))
    monkeypatch.setattr(cv, "image_tensorrt_version", lambda image: "11.3.0.99")
    monkeypatch.setattr(containers, "has_cdi", lambda: False)
    builds = []

    def build(m, build, out, **kw):
        builds.append(build)
        out.mkdir(parents=True)
        (out / "model.plan").write_bytes(b"plan")
        return out / "model.plan"
    monkeypatch.setattr(cv, "build_engine", build)
    d = _cv_folder(tmp_path / "models")
    (d / "spec.toml").write_text(SPEC.replace('input_name = "frames"', 'input_name = "x"'))
    for _ in range(2):
        out = CliRunner().invoke(cli.cli, ["run", "--ft", str(d), "--backend", "triton"])
        assert out.exit_code == 0, out.output
    assert len(builds) == 1 and "Engine: cached" in out.output
    assert builds[0].shapes["opt"] == "16x720x1280x4" and builds[0].input == "x"
    repo = Path(launched["cmd"][1].split("=", 1)[1])
    assert (repo / "det" / "1" / "model.plan").read_bytes() == b"plan"
    assert 'platform: "tensorrt_plan"' in (repo / "det" / "config.pbtxt").read_text()
    assert launched["runtime"]["gpus"] == "all"


@pytest.mark.parametrize("backend", ["vllm", "llama", "omni"])
def test_a_cv_model_runs_only_on_cv_backends(tmp_path, launched, backend):
    out = CliRunner().invoke(cli.cli, ["run", "--ft", str(_cv_folder(tmp_path)),
                                       "--backend", backend])
    assert out.exit_code != 0 and "--backend onnx or triton" in out.output


def test_triton_needs_a_cv_model_and_docker(tmp_path, launched):
    ckpt = tmp_path / "Qwen3-8B"
    ckpt.mkdir()
    (ckpt / "config.json").write_text("{}")
    (ckpt / "model.safetensors").write_text("")
    out = CliRunner().invoke(cli.cli, ["run", "--ft", str(ckpt), "--backend", "triton"])
    assert out.exit_code != 0 and "not a CV model" in out.output
    out = CliRunner().invoke(cli.cli, ["run", "--ft", str(_cv_folder(tmp_path)),
                                       "--backend", "triton", "--runtime", "host"])
    assert out.exit_code != 0 and "container only" in out.output


def test_a_cv_name_must_suit_tritons_api(tmp_path, launched):
    out = CliRunner().invoke(cli.cli, ["run", "--ft", str(_cv_folder(tmp_path)),
                                       "--name", "ft:det"])
    assert out.exit_code != 0 and "pass --name" in out.output


def test_triton_answers_on_kserves_health_path():
    assert backends.health_path("triton") == "/v2/health/ready"
    assert backends.health_path("onnx") == backends.health_path("vllm") == "/health"
    assert backends.with_bind_host(["tritonserver", "--http-address", "0.0.0.0"], "triton",
                                   "10.0.0.5")[-1] == "10.0.0.5"


# ------------------------------------------------------------ onnx server --

def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture
def server(tmp_path):
    pytest.importorskip("onnxruntime")
    d = _cv_folder(tmp_path)
    cfg = tmp_path / "config.json"
    cfg.write_text(json.dumps(cv.server_config(cv.load(d), "det", gpu=False)))
    port = _free_port()
    p = subprocess.Popen([sys.executable, backends.onnx_server_script(), "--model",
                          str(d / "1" / "model.onnx"), "--name", "det", "--host", "127.0.0.1",
                          "--port", str(port), "--config", str(cfg), "--device", "cpu"],
                         stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    for _ in range(100):
        try:
            socket.create_connection(("127.0.0.1", port), timeout=0.2).close()
            break
        except OSError:
            if p.poll() is not None:
                pytest.fail(p.stdout.read().decode())
            time.sleep(0.1)
    yield port
    p.terminate()
    p.wait(10)


def _post(port, path, body: bytes, headers: dict):
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
    conn.request("POST", path, body=body, headers=headers)
    r = conn.getresponse()
    return r.status, dict(r.getheaders()), r.read()


def test_the_server_speaks_kserve_v2(server):
    for path in ("/health", "/v2/health/ready", "/v2/models/det/ready"):
        conn = http.client.HTTPConnection("127.0.0.1", server, timeout=5)
        conn.request("GET", path)
        assert conn.getresponse().status == 200, path
    meta = _call(SimpleNamespace(port=server), "GET", "/v2/models/det")[1]
    assert meta["inputs"] == [{"name": "x", "datatype": "FP32", "shape": [-1, 3]}]
    cfg = _call(SimpleNamespace(port=server), "GET", "/v2/models/det/versions/1/config")[1]
    assert cfg["max_batch_size"] == 16 and "detector_spec" in cfg["parameters"]
    # JSON in, JSON out.
    status, _, body = _post(server, "/v2/models/det/infer", json.dumps({
        "id": "r1", "inputs": [{"name": "x", "datatype": "FP32", "shape": [2, 3],
                                "data": [0, 0, 0, 1, 1, 1]}]}).encode(),
        {"Content-Type": "application/json"})
    out = json.loads(body)
    assert status == 200 and out["id"] == "r1" and out["model_name"] == "det"
    assert out["outputs"] == [{"name": "y", "datatype": "FP32", "shape": [2, 3],
                               "data": [1, 2, 3, 2, 3, 4]}]
    # Binary in, binary out (the extension M0's client uses).
    raw = struct.pack("<3f", 10, 20, 30)
    head = json.dumps({"inputs": [{"name": "x", "datatype": "FP32", "shape": [1, 3],
                                   "parameters": {"binary_data_size": len(raw)}}],
                       "outputs": [{"name": "y", "parameters": {"binary_data": True}}]}).encode()
    status, hdrs, body = _post(server, "/v2/models/det/infer", head + raw, {
        "Content-Type": "application/octet-stream",
        "Inference-Header-Content-Length": str(len(head))})
    n = int(hdrs["Inference-Header-Content-Length"])
    reply = json.loads(body[:n])
    assert status == 200 and reply["outputs"][0]["parameters"] == {"binary_data_size": 12}
    assert struct.unpack("<3f", body[n:]) == (11, 22, 33)


def test_the_server_refuses_bad_requests(server):
    for body, needle in ((b"{}", "missing input"),
                         (json.dumps({"inputs": [{"name": "z", "datatype": "FP32",
                                                  "shape": [1], "data": [1]}]}), "unexpected"),
                         (json.dumps({"inputs": [{"name": "x", "datatype": "FP32",
                                                  "shape": [1, 3], "data": [1]}]}), "1 values")):
        status, _, out = _post(server, "/v2/models/det/infer",
                               body if isinstance(body, bytes) else body.encode(), {})
        assert status == 400 and needle in json.loads(out)["error"], needle
    status, _, out = _post(server, "/v2/models/other/infer", b"{}", {})
    assert status == 404


# ---------------------------------------------------------------- gateway --

def test_kserve_routes_apart_from_chat():
    chat = Instance("gemma", "rtx", 0, True, {"api": OPENAI, "status": "ready", "port": 1})
    det = Instance("det", "rtx", 0, True, {"api": KSERVE, "status": "ready", "port": 2})
    assert [c.id for c in pick([chat, det], "de", KSERVE)] == ["det"]
    assert [c.id for c in pick([chat, det], None, OPENAI)] == ["gemma"]
    with pytest.raises(Unroutable) as e:
        pick([chat, det], "gemma", KSERVE)
    assert e.value.code == "model_not_found" and "CV model" in e.value.message


def _kserve_handler(label):
    from http.server import BaseHTTPRequestHandler

    class H(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *a):
            pass

        def _reply(self, payload):
            body = json.dumps(payload).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            ok = self.path == "/v2/health/ready"
            self._reply({"who": label, "path": self.path, "ok": ok})

        def do_POST(self):
            n = int(self.headers.get("Content-Length") or 0)
            body = self.rfile.read(n)
            self._reply({"who": label, "path": self.path, "bytes": len(body),
                         "hlen": self.headers.get("Inference-Header-Content-Length")})
    return H


def _cv_rec(name, port, backend="onnx"):
    return {"name": name, "pid": os.getpid(), "port": port, "started_at": 1.7e9,
            "model_path": f"/m/{name}", "log_file": "", "cmd": ["x"], "backend": backend,
            "env": {}, "group": "detector"}


def test_the_gateway_routes_kserve_paths_across_nodes(tmp_path):
    orin_det = _serve(_kserve_handler("orin"))
    lab, lab_srv = _node(tmp_path, "lab", 5, [], serve=True)
    url = f"http://127.0.0.1:{lab.port}"
    orin, orin_srv = _node(tmp_path, "orin", 1,
                           [_cv_rec("det-trt", orin_det.server_address[1], "triton")],
                           registry_url=url)
    try:
        for _ in range(2):
            for g in (lab, orin):
                g.heartbeat_once()
        head = b'{"inputs": []}'
        conn = http.client.HTTPConnection("127.0.0.1", lab.port, timeout=5)
        # By group, from the registry node: the call goes to orin under the exact name.
        conn.request("POST", "/v2/models/detector/infer", body=head + b"\x00" * 100,
                     headers={"Inference-Header-Content-Length": str(len(head))})
        r = json.loads(conn.getresponse().read())
        assert r == {"who": "orin", "path": "/v2/models/det-trt/infer", "bytes": len(head) + 100,
                     "hlen": str(len(head))}
        status, cfg = _call(lab, "GET", "/v2/models/det-trt/config?x=1")
        assert status == 200 and cfg["path"] == "/v2/models/det-trt/config?x=1"
        status, err = _call(lab, "GET", "/v2/models/nope/config")
        assert status == 404 and err["error"]["code"] == "model_not_found"
        by_id = {m["id"]: m["llmctl"] for m in _call(orin, "GET", "/v1/models")[1]["data"]}
        assert by_id["det-trt"]["api"] == "kserve" and by_id["det-trt"]["status"] == "ready"
    finally:
        for s in (orin_det, lab_srv, orin_srv):
            s.shutdown()


# ---------------------------------------------------------------- import --

@pytest.fixture
def models(tmp_path, monkeypatch):
    m = tmp_path / "models"
    m.mkdir()
    monkeypatch.setattr(cli, "MODELS_DIR", m)
    monkeypatch.setattr(cli, "discover_models", lambda: discover_models(m))
    monkeypatch.setattr(proc, "STATE_DIR", tmp_path / "state")
    monkeypatch.setattr(proc, "STATE_FILE", tmp_path / "state" / "running.json")
    return m


def test_import_a_cv_fine_tune(tmp_path, models):
    src = tmp_path / "train" / "build" / "frames"
    onnx_add(src / "rfdetr-medium-fp16-frames.onnx")
    (src / "rfdetr-medium-fp16-frames.onnx.data").write_bytes(b"ext")
    (src / "spec.toml").write_text(SPEC)
    (src / "card.md").write_text("---\nlicense: Apache-2.0\nbase_model: rfdetr-medium\n"
                                 "model_version: 1.0.0\n---\n\n# ext1\n")
    out = CliRunner().invoke(cli.cli, ["import", str(src), "--name", "ext1"])
    assert out.exit_code == 0, out.output
    d = models / "ext1"
    assert sorted(str(p.relative_to(d)) for p in d.rglob("*") if p.is_file()) == [
        ".llmctl-source.json", "1/model.onnx", "1/rfdetr-medium-fp16-frames.onnx.data",
        "card.md", "model.toml", "spec.toml"]
    assert (d / "spec.toml").read_text() == SPEC                        # untouched
    m = cv.load(d)
    assert m.version == "1.0.0" and m.license == "Apache-2.0"
    assert m.meta["base_model"] == "rfdetr-medium" and m.meta["onnx_sha256"]
    src_rec = json.loads((d / ".llmctl-source.json").read_text())
    assert src_rec["source"] == "import" and src_rec["origin"] == str(src.resolve())
    assert discover_models(models)["ext1"].backend == "onnx"
    again = CliRunner().invoke(cli.cli, ["import", str(src), "--name", "ext1"])
    assert again.exit_code != 0 and "--replace" in again.output
    assert CliRunner().invoke(cli.cli, ["import", str(src), "--name", "ext1",
                                        "--version", "1.0.1", "--replace"]).exit_code == 0
    assert cv.load(d).version == "1.0.1"


def test_import_needs_a_licence(tmp_path, models):
    src = onnx_add(tmp_path / "det.onnx")
    out = CliRunner().invoke(cli.cli, ["import", str(src)])
    assert out.exit_code != 0 and "--license" in out.output
    out = CliRunner().invoke(cli.cli, ["import", str(src), "--license", "MIT", "--json"])
    assert out.exit_code == 0, out.output
    assert json.loads(out.output)["name"] == "det"
    assert (models / "det" / "1" / "model.onnx").is_file()


def test_import_a_gguf_and_a_checkpoint(tmp_path, models):
    gguf = tmp_path / "out" / "my-tune-Q4_K_M.gguf"
    gguf.parent.mkdir()
    gguf.write_bytes(b"GGUF")
    (gguf.parent / "mmproj-F16.gguf").write_bytes(b"GGUF")
    out = CliRunner().invoke(cli.cli, ["import", str(gguf), "--name", "my-tune",
                                       "--license", "apache-2.0"])
    assert out.exit_code == 0, out.output
    assert sorted(p.name for p in (models / "my-tune").iterdir()) == [
        ".llmctl-source.json", "mmproj-F16.gguf", "my-tune.gguf"]
    spec = discover_models(models)["my-tune"]
    assert spec.backend == "llama" and spec.mmproj_path is not None
    ckpt = tmp_path / "ckpt"
    (ckpt / ".git").mkdir(parents=True)
    (ckpt / "config.json").write_text("{}")
    (ckpt / "model.safetensors").write_bytes(b"")
    out = CliRunner().invoke(cli.cli, ["import", str(ckpt), "--license", "mit"])
    assert out.exit_code == 0, out.output
    assert not (models / "ckpt" / ".git").exists()
    assert discover_models(models)["ckpt"].backend == "vllm"
    junk = tmp_path / "notes.txt"
    junk.write_text("x")
    out = CliRunner().invoke(cli.cli, ["import", str(junk), "--license", "mit"])
    assert out.exit_code != 0 and "not a model" in out.output


def test_import_refuses_to_replace_a_served_model(tmp_path, models, monkeypatch):
    src = onnx_add(tmp_path / "det.onnx")
    assert CliRunner().invoke(cli.cli, ["import", str(src), "--license", "MIT"]).exit_code == 0
    monkeypatch.setattr(proc, "list_running", lambda: [SimpleNamespace(
        name="det", model_path=str(models / "det"))])
    out = CliRunner().invoke(cli.cli, ["import", str(src), "--license", "MIT", "--replace"])
    assert out.exit_code != 0 and "served by det" in out.output


# -------------------------------------------------------------- download --

def test_download_an_onnx_repo(tmp_path, models, monkeypatch):
    import huggingface_hub
    files = {"onnx/model.onnx": 100, "onnx/model_fp16.onnx": 50, "config.json": 1,
             "preprocessor_config.json": 1, "README.md": 1, "onnx/model_int8.onnx": 25}
    info = SimpleNamespace(
        siblings=[SimpleNamespace(rfilename=f, size=s) for f, s in files.items()],
        card_data={"license": "apache-2.0"}, pipeline_tag="object-detection", sha="abc123")
    monkeypatch.setattr(huggingface_hub.HfApi, "model_info", lambda self, repo, **kw: info)
    got = []

    def fetch(repo_id, filename, local_dir):
        got.append(filename)
        p = Path(local_dir) / filename
        if filename.endswith(".onnx"):
            onnx_add(p, dims=("N", 3, "H", "W"))
        elif filename == "preprocessor_config.json":
            p.write_text('{"size": {"height": 512, "width": 512}}')
        else:
            p.write_text("{}")
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", fetch)
    out = CliRunner().invoke(cli.cli, ["download", "onnx-community/rfdetr_small-ONNX",
                                       "--file", "onnx/model_fp16.onnx"])
    assert out.exit_code == 0, out.output
    assert sorted(got) == ["README.md", "config.json", "onnx/model_fp16.onnx",
                           "preprocessor_config.json"]
    d = models / "rfdetr_small-onnx"
    m = cv.load(d)
    assert m.onnx == d / "onnx" / "model_fp16.onnx" and m.license == "apache-2.0"
    assert m.meta["revision"] == "abc123" and m.meta["source"] == "onnx-community/rfdetr_small-ONNX"
    assert m.meta["tensorrt"] == {"min": "1x3x512x512", "opt": "16x3x512x512",
                                  "max": "16x3x512x512"}
    assert json.loads((d / ".llmctl-source.json").read_text())["format"] == "onnx"
    listing = CliRunner().invoke(cli.cli, ["list"]).output
    row = next(line for line in listing.splitlines() if line.startswith("rfdetr_small-onnx"))
    assert "onnx" in row and "abc123" in row and "apache-2.0" in row

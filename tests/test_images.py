"""Images delivered as files: chain checks, `images load`/`save`, run by ID."""
import base64
import hashlib
import io
import json
import re
import subprocess
import tarfile
from pathlib import Path

import pytest
from click.testing import CliRunner

from llm_runner import backends
from llm_runner import bundle
from llm_runner import cli
from llm_runner import containers
from llm_runner import images
from llm_runner import process as proc
from llm_runner import registry


def sha(b: bytes) -> str:
    return "sha256:" + hashlib.sha256(b).hexdigest()


CONFIG = b'{"architecture":"amd64","rootfs":{"diff_ids":["sha256:aa"]}}'
CONFIG_ID = sha(CONFIG)
MANIFEST = json.dumps({"schemaVersion": 2, "config": {"digest": CONFIG_ID},
                       "layers": []}).encode()
INDEX = json.dumps({"manifests": [
    {"digest": sha(MANIFEST), "platform": {"os": "linux", "architecture": "amd64"}},
    {"digest": "sha256:" + "1" * 64, "platform": {"os": "linux", "architecture": "arm64"}}]}
).encode()
REF = f"vllm/vllm-openai@{sha(INDEX)}"
# What a classic store's `docker save` puts in index.json: a rewritten manifest.
SAVED_MANIFEST = "sha256:" + "5" * 64


def archive(path: Path, *, config: bytes = CONFIG, sidecar: dict | None = None,
            images_n: int = 1) -> Path:
    cfg = hashlib.sha256(config).hexdigest()
    files = {
        "manifest.json": json.dumps([{"Config": f"blobs/sha256/{cfg}", "RepoTags": None,
                                      "Layers": []}] * images_n).encode(),
        "index.json": json.dumps({"manifests": [{"digest": SAVED_MANIFEST}]}).encode(),
        f"blobs/sha256/{cfg}": config,
    }
    if sidecar is not None:
        files[images.SIDECAR] = json.dumps(sidecar).encode()
    with tarfile.open(path, "w") as tf:
        for name, data in files.items():
            info = tarfile.TarInfo(name)
            info.size = len(data)
            tf.addfile(info, io.BytesIO(data))
    return path


def sidecar(index: bytes = INDEX, manifest: bytes = MANIFEST, ref: str = REF,
            platform: str = "linux/amd64") -> dict:
    return {"schema": images.SIDECAR_SCHEMA, "ref": ref, "platform": platform,
            "index": base64.b64encode(index).decode(),
            "manifest": base64.b64encode(manifest).decode()}


class FakeDocker:
    """Local images by name/ID, and what `docker load` reports."""

    def __init__(self):
        self.images: dict[str, dict] = {}     # name or ID -> inspect
        self.loads: list[str] = []
        self.tags: list[tuple[str, str]] = []
        self.load_reports = CONFIG_ID         # the ID `docker load` prints
        self.saved_config = CONFIG

    def add(self, image_id: str, *names: str, tags=()):
        info = {"Id": image_id, "RepoTags": list(tags)}
        for n in (image_id, *names):
            self.images[n] = info

    def inspect(self, ref):
        return self.images.get(ref)

    def run(self, argv, **kw):
        if argv[1] == "load":
            self.loads.append(argv[-1])
            self.add(self.load_reports)
            return subprocess.CompletedProcess(argv, 0, f"Loaded image ID: {self.load_reports}\n", "")
        if argv[1] == "tag":
            self.tags.append((argv[2], argv[3]))
            self.images[argv[2]]["RepoTags"].append(argv[3])
            return subprocess.CompletedProcess(argv, 0, "", "")
        if argv[1] == "save":
            archive(Path(argv[3]), config=self.saved_config)
            return subprocess.CompletedProcess(argv, 0, "", "")
        raise AssertionError(argv)


@pytest.fixture
def docker(monkeypatch, tmp_path):
    d = FakeDocker()
    monkeypatch.setattr(containers, "docker_bin", lambda: "/usr/bin/docker")
    monkeypatch.setattr(containers, "inspect_image", d.inspect)
    monkeypatch.setattr(images.subprocess, "run", d.run)
    monkeypatch.setattr(images, "host_platform", lambda: "linux/amd64")
    monkeypatch.setattr(containers, "expected_images", lambda: {"vllm": {
        "image": REF, "by_platform": {"linux/amd64": {"manifest": sha(MANIFEST),
                                                       "config": CONFIG_ID}}}})
    monkeypatch.setattr(proc, "STATE_DIR", tmp_path / "state")
    return d


# ---------------------------------------------------------------- archives --

def test_an_archive_gives_its_config_and_manifest(tmp_path):
    a = images.read_archive(archive(tmp_path / "a.tar"))
    assert a == {"config": CONFIG_ID, "manifest": SAVED_MANIFEST, "sidecar": None}
    (tmp_path / "x.tar").write_text("not a tar")
    with pytest.raises(images.BadArchive):
        images.read_archive(tmp_path / "x.tar")
    with pytest.raises(images.BadArchive, match="one image"):
        images.read_archive(archive(tmp_path / "two.tar", images_n=2))


def test_the_chain_checks_every_link():
    assert images.verify_chain(sidecar(), CONFIG_ID) == {
        "ref": REF, "platform": "linux/amd64", "manifest": sha(MANIFEST)}
    with pytest.raises(images.DigestMismatch, match="doesn't hash"):
        images.verify_chain(sidecar(index=INDEX + b" "), CONFIG_ID)
    other = json.dumps({"config": {"digest": "sha256:" + "9" * 64}}).encode()
    with pytest.raises(images.DigestMismatch, match="doesn't list"):
        images.verify_chain(sidecar(manifest=other), CONFIG_ID)
    with pytest.raises(images.DigestMismatch, match="doesn't list"):
        images.verify_chain(sidecar(platform="linux/arm64"), CONFIG_ID)
    with pytest.raises(images.DigestMismatch, match="has config"):
        images.verify_chain(sidecar(), "sha256:" + "8" * 64)
    with pytest.raises(images.BadArchive):
        images.verify_chain({**sidecar(), "schema": "x"}, CONFIG_ID)


# -------------------------------------------------------------------- load --

def test_a_plain_docker_save_of_a_shipped_image_loads(docker, tmp_path):
    r = images.load(archive(tmp_path / "v.tar"), proc.STATE_DIR)
    assert r["ok"] and r["ref"] == REF and r["image_id"] == CONFIG_ID
    assert r["verified_by"] == "images.json" and r["platform"] == "linux/amd64"
    assert docker.tags == [(CONFIG_ID, images.keep_tag(REF))]
    assert images.keep_tag(REF) == f"llmctl/vllm-openai:{sha(INDEX)[7:19]}"
    rec = images.loaded(proc.STATE_DIR)[REF]
    assert rec["image_id"] == CONFIG_ID and rec["archive_manifest"] == SAVED_MANIFEST


def test_the_containerd_store_id_is_the_archives_manifest(docker, tmp_path):
    docker.load_reports = SAVED_MANIFEST
    r = images.load(archive(tmp_path / "v.tar"), proc.STATE_DIR)
    assert r["image_id"] == SAVED_MANIFEST
    assert images.local_image(REF, proc.STATE_DIR) == (SAVED_MANIFEST, SAVED_MANIFEST)


def test_load_refuses_unknown_images_and_unexpected_ids(docker, tmp_path):
    with pytest.raises(images.BadArchive, match="no chain"):
        images.load(archive(tmp_path / "o.tar", config=b"{}"), proc.STATE_DIR)
    assert docker.loads == []                        # refused before loading
    docker.load_reports = "sha256:" + "7" * 64
    with pytest.raises(images.DigestMismatch):
        images.load(archive(tmp_path / "v.tar"), proc.STATE_DIR)
    assert images.loaded(proc.STATE_DIR) == {}


def test_an_unshipped_image_loads_through_its_sidecar(docker, tmp_path, monkeypatch):
    monkeypatch.setattr(containers, "expected_images", lambda: {})
    r = images.load(archive(tmp_path / "s.tar", sidecar=sidecar()), proc.STATE_DIR)
    assert r["verified_by"] == "sidecar" and r["ref"] == REF
    bad = archive(tmp_path / "t.tar", sidecar=sidecar(index=INDEX.replace(b"amd64", b"arm64")))
    with pytest.raises(images.DigestMismatch):
        images.load(bad, proc.STATE_DIR)


def test_cli_load_json_and_exit_codes(docker, tmp_path):
    r = CliRunner().invoke(cli.cli, ["images", "load", str(archive(tmp_path / "v.tar")), "--json"])
    assert r.exit_code == 0, r.output
    out = json.loads(r.stdout)
    assert out["ok"] and out["ref"] == REF and out["image_id"] == CONFIG_ID
    r = CliRunner().invoke(cli.cli, ["images", "load",
                                     str(archive(tmp_path / "o.tar", config=b"{}")), "--json"])
    assert r.exit_code == 3 and json.loads(r.stdout)["error"] == "bad_archive"
    docker.load_reports = "sha256:" + "7" * 64
    r = CliRunner().invoke(cli.cli, ["images", "load", str(tmp_path / "v.tar"), "--json"])
    assert r.exit_code == 4 and json.loads(r.stdout)["error"] == "digest_mismatch"


# --------------------------------------------------------------- resolving --

def test_resolution_prefers_the_ref_then_the_loaded_id_never_a_tag(docker, tmp_path):
    state = proc.STATE_DIR
    assert images.local_image(REF, state) is None
    docker.add(CONFIG_ID)                            # loaded by plain `docker load`
    assert images.local_image(REF, state) == (CONFIG_ID, CONFIG_ID)
    docker.add("sha256:" + "6" * 64, REF)            # pulled by digest
    assert images.local_image(REF, state) == (REF, "sha256:" + "6" * 64)
    docker.add("sha256:" + "4" * 64, "vllm/vllm-openai:v0.29.0")
    assert images.local_image("vllm/vllm-openai@sha256:" + "3" * 64, state) is None


def test_another_platforms_record_is_ignored(docker, tmp_path):
    images._record(proc.STATE_DIR, REF, {"platform": "linux/arm64", "image_id": SAVED_MANIFEST})
    docker.add(SAVED_MANIFEST)
    assert images.local_image(REF, proc.STATE_DIR) is None


def test_run_and_start_use_the_loaded_id(docker, tmp_path, monkeypatch):
    docker.add(CONFIG_ID)
    assert containers.resolve_digest(REF, proc.STATE_DIR) == sha(INDEX)
    calls = {}

    class Popen:
        pid = 4242

        def __init__(self, argv, **kw):
            calls["argv"] = argv
    monkeypatch.setattr(proc, "LOG_DIR", tmp_path / "logs")
    monkeypatch.setattr(proc, "reconcile", list)
    monkeypatch.setattr(proc, "_save", lambda procs: None)
    monkeypatch.setattr(proc, "autostart_add", lambda **kw: None)
    monkeypatch.setattr(proc.subprocess, "Popen", Popen)
    monkeypatch.setattr(containers, "stop_container", lambda n, timeout=20: None)
    monkeypatch.setattr(containers, "has_cdi", lambda: True)
    monkeypatch.setattr(containers, "gpu_arch", lambda index=0: "sm_89")
    model = tmp_path / "models" / "qwen3"
    model.mkdir(parents=True)
    monkeypatch.setattr(registry, "MODELS_DIR", model.parent)
    cmd = backends.build_vllm_cmd("vllm", model, name="qwen", port=8101, ctx_size=4096,
                                  host="127.0.0.1", extra=[])
    rec = proc.start(name="qwen", cmd=cmd, port=8101, model_path=str(model), backend="vllm",
                     runtime={"kind": "docker", "image": REF, "digest": sha(INDEX)})
    argv = calls["argv"]
    assert argv[argv.index("--entrypoint") + 2] == CONFIG_ID
    assert REF not in argv
    assert rec.runtime["image_id"] == CONFIG_ID and rec.runtime["digest"] == sha(INDEX)
    assert "run by ID" in Path(rec.log_file).read_text()
    assert proc.chosen_runtime(rec.runtime) == {"kind": "docker", "image": REF,
                                                "digest": sha(INDEX)}


def test_a_missing_image_points_at_images_load(docker):
    with pytest.raises(containers.ContainerError, match="llmctl images load"):
        containers.resolve_digest(REF, proc.STATE_DIR)


def test_images_lists_and_protects_untagged_images(docker):
    docker.add(CONFIG_ID)
    r = CliRunner().invoke(cli.cli, ["images"])
    assert f"loaded as {CONFIG_ID[7:19]}" in r.output and "prune would delete" in r.output
    r = CliRunner().invoke(cli.cli, ["images", "--protect"])
    assert docker.tags == [(CONFIG_ID, images.keep_tag(REF))] and "now tagged" in r.output
    r = CliRunner().invoke(cli.cli, ["images"])
    assert "prune" not in r.output


# -------------------------------------------------------------------- save --

def test_save_adds_a_sidecar_that_load_accepts(docker, tmp_path, monkeypatch):
    docker.add(CONFIG_ID)
    raw = {REF: INDEX, f"vllm/vllm-openai@{sha(MANIFEST)}": MANIFEST}
    monkeypatch.setattr(images, "_skopeo_raw", lambda ref: raw[ref])
    out = images.save(REF, tmp_path / "v.tar", proc.STATE_DIR)
    assert out["config"] == CONFIG_ID and out["manifest"] == sha(MANIFEST)
    a = images.read_archive(tmp_path / "v.tar")
    assert a["sidecar"]["ref"] == REF
    assert not (tmp_path / "v.tar.partial").exists()
    monkeypatch.setattr(containers, "expected_images", lambda: {})
    assert images.load(tmp_path / "v.tar", proc.STATE_DIR)["verified_by"] == "sidecar"


def test_save_refuses_a_local_image_that_isnt_the_registrys(docker, tmp_path, monkeypatch):
    docker.add(CONFIG_ID)
    docker.saved_config = b'{"other": 1}'
    raw = {REF: INDEX, f"vllm/vllm-openai@{sha(MANIFEST)}": MANIFEST}
    monkeypatch.setattr(images, "_skopeo_raw", lambda ref: raw[ref])
    with pytest.raises(images.DigestMismatch):
        images.save(REF, tmp_path / "v.tar", proc.STATE_DIR)
    assert not (tmp_path / "v.tar").exists() and not (tmp_path / "v.tar.partial").exists()


def test_save_needs_skopeo_a_pin_and_the_image(docker, tmp_path, monkeypatch):
    monkeypatch.setattr(images.shutil, "which", lambda name: None)
    docker.add(CONFIG_ID)
    with pytest.raises(bundle.SourceUnreachable, match="skopeo"):
        images.save(REF, tmp_path / "v.tar", proc.STATE_DIR)
    with pytest.raises(images.BadArchive):
        images.save("vllm/vllm-openai:latest", tmp_path / "v.tar", proc.STATE_DIR)
    r = CliRunner().invoke(cli.cli, ["images", "save", "vllm/vllm-openai@sha256:" + "3" * 64,
                                     "-o", str(tmp_path / "v.tar"), "--json"])
    assert r.exit_code == 6 and json.loads(r.stdout)["error"] == "source_unreachable"


def test_shipped_images_carry_their_platform_ids():
    containers.expected_images.cache_clear()
    for backend, e in containers.expected_images().items():
        assert set(e["by_platform"]) == set(e["platforms"]), backend
        for p in e["by_platform"].values():
            assert re.fullmatch(r"sha256:[0-9a-f]{64}", p["config"])
            assert re.fullmatch(r"sha256:[0-9a-f]{64}", p["manifest"])


"""Offline bundles: manifest checks, install, verify, the CLI contract."""
import hashlib
import json
from pathlib import Path

import pytest
from click.testing import CliRunner

from llm_runner import bundle
from llm_runner import cli
from llm_runner import process as proc


def _model(folder: Path) -> Path:
    folder.mkdir(parents=True)
    (folder / "config.json").write_text('{"architectures": ["Qwen3ForCausalLM"]}')
    (folder / "model.safetensors").write_bytes(b"\x00" * 4096)
    (folder / "tokenizer.json").write_text("{}")
    (folder / "LICENSE").write_text("Apache-2.0")
    return folder


def _bundle(tmp_path: Path, name="qwen3-0.6b", **kw) -> Path:
    b = _model(tmp_path / "bundle")
    data = bundle.build_manifest(b, name=name, model_id="Qwen/Qwen3-0.6B",
                                 revision="abc123", license="apache-2.0", fmt="hf",
                                 license_file="LICENSE", **kw)
    (b / bundle.MANIFEST_FILE).write_text(json.dumps(data))
    return b


def _manifest(b: Path) -> dict:
    return json.loads((b / bundle.MANIFEST_FILE).read_text())


def _rewrite(b: Path, data: dict) -> None:
    (b / bundle.MANIFEST_FILE).write_text(json.dumps(data))


def test_install_places_verified_files_and_provenance(tmp_path):
    b = _bundle(tmp_path)
    (b / "manifest.sig").write_text("signature")     # not listed: left behind
    models = tmp_path / "models"
    r = bundle.install(b, models)
    target = models / "qwen3-0.6b"
    assert r["ok"] and r["path"] == str(target) and r["files_verified"] == 4
    assert sorted(p.name for p in target.iterdir()) == sorted(
        ["config.json", "model.safetensors", "tokenizer.json", "LICENSE",
         bundle.INSTALLED_MANIFEST, bundle.SOURCE_FILE])
    src = json.loads((target / bundle.SOURCE_FILE).read_text())
    raw = (b / bundle.MANIFEST_FILE).read_bytes()
    assert src["source"] == "install" and src["repo_id"] == "Qwen/Qwen3-0.6B"
    assert src["revision"] == "abc123" and src["license"] == "apache-2.0"
    assert src["manifest_sha256"] == hashlib.sha256(raw).hexdigest() == r["manifest_sha256"]
    assert src["sha256"]["model.safetensors"] == hashlib.sha256(b"\x00" * 4096).hexdigest()
    assert (target / bundle.INSTALLED_MANIFEST).read_bytes() == raw
    assert not list(models.glob(".install-*"))


def test_a_changed_file_refuses_the_whole_install(tmp_path):
    b = _bundle(tmp_path)
    (b / "tokenizer.json").write_text("{ }")         # same size? no: one byte more
    (b / "config.json").write_text('{"architectures": ["Qwen3ForCausalLM"]}'.replace("Q", "X"))
    models = tmp_path / "models"
    with pytest.raises(bundle.ChecksumMismatch) as e:
        bundle.install(b, models)
    paths = {m["path"] for m in e.value.details["mismatches"]}
    assert paths == {"tokenizer.json", "config.json"}
    assert e.value.exit_code == 4
    assert not (models / "qwen3-0.6b").exists() and not list(models.glob(".install-*"))


def test_a_missing_file_is_a_mismatch(tmp_path):
    b = _bundle(tmp_path)
    (b / "tokenizer.json").unlink()
    with pytest.raises(bundle.ChecksumMismatch) as e:
        bundle.install(b, tmp_path / "models")
    assert e.value.details["mismatches"] == [
        {"path": "tokenizer.json", "expected": hashlib.sha256(b"{}").hexdigest(),
         "missing": True}]


def test_a_symlinked_file_is_refused(tmp_path):
    b = _bundle(tmp_path)
    (tmp_path / "elsewhere").write_text("{}")
    (b / "tokenizer.json").unlink()
    (b / "tokenizer.json").symlink_to(tmp_path / "elsewhere")
    with pytest.raises(bundle.ChecksumMismatch):
        bundle.install(b, tmp_path / "models")


@pytest.mark.parametrize("mutate, message", [
    (lambda d: d.update(schema="llmctl-manifest/2"), "unknown manifest schema"),
    (lambda d: d["files"][0].update(path="../escape"), "must be relative"),
    (lambda d: d["files"][0].update(path="/etc/passwd"), "must be relative"),
    (lambda d: d["files"][0].update(path=".llmctl-source.json"), "reserves"),
    (lambda d: d["files"].append(dict(d["files"][0])), "listed twice"),
    (lambda d: d["files"][0].update(sha256="ABC"), "64 lowercase hex"),
    (lambda d: d["model"].pop("revision"), "model.revision is required"),
    (lambda d: d["model"].update(format="onnx"), "model.format"),
    (lambda d: d["model"].update(license_file="NOTICE"), "license_file"),
    (lambda d: d.update(name="../x"), "name"),
    (lambda d: d.update(caches=[{"kind": "vllm", "path": "cache/vllm", "gpu_arch": "sm_89"}]),
     "holds none of the files"),
])
def test_bad_manifests_are_refused(tmp_path, mutate, message):
    b = _bundle(tmp_path)
    data = _manifest(b)
    mutate(data)
    _rewrite(b, data)
    with pytest.raises(bundle.BadManifest, match=message) as e:
        bundle.install(b, tmp_path / "models")
    assert e.value.exit_code == 3


def test_unknown_top_level_keys_are_ignored(tmp_path):
    b = _bundle(tmp_path)
    data = _manifest(b)
    data["ss6m_bundle"] = {"id": "x", "built": "2026-10-22"}
    _rewrite(b, data)
    assert bundle.install(b, tmp_path / "models")["ok"]


def test_an_existing_name_is_never_touched_without_replace(tmp_path):
    b = _bundle(tmp_path)
    models = tmp_path / "models"
    bundle.install(b, models)
    marker = models / "qwen3-0.6b" / "keep"
    marker.write_text("old")
    with pytest.raises(bundle.NameConflict) as e:
        bundle.install(b, models)
    assert e.value.exit_code == 5 and marker.exists()
    # Side by side under another name.
    assert bundle.install(b, models, name="qwen3-0.6b-2")["ok"]
    assert marker.exists()


def test_replace_is_refused_while_served_and_swaps_otherwise(tmp_path):
    b = _bundle(tmp_path)
    models = tmp_path / "models"
    bundle.install(b, models)
    target = models / "qwen3-0.6b"
    (target / "keep").write_text("old")
    with pytest.raises(bundle.NameConflict, match="being served"):
        bundle.install(b, models, replace=True, served={str(target)})
    r = bundle.install(b, models, replace=True, served={str(tmp_path / "other")})
    assert r["ok"] and not (target / "keep").exists()
    assert not list(models.glob(".replaced-*"))


def test_verify_finds_tampering(tmp_path):
    models = tmp_path / "models"
    bundle.install(_bundle(tmp_path), models)
    target = models / "qwen3-0.6b"
    assert bundle.verify(target)["files_verified"] == 4
    (target / "model.safetensors").write_bytes(b"\x01" * 4096)
    with pytest.raises(bundle.ChecksumMismatch) as e:
        bundle.verify(target)
    assert [m["path"] for m in e.value.details["mismatches"]] == ["model.safetensors"]


def test_build_manifest_skips_llmctl_files_and_lists_cache_descriptors(tmp_path):
    folder = _model(tmp_path / "m")
    (folder / bundle.SOURCE_FILE).write_text("{}")
    (folder / ".cache").mkdir()
    (folder / ".cache" / "x").write_text("hf leftovers")
    cache = folder / "cache" / "vllm"
    (cache / "torch_compile_cache" / ".hidden").mkdir(parents=True)
    (cache / "torch_compile_cache" / ".hidden" / "k").write_text("kernel")
    (cache / "llmctl-cache.json").write_text(json.dumps(
        {"kind": "vllm", "path": None, "engine": "vllm", "engine_version": "0.29.0",
         "image": "vllm/vllm-openai@sha256:" + "a" * 64, "gpu_arch": "sm_89",
         "model_folder": "m"}))
    data = bundle.build_manifest(folder, name="m", model_id="x/y", revision="r",
                                 license="mit", fmt="hf")
    paths = [f["path"] for f in data["files"]]
    assert "cache/vllm/torch_compile_cache/.hidden/k" in paths
    assert not any(p.startswith(".") for p in paths) and bundle.SOURCE_FILE not in paths
    assert data["caches"] == [{"kind": "vllm", "path": "cache/vllm", "engine": "vllm",
                               "engine_version": "0.29.0",
                               "image": "vllm/vllm-openai@sha256:" + "a" * 64,
                               "gpu_arch": "sm_89"}]


# ---------------------------------------------------------------- the CLI --

@pytest.fixture
def models(tmp_path, monkeypatch):
    d = tmp_path / "models"
    d.mkdir()
    monkeypatch.setattr(cli, "MODELS_DIR", d)
    monkeypatch.setattr(proc, "list_running", list)
    return d


def test_install_json_and_exit_codes(tmp_path, models):
    b = _bundle(tmp_path)
    r = CliRunner().invoke(cli.cli, ["install", str(b), "--json"])
    assert r.exit_code == 0, r.output
    out = json.loads(r.stdout)
    assert out["ok"] and out["name"] == "qwen3-0.6b" and out["files_verified"] == 4
    assert out["cache"] == {"shipped": False, "seeded": False}

    r = CliRunner().invoke(cli.cli, ["install", str(b), "--json"])
    assert r.exit_code == 5 and json.loads(r.stdout)["error"] == "name_conflict"

    (b / "tokenizer.json").write_text("[]")
    r = CliRunner().invoke(cli.cli, ["install", str(b), "--name", "other", "--json"])
    assert r.exit_code == 4
    out = json.loads(r.stdout)
    assert out["error"] == "checksum_mismatch" and out["mismatches"][0]["path"] == "tokenizer.json"

    (b / bundle.MANIFEST_FILE).write_text("{")
    r = CliRunner().invoke(cli.cli, ["install", str(b), "--json"])
    assert r.exit_code == 3 and json.loads(r.stdout)["error"] == "bad_manifest"


def test_install_from_oci_without_oras(tmp_path, models, monkeypatch):
    monkeypatch.setattr(bundle.shutil, "which", lambda name: None)
    r = CliRunner().invoke(cli.cli, ["install", "oci://zot:5000/m@sha256:" + "a" * 64, "--json"])
    assert r.exit_code == 6 and json.loads(r.stdout)["error"] == "source_unreachable"
    assert not list(models.glob(".pull-*"))


def test_install_from_oci_pulls_and_moves(tmp_path, models, monkeypatch):
    b = _bundle(tmp_path)
    calls = []

    def fake_run(cmd, **kw):
        calls.append(cmd)
        dest = Path(cmd[cmd.index("-o") + 1])
        for p in b.iterdir():
            (dest / p.name).write_bytes(p.read_bytes())
        return type("R", (), {"returncode": 0, "stdout": "", "stderr": ""})()
    monkeypatch.setattr(bundle.shutil, "which", lambda name: "/usr/bin/oras")
    monkeypatch.setattr(bundle.subprocess, "run", fake_run)
    ref = "zot:5000/models/qwen@sha256:" + "b" * 64
    r = CliRunner().invoke(cli.cli, ["install", f"oci://{ref}", "--plain-http", "--json"])
    assert r.exit_code == 0, r.output
    assert calls[0][:3] == ["/usr/bin/oras", "pull", ref] and "--plain-http" in calls[0]
    assert (models / "qwen3-0.6b" / "model.safetensors").exists()
    assert not list(models.glob(".pull-*"))


def test_update_refuses_an_installed_model(tmp_path, models, monkeypatch):
    bundle.install(_bundle(tmp_path), models)
    from llm_runner import registry
    monkeypatch.setattr(cli, "discover_models", lambda: registry.discover_models(models))
    monkeypatch.setattr(cli, "_require_online", lambda what: None)
    r = CliRunner().invoke(cli.cli, ["update", "qwen3-0.6b"])
    assert r.exit_code != 0 and "installed from a bundle" in r.output


def test_discovery_skips_an_install_in_progress(tmp_path):
    from llm_runner import registry
    models = tmp_path / "models"
    (models / ".install-x.123").mkdir(parents=True)
    (models / ".install-x.123" / "m.gguf").write_bytes(b"GGUF")
    _model(models / ".install-y.456")
    assert registry.discover_models(models) == {}


def test_verify_cli(tmp_path, models):
    bundle.install(_bundle(tmp_path), models)
    r = CliRunner().invoke(cli.cli, ["verify", str(models / "qwen3-0.6b"), "--json"])
    assert r.exit_code == 0 and json.loads(r.stdout)["files_verified"] == 4
    (models / "qwen3-0.6b" / "LICENSE").write_text("changed")
    r = CliRunner().invoke(cli.cli, ["verify", str(models / "qwen3-0.6b")])
    assert r.exit_code == 4 and "LICENSE" in r.output


def test_manifest_cli_round_trips_through_install(tmp_path, models):
    folder = _model(tmp_path / "src")
    r = CliRunner().invoke(cli.cli, ["manifest", str(folder), "--id", "Qwen/Qwen3-0.6B",
                                     "--revision", "r1", "--license", "apache-2.0",
                                     "--license-file", "LICENSE", "--name", "q"])
    assert r.exit_code == 0, r.output
    r = CliRunner().invoke(cli.cli, ["install", str(folder), "--json"])
    assert r.exit_code == 0 and json.loads(r.stdout)["name"] == "q"

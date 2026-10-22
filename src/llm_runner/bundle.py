"""Offline model bundles: ``llmctl install``, ``verify`` and ``manifest``.

A bundle is a directory holding ``llmctl-manifest.json`` and the files it
lists. The manifest names the model (id, revision, licence) and gives every
file's size and sha256. ``install`` copies only the listed files into
``MODELS_DIR/<name>``, hashing them on the way, and refuses the whole
install if any file doesn't match. Nothing is fetched: a bundle comes from
removable media, or from a local OCI registry through the ``oras`` CLI.

llmctl doesn't verify signatures. Whoever signs a bundle signs the manifest
bytes, and the per-file hashes extend that to every file. The installed
folder keeps a copy of the manifest (``.llmctl-manifest.json``) and records
its sha256 in the provenance sidecar (``.llmctl-source.json``).

Schema 1::

    {
      "schema": "llmctl-manifest/1",
      "name": "qwen3.5-9b",
      "model": {"id": "Qwen/Qwen3.5-9B", "revision": "<commit>", "format": "hf",
                "license": "apache-2.0", "license_file": "LICENSE"},
      "files": [{"path": "config.json", "size": 1534, "sha256": "<hex>"}, ...],
      "caches": [{"kind": "vllm", "path": "cache/vllm", "engine": "vllm",
                  "engine_version": "0.29.0", "image": "repo@sha256:...",
                  "gpu_arch": "sm_89"}]
    }

Unknown top-level keys are ignored, so a bundle builder can add its own.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import stat
import subprocess
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath

SCHEMA = "llmctl-manifest/1"
MANIFEST_FILE = "llmctl-manifest.json"
# Kept in the installed folder: the manifest exactly as installed, and the
# provenance sidecar (shared with `download`).
INSTALLED_MANIFEST = ".llmctl-manifest.json"
SOURCE_FILE = ".llmctl-source.json"
FORMATS = ("hf", "gguf")
CACHE_KINDS = ("vllm",)
NAME_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*")
_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_CHUNK = 8 << 20


class BundleError(Exception):
    """An install, verify or export that can't go ahead. ``exit_code`` and
    ``error`` are part of the CLI contract (docs/compatibility.md)."""
    exit_code = 1
    error = "error"

    def __init__(self, message: str, **details):
        super().__init__(message)
        self.details = details

    def to_json(self) -> dict:
        return {"ok": False, "error": self.error, "message": str(self), **self.details}


class BadManifest(BundleError):
    exit_code = 3
    error = "bad_manifest"


class ChecksumMismatch(BundleError):
    exit_code = 4
    error = "checksum_mismatch"


class NameConflict(BundleError):
    exit_code = 5
    error = "name_conflict"


class SourceUnreachable(BundleError):
    exit_code = 6
    error = "source_unreachable"


@dataclass(frozen=True)
class FileEntry:
    path: str
    size: int
    sha256: str


@dataclass
class Manifest:
    name: str
    model: dict
    files: list[FileEntry]
    caches: list[dict] = field(default_factory=list)
    sha256: str = ""          # of the manifest file's bytes
    raw: bytes = b""


# ------------------------------------------------------------------ parsing --

def safe_relpath(p: object, what: str = "path") -> str:
    """``p`` as a normalised relative POSIX path, or BadManifest. No absolute
    paths, no ``..``, no empty parts, nothing under llmctl's own names."""
    if not isinstance(p, str) or not p or "\\" in p or "\0" in p:
        raise BadManifest(f"{what} {p!r} is not a relative POSIX path")
    pp = PurePosixPath(p)
    if pp.is_absolute() or any(part in ("", ".", "..") for part in p.split("/")):
        raise BadManifest(f"{what} {p!r} must be relative, without '.' or '..'")
    if pp.parts[0].startswith(".llmctl-"):
        raise BadManifest(f"{what} {p!r} uses a name llmctl reserves")
    return str(pp)


def parse_manifest(raw: bytes) -> Manifest:
    try:
        data = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as e:
        raise BadManifest(f"manifest is not valid JSON: {e}") from None
    if not isinstance(data, dict):
        raise BadManifest("manifest must be a JSON object")
    if data.get("schema") != SCHEMA:
        raise BadManifest(f"unknown manifest schema {data.get('schema')!r} "
                          f"(this llmctl reads {SCHEMA!r})")
    name = data.get("name")
    if not isinstance(name, str) or not NAME_RE.fullmatch(name):
        raise BadManifest(f"name {name!r} must be letters, digits, '.', '_' and '-'")
    model = data.get("model")
    if not isinstance(model, dict):
        raise BadManifest("model must be an object")
    for key in ("id", "revision", "license"):
        if not isinstance(model.get(key), str) or not model[key]:
            raise BadManifest(f"model.{key} is required (a non-empty string)")
    if model.get("format") not in FORMATS:
        raise BadManifest(f"model.format must be one of {', '.join(FORMATS)}")

    files_raw = data.get("files")
    if not isinstance(files_raw, list) or not files_raw:
        raise BadManifest("files must be a non-empty list")
    files: list[FileEntry] = []
    seen: set[str] = set()
    for i, f in enumerate(files_raw):
        if not isinstance(f, dict):
            raise BadManifest(f"files[{i}] must be an object")
        path = safe_relpath(f.get("path"), f"files[{i}].path")
        if path in seen:
            raise BadManifest(f"files[{i}].path {path!r} is listed twice")
        seen.add(path)
        size = f.get("size")
        if not isinstance(size, int) or isinstance(size, bool) or size < 0:
            raise BadManifest(f"files[{i}].size must be a non-negative integer")
        digest = f.get("sha256")
        if not isinstance(digest, str) or not _SHA256_RE.fullmatch(digest):
            raise BadManifest(f"files[{i}].sha256 must be 64 lowercase hex digits")
        files.append(FileEntry(path, size, digest))
    # A file can't also be a directory another file lives in.
    for p in seen:
        for parent in PurePosixPath(p).parents:
            if str(parent) in seen:
                raise BadManifest(f"{str(parent)!r} is listed as a file and as a "
                                  f"directory of {p!r}")

    lic = model.get("license_file")
    if lic is not None and lic not in seen:
        raise BadManifest(f"model.license_file {lic!r} is not one of the files")

    caches = data.get("caches") or []
    if not isinstance(caches, list):
        raise BadManifest("caches must be a list")
    for i, c in enumerate(caches):
        if not isinstance(c, dict) or c.get("kind") not in CACHE_KINDS:
            raise BadManifest(f"caches[{i}].kind must be one of {', '.join(CACHE_KINDS)}")
        cpath = safe_relpath(c.get("path"), f"caches[{i}].path")
        if not any(p.startswith(cpath + "/") for p in seen):
            raise BadManifest(f"caches[{i}].path {cpath!r} holds none of the files")
        image = c.get("image")
        if image is not None and (not isinstance(image, str) or "@sha256:" not in image):
            raise BadManifest(f"caches[{i}].image must be pinned by digest (…@sha256:…)")
        if not isinstance(c.get("gpu_arch"), str):
            raise BadManifest(f"caches[{i}].gpu_arch is required (e.g. sm_89)")
    return Manifest(name=name, model=model, files=files, caches=caches,
                    sha256=hashlib.sha256(raw).hexdigest(), raw=raw)


def load_manifest(bundle_dir: Path) -> Manifest:
    path = bundle_dir / MANIFEST_FILE
    try:
        raw = path.read_bytes()
    except FileNotFoundError:
        raise BadManifest(f"no {MANIFEST_FILE} in {bundle_dir}") from None
    except OSError as e:
        raise BadManifest(f"can't read {path}: {e}") from None
    return parse_manifest(raw)


# ------------------------------------------------------------------ hashing --

def _copy_hashing(src: Path, dst: Path) -> tuple[int, str]:
    h = hashlib.sha256()
    n = 0
    with open(src, "rb") as fi, open(dst, "wb") as fo:
        while chunk := fi.read(_CHUNK):
            h.update(chunk)
            fo.write(chunk)
            n += len(chunk)
    return n, h.hexdigest()


def hash_file(path: Path) -> tuple[int, str]:
    h = hashlib.sha256()
    n = 0
    with open(path, "rb") as f:
        while chunk := f.read(_CHUNK):
            h.update(chunk)
            n += len(chunk)
    return n, h.hexdigest()


def _regular_file(path: Path) -> bool:
    try:
        return stat.S_ISREG(os.lstat(path).st_mode)
    except OSError:
        return False


# ------------------------------------------------------------------ install --

def _served(target: Path, served: set[str]) -> bool:
    for s in served:
        try:
            p = Path(s).resolve()
        except OSError:
            continue
        if p == target or p.is_relative_to(target):
            return True
    return False


def install(bundle_dir: Path, models_dir: Path, *, name: str | None = None,
            replace: bool = False, served: set[str] | None = None,
            move: bool = False, progress=None) -> dict:
    """Verify ``bundle_dir`` and place it at ``models_dir/<name>``.

    Only the listed files are placed. A missing file, a size or a sha256
    that doesn't match refuses the whole install, and nothing is left behind.
    An existing ``name`` is never touched unless ``replace``; even then a
    folder that's being served (``served``: model paths of running models)
    is refused. ``move`` renames files out of ``bundle_dir`` instead of
    copying them (for a temporary pull). Returns the JSON result."""
    m = load_manifest(bundle_dir)
    name = name or m.name
    if not NAME_RE.fullmatch(name):
        raise NameConflict(f"--name {name!r} must be letters, digits, '.', '_' and '-'",
                           name=name)
    models_dir = models_dir.resolve()
    target = models_dir / name
    if target.exists() or target.is_symlink():
        if not replace:
            raise NameConflict(f"{target} already exists. Install side by side under "
                               "another --name, or pass --replace.",
                               name=name, path=str(target))
        if _served(target.resolve(), served or set()):
            raise NameConflict(f"{target} is being served; stop it before --replace.",
                               name=name, path=str(target))

    models_dir.mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.mkdtemp(prefix=f".install-{name}.", dir=models_dir))
    try:
        mismatches: list[dict] = []
        hashes: dict[str, str] = {}
        total = 0
        for i, f in enumerate(m.files):
            src = bundle_dir / f.path
            if progress:
                progress(i, len(m.files), f)
            if not _regular_file(src):
                mismatches.append({"path": f.path, "expected": f.sha256, "missing": True})
                continue
            size = src.stat().st_size
            if size != f.size:
                mismatches.append({"path": f.path, "expected": f.sha256,
                                   "expected_size": f.size, "actual_size": size})
                continue
            dst = tmp / f.path
            dst.parent.mkdir(parents=True, exist_ok=True)
            if move:
                n, digest = hash_file(src)
            else:
                n, digest = _copy_hashing(src, dst)
            if digest != f.sha256 or n != f.size:
                mismatches.append({"path": f.path, "expected": f.sha256, "actual": digest})
                continue
            if move:
                os.replace(src, dst)
            hashes[f.path] = digest
            total += n
        if mismatches:
            raise ChecksumMismatch(
                f"{len(mismatches)} of {len(m.files)} files don't match the manifest; "
                "nothing was installed.", mismatches=mismatches)

        (tmp / INSTALLED_MANIFEST).write_bytes(m.raw)
        source = {
            "source": "install",
            "repo_id": m.model["id"],
            "revision": m.model["revision"],
            "license": m.model["license"],
            "license_file": m.model.get("license_file"),
            "format": m.model["format"],
            "files": sorted(hashes),
            "sha256": hashes,
            "manifest_sha256": m.sha256,
            "manifest_name": m.name,
            "caches": m.caches,
            "installed_at": int(time.time()),
        }
        (tmp / SOURCE_FILE).write_text(json.dumps(source, indent=2) + "\n")
        tmp.chmod(0o755)

        old: Path | None = None
        if target.exists() or target.is_symlink():
            old = models_dir / f".replaced-{name}.{os.getpid()}"
            os.replace(target, old)
        os.replace(tmp, target)
        if old is not None:
            shutil.rmtree(old, ignore_errors=True) if old.is_dir() else old.unlink()
    except BaseException:
        shutil.rmtree(tmp, ignore_errors=True)
        raise

    return {"ok": True, "name": name, "path": str(target),
            "manifest_sha256": m.sha256, "files_verified": len(hashes), "bytes": total,
            "model": {k: m.model.get(k) for k in ("id", "revision", "format", "license")},
            "caches": m.caches}


# ---------------------------------------------------------------------- OCI --

def pull_oci(ref: str, dest: Path, *, plain_http: bool = False) -> Path:
    """``oras pull`` ``ref`` (without its ``oci://``) into ``dest``. The
    artifact's files are the bundle directory as pushed with ``oras push``."""
    oras = shutil.which("oras")
    if oras is None:
        raise SourceUnreachable("`oras` is not on PATH; install the ORAS CLI to "
                                "install from an OCI registry, or pass a bundle directory.",
                                ref=ref)
    if "@sha256:" not in ref:
        raise SourceUnreachable(f"{ref} is not pinned by digest (…@sha256:…)", ref=ref)
    cmd = [oras, "pull", ref, "-o", str(dest)]
    if plain_http:
        cmd.append("--plain-http")
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise SourceUnreachable(f"oras pull {ref} failed: {(r.stderr or r.stdout).strip()}",
                                ref=ref)
    return dest


# ------------------------------------------------------------------- verify --

def read_source(folder: Path) -> dict:
    try:
        data = json.loads((folder / SOURCE_FILE).read_text())
    except (OSError, json.JSONDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


def verify(folder: Path) -> dict:
    """Re-hash an installed folder against its sidecar."""
    src = read_source(folder)
    hashes = src.get("sha256")
    if src.get("source") != "install" or not isinstance(hashes, dict):
        raise BadManifest(f"{folder} wasn't installed from a bundle (no recorded "
                          "checksums to verify against)")
    mismatches = []
    for path, expected in sorted(hashes.items()):
        p = folder / path
        if not _regular_file(p):
            mismatches.append({"path": path, "expected": expected, "missing": True})
            continue
        _, digest = hash_file(p)
        if digest != expected:
            mismatches.append({"path": path, "expected": expected, "actual": digest})
    man = folder / INSTALLED_MANIFEST
    man_ok = man.is_file() and hashlib.sha256(man.read_bytes()).hexdigest() == \
        src.get("manifest_sha256")
    if not man_ok:
        mismatches.append({"path": INSTALLED_MANIFEST,
                           "expected": src.get("manifest_sha256"),
                           **({"actual": hashlib.sha256(man.read_bytes()).hexdigest()}
                              if man.is_file() else {"missing": True})})
    if mismatches:
        raise ChecksumMismatch(f"{len(mismatches)} files in {folder} don't match what "
                               "was installed", mismatches=mismatches, path=str(folder))
    return {"ok": True, "path": str(folder), "files_verified": len(hashes),
            "manifest_sha256": src.get("manifest_sha256")}


# ----------------------------------------------------------------- manifest --

def find_cache_descriptors(folder: Path) -> list[dict]:
    """``caches`` entries for the subtrees ``cache export`` wrote under ``folder``."""
    from llm_runner.containers import CACHE_DESCRIPTOR
    out = []
    for d in sorted(folder.rglob(CACHE_DESCRIPTOR)):
        rel = d.parent.relative_to(folder)
        if rel.parts and rel.parts[0].startswith("."):
            continue
        try:
            entry = json.loads(d.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        entry = {k: v for k, v in entry.items() if v is not None and k != "model_folder"}
        entry["path"] = rel.as_posix()
        out.append(entry)
    return out

# What `download` and llmctl leave in a model folder that isn't the model.
_NOT_MODEL = {SOURCE_FILE, INSTALLED_MANIFEST, MANIFEST_FILE}


def build_manifest(folder: Path, *, name: str, model_id: str, revision: str,
                   license: str, fmt: str, license_file: str | None = None,
                   caches: list[dict] | None = None, progress=None) -> dict:
    """The manifest for an existing folder: every regular file under it
    except llmctl's own files and hidden top-level entries (``.cache`` …).
    Hidden names deeper down are kept: compile caches use them."""
    paths = []
    for p in sorted(folder.rglob("*")):
        rel = p.relative_to(folder)
        if rel.parts[0].startswith(".") or (len(rel.parts) == 1 and rel.name in _NOT_MODEL):
            continue
        if p.is_symlink():
            raise BadManifest(f"{p} is a symlink; a bundle holds regular files only")
        if p.is_file():
            paths.append(rel)
    files = []
    for i, rel in enumerate(paths):
        if progress:
            progress(i, len(paths), rel)
        size, digest = hash_file(folder / rel)
        files.append({"path": rel.as_posix(), "size": size, "sha256": digest})
    model = {"id": model_id, "revision": revision, "format": fmt, "license": license}
    if license_file:
        model["license_file"] = license_file
    data: dict = {"schema": SCHEMA, "name": name, "model": model, "files": files}
    if caches is None:
        caches = find_cache_descriptors(folder)
    if caches:
        data["caches"] = caches
    # Round-trip through the parser so a bad argument fails here, not at install.
    parse_manifest(json.dumps(data).encode())
    return data

"""Container images delivered as files: ``llmctl images save``/``load``.

llmctl pins images by their registry (multi-arch index) digest. A host that
gets the image from a ``docker save`` archive loses that name: ``docker
load`` leaves it with no tag and no RepoDigest, so ``repo@sha256:<index>``
doesn't resolve. What survives is the image ID, which the registry links to
the index digest:

    sha256(index) = index digest -> the index lists the platform manifest
    -> sha256(manifest) = its digest -> the manifest names the config digest

On the classic store (overlay2) the loaded ID is the config digest. On the
containerd store it is the digest of the manifest *in the archive*, which a
classic host's ``docker save`` rewrites (uncompressed layers), so only the
config digest inside the archive is on the registry chain.

llmctl finds the ID for an index digest from:

- ``images.json`` ``by_platform``, for the images it ships (the config and
  registry manifest digest per platform), and
- the record ``llmctl images load`` writes after checking an archive
  (``<state>/images.json``), for any other image or store.

and runs ``docker run <ID>``. The runtime record keeps the index digest, so
compile caches keyed on it still match. There is no tag fallback; the
``llmctl/<repo>:<digest12>`` tag that ``load`` adds only keeps ``docker image
prune`` from deleting an image with no other tag.
"""
from __future__ import annotations

import base64
import hashlib
import io
import json
import os
import platform
import re
import shutil
import subprocess
import tarfile
import time
from pathlib import Path

from llm_runner import bundle, containers

SIDECAR = "llmctl-image.json"
SIDECAR_SCHEMA = "llmctl-image/1"
LOADED_SCHEMA = "llmctl-loaded/1"
_DIGEST_RE = re.compile(r"sha256:[0-9a-f]{64}")
_ARCH = {"x86_64": "amd64", "amd64": "amd64", "aarch64": "arm64", "arm64": "arm64"}


class BadArchive(bundle.BadManifest):
    error = "bad_archive"


class DigestMismatch(bundle.ChecksumMismatch):
    error = "digest_mismatch"


def host_platform() -> str:
    m = platform.machine().lower()
    return f"linux/{_ARCH.get(m, m)}"


def _sha(data: bytes) -> str:
    return "sha256:" + hashlib.sha256(data).hexdigest()


def keep_tag(ref: str) -> str:
    """The tag that protects ``ref``'s image from ``docker image prune``."""
    repo, _, digest = ref.partition("@")
    name = repo.rsplit("/", 1)[-1].split(":")[0].lower()
    return f"llmctl/{name}:{digest.removeprefix('sha256:')[:12]}"


# ------------------------------------------------------------ the record --

def _loaded_file(state_dir: Path) -> Path:
    return state_dir / "images.json"


def loaded(state_dir: Path) -> dict[str, dict]:
    """What ``images load`` recorded: ref -> {platform, config, manifest, image_id, …}."""
    try:
        data = json.loads(_loaded_file(state_dir).read_text())
    except (OSError, json.JSONDecodeError):
        return {}
    imgs = data.get("images") if isinstance(data, dict) else None
    return imgs if isinstance(imgs, dict) else {}


def _record(state_dir: Path, ref: str, entry: dict) -> None:
    imgs = loaded(state_dir)
    imgs[ref] = entry
    path = _loaded_file(state_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps({"schema": LOADED_SCHEMA, "images": imgs}, indent=2) + "\n")
    os.replace(tmp, path)


def shipped_entry(ref: str) -> dict | None:
    """The ``images.json`` entry pinned to the same index digest as ``ref``."""
    digest = containers.pinned_digest(ref)
    for entry in containers.expected_images().values():
        if digest and containers.pinned_digest(entry.get("image") or "") == digest:
            return entry
    return None


def candidate_ids(ref: str, state_dir: Path, plat: str | None = None) -> list[str]:
    """Image IDs ``ref`` may have been loaded as on this host, most specific
    first: the recorded load, then the shipped config and manifest digests."""
    plat = plat or host_platform()
    out = []
    rec = loaded(state_dir).get(ref)
    if rec and rec.get("platform") == plat and rec.get("image_id"):
        out.append(rec["image_id"])
    entry = shipped_entry(ref)
    if entry:
        p = (entry.get("by_platform") or {}).get(plat) or {}
        out += [p[k] for k in ("config", "manifest") if p.get(k)]
    return list(dict.fromkeys(i for i in out if _DIGEST_RE.fullmatch(i)))


def local_image(ref: str, state_dir: Path) -> tuple[str, str] | None:
    """What to hand ``docker run`` for ``ref``, and that image's ID.

    ``ref`` itself when it resolves (pulled by digest); else an image ID
    linked to ``ref``'s index digest (loaded from an archive); else None.
    A tag is never consulted."""
    info = containers.inspect_image(ref)
    if info is not None:
        return ref, info.get("Id", "")
    if not containers.is_pinned(ref):
        return None
    for image_id in candidate_ids(ref, state_dir):
        info = containers.inspect_image(image_id)
        if info is not None and info.get("Id") == image_id:
            return image_id, image_id
    return None


def dangling(ref: str, state_dir: Path) -> bool:
    """Whether ``ref``'s image has no tag, so ``docker image prune`` deletes it."""
    found = local_image(ref, state_dir)
    info = containers.inspect_image(found[1]) if found else None
    return info is not None and not info.get("RepoTags")


def protect(ref: str, image_id: str) -> str:
    tag = keep_tag(ref)
    subprocess.run([containers.docker_bin() or "docker", "tag", image_id, tag],
                   check=True, capture_output=True)
    return tag


# ------------------------------------------------------------ archives --

def read_archive(path: Path) -> dict:
    """The image a ``docker save`` archive holds: its config digest, the
    digest of the manifest it carries (OCI layout), and the llmctl sidecar
    if ``images save`` wrote one. Reads headers only."""
    try:
        tf = tarfile.open(path, "r:")
    except (OSError, tarfile.TarError) as e:
        raise BadArchive(f"{path} is not a docker save archive: {e}") from None
    with tf:
        def read(name: str) -> bytes | None:
            try:
                f = tf.extractfile(name)
            except KeyError:
                return None
            return f.read() if f else None
        try:
            manifest = json.loads(read("manifest.json") or b"null")
            index = json.loads(read("index.json") or b"null")
            sidecar = json.loads(read(SIDECAR) or b"null")
        except json.JSONDecodeError as e:
            raise BadArchive(f"{path}: unreadable metadata: {e}") from None
    if not isinstance(manifest, list) or len(manifest) != 1:
        raise BadArchive(f"{path}: expected one image in manifest.json, found "
                         f"{len(manifest) if isinstance(manifest, list) else 'none'}")
    cfg = str(manifest[0].get("Config") or "")
    m = re.search(r"([0-9a-f]{64})(?:\.json)?$", cfg)
    if not m:
        raise BadArchive(f"{path}: manifest.json has no config digest")
    mdigest = None
    if isinstance(index, dict):
        ds = [d.get("digest") for d in index.get("manifests") or []]
        if len(ds) == 1 and _DIGEST_RE.fullmatch(ds[0] or ""):
            mdigest = ds[0]
    return {"config": "sha256:" + m.group(1), "manifest": mdigest,
            "sidecar": sidecar if isinstance(sidecar, dict) else None}


def verify_chain(sidecar: dict, config: str) -> dict:
    """Check an ``images save`` sidecar: the index hashes to the ref's digest,
    lists the platform's manifest, which hashes to its digest and names
    ``config``. Returns {ref, platform, manifest}."""
    if sidecar.get("schema") != SIDECAR_SCHEMA:
        raise BadArchive(f"{SIDECAR}: unknown schema {sidecar.get('schema')!r}")
    ref, plat = sidecar.get("ref") or "", sidecar.get("platform") or ""
    digest = containers.pinned_digest(ref)
    if not digest:
        raise BadArchive(f"{SIDECAR}: ref {ref!r} is not pinned by digest")
    try:
        index = base64.b64decode(sidecar["index"], validate=True)
        manifest = base64.b64decode(sidecar["manifest"], validate=True)
        listed = json.loads(index)
        man = json.loads(manifest)
    except (KeyError, ValueError, TypeError) as e:
        raise BadArchive(f"{SIDECAR}: unreadable index or manifest: {e}") from None
    if _sha(index) != digest:
        raise DigestMismatch(f"the index in {SIDECAR} doesn't hash to {digest}",
                             expected=digest, actual=_sha(index))
    mdigest = _sha(manifest)
    if "manifests" in listed:
        want = [d.get("digest") for d in listed["manifests"] if _plat(d) == plat]
        if mdigest not in want:
            raise DigestMismatch(f"the index of {ref} doesn't list manifest {mdigest} "
                                 f"for {plat}", expected=want, actual=mdigest)
    elif mdigest != digest:      # the ref names a single-platform manifest
        raise DigestMismatch(f"{ref} is a manifest, but {SIDECAR} carries another one",
                             expected=digest, actual=mdigest)
    named = (man.get("config") or {}).get("digest")
    if named != config:
        raise DigestMismatch(f"{ref} ({plat}) has config {named}; the archive holds {config}",
                             expected=named, actual=config)
    return {"ref": ref, "platform": plat, "manifest": mdigest}


def _plat(desc: dict) -> str:
    p = desc.get("platform") or {}
    return f"{p.get('os')}/{p.get('architecture')}"


def match_shipped(config: str) -> dict | None:
    """The shipped image and platform whose config digest is ``config``."""
    for entry in containers.expected_images().values():
        for plat, p in (entry.get("by_platform") or {}).items():
            if p.get("config") == config:
                return {"ref": entry["image"], "platform": plat, "manifest": p.get("manifest")}
    return None


def load(path: Path, state_dir: Path) -> dict:
    """Check an archive's image against its chain, ``docker load`` it,
    confirm the ID, record it and protect it from prune."""
    d = containers.docker_bin()
    if d is None:
        raise bundle.SourceUnreachable("the docker CLI is not on PATH")
    arc = read_archive(path)
    if arc["sidecar"] is not None:
        chain = verify_chain(arc["sidecar"], arc["config"])
        how = "sidecar"
    else:
        chain = match_shipped(arc["config"])
        how = "images.json"
        if chain is None:
            raise BadArchive(
                f"{path}: no chain names config {arc['config']}. It's not an image this "
                "llmctl ships; make the archive with `llmctl images save` so it carries one.",
                config=arc["config"])
    accepted = [i for i in (arc["config"], arc["manifest"]) if i]
    r = subprocess.run([d, "load", "-i", str(path)], capture_output=True, text=True)
    if r.returncode != 0:
        raise bundle.BundleError(f"docker load failed: {(r.stderr or r.stdout).strip()}")
    ids = re.findall(r"Loaded image ID: (sha256:[0-9a-f]{64})", r.stdout)
    if ids:
        image_id = ids[-1]
    else:
        image_id = next((i for i in accepted
                         if (containers.inspect_image(i) or {}).get("Id") == i), None)
    if image_id not in accepted:
        raise DigestMismatch(f"docker loaded {image_id or 'an unknown image'}, expected one of "
                             f"{', '.join(accepted)}", expected=accepted, actual=image_id)
    tag = protect(chain["ref"], image_id)
    entry = {"platform": chain["platform"], "config": arc["config"],
             "manifest": chain["manifest"], "archive_manifest": arc["manifest"],
             "image_id": image_id, "file": str(path.resolve()), "verified_by": how,
             "loaded_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "tag": tag}
    _record(state_dir, chain["ref"], entry)
    return {"ok": True, "ref": chain["ref"], "image_id": image_id, "tag": tag,
            "platform": chain["platform"], "config": arc["config"],
            "manifest": chain["manifest"], "verified_by": how,
            "host_platform": host_platform()}


def _skopeo_raw(ref: str) -> bytes:
    sk = shutil.which("skopeo")
    if sk is None:
        raise bundle.SourceUnreachable("`images save` needs skopeo to read the registry's "
                                       "index and manifest (e.g. `pacman -S skopeo`, "
                                       "`apt install skopeo`)")
    r = subprocess.run([sk, "inspect", "--raw", f"docker://{ref}"], capture_output=True)
    if r.returncode != 0:
        raise bundle.SourceUnreachable(f"skopeo inspect {ref}: "
                                       f"{r.stderr.decode(errors='replace').strip()}")
    return r.stdout


def save(ref: str, out: Path, state_dir: Path, plat: str | None = None) -> dict:
    """``docker save`` ``ref``'s local image to ``out`` with a sidecar that
    carries the registry's index and manifest, so ``load`` can check the
    chain offline."""
    plat = plat or host_platform()
    digest = containers.pinned_digest(ref)
    if not digest:
        raise BadArchive(f"{ref} is not pinned by digest (repo@sha256:…)")
    found = local_image(ref, state_dir)
    if found is None:
        raise bundle.SourceUnreachable(f"{ref} is not on this host")
    index = _skopeo_raw(ref)
    if _sha(index) != digest:
        raise DigestMismatch(f"the registry's index for {ref} doesn't hash to its digest",
                             expected=digest, actual=_sha(index))
    listed = json.loads(index)
    if "manifests" in listed:
        want = [m["digest"] for m in listed["manifests"] if _plat(m) == plat]
        if not want:
            raise BadArchive(f"{ref} has no {plat} image")
        manifest = _skopeo_raw(f"{ref.partition('@')[0]}@{want[0]}")
    else:
        manifest = index
    config = (json.loads(manifest).get("config") or {}).get("digest")
    out = out.expanduser()
    partial = out.with_name(out.name + ".partial")
    d = containers.docker_bin() or "docker"
    try:
        r = subprocess.run([d, "save", "-o", str(partial), found[0]],
                           capture_output=True, text=True)
        if r.returncode != 0:
            raise bundle.BundleError(f"docker save failed: {r.stderr.strip()}")
        held = read_archive(partial)["config"]
        if held != config:
            raise DigestMismatch(f"the local image of {ref} has config {held}, but its "
                                 f"{plat} manifest names {config}", expected=config, actual=held)
        sidecar = {"schema": SIDECAR_SCHEMA, "ref": ref, "platform": plat,
                   "index": base64.b64encode(index).decode(),
                   "manifest": base64.b64encode(manifest).decode()}
        body = json.dumps(sidecar, indent=2).encode()
        with tarfile.open(partial, "a:") as tf:
            info = tarfile.TarInfo(SIDECAR)
            info.size, info.mtime, info.mode = len(body), int(time.time()), 0o644
            tf.addfile(info, io.BytesIO(body))
        os.replace(partial, out)
    finally:
        partial.unlink(missing_ok=True)
    return {"ok": True, "ref": ref, "platform": plat, "config": config,
            "manifest": _sha(manifest), "file": str(out), "size": out.stat().st_size}

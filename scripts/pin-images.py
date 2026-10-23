#!/usr/bin/env python3
"""Fill (or check) ``by_platform`` in src/llm_runner/images.json.

For every image, pinned by its index digest, this fetches the index and each
listed platform's manifest from the registry with ``skopeo inspect --raw``,
checks that every hash in the chain matches, and records per platform the
manifest digest and the config digest (the image ID ``docker load`` gives on
the classic store). An offline host can then recognise a loaded image by
its ID without trusting anything but the llmctl package.

    scripts/pin-images.py           # rewrite images.json
    scripts/pin-images.py --check   # fail if images.json is stale
"""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

IMAGES = Path(__file__).resolve().parent.parent / "src" / "llm_runner" / "images.json"


def raw(ref: str) -> bytes:
    return subprocess.run(["skopeo", "inspect", "--raw", f"docker://{ref}"],
                          check=True, capture_output=True).stdout


def by_platform(ref: str, platforms: list[str]) -> dict:
    repo, _, digest = ref.partition("@")
    index = raw(ref)
    if "sha256:" + hashlib.sha256(index).hexdigest() != digest:
        sys.exit(f"{ref}: the registry's index doesn't hash to its digest")
    listed = {f"{m['platform']['os']}/{m['platform']['architecture']}": m["digest"]
              for m in json.loads(index).get("manifests", [])
              if m.get("platform", {}).get("os") != "unknown"}
    out = {}
    for plat in platforms:
        if plat not in listed:
            sys.exit(f"{ref}: the index has no {plat} manifest")
        mdigest = listed[plat]
        manifest = raw(f"{repo}@{mdigest}")
        if "sha256:" + hashlib.sha256(manifest).hexdigest() != mdigest:
            sys.exit(f"{repo}@{mdigest}: manifest doesn't hash to its digest")
        out[plat] = {"manifest": mdigest, "config": json.loads(manifest)["config"]["digest"]}
    return out


def main() -> None:
    check = "--check" in sys.argv[1:]
    data = json.loads(IMAGES.read_text())
    stale = []
    for backend, entry in data["images"].items():
        want = by_platform(entry["image"], entry["platforms"])
        if entry.get("by_platform") != want:
            stale.append(backend)
            entry["by_platform"] = want
        print(f"{backend}: " + ", ".join(f"{p} config {v['config'][7:19]}"
                                        for p, v in want.items()), file=sys.stderr)
    if check:
        if stale:
            sys.exit(f"images.json by_platform is stale for {', '.join(stale)}; "
                     "run scripts/pin-images.py")
        return
    IMAGES.write_text(json.dumps(data, indent=2) + "\n")


if __name__ == "__main__":
    main()

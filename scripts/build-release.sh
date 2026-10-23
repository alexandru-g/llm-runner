#!/usr/bin/env bash
# Build the release artifacts for the version in pyproject.toml:
#
#   dist/release-<v>/
#     llm_runner-<v>-py3-none-any.whl, llm_runner-<v>.tar.gz
#     systemone_proxy-<pv>-py3-none-any.whl
#     llm-runner-<v>-wheelhouse-py3XX-<arch>.tar.gz   (x86_64, aarch64; 3.11, 3.12)
#     images.json      the container images this version was tested with
#     sha256sums.txt
#
# A wheelhouse holds the wheels for llm-runner and every dependency for one
# Python version and architecture, plus install.sh, so an air-gapped host
# installs with `./install.sh --offline .`. Dependencies are the versions
# pinned in uv.lock, with their hashes checked on download. systemone-proxy
# (decision endpoints) pins its own versions and ships as a separate wheel.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"
VERSION="$(sed -n 's/^version = "\(.*\)"/\1/p' pyproject.toml | head -1)"
OUT="$REPO/dist/release-$VERSION"
PYTHONS=(3.11 3.12)
ARCHES=(x86_64 aarch64)

say() { printf '\033[1;34m==>\033[0m %s\n' "$*" >&2; }

rm -rf "$OUT"
mkdir -p "$OUT"
say "Checking images.json against the registry"
python3 scripts/pin-images.py --check
say "Building llm-runner $VERSION"
uv build --quiet --out-dir "$OUT" .
uv build --quiet --wheel --out-dir "$OUT" packages/systemone-proxy
cp src/llm_runner/images.json "$OUT/images.json"

work="$(mktemp -d)"
trap 'rm -rf "$work"' EXIT
uv export --quiet --frozen --no-dev --no-emit-project -o "$work/requirements.txt"

for py in "${PYTHONS[@]}"; do
    for arch in "${ARCHES[@]}"; do
        tag="py${py/./}-$arch"
        name="llm-runner-$VERSION-wheelhouse-$tag"
        dir="$work/$name"
        say "Wheelhouse $tag"
        mkdir -p "$dir"
        uvx --quiet --from pip pip download --quiet --disable-pip-version-check \
            --only-binary=:all: --implementation cp --python-version "$py" \
            --platform "manylinux_2_17_$arch" --platform "manylinux2014_$arch" \
            --platform "manylinux_2_28_$arch" --platform "manylinux_2_34_$arch" \
            --platform "linux_$arch" \
            --require-hashes -r "$work/requirements.txt" -d "$dir"
        cp "$OUT"/llm_runner-"$VERSION"-py3-none-any.whl "$dir/"
        cp install.sh src/llm_runner/images.json LICENSE "$work/requirements.txt" "$dir/"
        cat > "$dir/README.txt" <<EOF
llm-runner $VERSION offline install (Python $py, $arch)

  ./install.sh --offline .

Needs Python $py and uv (or pip) on the host. Puts llmctl in
~/.local/bin. requirements.txt is the pinned dependency set, with
hashes. images.json lists the container images this version was
tested with; pull or load them by digest before
'llmctl config runtime <backend> docker'.
EOF
        tar -C "$work" -czf "$OUT/$name.tar.gz" "$name"
    done
done

(cd "$OUT" && sha256sum -- *.whl *.tar.gz images.json > sha256sums.txt)
say "Done: $OUT"
ls -la "$OUT" >&2

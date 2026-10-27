#!/usr/bin/env bash
# Install `llmctl` globally so it can be run from any directory.
#
# Prefers `uv tool install` (fast, isolated). Falls back to `pipx`, then to a
# plain `python -m venv` + symlink. In all cases the binary lands in a
# directory that should be on your PATH (typically ~/.local/bin).

set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_DIR"

# --offline DIR: install from a release wheelhouse (wheels for llm-runner and
# every dependency), with no package index. Used on air-gapped hosts.
OFFLINE_DIR=""
if [[ "${1:-}" == "--offline" ]]; then
    [[ -n "${2:-}" && -d "$2" ]] || { echo "usage: $0 [--offline WHEELHOUSE_DIR]" >&2; exit 2; }
    OFFLINE_DIR="$(cd "$2" && pwd)"
fi

say() { printf '\033[1;34m==>\033[0m %s\n' "$*" >&2; }
warn() { printf '\033[1;33m!!\033[0m %s\n' "$*" >&2; }

# What to install: this checkout, or (offline) the wheel in the wheelhouse.
src_args() {  # $1 = package name, $2 = source dir
    if [[ -n "$OFFLINE_DIR" ]]; then
        echo "--offline --no-index --find-links $OFFLINE_DIR $1"
    else
        echo "--from $2 $1"
    fi
}

install_with_uv() {
    say "Installing via 'uv tool install' from ${OFFLINE_DIR:-$REPO_DIR}"
    # shellcheck disable=SC2046
    uv tool install --force --reinstall $(src_args llm-runner "$REPO_DIR")
    BIN_DIR="$(uv tool dir --bin 2>/dev/null || echo "$HOME/.local/bin")"
    echo "$BIN_DIR"
}

install_with_pipx() {
    say "Installing via 'pipx install' from ${OFFLINE_DIR:-$REPO_DIR}"
    if [[ -n "$OFFLINE_DIR" ]]; then
        pipx install --force --pip-args="--no-index --find-links $OFFLINE_DIR" llm-runner
    else
        pipx install --force "$REPO_DIR"
    fi
    echo "$HOME/.local/bin"
}

install_with_venv() {
    local venv="$HOME/.local/share/llm-runner/venv"
    say "Installing via python venv at $venv"
    python3 -m venv "$venv"
    if [[ -n "$OFFLINE_DIR" ]]; then
        "$venv/bin/pip" install --quiet --upgrade --no-index --find-links "$OFFLINE_DIR" llm-runner
    else
        "$venv/bin/pip" install --quiet --upgrade pip
        "$venv/bin/pip" install --quiet --upgrade "$REPO_DIR"
    fi
    mkdir -p "$HOME/.local/bin"
    ln -sf "$venv/bin/llmctl" "$HOME/.local/bin/llmctl"
    echo "$HOME/.local/bin"
}

if command -v uv >/dev/null 2>&1; then
    BIN_DIR="$(install_with_uv)"
elif command -v pipx >/dev/null 2>&1; then
    BIN_DIR="$(install_with_pipx)"
else
    warn "Neither 'uv' nor 'pipx' found — falling back to a plain venv."
    BIN_DIR="$(install_with_venv)"
fi

say "Installed. Binary should be at: $BIN_DIR/llmctl"

# systemone-proxy: the decision endpoint `llmctl decide` starts. Its own
# package (packages/systemone-proxy), installed the same way.
PROXY_DIR="$REPO_DIR/packages/systemone-proxy"
if [[ -n "$OFFLINE_DIR" ]] && ! ls "$OFFLINE_DIR"/systemone_proxy-*.whl >/dev/null 2>&1; then
    say "No systemone-proxy wheel in the wheelhouse; skipping it"
elif [[ -n "$OFFLINE_DIR" ]] && command -v uv >/dev/null 2>&1; then
    say "Installing systemone-proxy from the wheelhouse"
    # shellcheck disable=SC2046
    uv tool install --force --reinstall $(src_args systemone-proxy "$PROXY_DIR") >&2
elif [[ -n "$OFFLINE_DIR" ]]; then
    warn "Offline systemone-proxy install needs uv; skipping it"
elif command -v uv >/dev/null 2>&1; then
    say "Installing systemone-proxy via 'uv tool install'"
    uv tool install --force --reinstall --from "$PROXY_DIR" systemone-proxy >&2
elif command -v pipx >/dev/null 2>&1; then
    say "Installing systemone-proxy via 'pipx install'"
    pipx install --force "$PROXY_DIR" >&2
else
    venv="$HOME/.local/share/llm-runner/systemone-proxy-venv"
    say "Installing systemone-proxy via python venv at $venv"
    python3 -m venv "$venv"
    "$venv/bin/pip" install --quiet --upgrade pip
    "$venv/bin/pip" install --quiet --upgrade "$PROXY_DIR"
    ln -sf "$venv/bin/systemone-proxy" "$HOME/.local/bin/systemone-proxy"
fi

if ! command -v llmctl >/dev/null 2>&1; then
    warn "'llmctl' is not on your PATH yet."
    warn "Add this to your shell rc and re-source it:"
    warn "    export PATH=\"$BIN_DIR:\$PATH\""
else
    say "Verify: $(command -v llmctl)"
    llmctl --help | head -n 5 || true
fi

# A running gateway keeps the code it started with, so restart it on the new
# one (deploy.sh runs this script on the remote). Models are separate servers
# and keep running; a gateway that isn't running stays stopped.
if [[ -x "$BIN_DIR/llmctl" ]]; then
    say "Restarting the gateway on the new code (if it is running)"
    "$BIN_DIR/llmctl" gateway restart >&2 \
        || warn "Gateway restart failed; run 'llmctl gateway restart' and check 'llmctl logs gateway'."
fi

cat <<EOF

Done. Next steps:
  1. Make sure 'llama-server' is on PATH (see README).
  2. Either set LLM_RUNNER_MODELS_DIR=/path/to/models, or place GGUFs under
     ~/.local/share/llm-runner/models/ following the layout in the README.
  3. Try: llmctl list
EOF

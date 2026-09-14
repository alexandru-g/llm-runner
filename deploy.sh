#!/usr/bin/env bash
# Deploy llm-runner to a prod server with rsync.
#
# Syncs the repo (code only — GGUF models are excluded) to a remote host over
# SSH, then runs ./install.sh on the remote so `llmctl` is (re)installed.
#
# Usage:
#   ./deploy.sh user@host                 # deploy to ~/llm-runner on the host
#   ./deploy.sh user@host /opt/llm-runner # deploy to a specific remote dir
#   DEPLOY_HOST=user@host ./deploy.sh     # host via env instead of arg
#   ./deploy.sh --dry-run user@host       # show what would change, copy nothing
#   ./deploy.sh --no-install user@host    # sync only, skip remote install
#
# Env overrides:
#   DEPLOY_HOST   default remote (user@host or ~/.ssh/config alias)
#   DEPLOY_DIR    default remote target dir (default: llm-runner, i.e. ~/llm-runner)

set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_DIR"

say()  { printf '\033[1;34m==>\033[0m %s\n' "$*" >&2; }
warn() { printf '\033[1;33m!!\033[0m %s\n'  "$*" >&2; }
die()  { printf '\033[1;31mxx\033[0m %s\n'  "$*" >&2; exit 1; }

DRY_RUN=0
RUN_INSTALL=1
POSITIONAL=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        -n|--dry-run)    DRY_RUN=1; shift ;;
        --no-install)    RUN_INSTALL=0; shift ;;
        -h|--help)
            # Print the contiguous header comment (skip the shebang on line 1).
            awk 'NR==1{next} /^#/{sub(/^# ?/,""); print; next} {exit}' "${BASH_SOURCE[0]}"
            exit 0 ;;
        -*) die "Unknown option: $1" ;;
        *)  POSITIONAL+=("$1"); shift ;;
    esac
done

HOST="${POSITIONAL[0]:-${DEPLOY_HOST:-}}"
REMOTE_DIR="${POSITIONAL[1]:-${DEPLOY_DIR:-llm-runner}}"

[[ -n "$HOST" ]] || die "No destination. Pass user@host (or set DEPLOY_HOST). See --help."
command -v rsync >/dev/null 2>&1 || die "rsync not found on this machine."

# Files that must never go to prod: VCS, local venvs/state, build junk, and the
# (potentially huge) GGUF models — those live in LLM_RUNNER_MODELS_DIR on prod.
EXCLUDES=(
    --exclude '.git/'
    --exclude '.venv/'
    --exclude '.state/'
    --exclude '__pycache__/'
    --exclude '*.pyc'
    --exclude '*.egg-info/'
    --exclude 'dist/'
    --exclude 'build/'
    --exclude '.python-version'
    --exclude 'models/'
)

RSYNC_OPTS=(-az --delete --human-readable "${EXCLUDES[@]}")
if [[ "$DRY_RUN" -eq 1 ]]; then
    RSYNC_OPTS+=(--dry-run --itemize-changes)
    say "DRY RUN — no files will be copied and install will be skipped."
fi

say "Syncing $REPO_DIR/  ->  $HOST:$REMOTE_DIR/"

# Ensure the remote dir exists (skip in dry-run so we touch nothing remote).
if [[ "$DRY_RUN" -eq 0 ]]; then
    ssh "$HOST" "mkdir -p -- '$REMOTE_DIR'"
fi

# Trailing slash on the source: copy the *contents* of REPO_DIR into REMOTE_DIR.
rsync "${RSYNC_OPTS[@]}" "$REPO_DIR/" "$HOST:$REMOTE_DIR/"

if [[ "$DRY_RUN" -eq 1 ]]; then
    say "Dry run complete."
    exit 0
fi

if [[ "$RUN_INSTALL" -eq 1 ]]; then
    say "Running ./install.sh on $HOST"
    # -t for a TTY so install.sh's PATH checks behave; bash -lc to load profile
    # (so uv/pipx on ~/.local/bin are found).
    ssh -t "$HOST" "cd -- '$REMOTE_DIR' && chmod +x install.sh && bash -lc './install.sh'"
else
    warn "Skipped remote install (--no-install). Run it yourself with:"
    warn "    ssh $HOST 'cd $REMOTE_DIR && ./install.sh'"
fi

say "Done."
cat >&2 <<EOF

Deployed to $HOST:$REMOTE_DIR
Next on prod:
  - Ensure 'llama-server' is on PATH (see README install section 2).
  - Point at your models: export LLM_RUNNER_MODELS_DIR=/path/to/models
    (or persist with: llmctl config models-dir /path/to/models)
  - Verify: ssh $HOST 'llmctl list'
EOF

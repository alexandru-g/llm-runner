# Compatibility promise

llm-runner follows [semantic versioning](https://semver.org) from 1.0.0. This
page lists what the version number protects, so scripts and clients can pin
against it.

- **Patch** releases (1.0.x) fix bugs. Nothing listed here changes.
- **Minor** releases (1.x.0) add things: new commands, options, JSON keys,
  manifest or serving-file keys, backends. Existing ones keep working.
- **Major** releases (2.0.0) may remove or change what's listed here. The
  release notes name every such change, and a removal is announced as
  deprecated (with a warning on use) in at least one minor release first.

## What's covered

### Command line

- The commands and their options as documented in `llmctl <command> --help`.
- Exit codes:

  | Code | Meaning | Commands |
  |---|---|---|
  | 0 | success (or nothing to do) | all |
  | 1 | other failure (for `apply`: a start failed) | all |
  | 2 | usage error | all |
  | 3 | bad manifest / invalid serving files | `install`, `verify`, `cache export`, `apply` |
  | 4 | checksum or size mismatch, or a listed file missing | `install`, `verify` |
  | 5 | name conflict | `install` |
  | 6 | source unreachable (`oras` missing, pull failed) | `install` |

- `--json` output of `install`, `verify`, `cache export`, `apply` and
  `images`: the keys documented in the README keep their names, types and
  meaning. New keys may appear; parse leniently.

### File formats

- **Bundle manifest** `llmctl-manifest.json`, schema `llmctl-manifest/1`
  (README, "Install a model bundle"). llmctl 1.x reads schema 1. A new
  schema gets a new name; 1.x keeps reading `/1`.
- **Serving files** for `llmctl apply`, `version: 1` (README, "Serving
  files"). New keys may be added; existing keys keep their meaning.
- **`images.json`**, schema `llmctl-images/1`: per backend, `image` (pinned
  by digest), `tag`, `engine_version`, `entrypoint`, `platforms`, `license`.
  The images themselves change between releases; that's what the file is
  for.
- **Provenance sidecar** `.llmctl-source.json` of an installed model: the
  keys `source`, `repo_id`, `revision`, `license`, `format`, `files`,
  `sha256`, `manifest_sha256`, `caches`.

### Gateway HTTP API

- The OpenAI-compatible routes the gateway forwards (`/v1/...`), the
  per-model prefixes `/m/<name>/...`, and its error format
  (`{"error": {"code", "message", "type"}}`) with HTTP status codes.
- `/healthz` (the gateway is up) and `/readyz` (200 only when every
  autostart model is ready).
- In `GET /v1/models`, each model's `llmctl` object: `backend`, `api`,
  `port`, `path_prefix`, `status`, `group`, `model_path`, `runtime`,
  `image`, `image_digest`, `context`, `vision`, `embeddings`, `decision`,
  `parallel`, `node`, `local`, and `nodes` (each with `node`, `priority`,
  `local`, `status`, `port`, `context`, `vision`, `runtime`, `image`,
  `image_digest`).
- API keys: `Authorization: Bearer <key>` as configured with `llmctl
  gateway auth`.

### Cluster protocol

Nodes and the registry speak protocol version 1 (`PROTOCOL` in
`cluster.py`). A 1.x node works with a 1.x registry. A change that breaks
that bumps the protocol version and is a major release.

## What's not covered

- Human-readable output: tables, messages, wording, column order. Use
  `--json` where it exists.
- Log formats, of llmctl and of the engines.
- State files under the state directory (`running.json`, `autostart.json`,
  `profiles.json`, `cache/`) and `config.json`. Use the commands that manage
  them.
- The exact engine arguments llmctl generates, and the images and engine
  versions it was tested with (`images.json`, the README's install pins).
- Python APIs inside the `llm_runner` package. It's a CLI, not a library.
- The interactive menus and their profiles.
- `systemone-proxy`, which has its own version.

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
  | 3 | bad manifest / invalid serving files / bad image archive | `install`, `verify`, `cache export`, `apply`, `images load`, `images save` |
  | 4 | checksum or size mismatch, a listed file missing, an image digest mismatch | `install`, `verify`, `images load`, `images save` |
  | 5 | name conflict | `install` |
  | 6 | source unreachable (`oras`/`skopeo`/docker missing, pull failed, image not on the host) | `install`, `images load`, `images save` |

  `llmctl ps --json` prints `{"models": [...]}`, each with `name`, `pid`,
  `port`, `backend`, `started_at`, `group`, `model_path`, `runtime`,
  `state`, `restarts`, `next_restart_in`, `last_error`, (since 1.5)
  `owner`, and (since 1.6) `groups` for a model with several groups
  (`group` is then the first).

- `--json` output of `install`, `verify`, `cache export`, `apply`,
  `images`, `images load`, `images save` and (since 1.2) `ps`: the keys documented in the README keep their names, types and
  meaning. New keys may appear; parse leniently.

### File formats

- **Bundle manifest** `llmctl-manifest.json`, schema `llmctl-manifest/1`
  (README, "Install a model bundle"). llmctl 1.x reads schema 1. A new
  schema gets a new name; 1.x keeps reading `/1`.
- **Serving files** for `llmctl apply`, `version: 1` (README, "Serving
  files"). New keys may be added; existing keys keep their meaning
  (`restart` and `restart_max` since 1.2).
- **`images.json`**, schema `llmctl-images/1`: per backend, `image` (pinned
  by digest), `tag`, `engine_version`, `entrypoint`, `platforms`, `license`,
  and (since 1.1) `by_platform` (per platform, `manifest` and `config`
  digests).
- **Image archive sidecar** `llmctl-image.json`, schema `llmctl-image/1`:
  `ref`, `platform`, and the registry's `index` and `manifest`, base64.
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
  autostart model is ready; 503 when no model is declared or running).
- Since 1.2: a call for a model that is configured on the node but down
  answers **503 `model_unavailable`** (with `Retry-After` when a restart is
  due); a stream whose engine dies ends with an SSE event
  `data: {"error": {"code": "model_unavailable", …}}`. Before 1.2 these
  were 404/502 and a cut stream; 1.2 treats the old behaviour as a bug.
  Since 1.3 the same 503 also covers a model that `llmctl apply` is
  restarting, which used to get a 404 for a moment, and a local engine
  that answers its own 503 while it loads.
- Since 1.3: `llmctl events --json` lines. Each has `ts`, `model`, `event`
  and `detail`, and `pid`, `port`, `by`, `reason`, `log` and (since 1.5)
  `owner` when known.
  The event names are those in the README table. New events and keys may
  be added in a minor release.
- `X-Request-Id` on every reply (the client's, when valid, else a new one),
  and W3C `traceparent` continued to the engines.
- Metrics (since 1.2): `/metrics`, `/metrics/sd` (HTTP SD JSON) and
  `/metrics/backends/<model>`, the `llmctl_*` metric names, types and
  labels listed in the README ("Monitoring and operations"). New metrics
  and label values may appear; the engines' own metrics are theirs.
- In `GET /v1/models`, each model's `llmctl` object: `backend`, `api`,
  `port`, `path_prefix`, `status` (`ready`, `loading`, `down`, and since 1.2
  `restarting`, `crashed`), `restarts` (1.2), `group`, `model_path`, `runtime`,
  `image`, `image_digest`, `context`, `vision`, `embeddings`, `decision`,
  `parallel`, `node`, `local`, and `nodes` (each with `node`, `priority`,
  `local`, `status`, `port`, `context`, `vision`, `runtime`, `image`,
  `image_digest`), and since 1.1 `image_id` (in both). Since 1.6 a model in
  several groups also has `groups` (all of them; `group` stays the first).
- API keys: `Authorization: Bearer <key>` as configured with `llmctl
  gateway auth`.

### Cluster protocol

Nodes and the registry speak protocol version 1 (`PROTOCOL` in
`cluster.py`). A 1.x node works with a 1.x registry. A change that breaks
that bumps the protocol version and is a major release.

## What's not covered

- Human-readable output: tables, messages, wording, column order. Use
  `--json` where it exists.
- Log formats, of llmctl and of the engines, except the gateway's JSON
  lines (since 1.2), whose keys are covered: `ts`, `level`, `node`, `event`,
  `msg`, and for `event: "request"` the keys listed in the README. They
  never contain prompt or reply text.
- State files under the state directory (`running.json`, `autostart.json`,
  `profiles.json`, `images.json`, `supervisor.json`, `events.jsonl`,
  `holds.json`, `cache/`) and `config.json`. Use `llmctl ps --json` and `llmctl events
  --json`. Use the commands that manage
  them.
- The exact engine arguments llmctl generates, and the images and engine
  versions it was tested with (`images.json`, the README's install pins).
- Python APIs inside the `llm_runner` package. It's a CLI, not a library.
- The interactive menus and their profiles.
- `systemone-proxy`, which has its own version.

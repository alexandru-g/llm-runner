# Changelog

All notable changes to llm-runner (`llmctl`). The format follows
[Keep a Changelog](https://keepachangelog.com), and versions follow
[semantic versioning](https://semver.org); what the version protects is in
[docs/compatibility.md](docs/compatibility.md).

## [Unreleased]

## [1.6.0] - 2026-10-28

### Added

- **Several groups per model:** `llmctl run --group assistant,gemma-4-26b-a4b-it`
  (or `group: [assistant, gemma-4-26b-a4b-it]` in a serving file) makes one
  running model answer to each name. Routing, `/m/<group>/`, `/v1/models`
  group entries and spill-over work for every one. `group` stays the first
  name in `ps --json` and `/v1/models`, and `groups` lists them all. Nodes
  before 1.6 route by the first only.
- **Gateway traces into Phoenix:** `llmctl config gateway-otlp URL
  --protocol protobuf` exports OTLP/HTTP protobuf (hand-encoded, standard
  library only), and `--header` adds headers such as Phoenix's API key.
  `llmctl config gateway-otlp-project KEY PROJECT` puts the spans of calls
  made with an API key in a Phoenix project (`openinference.project.name`).
  `'*'` covers every other call, and the project `none` exports none of
  them.
- Gateway spans now carry llama-server's prefill and decode timings
  (`llmctl.prefill_ms`, `prefill_tokens`, `cached_tokens`, `decode_ms`,
  `decode_tokens_per_s`). The README lists every span attribute; there is
  still no content. A test checks that across chat, streams, tools,
  embeddings, `/v1/systemone` and failover.

## [1.5.0] - 2026-10-28

### Added

- **Decision models:** Xevn-4B and iota-decide-4b v0.3 profiles in
  systemone-proxy. Both are ported from the vendors' code with golden
  prompt token ids; iota needs a locally converted GGUF.
- **Natively served decision GGUFs** (llama.cpp b11361+, e.g. vllm-d3-mini):
  `llmctl decide` runs the proxy in native mode, forwarding to llama-server's
  own `/v1/systemone` under the same contract. `systemone-proxy detect`
  prints `native` for such files.
- `llmctl run --batch-size/--ubatch-size` now reach llama-server for every
  model (they used to apply to `--embeddings` only). For non-causal GGUFs
  (vllm-d3) both default to the context size, since each prompt has to fit
  one batch.
- **Owner tags:** `llmctl run --owner TAG`, `$LLMCTL_OWNER`, or `owner:` in
  serving files records who started a model. It shows in `ps` (an OWNER
  column, and `owner` in `--json`), in events, and in `/v1/models`. Restarts,
  `restore`, the supervisor and decision endpoints keep it, and `apply`
  doesn't restart a model over a different tag. An unset owner writes no key,
  so older llmctl versions can still read the state files. llmctl now also
  ignores unknown keys in `running.json`.

### Changed

- **H2O-Lightning-4B profile follows v1.4.1:** a score's confidence is now
  its most likely level's probability, as in the vendor's shim since v1.4.1.
  The prompt, temperatures and yes/no floor are unchanged since v1.2.3. Use
  v1.4.1 weights: mradermacher/h2o-lightning-4b-GGUF holds older ones, and
  its answers differ by up to 0.6 from v1.4.1's. h2oai's own GGUF is served
  natively (llama.cpp b11773+), which skips the 0.801 yes/no floor.

### Fixed

- **Gateway auth with a single bind address:** the bind address now counts as
  local, like loopback, because llmctl's own calls come from it. Before, with
  `gateway auth on`, local networks narrowed to loopback and `config bind
  172.19.0.1`, `llmctl gateway` got 401. Containers on that bridge still need
  a key.
- `install.sh` without uv or pipx (plain venv) now upgrades an existing
  install: offline installs from a newer wheelhouse used to keep the old
  version.

## [1.4.1] - 2026-10-27

### Fixed

- **`llmctl config bind` to a single non-loopback address** (e.g. a Docker
  bridge address, 172.19.0.1): llmctl dialled 127.0.0.1 to reach its own
  listeners, which refuse it on such a bind. `llmctl gateway` reported "not
  answering" and exited 1, and `gateway restart` failed. Worse, the gateway
  couldn't reach local models, and health waits, the supervisor and port
  checks failed too. They now dial the bind address. Loopback is still used
  for 0.0.0.0, `::` and `localhost`. A decision endpoint restarted after a
  bind change follows its llama-server to the new address.

## [1.4.0] - 2026-10-27

### Added

- **DNS for containers.** `llmctl config runtime <backend> docker --dns
  IP[,IP]`, `run --dns`, and `dns:` in serving files (a list or one string)
  pass `--dns` to `docker run`. On a network without egress, `--dns
  127.0.0.1` stops Docker's embedded resolver from answering outside names,
  which would otherwise be a DNS channel out, while container names still
  resolve. The default is unchanged (Docker's resolver). The list is kept
  in each launch's spec, so restarts and `restore` keep it, and changing it
  in a serving file restarts the model.

## [1.3.0] - 2026-10-25

Planned restarts and model events, from the M0.4 runbook runs.

### Fixed

- `llmctl apply` restarting a model no longer makes the gateway answer 404
  `model_not_found` between stopping the old instance and starting the new
  one. The model is held for the restart, and calls get 503
  `model_unavailable` with `Retry-After: 30` until the new instance is
  ready. A held model stays in the autostart manifest, and the supervisor
  leaves it alone.
- A local engine that answers 503 while it loads (llama-server's "Loading
  model") now gives callers the gateway's 503 `model_unavailable` with
  `Retry-After: 10`, as a model that isn't listening yet does.
- The supervisor takes an instance started by someone else (`apply`, `run`)
  as a new one. Before, if the swap happened between two of its checks,
  it counted the new instance as the old, ready one. That skipped the new
  instance's `ready` event, and a failed new instance would have been
  restarted as if it had crashed.

### Added

- `llmctl events` lists more than the supervisor's actions. It now also
  shows `started`, `ready`, `start_failed` and `stopped` from `run`,
  `apply` and `stop`. Events carry `pid`, `port`, `by`, `reason` and `log`
  keys where they apply. A start that fails names its log in a
  `start_failed` event, including one under `apply --no-wait`, which the
  supervisor reports.
- `llmctl logs NAME` shows the last log of a model that isn't running, for
  example one whose start failed.

## [1.2.0] - 2026-10-23

Monitoring and operations: metrics, crash recovery, structured logs and
traces, for a production host watched from another machine.

### Added

- **Metrics.** `GET /metrics` on every gateway, in Prometheus text format,
  without a new dependency:
  - calls by status and error code, latency and time-to-first-token
    histograms;
  - tokens from `usage`, in-flight and queued calls;
  - spills, failovers, node up, each model's state and its restarts.
  `GET /metrics/backends/<model>` re-exports each engine's own metrics with
  `model`, `node` and `backend` labels, and `GET /metrics/sd` lists them for
  Prometheus HTTP service discovery. llama-server now runs with `--metrics`.
- `llmctl gateway metrics-token`: a scrape-only token, so a monitoring host
  needs no key that can run models. Without one, scrapes follow the
  API-key rules of `/v1`.
- **The gateway restarts crashed models.** A model in the autostart
  manifest that dies after it was ready is restarted after 2, 4, 8 … s
  (capped at 60 s). After 5 restarts in 10 minutes it is left down as
  `crashed`. Per model: `run --restart on-failure|never`, `--restart-max N`,
  and `restart:`/`restart_max:` in serving files.
- `llmctl ps` shows STATE and RESTARTS and takes `--json`. `llmctl events
  [-f]` follows the supervisor, and `/v1/models` lists down models as
  `restarting` or `crashed`.
- `scripts/chaos-test.py MODEL`: kills a model mid-stream and checks the
  client gets an error within 10 s and the model serves again within 120 s.
  Measured on ws: error at once, serving again in 5 s (llama.cpp on the
  host) and 7 s (in Docker).
- **JSON logs.** `llmctl config gateway-log json` writes one object per
  line. Each call logs its request id, trace id, model, status, timings and
  tokens, and never prompt or reply text, in any mode.
- **Request ids.** `X-Request-Id` is kept from the client (or made), sent
  to the engine and returned on every reply.
- **Traces.** The gateway continues the client's W3C trace and gives each
  engine a `traceparent`. `llmctl config gateway-otlp URL` exports gateway
  spans over OTLP/HTTP (off by default).

### Changed

- A call for a model that is configured but down answers **503
  `model_unavailable`** with `Retry-After`, instead of 404 `model_not_found`
  (process gone) or 502 `upstream_unavailable` (port refusing). This counts
  as a bug fix within 1.x (docs/compatibility.md).
- A stream whose engine dies ends with an SSE error event and a clean end
  of the response, instead of a cut connection.
- `llmctl stop` takes a model out of the autostart manifest before stopping
  it, so the gateway never restarts a model that was stopped on purpose.

## [1.1.1] - 2026-10-23

### Fixed

- The embedding-mode guess matched name hints as substrings, so a chat model
  whose folder name merely contained "gte", "e5", "bge" or "minilm" (e.g.
  `qwen3.5-4b-imgtest`) started as an embedding model: vLLM `--runner
  pooling`, which also changed its compile-cache key, so a shipped cache
  missed. Hints now match whole name tokens ("embed" as a prefix), and an HF
  folder whose `config.json` declares a text-generating architecture is
  never guessed as an embedding model.

## [1.1.0] - 2026-10-23

Images delivered as files run offline, on either Docker image store.

### Added

- **Images loaded from archives run.** After `docker load`, an image saved
  by digest has no name, so `repo@sha256:<index>` didn't resolve and the
  start failed. llmctl now runs the pinned ref by the image ID the registry
  links it to (index → platform manifest → config). The identity stays the
  index digest, so `ps`, `apply` and compile caches (`cache rule: digest`)
  are unchanged. There is no tag fallback.
- `images.json` lists each shipped image's manifest and config digest per
  platform (`by_platform`), checked against the registry with
  `scripts/pin-images.py`.
- `llmctl images load FILE` checks an archive against that chain, loads it,
  confirms the ID (the config digest on the classic store, the archive's
  manifest digest on the containerd store) and records it.
- `llmctl images save REF|BACKEND -o FILE` writes a `docker save` archive
  that carries the registry's index and manifest (`llmctl-image.json`), so
  images llmctl doesn't ship can be checked offline too. Needs skopeo.
- `llmctl images --protect`: tags images that have no tag
  (`llmctl/<repo>:<digest12>`), since `docker image prune` deletes those;
  `images` warns about them and `images load` tags what it loads.
- `image_id` in `/v1/models` (and per node), the model log and the gateway
  log: what docker ran.

## [1.0.0] - 2026-10-22

The first stable release: llmctl can now run a production host from pinned
images, offline bundles and a serving file, with no network access.

### Added

- **Container runtime.** vLLM, SGLang and llama.cpp servers can run in Docker
  from an image pinned by digest: `llmctl config runtime <backend> docker`, or
  `run --runtime docker [--image repo@sha256:…]`. GPUs come through CDI (or
  `--gpus`), the model is mounted read-only, the port is published on the bind
  address only, and the container runs as the calling user. The image and
  digest show in `ps`, `/v1/models`, the model log and the gateway log. Host
  processes remain the default.
- `llmctl images` and `images.json`: the container images this release was
  tested with, pinned by digest (vLLM v0.29.0, llama.cpp server-cuda b11804).
- `run --gpu all|N[,N…]` picks GPUs, for containers and host processes.
- **Offline bundles.** `llmctl install <dir|oci://…@sha256:…>` verifies every
  file's size and sha256 against `llmctl-manifest.json` (schema
  `llmctl-manifest/1`) and installs atomically, or refuses with nothing left
  behind. Versions install side by side; `--replace` is explicit and refused
  while the model is served. `oci://` pulls with the `oras` CLI.
  `llmctl manifest` writes a manifest for a folder, `llmctl verify` re-hashes an
  installed model, and `list` shows each model's revision and licence.
- **Compile caches.** Container starts keep vLLM's torch.compile, Triton and
  FlashInfer caches per model, mounted at `/cache`. `llmctl cache export` puts
  a warm cache into a bundle; an installed model's first start seeds from it
  when the image digest and GPU architecture match (logged as `cache rule:
  digest`). Qwen3-0.6B: torch.compile 17.4 s → 0.2 s, +0.5 GiB of KV cache.
- **Serving files.** `llmctl apply <file.yaml…>` starts, restarts or (with
  `--prune`) stops models to match YAML files, and leaves unchanged ones
  alone. Fragments merge; every error names the file and line; `--dry-run`
  and `--json` for automation. A rollback is reapplying the previous file.
- `--json` results and fixed exit codes (3 bad manifest, 4 checksum mismatch,
  5 name conflict, 6 source unreachable) for `install`, `verify`,
  `cache export` and `apply`.
- `run --kv-cache-mem SIZE` sets vLLM's `--kv-cache-memory-bytes`, and llmctl
  warns when a hybrid model (Qwen3.5/3.6) starts on vLLM without a fixed KV
  size.
- `install.sh --offline <wheelhouse>`; each release ships wheelhouses for
  Python 3.11 and 3.12 on x86_64 and aarch64 (`scripts/build-release.sh`).
- [docs/compatibility.md](docs/compatibility.md): what 1.x keeps stable.

### Changed

- The gateway's listen backlog is 128 instead of 5, so bursts of new
  connections are queued rather than dropped.
- `restore` waits up to a minute for the Docker daemon when the autostart
  manifest holds container models.
- PyYAML is a new dependency (for `apply`).

### Earlier

Changes before 1.0.0 were not tracked in a changelog; see the git history.
Highlights of the 0.x series: the SGLang backend and uv-installed engines
(0.12), the all-models gateway, clusters through a registry, model groups,
API keys, offline mode, telemetry off, `/healthz` and `/readyz`, launch
profiles, decision endpoints, vLLM and decider support.

[1.6.0]: https://github.com/alexandru-g/llm-runner/releases/tag/v1.6.0
[1.5.0]: https://github.com/alexandru-g/llm-runner/releases/tag/v1.5.0
[1.4.1]: https://github.com/alexandru-g/llm-runner/releases/tag/v1.4.1
[1.4.0]: https://github.com/alexandru-g/llm-runner/releases/tag/v1.4.0
[1.3.0]: https://github.com/alexandru-g/llm-runner/releases/tag/v1.3.0
[1.2.0]: https://github.com/alexandru-g/llm-runner/releases/tag/v1.2.0
[1.1.1]: https://github.com/alexandru-g/llm-runner/releases/tag/v1.1.1
[1.1.0]: https://github.com/alexandru-g/llm-runner/releases/tag/v1.1.0
[1.0.0]: https://github.com/alexandru-g/llm-runner/releases/tag/v1.0.0

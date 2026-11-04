# Changelog

All notable changes to llm-runner (`llmctl`). The format follows
[Keep a Changelog](https://keepachangelog.com), and versions follow
[semantic versioning](https://semver.org); what the version protects is in
[docs/compatibility.md](docs/compatibility.md).

## [Unreleased]

## [1.12.0] - 2026-11-04

### Changed

- **Omni on a Jetson Orin:** when the weights don't fit, `llmctl run` now also turns
  off pinned memory for the offload and tiles the VAE, so Qwen-Image-2512 and
  Edit-2511 start with no extra flags. Other models and hosts launch as before.
  An offload flag in `-X` still replaces all of this.

### Fixed

- **vLLM and Omni models read `down` while they load:** they don't listen until
  they've loaded, so the gateway couldn't tell a load from a failure. A running
  model that hasn't answered yet now reads `loading`, as llama-server does.
  Once it has answered, a refused connection is still `down` at once.
- **A client leaving could keep a model busy for nobody:** the gateway only learned
  a client had left when it next wrote to it, so a request still queued, or a reply
  with nothing to send, kept going and held the model's slot. It now watches the
  client while it waits and hangs up on the model at once, across nodes too, without
  failing over. A request dropped before any reply logs status 499 and code
  `client_left`. A client that half-closes its connection after sending a request
  counts as having left.

### Known issues

- **Gemma 4 can lose its answer after a stray `<channel|>`** on llama-server b11805
  with `--reasoning-budget 0`: the reply stops after a plan, with `finish_reason:
  stop`. See [docs/known-issues.md](docs/known-issues.md).

## [1.11.0] - 2026-11-04

### Added

- **`llmctl chat`:** chat in the terminal with a running model anywhere in the
  cluster, through this machine's gateway. Text models stream their replies
  (reasoning dimmed) and keep the conversation, with `/clear`, `/retry`, `/system`
  and `/set` for any request field. Image pipelines take each message as a prompt
  and save the images on this machine, opening them in the default viewer when run
  from a terminal (`--open`, `--no-open`), and `--image` or `/image` turns prompts into
  edits. With no name, it uses the only model that can chat, or opens a picker
  listing each model's nodes; `NAME@NODE` keeps it to one node. Decision, embedding
  and CV models are left out, and naming one says why. `-m` or piped stdin sends one
  message and exits, with the answer alone on stdout.
- **`llmctl setup vllm` and `setup omni` on a Jetson Orin with JetPack 6:** they
  unpack NVIDIA's `cuda-compat-orin`, which runs PyPI's CUDA 13 wheels on the
  CUDA 12.6 driver with no change to the host, install the versions tested there
  (vLLM 0.33, vLLM-Omni 0.32) and finish with a real operation on the GPU. `llmctl
  run` adds the libraries for vLLM and Omni only. `setup vllm --runtime docker`
  builds the image there from a Dockerfile shipped with llmctl and pins the runtime
  to it. Other hosts keep their versions.

### Fixed

- A model too busy to answer its health check no longer reads as down. vLLM-Omni
  stops answering for about 2.5 s at the end of each image, so `/v1/models`,
  `/readyz` and the metrics flipped to `down`, `llmctl ps` to `starting`, a spurious
  `ready` event was logged, and the time counted towards clearing restarts started
  over. A model that has answered stays ready through up to 60 s of timed-out checks.
  A refused connection still reads as down at once.
- vLLM and Omni are sized from `/proc/meminfo` on a Jetson, where `nvidia-smi`
  reports no memory and vLLM otherwise took its default share of the whole board.
- A `~/.cache/vllm` the user can't write, left by a run as root, no longer stops
  vLLM and Omni at their first compile: they use a cache folder of their own.

## [1.10.0] - 2026-11-02

### Added

- **`llmctl setup`:** with no argument, one row per backend with the version
  found on this host and the one llmctl was tested with. `llmctl setup
  BACKEND` shows a plan and runs it once confirmed (`--yes`, `--dry-run`):
  vllm and sglang as uv tools and vLLM-Omni and ONNX Runtime as venvs under
  `~/.venvs`, at the versions in the new `engines.json` (onnx: the GPU build
  on an NVIDIA host, CPU otherwise or with `--cpu`, a Jetson's wheel index
  with `--index-url`); triton's image, and vllm, sglang and llama's with
  `--runtime docker`, pulled by digest or loaded with `--archive`, plus the
  runtime setting. An engine at another version is left alone unless
  `--upgrade`; `--find-links` installs offline. llama.cpp and decider are
  checked only.
- `images.json` has `triton-igpu` (NGC's Triton 26.02 for Jetson), which a
  Jetson takes instead of `triton` without any configuration.

## [1.9.0] - 2026-11-02

### Added

- **`llmctl speedtest`:** measures a running model on its own port. Text
  models report time to first token, prefill and decode tokens/s, and drafter
  acceptance. Embedding models report inputs and tokens/s, decision endpoints
  report decisions/s, and image pipelines report seconds per image. Prompts
  are fresh random text sized to the requested token counts, so no prompt
  cache helps. Several names, or several models ticked with space in the
  picker, are tested one after another with the same options and compared in
  one table per prompt size. With no name, it opens the picker and then an
  options menu with profiles (`last`, `default`, `quick`, `long`, `load`, and
  your own). It asks before loading a model that is already serving requests
  (`--force` skips that), and `--json` prints the results as JSON.
- **Computer-vision models (backends `onnx` and `triton`):** a folder holding
  an ONNX graph in Triton's layout (`1/model.onnx`, optional `spec.toml`,
  `config.pbtxt` and `model.toml`) is listed as one model with backend `onnx`.
  `run` serves it on ONNX Runtime (llmctl's own server, CUDA or CPU, in an
  ONNX Runtime venv), or with `--backend triton` on Triton in Docker:
  TensorRT on a GPU, with the engine built on the host by trtexec in the
  Triton image, strongly typed (refused if trtexec didn't build it so), and
  cached per graph, TensorRT version, GPU, build shapes and flags (with a
  `build.json` recording its precision), and ONNX Runtime on a host without
  one. Both answer
  Triton's HTTP API (KServe v2, the binary tensor extension, and
  `/v2/models/<m>/config` with `detector_spec` and `model_version`
  parameters), so one client works against either. Triton 26.09 is in
  `images.json`; a Jetson takes NGC's iGPU image through `config runtime`.
- The gateway routes `/v2/models/<name>/...` to CV models only (`llmctl.api`
  `kserve`), by id, group, prefix or `name@node`, across the cluster.
- **`llmctl import`:** copies a fine-tune from a local folder or file into
  `MODELS_DIR`: a CV model (into Triton's layout), a GGUF file or a
  checkpoint folder, recording its licence (required, or from a model card),
  version, base model and origin. `list` shows the licence and version.
- `llmctl download` of a repo of ONNX exports (onnx-community's) fetches one
  graph as a CV model, with a `model.toml` naming it and, for a graph of
  dynamic size, TensorRT build shapes at the preprocessor's size.

### Changed

- Health checks use each backend's own path: Triton's `/v2/health/ready`,
  `/health` for the rest.

## [1.8.0] - 2026-11-01

### Added

- **`llmctl pause` / `llmctl resume`:** `pause` stops a model but keeps its
  autostart entry, marked paused, and `resume` starts it again with the same
  args and port. `llmctl ps` shows it with STATE `paused`. The supervisor and
  `restore` leave it down, `run` doesn't give its port to another model, and the
  gateway answers 503 for it. Its decision endpoint is paused and resumed with
  it. With no name, both pick the only candidate or show a picker. `llmctl stop`
  forgets a paused model.
- `llmctl ls` is an alias for `llmctl list`, and `llmctl start` for `llmctl run`.
- **Image generation (backend `omni`):** diffusers pipelines such as FLUX.1-dev
  and FLUX.1-schnell run on vLLM-Omni (`vllm serve --omni`) and answer
  `POST /v1/images/generations`. A folder with `model_index.json` is listed as
  one model; its `transformer/`, `vae/`, ... folders are no longer listed as
  vllm models of their own. `run` keeps the pipeline on the GPU when it fits
  and otherwise streams the DiT from CPU a block at a time; an offload flag in
  `-X` replaces that. vLLM-Omni lives in its own venv (`~/.venvs/vllm-omni`,
  or `llmctl config omni-bin`); the `vllm` on `PATH` is never used for it.
- `llmctl download` of a diffusers repo skips the single-file weights at its top
  (`flux1-dev.safetensors`, `ae.safetensors`), ~24 GB that the pipeline never reads.
- The gateway routes `/v1/images/*` to image models only (`llmctl.api` `images`
  in `/v1/models`), so a chat call without `"model"` still finds the one chat
  model when an image model runs beside it.

### Fixed

- `llmctl update` of a decider checkpoint looked for its recorded source in the
  parent folder.

## [1.7.1] - 2026-10-29

### Added

- Serving files take `batch_size` and `ubatch_size` (`run --batch-size/--ubatch-size`).
  2048/2048 raised prefill 20–30% for Gemma 26B-A4B on orin.

## [1.7.0] - 2026-10-28

### Added

- **Decision endpoints in serving files:** a model entry's `decide:` (with
  `port`, and optional `profile`, `cache`, `concurrency`, `args`) makes
  `llmctl apply` start `<name>.decide` with its model. It restarts when the
  model restarts or its own keys change, and `--prune` also stops a pruned
  model's endpoint. One file now holds a setup like PA's.

### Fixed

- `llmctl stop NAME` on a model that crashed or failed to start used to say
  "No running model" and leave it in autostart, so the supervisor kept
  restarting it and a reboot brought it back. It now removes it from
  autostart (and stops its decision endpoint), with a `stopped` event.
- vLLM's default `--gpu-mem` (from free VRAM) now leaves 1.5 GiB instead of
  768 MiB. vLLM holds up to ~0.85 GiB more than its share (non-torch memory
  and CUDA graphs), so on a busy 24 GB desktop card a start could run out of
  memory. For a 14B FP8 model on ws, 0.79 becomes 0.75. The share still
  follows free memory, so pin `--gpu-mem` or `--kv-cache-mem` for repeatable
  runs.

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

[1.12.0]: https://github.com/alexandru-g/llm-runner/releases/tag/v1.12.0
[1.11.0]: https://github.com/alexandru-g/llm-runner/releases/tag/v1.11.0
[1.10.0]: https://github.com/alexandru-g/llm-runner/releases/tag/v1.10.0
[1.9.0]: https://github.com/alexandru-g/llm-runner/releases/tag/v1.9.0
[1.8.0]: https://github.com/alexandru-g/llm-runner/releases/tag/v1.8.0
[1.7.1]: https://github.com/alexandru-g/llm-runner/releases/tag/v1.7.1
[1.7.0]: https://github.com/alexandru-g/llm-runner/releases/tag/v1.7.0
[1.6.0]: https://github.com/alexandru-g/llm-runner/releases/tag/v1.6.0
[1.5.0]: https://github.com/alexandru-g/llm-runner/releases/tag/v1.5.0
[1.4.1]: https://github.com/alexandru-g/llm-runner/releases/tag/v1.4.1
[1.4.0]: https://github.com/alexandru-g/llm-runner/releases/tag/v1.4.0
[1.3.0]: https://github.com/alexandru-g/llm-runner/releases/tag/v1.3.0
[1.2.0]: https://github.com/alexandru-g/llm-runner/releases/tag/v1.2.0
[1.1.1]: https://github.com/alexandru-g/llm-runner/releases/tag/v1.1.1
[1.1.0]: https://github.com/alexandru-g/llm-runner/releases/tag/v1.1.0
[1.0.0]: https://github.com/alexandru-g/llm-runner/releases/tag/v1.0.0

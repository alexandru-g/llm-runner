# Changelog

All notable changes to llm-runner (`llmctl`). The format follows
[Keep a Changelog](https://keepachangelog.com), and versions follow
[semantic versioning](https://semver.org); what the version protects is in
[docs/compatibility.md](docs/compatibility.md).

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

[1.0.0]: https://github.com/alexandru-g/llm-runner/releases/tag/v1.0.0

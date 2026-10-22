# SS6M production readiness

What llm-runner needs before it can serve models in SS6M production on the
PS6 AI server. SS6M (Smart Sector 6 Management) is SmartRDI's operational
dashboard for Primăria Sectorului 6 București. Its AI workstream adopts
llmctl as the development cluster on ws, orin and lab from 2026-10-21. The
SS6M plan A0 (AI serving foundation) decides in its serving ADR (milestone
M0.2) whether llmctl also becomes the production routing layer for LLMs and
embedding models.

This page lists the gap, not a design. Items are ordered by how hard they
block. "A0 M0.x" points to the SS6M milestone that needs the item.

## Target environment

- Two NVIDIA L40S (48 GB each), scalable to four, on a PS6 AI server; an
  application server without a GPU in front of it. lab, ws and orin stand in
  for that layout until the hardware exists.
- **No internet.** Nothing reaches the Hugging Face hub, PyPI or any vendor
  at runtime. Models and images arrive as signed bundles on removable media
  and are installed from a local OCI registry (zot, pushed with ORAS).
- Everything is delivered with source and documentation and transfers to
  PS6, who may operate and modify it without SmartRDI.
- The AI-side stack is fixed by the SS6M Decisions Log entry "2026-10-21:
  AI-side stack addendum": vLLM for LLM serving, Triton for CV, Docker
  Compose per module, images pinned by digest with SBOMs, offline with
  telemetry off. Adopting llmctl in production amends that entry.

## Must have

### 1. Container backends

**Done in 1.0.0:** `llmctl config runtime <backend> docker` and `run
--runtime docker` for vLLM, SGLang and llama.cpp; images pinned by digest
(`llmctl images`); CDI or `--gpus`; models read-only; the port published on
the bind address; image and digest in `ps`, `/v1/models` and the logs. Egress
is blocked on the host (a `DOCKER-USER` rule), since Docker can't publish a
port from a container with no network. Still open: a Jetson image for orin.

Before 1.0, llama-server, vLLM and decider ran as host processes, from the
Arch package, a uv venv or a source build. Production needs each backend to run
as a container started from an image pinned by digest:
`llama.cpp@sha256:…`, `vllm/vllm-openai@sha256:…`, and a Jetson llama.cpp
image for orin.

- A `docker` launcher next to the process launcher, picked per backend in
  config. It needs `--gpus` or `--device nvidia.com/gpu=N` (CDI), the models
  directory mounted read-only, a port, and `--network` limited to what the
  backend needs (none, plus the published port).
- The image reference and digest are reported in `ps`, `/v1/models` and the
  logs, so benchmark results can record exactly what ran.
- Host processes stay available for development.

Why: reproducible numbers (the SS6M benchmarks record the image digest), SBOMs
and vulnerability scans per image, and air-gapped delivery of a known runtime.
The PS6 host then needs only the driver, Docker and the NVIDIA Container
Toolkit. Needed by A0 M0.2 and M0.3.

### 2. Offline model source

Models are downloaded once, but at PS6 there is nothing to download from.

- **Done 2026-10-22:** `llmctl config offline on` (or
  `LLM_RUNNER_OFFLINE=1`) makes `download` and `update` refuse, and starts
  every backend with `HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1`.
- **Done in 1.0.0:** `llmctl install <bundle-or-oci-ref>` places a model
  from a bundle directory or a local OCI registry into `MODELS_DIR`,
  verifying sha256 per file against `llmctl-manifest.json`, and refuses the
  whole install on any mismatch. Versions install side by side.
- **Done in 1.0.0:** provenance (model revision, licence, per-file and
  manifest checksums) in the `.llmctl-source.json` sidecar; `list` shows
  revision and licence; `verify` re-hashes.
- **Done in 1.0.0:** warm compile caches shipped in bundles (`cache export`,
  seeded when the image digest matches).

Needed by A0 M0.3.

### 3. Telemetry off

- **Done 2026-10-22:** every process llmctl starts gets
  `VLLM_NO_USAGE_STATS=1`, `VLLM_DO_NOT_TRACK=1`, `DO_NOT_TRACK=1` and
  `HF_HUB_DISABLE_TELEMETRY=1`. llama-server has no telemetry. `-E` can
  still override them.
- A test that starts each backend with networking disabled and fails if
  startup needs the network.

Needed by A0 M0.5 (zero non-allowlisted egress).

### 4. Network exposure and authentication

SS6M runs llmctl on one host, not as a cluster, so the registry and the
cluster token are out of play.

- **Done 2026-10-21:** `llmctl config bind 127.0.0.1` makes the gateway and
  every backend listen on loopback only (default `0.0.0.0`). `restore` and
  `restart` move recorded processes to the current setting. The SS6M host
  uses it, so nothing off the host reaches llmctl except through the proxy
  that fronts it on the same host.
- With loopback binding, TLS 1.3 and client authentication belong to that
  proxy (the M0 edge or a reverse proxy on the AI server): it terminates TLS
  with a certificate from the M0 internal CA and authenticates callers.
  llmctl itself stays plain HTTP. This makes "AI ports are reachable only
  through the M0 edge" enforceable, not just conventional.
- If the proxy runs in a Docker bridge network it cannot reach the host's
  `127.0.0.1`. Then either run it with `network_mode: host`, or bind llmctl
  to the bridge address (for example `172.17.0.1`) instead.
- **Done 2026-10-22:** `llmctl gateway auth on` with `gateway key add`.
  Callers outside the local networks need `Authorization: Bearer <key>`;
  keys are stored hashed and revoked at once. For SS6M, where the edge proxy
  is the only outside caller, `--local 127.0.0.0/8` plus a key for the edge
  makes even LAN callers authenticate. The key name is the caller identity
  that item 5's audit events need.

Security baseline from the SS6M architecture: TLS 1.3 in transit, RBAC with
least privilege, full audit logging. Needed by A0 M0.5.

### 5. Audit events without content

- One structured event per request: time, caller identity (from item 4),
  model, node, backend, status, token counts, duration and request id.
- Sent to a configured sink (HTTP endpoint or JSON lines file) for the SS6M
  M0 audit service.
- **Never prompt or completion text, and never images.** A debug mode that
  logs content must be off by default, separately switched, and itself
  audited.

Needed by the SS6M audit requirement and A1 M1.3 (100% audit coverage).

### 6. Declarative configuration

**Partly done in 1.0.0:** `llmctl apply <file.yaml…>` covers the models:
idempotent, with `--dry-run`, `--prune`, merged fragments and errors that
name the file and line. Still open: the gateway and registry settings in the
same file, and booting from the file (a unit that runs `apply`) instead of
the autostart manifest.

The interactive menus, profiles and `launch.json` state suit a desk. A
server wants one file per host that says what runs:

- `llmctl apply hosts/ai-gpu1.yaml`: models, backend, image, port, group,
  priority, context, GPU and memory share, extra args. Idempotent, with a
  diff. Starts and stops only what changed.
- The same file drives autostart (a systemd unit or a Compose service for the
  gateway and the registry). No hidden state needed to reproduce a host.
- Validation of the file with errors that name the line.

Needed by A0 M0.5 (install guide followed by a non-author, cold start in at
most 15 minutes).

### 7. Metrics and health

- `/metrics` in Prometheus format on the gateway, labelled by model, node
  and backend: requests, errors by status, latency (time to first token and
  total), in-flight and queued requests, spill-overs, failovers, node up or
  down.
- Re-export or document the backends' own metrics (vLLM and llama-server
  both expose `/metrics`).
- **Done 2026-10-22:** `/healthz` (process alive) and `/readyz` on the
  gateway, for Compose and the M0 edge. `/readyz` answers 200 only when
  every model in the autostart manifest is running and ready. A crashed
  backend stays in the manifest, so it turns readiness off.

Needed by A0 M0.4 (dashboards in Perses over VictoriaMetrics or
Prometheus).

### 8. Failure behaviour, tested

A0 M0.4 runs chaos tests the router must pass:

- A killed LLM backend gives clients an explicit "unavailable" error within
  10 s, not a hang. Recovery (restart and ready) within 120 s.
- A full GPU, a node that disappears mid-stream, and a registry restart each
  have a defined outcome and a test.
- Restart policy and backoff per instance are configured, not implicit.
- **Observed 2026-10-21, cause open.** SS6M's BM-LLM-01 sent 256 streaming chat
  requests at concurrency 16 from ws to orin's gateway, with Gemma 4 12B and
  26B-A4B on llama-server. 6 and 7 requests failed with a client-side read
  error: the connection closed with no HTTP error. Neither the gateway log
  nor llama-server's log shows anything. The same load through ws's own
  gateway, over localhost, had no failures. Part of the silence was the
  gateway: when a relayed stream broke, it closed the connection without
  logging. Since 2026-10-21 it logs a `relay:` line saying which side broke
  (`client left` or `upstream broke`), the request path, and how many bytes
  had gone through. Remaining candidates: connection reuse after the server
  closed it (the gateway speaks HTTP/1.1 keep-alive), cluster membership
  changing during the run (ws's gateway started and stopped in parallel),
  or the LAN. The SS6M harness now records the phase and time of each
  failure; reproduce with the new logging before M0.2 decides. Since 1.0.0
  the gateway's listen backlog is 128 (it was 5), which removes one more
  candidate: a burst of 16 new connections overflowing it.

### 9. Reproducible builds and supply chain

- `uv.lock` committed for the CLI and `systemone-proxy` (no longer
  ignored since 2026-10-21). Pinned dependencies for the decider
  integration.
- An SBOM (syft) and a vulnerability scan for every image llm-runner ships
  or launches.
- The licence of every external piece recorded: llama.cpp (MIT), vLLM
  (Apache 2.0), decider (check), and model weights per model card.

## Should have

- **GPU placement.** Pin an instance to a GPU index (`CUDA_VISIBLE_DEVICES`
  or CDI device) and report VRAM per instance. On two L40S the LLM gets GPU 1
  alone, and small models share GPU 2 under measured budgets.
- **Version and protocol stability.** **Done in 1.0.0:**
  [compatibility.md](compatibility.md) documents what 1.x keeps stable: the
  gateway API, `/v1/models` fields, exit codes, JSON outputs, the manifest
  and serving-file formats, and the cluster protocol.
- **Request limits.** Maximum body size, maximum tokens per request,
  per-client rate limits and queue limits, with OpenAI-style errors.
- **Structured logs.** JSON lines with request ids, matching the audit event
  ids.
- **Python 3.10 support,** or a clear statement that the CLI runs from its
  container only. orin's Ubuntu 22.04 ships Python 3.10, and the project
  requires 3.11 or later.

## Documentation for transfer

PS6 receives everything under the contract's transfer article. In English:

- An administration guide: install from bundles, the host file, start, stop,
  upgrade, rotate the token and certificates.
- Runbooks: a backend down, a node lost, adding a model, rolling back a
  model.
- An architecture page: the gateway, registry, nodes, routing and failover
  rules, and the security model.

## Out of scope

- **Computer-vision pipelines.** The SS6M video analytics stream frames to a
  shared detector served by Triton with dynamic batching, which is not
  request routing. The one candidate for llmctl is the optional
  vision-language verifier that checks a detection candidate. It is called
  like a chat model with an image, which the gateway already routes.
- **Fine-tuning and model registry features.** These belong to the SS6M A0
  registry (MLflow plus the OCI registry). llmctl only installs what the
  registry publishes.

## Open question for SmartRDI

llm-runner is MIT-licensed and owned by its author. Shipping it inside a
deliverable that transfers to PS6 is legally fine under MIT, but it sets a
precedent for SmartRDI engagements. Before llmctl becomes part of an SS6M
deliverable, the project owner (Radu) decides how it is held: as an
external MIT dependency, a copy vendored into the SS6M repository, or
contributed to SmartRDI.

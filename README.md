# llm-runner

Background runner for local model servers.

CLI (`llmctl`) to download, list, start, stop, and tail models served via
[llama.cpp](https://github.com/ggml-org/llama.cpp)'s `llama-server` (GGUF
files) or [vLLM](https://docs.vllm.ai) (Hugging Face safetensors
checkpoints). Works with any GGUF or HF checkpoint (Gemma, Llama, Qwen,
Mistral, etc.) and arbitrary local files/dirs via `--ft`. Embedding models
(e.g. bge-m3) are supported too via `--embeddings`.

The backend is picked from the model format: a `.gguf` runs on
`llama-server`, a checkpoint directory (`config.json` + `*.safetensors`) runs
on `vllm serve`. Both expose the same OpenAI-compatible `/v1` API, so clients
don't care which one is behind a port.

## Install

Two separate installs are required: the Python CLI **and** the native
`llama-server` binary on the host. The CLI install does **not** install llama.cpp.

### 1. Python CLI (`llmctl`)

The bundled `install.sh` puts `llmctl` on your `PATH` so it works from any
directory. It prefers `uv tool install`, then falls back to `pipx`, then to a
plain venv + symlink in `~/.local/bin`.

```bash
cd llm-runner
./install.sh
```

After installing, make sure `~/.local/bin` (or whatever the script reports) is
on your `PATH`. Then:

```bash
llmctl --help
```

To upgrade after pulling changes: re-run `./install.sh`. To uninstall:
`uv tool uninstall llm-runner` (or `pipx uninstall llm-runner`).

**Hacking on the code instead?** Use `uv sync` to create a local `.venv/` and
invoke via `uv run llmctl ...` from inside the repo — that picks up edits
without reinstalling.

### 2. `llama-server` (native, on the host)

`llmctl run` shells out to `llama-server`, so the binary must be on `PATH`.
Pick the build that matches your accelerator.

**Arch Linux** (recommended — prebuilt with GPU support):

```bash
# CUDA (NVIDIA)
yay -S llama.cpp-cuda

# Vulkan (AMD / Intel / generic)
sudo pacman -S llama.cpp-vulkan

# CPU-only
sudo pacman -S llama.cpp
```

**Ubuntu** (22.04 / 24.04) — no official package. llama.cpp's GitHub
releases ship prebuilt tarballs for **CPU**, **Vulkan**, **ROCm**, and
**SYCL**, but **not CUDA**. For NVIDIA GPUs you must build from source.

*NVIDIA (CUDA) — build from source*:

```bash
# 1. Prerequisites: NVIDIA driver + CUDA toolkit (nvcc).
#    Verify the driver first:
nvidia-smi

# 2. Install build deps + CUDA toolkit.
sudo apt update
sudo apt install -y build-essential cmake git libcurl4-openssl-dev ccache

# CUDA toolkit — pick ONE:
#   (a) Ubuntu's packaged toolkit (simplest; version tied to Ubuntu release):
sudo apt install -y nvidia-cuda-toolkit
#   (b) Or NVIDIA's official repo for the latest CUDA — see
#       https://developer.nvidia.com/cuda-downloads
#       Then ensure nvcc is on PATH:
#         export PATH=/usr/local/cuda/bin:$PATH
#         export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

# 3. Build llama.cpp with CUDA enabled.
git clone https://github.com/ggml-org/llama.cpp
cd llama.cpp
cmake -B build -DGGML_CUDA=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release -j$(nproc)

# 4. Install into /opt/llama.cpp (keeps the binary next to its .so files —
#    the binaries use RUNPATH=$ORIGIN, so the directory must be intact).
sudo mkdir -p /opt/llama.cpp
sudo cp -a build/bin/. /opt/llama.cpp/

# 5. Put it on PATH (system-wide):
echo 'export PATH=/opt/llama.cpp:$PATH' | sudo tee /etc/profile.d/llama-cpp.sh
sudo chmod +x /etc/profile.d/llama-cpp.sh
# Apply in current shell:
export PATH=/opt/llama.cpp:$PATH

# 6. Verify.
which llama-server && llama-server --version
```

If `cmake` errors out finding CUDA, point it explicitly:

```bash
cmake -B build -DGGML_CUDA=ON -DCMAKE_CUDA_COMPILER=$(which nvcc) -DCMAKE_BUILD_TYPE=Release
```

To rebuild after `git pull`: `cmake --build build --config Release -j$(nproc)`
(no flags needed — CMake reuses the cached configuration). Only a fresh
configure after `rm -rf build` needs `-DGGML_CUDA=ON` again.

*Non-NVIDIA on Ubuntu — prebuilt tarball:*

Releases are tagged by build number (`bNNNN`); fetch the latest tag and use
the matching tarball. Pick the asset for your accelerator:

| GPU              | Asset substring               |
|------------------|-------------------------------|
| AMD / Intel / generic GPU (Vulkan) | `ubuntu-vulkan-x64.tar.gz` |
| AMD ROCm         | `ubuntu-rocm-7.2-x64.tar.gz`  |
| Intel SYCL       | `ubuntu-sycl-fp16-x64.tar.gz` |
| CPU only         | `ubuntu-x64.tar.gz`           |
| ARM64            | `ubuntu-arm64.tar.gz`         |

```bash
sudo apt update && sudo apt install -y curl tar libcurl4 libgomp1

ASSET_SUBSTR='ubuntu-vulkan-x64.tar.gz'   # change to match your GPU
URL=$(curl -s https://api.github.com/repos/ggml-org/llama.cpp/releases/latest \
  | grep browser_download_url | grep "$ASSET_SUBSTR" | head -1 | cut -d '"' -f 4)
echo "Downloading: $URL"
curl -L "$URL" -o /tmp/llama.tgz

# Tarball top-level is llama-b<BUILD>/; strip it and install to /opt/llama.cpp.
sudo mkdir -p /opt/llama.cpp
sudo tar -xzf /tmp/llama.tgz -C /opt/llama.cpp --strip-components=1
echo 'export PATH=/opt/llama.cpp:$PATH' | sudo tee /etc/profile.d/llama-cpp.sh
sudo chmod +x /etc/profile.d/llama-cpp.sh
export PATH=/opt/llama.cpp:$PATH
llama-server --version
```

**Other distros** — same source build as the NVIDIA path above; install the
equivalent of `build-essential`, `cmake`, `git`, and `libcurl4-openssl-dev`
via your package manager, plus your accelerator's driver/toolkit.

### 3. `vllm` (optional, for Hugging Face checkpoints)

vLLM is a Python package with a CUDA-specific torch build, so give it its own
env rather than installing it next to `llmctl`:

```bash
conda create -n vllm python=3.12 -y
conda activate vllm
pip install vllm
vllm --version
```

`llmctl` finds the binary without the env being active: it checks
`$LLM_RUNNER_VLLM_BIN`, then the persisted `vllm_bin` setting, then `PATH`,
then a conda/venv env named `vllm`, then any conda env. If it lives somewhere
else:

```bash
llmctl config vllm-bin ~/some/env/bin/vllm
llmctl config show          # lists both backends and where they resolve
```

vLLM JIT-compiles a few kernels (flashinfer sampler) on first start, which
needs a working `nvcc` **and** a host C++ compiler it accepts. If a start dies
with `cannot execute 'cc1plus'` or `Ninja build failed` in `llmctl logs`,
either install the C++ half of whatever `gcc` is first on your PATH (a conda
`gcc_linux-64` without `gxx_linux-64` is the usual culprit:
`conda install -n base -c conda-forge gxx=<same version>`), or skip that JIT:

```bash
llmctl run <model> -E VLLM_USE_FLASHINFER_SAMPLER=0
```

### 4. `decider` (optional, for decision servers)

[Mapika/decider](https://github.com/Mapika/decider) serves *decision* models
— typed questions in, calibrated probabilities out, no text generation — on
the Jev wire (`POST /v1/systemone`). It is a Python package with its own
torch build, so clone it and give it a venv, then point `llmctl` at the
clone:

```bash
git clone https://github.com/Mapika/decider /fast/ml/jev/decider
cd /fast/ml/jev/decider
uv venv --python 3.12 .venv312 && uv pip install -p .venv312/bin/python -e ".[serve]"
llmctl config decider-dir /fast/ml/jev/decider   # or $LLM_RUNNER_DECIDER_DIR
llmctl download Mapika/decider-2b --hf           # an HF checkpoint + decider_config.json
llmctl run decider-2b -p 8009 --wait             # backend auto-detected from decider_config.json
curl -s localhost:8009/v1/systemone -H 'content-type: application/json' \
  -d '{"state": "My card was charged twice.", "questions": {"refund": {"type": "noul", "instructions": "A refund is due."}}}'
```

A checkpoint directory that carries `decider_config.json` is listed with
backend `decider` and served by `decider.serve` (uvicorn from the clone's
`.venv*/bin`). The launch profile is sized for a GPU shared with a
llama-server: CUDA graphs only up to 8 × 1024 tokens, batch ≤ 8, eager
forwards capped at 4,608 padded tokens, bf16 linears,
`DECIDER_TEMPERATURE=1.0` (raw model — apply your own fitted scaling
client-side); decider-2b then takes ~5.6 GB loaded / ~6.1 GB peak on a 4090.
`--n-parallel N` sets the batch, `-E KEY=VALUE` overrides any `DECIDER_*`
knob (e.g. `-E DECIDER_FP8=1` for fp8 linears: ~0.7 GB less, half the
throughput without torch.compile), `-X` appends uvicorn flags. The graph-capture caps
(`DECIDER_WARMUP_MAX_B`, `DECIDER_GRAPH_MAX_T`) need the small `serve.py`
patch kept in enersec-agentic (`benchmark/decisions/serving-decider-footprint.patch`)
until it is upstream; without it they are ignored and the server captures
its default 72 graphs (~20 GB peak for the 2B).

## Models

Models are discovered dynamically under `MODELS_DIR`:

- any `*.gguf` file is a **llama** model, keyed by its filename stem. Any
  `*mmproj*.gguf` in the same folder is auto-attached as the vision projector;
- any directory holding a Hugging Face checkpoint (`config.json` plus
  `*.safetensors`/`*.bin` weights) is a **vllm** model, keyed by the directory
  name.

`llmctl list` shows which backend each key resolves to.

### Get a model with `llmctl download`

```bash
llmctl download unsloth/gemma-4-E4B-it-GGUF   # GGUF repo → pick a quant → llama
llmctl download Qwen/Qwen3-8B                 # no GGUFs → whole checkpoint → vllm
llmctl download some/repo --hf                # force the checkpoint even if GGUFs exist
```

A checkpoint repo is snapshotted whole (safetensors, config, tokenizer),
skipping legacy `.bin`/`.pth` weights when safetensors are present and any
`original/` or `consolidated*` files. Gated repos need `hf auth login` or
`$HF_TOKEN`. Local layout: `MODELS_DIR/<repo-name-lowercased>/`, so
`Qwen/Qwen3-8B` becomes the key `qwen3-8b`.

You'll get an arrow-key picker listing every quant in the repo with file size.
After selecting, the matching `mmproj` (vision projector) and any
speculative/MTP drafter (e.g. an `MTP/*-MTP.gguf`) are auto-fetched alongside —
the drafter is flattened into the model folder so `run` can pair it
automatically (see [Speculative decoding](#speculative-decoding-mtp-drafters)).

Skip the picker for scripting with `--file`:

```bash
llmctl download unsloth/gemma-4-E4B-it-GGUF --file gemma-4-E4B-it-Q4_K_M.gguf
```

Other flags:

- `--no-mmproj` — don't download the vision projector
- `--no-draft` — don't download the speculative/MTP drafter
- `--subdir <name>` — override the target subdir (default: repo name, lowercased, `-GGUF` stripped)

### Update a downloaded model

`download` records the source repo in a `.llmctl-source.json` sidecar in the
model folder, so you can re-check it later without retyping the repo:

```bash
llmctl update                          # pick from models with a recorded source
llmctl update gemma-4-31b-it-ud-q4_k_xl # by model key (uses the recorded repo)
llmctl update unsloth/gemma-4-31B-it-GGUF  # by repo id (for manually-placed models)
```

`update` revalidates against the remote and re-downloads **only what changed** —
`hf_hub_download` compares the commit/etag, so unchanged files are reported `up
to date` and skipped. Handy when a repo fixes a quant or **adds an `mtp-`
drafter after you first pulled** (the new drafter lands beside the model and
auto-attaches on `run`). Models downloaded before provenance tracking (or placed
by hand) have no recorded repo — pass the repo id once and it's recorded going
forward.

### Layout convention

`llmctl download` writes one folder per HF repo; multiple quants of the same
base model share a folder along with a single shared mmproj:

```
$MODELS_DIR/
├── gemma-4-e4b-it/
│   ├── gemma-4-E4B-it-Q4_K_M.gguf
│   ├── gemma-4-E4B-it-Q8_0.gguf
│   └── mmproj-F16.gguf                  ← shared by both quants
├── gemma-4-31b-it/
│   ├── gemma-4-31B-it-Q4_K_M.gguf
│   └── gemma-4-31B-it-Q8_0-MTP.gguf     ← MTP drafter, auto-attached on run
└── qwen3-8b/
    └── Qwen3-8B-Q5_K_M.gguf
```

If you place GGUFs manually, follow the same convention: each base model in
its own folder, with its `mmproj` and/or drafter alongside. A GGUF whose name
contains `mtp`, `-assistant`, `-draft`, or `-eagle` is treated as a drafter for
its sibling base model, not as a model in its own right.

### Reuse an existing collection

```bash
export LLM_RUNNER_MODELS_DIR=/fast/ml/models
```

## Usage

### List on-disk models

```bash
llmctl list
```

Shows discovered models with key, size, vision/draft markers, and path:

```
KEY                     SIZE  VISION  DRAFT  PATH
gemma-4-e4b-it-q4_k_m   2.4G  yes            gemma-4-e4b-it/gemma-4-E4B-it-Q4_K_M.gguf
gemma-4-e4b-it-q8_0     5.0G  yes            gemma-4-e4b-it/gemma-4-E4B-it-Q8_0.gguf
gemma-4-31b-it-q4_k_m   17G           yes    gemma-4-31b-it/gemma-4-31B-it-Q4_K_M.gguf
qwen3-8b-q5_k_m         5.4G                 qwen3-8b/Qwen3-8B-Q5_K_M.gguf
```

The key is the filename stem lowercased; collisions across folders are
disambiguated by prefixing with the parent dir name. A `DRAFT` marker means a
speculative/MTP drafter sits alongside and is auto-attached on `run` (see
[Speculative decoding](#speculative-decoding-mtp-drafters)).

### Start a model

```bash
llmctl run                          # interactive picker
llmctl run gemma-4-e4b-it-q4_k_m    # by key → llama-server
llmctl run qwen3-8b --wait          # HF checkpoint → vllm; --wait blocks until /health
```

- Backend: from the model format (`--backend llama|vllm` to force; see
  [vLLM specifics](#vllm-specifics)).
- Default port: first free port from `8080`.
- Default context size: `32768`.
- Override either: `-p 8081 -c 65536`.
- Reasoning is **disabled by default** on llama (`--reasoning-budget 0`). Pass
  `--reasoning` to let the model's default reasoning behavior apply. On vllm
  use `--reasoning-parser <name>` to get `reasoning_content` split out.
- Serve an embedding model with `--embeddings` (see [Embedding models](#embedding-models)).
- Process is detached (own session, survives the CLI exit). Stdout/stderr go
  to a per-instance log file under the state dir. `--wait` polls `/health`
  and exits non-zero if the server dies first — useful for vllm, which takes
  a minute or more to load and compile.
- `-X '<args>'` appends anything verbatim to the server command; `-E K=V`
  sets an env var for it. Both are remembered for `restart`/`restore`.

Run an arbitrary GGUF or checkpoint dir outside `MODELS_DIR`:

```bash
llmctl run --ft /path/to/finetune.gguf
llmctl run --ft /path/to/finetune.gguf --name my-ft -p 8082 -c 16384
llmctl run --ft /path/to/hf-checkpoint-dir --name my-ft     # → vllm
```

### Speculative decoding (MTP drafters)

Models that ship a draft head — e.g. Gemma 4's
[Multi-Token Prediction](https://huggingface.co/unsloth/gemma-4-12b-it-GGUF/blob/main/MTP/README.md)
drafter — can generate substantially faster: the small drafter proposes several
tokens per step and the base model verifies them in one pass (often >1.4× on
dense models like Gemma-4-31B). This needs a `llama.cpp` build from **after
2026-06-07**, when MTP landed.

If a drafter sits next to the model (a sibling GGUF whose name contains `mtp`,
`-assistant`, `-draft`, or `-eagle` — `llmctl list` shows a `DRAFT` marker), it
is wired up automatically:

```bash
llmctl run gemma-4-31b-it-q4_k_m          # auto-attaches the MTP drafter
```

This emits:

```bash
llama-server -m …/gemma-4-31B-it-Q4_K_M.gguf \
  --model-draft …/gemma-4-31B-it-Q8_0-MTP.gguf \
  --spec-type draft-mtp --spec-draft-n-max 3 \
  --flash-attn on --ctx-size 32768 …
```

Point at a drafter explicitly (by path or by registry key), or tune the spec
flags:

```bash
# Explicit base + drafter by path (no registry needed)
llmctl run --ft ~/models/gemma-4/gemma-4-31B-it-Q4_K_M.gguf \
  --draft ~/models/gemma-4/gemma-4-31B-it-assistant-Q8_0.gguf

# Tune how many tokens the drafter proposes per step (try 1–6)
llmctl run gemma-4-31b-it-q4_k_m --spec-draft-n-max 4
```

- `--draft <path|key>` — drafter to pair with the model (overrides auto-detect).
- `--no-draft` — ignore an auto-detected drafter and run the base model alone.
- `--spec-type <type>` — `llama-server --spec-type` (default `draft-mtp`).
- `--spec-draft-n-max <n>` — max draft tokens per step (default `3`; try 1–6).
- `--n-parallel <n>` — server slots (`llama-server --parallel`). **Auto-set to
  `1` when a drafter is attached**, because MTP acceptance and prompt-prefix
  reuse both work best single-stream; raise it only if you need concurrent
  requests. For a client that keeps N short requests in flight (the enersec
  v2 document analysis with `concurrency: N`) use `--n-parallel N`; decode is
  bandwidth-bound so N sequences cost about the same per step as one.
- `--kv-unified` / `--no-kv-unified` — share one KV buffer across the slots
  (`llama-server --kv-unified`), so a slot can use the full `--ctx-size`
  instead of ctx/N. Default on when `--n-parallel > 1`.
- `--cache-ram <MiB>` — host-RAM prompt cache (`llama-server --cache-ram`),
  default `0` (off). Enable it (e.g. `16384`) only for workloads whose prompts
  are long and share a long prefix. On Gemma 4 each saved prompt also carries
  its sliding-window context checkpoints (~1.7 GB per ~900-token prompt), so
  for many short, distinct prompts the save costs more than the prefill it
  spares.

Drafters are ignored in `--embeddings` mode.

> **⚠️ Don't combine `--kv-quant` with a drafter.** Quantizing the *target* KV
> cache (`-ctk/-ctv q8_0`) is a known llama.cpp bug that drops MTP/speculative
> acceptance to ~0% — the drafter runs but nothing it proposes is accepted, so
> you get *no* speedup (and pay the drafter's overhead, ending up slower than
> the base model). `llmctl run` warns when you request both. Pick one:
> **f16 KV + MTP** (faster generation, but the larger KV fits less context), or
> **quantized KV + `--no-draft`** (long context, no MTP). You can't have both
> until the upstream bug is fixed.

### Embedding models

Embedding models (e.g. [bge-m3](https://huggingface.co/gpustack/bge-m3-GGUF))
are served in embedding mode, which switches `llama-server` into embedding mode
and drops the chat-only flags (flash-attn, prompt cache, reasoning) that don't
apply to BERT-style models.

Embedding mode is **auto-enabled** when the model's filename matches a known
embedding family (`bge`, `e5`, `gte`, `nomic-embed`, `mxbai-embed`,
`arctic-embed`, `minilm`, or anything containing `embed`), so no flag is needed
for bge-m3:

```bash
llmctl download gpustack/bge-m3-GGUF --file bge-m3-Q8_0.gguf --no-mmproj
llmctl run bge-m3-q8_0
```

Force it either way with `--embeddings` / `--no-embeddings` (e.g. to embed with a
model whose name isn't recognized, or to run a recognized one as a chat server).

In embedding mode the defaults are tuned for bge-m3, so the bare command
above is equivalent to:

```bash
llama-server -m bge-m3-Q8_0.gguf --embeddings --pooling cls -c 8192 -b 8192 -ub 8192
```

- `--pooling [none|mean|cls|last|rank]` sets the pooling strategy (default
  `cls`, which bge-m3 expects). Ignored without `--embeddings`.
- `-c/--ctx-size` defaults to `8192` in embedding mode (vs `32768` for chat).
- `-b/--batch-size` and `-ub/--ubatch-size` default to the context size so a
  full-length input embeds in a single pass. Override either if needed.
- Port auto-selection still applies: with a chat model already on `8080`, the
  embedding server lands on `8081`. Query it at the OpenAI-compatible
  `http://localhost:<port>/v1/embeddings`.

### Fitting large models into limited VRAM

Three knobs trade quality/speed for memory, in rough order of bang-for-buck:

```bash
# Quantize the KV cache. q8_0 is ~half of f16 and near-lossless; q4_0 halves
# it again. (flash-attn is always on, which q8_0/q4_0 V-cache requires.)
llmctl run gemma-4-31b-it-q4_k_m --kv-quant q8_0

# Shrink the context window — KV cache scales linearly with it.
llmctl run gemma-4-31b-it-q4_k_m --kv-quant q8_0 -c 16384

# Skip the vision projector on a vision model you're using for text only.
llmctl run gemma-4-31b-it-q4_k_m --no-vision

# Last resort: spill some layers to CPU (it fits, but it's slower).
llmctl run gemma-4-31b-it-q4_k_m --ngl 48
```

For a 17 GB Q4 weight file on a 24 GB card, `--kv-quant q8_0 -c 16384`
(optionally `--no-vision`) is usually enough to fit comfortably.

### List running models

```bash
llmctl ps
```

Prints PID, port, start time, and name. Dead entries are pruned automatically.

### Stop a running model

```bash
llmctl stop                        # auto-pick if 1 running; picker if many
llmctl stop gemma-4-e4b-it-q4_k_m  # by name
llmctl stop 12345                  # by PID
```

Sends `SIGTERM` to the process group; escalates to `SIGKILL` after 10 s.

### Re-run a model with its last arguments

Every `llmctl run` records the exact invocation (model, flags, and the port it
landed on) in a manifest, so you never have to remember the args again:

```bash
llmctl autostart list                        # shows each entry's `llmctl run …` line
llmctl restart gemma-4-26b-a4b-it-ud-q4_k_m  # stop (if running) + relaunch, same args, same port
llmctl restart gemma-4-26b-a4b-it-ud-q4_k_m --n-parallel 8   # …with one flag changed
llmctl restore gemma-4-26b-a4b-it-ud-q4_k_m  # relaunch only if not already running
```

Overrides passed to `restart` win over the remembered flags and become the new
remembered set. **Ports are pinned**: a model always comes back on the port it
was first started on. If that port is taken, `restart`/`restore` report an
error rather than silently moving it — pass `--port N` to `restart` to move
it deliberately.

### Auto-restart running models on boot

The same manifest (updated on every `run` and `stop`) brings the whole set
back after a reboot:

```bash
llmctl restore            # relaunch everything that was running pre-reboot
llmctl restore NAME...    # only these entries
llmctl autostart list     # show what restore would bring back
```

`restore` is idempotent — anything already running is left alone. To run it
automatically at boot, install a **systemd user service**:

```bash
llmctl autostart install    # writes + enables the unit, enables linger
llmctl autostart status     # unit + manifest state
llmctl autostart uninstall  # remove the unit (manifest is kept)
```

The installer detects an active **conda** env and bakes a `bash -lc 'source
…/conda.sh && conda activate <env> && llmctl restore'` `ExecStart` into the
unit, so `llama-server` and its CUDA libraries resolve at boot exactly as they
do in your shell (without conda it bakes the current `PATH`/`LD_LIBRARY_PATH`
instead). `LLM_RUNNER_*` overrides are passed through. Test it without
rebooting:

```bash
systemctl --user start llmctl-restore.service && llmctl ps
```

A model you explicitly `stop` is dropped from the manifest, so it won't come
back on the next boot.

### Tail a model's log

```bash
llmctl logs                        # auto-pick if 1 running; picker if many
llmctl logs gemma-4-e4b-it-q4_k_m
llmctl logs 12345 -n 200
llmctl logs gemma-4-e4b-it-q4_k_m -f   # follow (like `tail -f`); Ctrl-C to stop
```

### Chatting with a running model

`llama-server` already exposes a built-in web chat UI at
`http://localhost:<port>` — just open the port `llmctl ps` shows. For a
terminal-native option, [aichat](https://github.com/sigoden/aichat) speaks
the OpenAI-compatible API:

```yaml
# ~/.config/aichat/config.yaml
clients:
  - type: openai-compatible
    name: local
    api_base: http://localhost:8080/v1
    models:
      - name: local
model: local:local
```

### vLLM specifics

```bash
llmctl run qwen3-8b --wait                  # block until /health answers (vllm takes ~1 min)
llmctl run qwen3-8b --gpu-mem 0.5           # cap at 50% of *total* VRAM (vllm default 0.9)
llmctl run qwen3-8b --tool-parser hermes    # OpenAI tool calling (--enable-auto-tool-choice)
llmctl run qwen3-8b --reasoning-parser qwen3   # split thinking into reasoning_content
llmctl run qwen3-8b --kv-quant q8_0         # → --kv-cache-dtype fp8 (no 4-bit KV in vllm)
llmctl run qwen3-8b --n-parallel 16         # → --max-num-seqs
llmctl run qwen3-8b -X '--max-num-batched-tokens 8192'   # anything else, verbatim
llmctl run qwen3-8b -E VLLM_USE_FLASHINFER_SAMPLER=0     # env var for the server process
```

How the shared `run` flags map:

| `llmctl run`        | llama-server                  | vllm serve                         |
|---------------------|-------------------------------|------------------------------------|
| `--ctx-size N`      | `--ctx-size N`                | `--max-model-len N`                |
| `--kv-quant q8_0`   | `--cache-type-k/v q8_0`       | `--kv-cache-dtype fp8`             |
| `--n-parallel N`    | `--parallel N`                | `--max-num-seqs N`                 |
| `--embeddings`      | `--embeddings --pooling ...`  | `--runner pooling`                 |
| `--draft M`         | `--model-draft M`             | `--speculative-config {draft_model}` |
| `--spec-type ngram` | (llama types)                 | `--speculative-config {ngram}`     |
| `--name X`          | display name only             | also `--served-model-name X`       |
| `-X ARGS` / `-E K=V`| appended / env                | appended / env                     |

`--gpu-mem`, `--tool-parser`, `--reasoning-parser` and `--trust-remote-code`
are vllm-only; `--n-gpu-layers`, `--no-vision`, `--kv-unified`, `--cache-ram`,
`--reasoning` and the embedding batch flags are llama-only. Passing one to the
other backend prints a warning and ignores it.

`--gpu-mem` is how vllm shares the card. It reserves that fraction of
**total** GPU memory up front (all of it, as KV cache — even for a tiny
model) and refuses to start if it isn't free. When omitted, llmctl computes
it from the VRAM free at launch (minus ~0.75 GB headroom, capped at vllm's
own 0.9) and prints the value it chose. So:

- vllm alone → it takes ~everything. Fine.
- llama model already running → the default fits in what's left. Fine.
- vllm first, llama model later → pass a deliberate `--gpu-mem 0.3` or the
  llama model won't fit.

`restart` recomputes the default; `restore` replays the value recorded at
launch.

The model id clients send is the registry key (or `--name`), e.g.
`"model": "qwen3-8b"`; check with `curl localhost:<port>/v1/models`.

## Running multiple models concurrently

```bash
llmctl run gemma-4-e4b-it-q4_k_m       # → 8080
llmctl run gemma-4-31b-it-q4_k_m -p 8081
llmctl run qwen3-8b --gpu-mem 0.3      # vllm alongside the llama instances
llmctl ps
```

Each instance gets its own port and log file. The auto-port picker walks up
from 8080, so omitting `-p` for the second model also works. Backends mix
freely; `ps`, `show` and `autostart list` report which one each instance uses.

## Configuration

### `MODELS_DIR` resolution order

1. `$LLM_RUNNER_MODELS_DIR` if set (e.g. `/fast/ml/models`).
2. The persisted `models_dir` from `llmctl config models-dir <path>` (see below).
3. `<repo>/models/` if it already exists (developer convenience).
4. `$XDG_DATA_HOME/llm-runner/models` (default `~/.local/share/llm-runner/models`).

### Persisting the models dir with `llmctl config`

Save a models directory so you don't have to export `LLM_RUNNER_MODELS_DIR`
every session:

```bash
llmctl config models-dir /fast/ml/models          # persist it
llmctl config models-dir /fast/ml/models --create # ... creating it if missing
llmctl config models-dir                          # print the stored value
llmctl config show                                # config file + effective dir
```

Settings are written to `$XDG_CONFIG_HOME/llm-runner/config.json` (default
`~/.config/llm-runner/config.json`); override the location with
`$LLM_RUNNER_CONFIG_DIR`. The `$LLM_RUNNER_MODELS_DIR` env var, if set, still
takes precedence over the persisted value.

The other persisted setting is `vllm_bin` (`llmctl config vllm-bin <path>`,
`--clear` to forget); `$LLM_RUNNER_VLLM_BIN` overrides it. See the vllm
install section for the full lookup order.

### State and logs

State (`running.json` + per-instance logs) is resolved in this order:

1. `$LLM_RUNNER_STATE_DIR` if set.
2. `<repo>/.state/` if it already exists (handy when developing in-repo).
3. `$XDG_STATE_HOME/llm-runner` (default `~/.local/state/llm-runner`).

Contents:

- `running.json` — registry of active PIDs (reconciled on every `ps`/`stop`).
- `autostart.json` — manifest of models to bring back via `llmctl restore`
  (survives reboots; updated on `run`/`stop`).
- `logs/` — per-instance log files, named `<key>-<port>-<timestamp>.log`.

Wipe state if it ever desyncs from reality:

```bash
rm -rf ~/.local/state/llm-runner
```

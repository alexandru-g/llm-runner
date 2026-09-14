# llm-runner

Background runner for local `llama-server` instances.

CLI (`llmctl`) to download, list, start, stop, and tail GGUF models served via
[llama.cpp](https://github.com/ggml-org/llama.cpp)'s `llama-server`. Works
with any GGUF from Hugging Face (Gemma, Llama, Qwen, Mistral, etc.) and
arbitrary local files via `--ft`. Embedding models (e.g. bge-m3) are supported
too via `--embeddings`.

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

## Models

Models are discovered dynamically: any `*.gguf` file under `MODELS_DIR` is a
candidate, with a friendly key derived from its filename. Any `*mmproj*.gguf`
in the same folder is auto-attached as the vision projector.

### Get a model with `llmctl download`

```bash
llmctl download unsloth/gemma-4-E4B-it-GGUF
```

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
llmctl run gemma-4-e4b-it-q4_k_m    # by key
```

- Default port: first free port from `8080`.
- Default context size: `32768`.
- Override either: `-p 8081 -c 65536`.
- Reasoning is **disabled by default** (`--reasoning-budget 0`). Pass
  `--reasoning` to let the model's default reasoning behavior apply.
- Serve an embedding model with `--embeddings` (see [Embedding models](#embedding-models)).
- Process is detached (own session, survives the CLI exit). Stdout/stderr go
  to a per-instance log file under the state dir.

Run an arbitrary GGUF outside `MODELS_DIR`:

```bash
llmctl run --ft /path/to/finetune.gguf
llmctl run --ft /path/to/finetune.gguf --name my-ft -p 8082 -c 16384
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

### Auto-restart running models on boot

`llmctl` keeps a manifest of what you have running (updated on every `run` and
`stop`) so the set can be brought back after a reboot:

```bash
llmctl restore            # relaunch everything that was running pre-reboot
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

## Running multiple models concurrently

```bash
llmctl run gemma-4-e4b-it-q4_k_m       # → 8080
llmctl run gemma-4-31b-it-q4_k_m -p 8081
llmctl ps
```

Each instance gets its own port and log file. The auto-port picker walks up
from 8080, so omitting `-p` for the second model also works.

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

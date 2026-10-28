# llm-runner

Background runner for local model servers.

CLI (`llmctl`) to download, list, start, stop, and tail models served via
[llama.cpp](https://github.com/ggml-org/llama.cpp)'s `llama-server` (GGUF
files) or [vLLM](https://docs.vllm.ai) / [SGLang](https://docs.sglang.io)
(Hugging Face safetensors checkpoints). Works with any GGUF or HF checkpoint (Gemma, Llama, Qwen,
Mistral, etc.) and arbitrary local files/dirs via `--ft`. Embedding models
(e.g. bge-m3) are supported too via `--embeddings`.

The backend is picked from the model format: a `.gguf` runs on
`llama-server`, a checkpoint directory (`config.json` + `*.safetensors`) runs
on `vllm serve` (or `sglang serve` with `--backend sglang`). All of them expose
the same OpenAI-compatible `/v1` API, so clients don't care which one is behind
a port.

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

`install.sh` also installs `systemone-proxy` (the decision endpoint behind
`llmctl decide`, from `packages/systemone-proxy`).

To upgrade after pulling changes: re-run `./install.sh`. It restarts a running
gateway on the new code (`llmctl gateway restart`); models keep running, and
decision endpoints keep their old code until `llmctl restart`. To uninstall:
`uv tool uninstall llm-runner systemone-proxy` (or `pipx uninstall …`).

**Offline host?** Each release ships a wheelhouse: the `llmctl` wheel plus
every dependency, per platform. Copy it over and install without an index:

```bash
sha256sum -c --ignore-missing sha256sums.txt
tar xf llm-runner-1.0.0-wheelhouse-py312-x86_64.tar.gz
cd llm-runner-1.0.0-wheelhouse-py312-x86_64 && ./install.sh --offline .
```

That runs `uv tool install --offline --no-index --find-links .` (or pip with
`--no-index` when uv is missing), with the dependency versions pinned in
`uv.lock`. Wheelhouses exist for Python 3.11 and 3.12 on x86_64 and aarch64.
systemone-proxy isn't in them; its wheel is a separate release asset.

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

### 3. `vllm` / `sglang` (optional, for Hugging Face checkpoints)

Both are Python packages with a CUDA-specific torch build, so each gets its
own venv rather than going next to `llmctl`. `uv tool install` does that and
puts the CLI in `~/.local/bin`; `--torch-backend` picks the CUDA build of torch
(`auto` detects it from the driver, or name one, e.g. `cu130`):

```bash
uv tool install --python 3.12 --torch-backend auto vllm
vllm --version
```

vLLM only sets a floor on `transformers`, so uv installs the newest one,
which can be newer than the vLLM release was built against. vLLM 0.29 needs
`transformers<5.17` for Mistral 3 / Pixtral (5.17 renamed a class it
imports), so pin it there; `uv tool upgrade` keeps a `--with` pin:

```bash
uv tool install --python 3.12 --torch-backend auto vllm==0.29.0 --with transformers==5.16.1
```

SGLang pins two pre-release dependencies, which uv only installs when they
are named explicitly (check `Requires-Dist` in the release's metadata when
upgrading; for 0.5.22 they are):

```bash
uv tool install --python 3.12 --torch-backend auto sglang==0.5.22 \
    --with cuda-tile==1.6.0rc5 --with 'flash-attn-4>=4.0.0b18'
```

`uv tool upgrade vllm` / `uv tool uninstall sglang` manage them later.
`llmctl` finds each CLI without anything activated: it checks
`$LLM_RUNNER_VLLM_BIN` / `$LLM_RUNNER_SGLANG_BIN`, then the persisted
`vllm_bin` / `sglang_bin` setting, then `PATH`, then the uv tool venv, then a
venv at `~/.venvs/<name>`. If it lives somewhere else:

```bash
llmctl config vllm-bin ~/some/venv/bin/vllm
llmctl config show          # lists the backends, where they resolve, and the toolchain
```

Both JIT-compile CUDA kernels (FlashInfer, sgl-kernel) on first use, which
needs `nvcc` **and** a host C++ compiler that nvcc accepts. llmctl sets one
for every vllm/sglang launch: `CUDA_HOME` from the `nvcc` it finds, and
`CC`/`CXX`/`NVCC_CCBIN` to the newest system GCC (`/usr/bin/g++`,
`/usr/bin/g++-N`) that can compile C++ and that the CUDA headers allow.
That keeps a compiler earlier on `PATH` (e.g. conda's `gcc` without
`cc1plus`) out of the build. `llmctl config show` prints the pair it picked;
`-E CXX=…` (etc.) overrides it per model. If no system GCC is old enough for
your nvcc, install one (Arch: `gcc15`; Ubuntu: `g++-13`) and llmctl picks it
up.

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
  name. `--backend sglang` serves the same directory on SGLang.

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

### Delete a model with `llmctl rm`

```bash
llmctl rm                              # pick from the models on disk
llmctl rm gemma-4-e4b-it-q8_0          # by model key; lists files + size, then asks
llmctl rm gemma-4-e4b-it-q8_0 -y       # skip the confirmation
```

For a GGUF, the shared mmproj / drafter are only deleted along with the last
quant in their folder, and the folder goes too once nothing but the
provenance sidecar is left. A vllm checkpoint is deleted as a whole directory.
`rm` refuses while the model is running and drops it from the autostart
manifest.

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
KEY                     SIZE  VISION  DRAFT  DECISION  PATH
gemma-4-e4b-it-q4_k_m   2.4G  yes                      gemma-4-e4b-it/gemma-4-E4B-it-Q4_K_M.gguf
gemma-4-e4b-it-q8_0     5.0G  yes                      gemma-4-e4b-it/gemma-4-E4B-it-Q8_0.gguf
gemma-4-31b-it-q4_k_m   17G           yes              gemma-4-31b-it/gemma-4-31B-it-Q4_K_M.gguf
qwen3-8b-q5_k_m         5.4G                           qwen3-8b/Qwen3-8B-Q5_K_M.gguf
```

The key is the filename stem lowercased; collisions across folders are
disambiguated by prefixing with the parent dir name. A `DRAFT` marker means a
speculative/MTP drafter sits alongside and is auto-attached on `run` (see
[Speculative decoding](#speculative-decoding-mtp-drafters)).

A `DECISION` marker flags a calibrated decision model (e.g. Quyet): an
ordinary checkpoint whose answer is the next-token probability of option
letters under a vendor-specific prompt, rescaled with fitted temperatures. It
is detected from a `quyet_config.json` beside the weights or a
`decision-model` tag in the GGUF header. `run` still serves it on llama/vllm
(and prints a note), but only a client that builds its prompt and reads the
letter logprobs gets meaningful answers.

### Start a model

```bash
llmctl run                          # interactive picker, then the launch-options menu
llmctl run gemma-4-e4b-it-q4_k_m    # by key → llama-server
llmctl run gemma-4-e4b-it-q4_k_m -i # by key, with the launch-options menu
llmctl run qwen3-8b --wait          # HF checkpoint → vllm; --wait blocks until /health
```

After an interactive pick, a launch-options menu lists the settings that apply
to that model, with **Start** on top so Enter launches with what's shown:

```
? Launch h2o-lightning-4b.q8_0 on llama  (Enter starts; * = not default)
 » ▶ Start
     Profile         last  ▸
   * Vision          off (saves ~1 GB)
     Reasoning       off
   * Context         8,192
     KV cache        f16
     Checkpoints     auto (off: decision model)
     Parallel slots  auto
     Group           auto (h2o-lightning-4b)
   ↺ Reset to defaults
```

Enter on a two-valued setting toggles it; the others open a value list. The
choices are remembered per model (state dir, `launch.json`) and preselected
next time, and they're recorded like typed flags, so `restart` and autostart
reuse them. `-i` shows the menu for a named model, `--no-menu` skips it. Flags
you type still apply and show in the menu. **Group** offers auto (the
model's folder), none, the groups already running in the cluster, and
"other…" to type a name (see [the registry](#several-machines-as-one-the-registry)).

For a Hugging Face checkpoint, a **Backend** row comes first when SGLang is
installed too (`llmctl config show` lists what is): Enter toggles between
vllm (the default) and sglang. The menu title and the rows that follow switch
to the chosen engine, and values that still fit are kept. The choice is
remembered like the others, so the next menu opens on it, and it is recorded
as `--backend sglang` for `restart` and autostart:

```
? Launch qwen3.5-4b on sglang  (Enter starts; * = not default)
 » ▶ Start
     Profile         default  ▸
   * Backend         sglang
     Context         default (32768)
     KV cache        f16
     Parallel slots  auto
     Group           auto (none)
   ＋ Save as profile…
   ↺ Reset to defaults
```

Settings are a table in
`src/llm_runner/launch.py`; adding one is one entry.

**Profiles** set every row at once. The **Profile** row, right under Start,
opens the list; pick one and press Enter on Start. The menu opens on `last`.

| Profile | For | Sets |
|---|---|---|
| `last` | any model | what this model ran with last time (including its group and backend) |
| `default` | any model | llmctl's defaults |
| `fast` | chat models with a drafter | MTP on, f16 KV, one slot |
| `long` | chat models | 128K context, q8_0 KV, drafter off, one slot |
| `multi` | chat models on llama | 8 GB RAM prompt cache, one slot: several long conversations take turns without reprocessing |
| `serve` | chat models | 4 slots sharing 64K, q8_0 KV, drafter off, 4 checkpoints |
| `lean` | chat models | 8K context, no vision, q8_0 KV, drafter off, no checkpoints |

"Chat models" means llama, vLLM or SGLang, not decision or embedding models; those
get only `last` and `default`. A profile resets the rows it doesn't set to
their defaults, except the group and the backend. The profiles that quantize the KV cache turn
the drafter off, because quantized KV drops MTP acceptance to ~0% (the menu
warns if you combine them by hand). After a profile, change any row; the
Profile row then reads `long (modified)`.

**Save as profile…** stores the current rows under a name in `profiles.json`.
Your profile is offered for models on the same backend that have every row it
sets. `llmctl profile` lists the profiles, `llmctl profile rm NAME` deletes one,
and `llmctl run MODEL --profile NAME` applies one without the menu (flags typed
alongside it win). `restart` and autostart record the flags a profile set, not
its name, so editing a profile never changes a running model.

- Backend: from the model format (`--backend llama|vllm|sglang` to force; see
  [vLLM and SGLang specifics](#vllm-and-sglang-specifics)).
- Default port: first free port from `8080`.
- Default context size: `32768`.
- Override either: `-p 8081 -c 65536`.
- Reasoning is **disabled by default** on llama (`--reasoning-budget 0`). Pass
  `--reasoning` to let the model's default reasoning behavior apply. On
  vllm/sglang use `--reasoning-parser <name>` to get `reasoning_content` split out.
- Serve an embedding model with `--embeddings` (see [Embedding models](#embedding-models)).
- Process is detached (own session, survives the CLI exit). Stdout/stderr go
  to a per-instance log file under the state dir. `--wait` polls `/health`
  and exits non-zero if the server dies first — useful for vllm/sglang, which
  take a minute or more to load and compile.
- `-X '<args>'` appends anything verbatim to the server command; `-E K=V`
  sets an env var for it. Both are remembered for `restart`/`restore`.

Run an arbitrary GGUF or checkpoint dir outside `MODELS_DIR`:

```bash
llmctl run --ft /path/to/finetune.gguf
llmctl run --ft /path/to/finetune.gguf --name my-ft -p 8082 -c 16384
llmctl run --ft /path/to/hf-checkpoint-dir --name my-ft     # → vllm
llmctl run --ft /path/to/hf-checkpoint-dir --backend sglang
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
  default `0` (off). Turn it on (the `multi` profile uses `8192`) when several
  long conversations take turns on one slot: a 9K-token Gemma conversation
  came back in 1.3 s instead of 16 s on the Orin, holding ~1.1 GB of RAM.
  Leave it off for many short, distinct prompts: each call saves its state
  (~75 MB on Gemma 26B), +33% per call on the RTX, and the cache fills its
  limit.

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

Prints PID, port, start time, name and runtime (`host`, or `docker` with the
image's short digest). Dead entries are pruned automatically.
When the gateway is up, its all-models URL is printed underneath.

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
llmctl restart                               # everything running, one at a time
llmctl restore gemma-4-26b-a4b-it-ud-q4_k_m  # relaunch only if not already running
```

Overrides passed to `restart` win over the remembered flags and become the new
remembered set. **Ports are pinned**: a model always comes back on the port it
was first started on. If that port is taken, `restart`/`restore` report an
error rather than silently moving it — pass `--port N` to `restart` to move
it deliberately.

Bare `llmctl restart` restarts everything that is running: the models first,
each back and answering `/health` before the next one stops (so memory never
holds two copies of one, which matters on the Orin), then the decision
endpoints, then the gateway. Handy after `./install.sh` or a deploy, so every
process runs the new llmctl and picks up launch defaults added since (e.g.
`--alias`, groups).

### Auto-restart running models on boot

The same manifest (updated on every `run` and `stop`) brings the whole set
back after a reboot:

```bash
llmctl restore            # relaunch everything that was running pre-reboot
llmctl restore NAME...    # only these entries
llmctl autostart list     # show what restore would bring back
llmctl autostart sync     # reset the manifest to exactly what's running now
```

`restore` is idempotent — anything already running is left alone. To run it
automatically at boot, install a **systemd user service**:

```bash
llmctl autostart install    # writes + enables the unit, enables linger
llmctl autostart status     # unit + manifest state
llmctl autostart uninstall  # remove the unit (manifest is kept)
```

The installer bakes the current `PATH` and `LD_LIBRARY_PATH` into the unit,
so `llama-server` and its CUDA libraries resolve at boot as they do in your
shell; vllm/sglang are recorded by absolute path and get their compilers from
llmctl. `LLM_RUNNER_*` overrides are passed through. Test it without
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

### Decision endpoints with `llmctl decide`

Calibrated decision models (`DECISION yes` in `llmctl list`) answer typed
questions with probabilities, not chat. Each vendor ships its own prompt
format and calibration, served by its own wrapper. `llmctl decide` starts
**systemone-proxy** (`packages/systemone-proxy`, installed by `./install.sh`)
next to the model's llama-server. It rebuilds the vendor's exact prompt (as
token ids), reads the option-letter probabilities from llama-server, applies
the vendor's temperatures, and answers on one `POST /v1/systemone` API that is
the same for every supported model:

```bash
llmctl run rune-26b-a4b-v3-iq4_xs-dyn -p 8180 -c 16384   # the model, as usual
llmctl decide rune-26b-a4b-v3-iq4_xs-dyn                 # + endpoint on :8090
llmctl decide spec rune-26b-a4b-v3-iq4_xs-dyn            # the client contract
llmctl decide profiles                                   # supported models
```

```bash
curl -s localhost:8090/v1/systemone -H 'Content-Type: application/json' -d '{
  "state": {"message": "Please close my card, I lost it yesterday."},
  "questions": {
    "urgent": {"type": "noul", "instructions": "The request is urgent."},
    "intent": {"type": "choice", "instructions": "What does the customer want?",
               "criteria": {"cancel": "close the card", "limit": "change the limit"}}}}'
```

- **Supported:** Quyet-1.0-Large, Rune 26B-A4B v3, Xor 26B-A4B, Bobcat Flash
  1.2, H2O-Lightning-4B, Xevn-4B and iota-decide-4b v0.3. The profile is
  picked from the GGUF's metadata.
- **Served by llama-server itself:** since build b11361, llama-server
  answers `/v1/systemone` for GGUFs that carry llama.cpp's decision metadata
  (`<arch>.decision.*`), for example `ggml-org/vllm-d3-mini-GGUF` (b11572 and
  later) and h2oai's own `h2o-lightning-4b-gguf` files (b11773 and later).
  It then refuses plain completions, so no profile can read them. For these
  models, `llmctl decide` starts the proxy in *native* mode: it forwards each
  request and gives the same contract (validation, error codes, noul
  confidence, `warnings`, `/v1/spec`), with llama-server's numbers. Text
  only. vllm-d3 attends to the whole prompt at once, so `llmctl run` sets
  `-b`/`-ub` to the context size for it (each prompt must fit one batch).
  The h2o profile stays on the metadata-free GGUF, which reproduces h2o's
  shim exactly, 0.801 noul floor included; native h2o skips the floor.
- **iota-decide-4b v0.3** has no published GGUF (the iota GGUFs on the Hub
  are v0.2, a different architecture). Convert the checkpoint with
  llama.cpp's `convert_hf_to_gguf.py` after moving `joint_head_config.json`
  aside, with `--no-mtp`; the profile's docstring has the steps.
- **One contract:** `llmctl decide spec [MODEL] [--json]` prints the request
  and response format plus the model's specifics (label order, temperatures,
  limits, state truncation, extras). A running endpoint serves the same at
  `GET /v1/spec`. The [client guide](packages/systemone-proxy/docs/client-guide.md)
  explains how to read it and write a client;
  [adding a profile](packages/systemone-proxy/docs/adding-a-profile.md) covers
  supporting a new model.
- **Self-test:** at startup the proxy checks that the GGUF tokenizes exactly
  like the vendor's tokenizer; it refuses to serve a mismatched model.
- **Lifecycle:** the endpoint is a tracked instance named `<model>.decide`
  (backend `decide` in `llmctl ps`). `llmctl stop <model>` stops its endpoint
  too, and autostart brings both back (the proxy waits for llama-server to
  load).
- **Prompt cache:** off by default (`--cache off`), so every call is computed
  fresh. Gemma 4's sliding-window attention made cached prefixes change
  answers. `--cache request` reuses the state prefix within one request only.
- **Memory:** `llmctl run` starts decision models with `--ctx-checkpoints 0`.
  The endpoint never reuses llama-server's prompt cache, and the checkpoints
  it would otherwise keep (up to 32 per slot, large on long prompts) once
  ran the Orin out of memory. Override with `--ctx-checkpoints N` or the
  Checkpoints setting in the launch menu.
- **Quantisation:** temperatures were fitted on bf16/fp8 weights. Prefer Q8_0
  where it fits; Q4 keeps the chosen option but shifts the probabilities.
- **Images (H2O-Lightning-4B only):** send them in a top-level `images` list,
  inline in `state` as data URIs, or as multipart file parts (up to 4, see
  `llmctl decide spec`). llama-server needs a vision projector. The h2o GGUF
  repo has none, but its vision tower is identical to Qwen3.5-4B's, so fetch
  that one into the model's folder and run without `--no-vision`:

  ```bash
  llmctl download unsloth/Qwen3.5-4B-GGUF --file mmproj-F16.gguf --subdir h2o-lightning-4b
  ```

  Text-only models reject the `images` field. Without a projector, an image
  request gets `503 vision_unavailable`.

### vLLM and SGLang specifics

A checkpoint runs on vLLM unless `--backend sglang` asks for SGLang. Every
example below works on either; `restart` and `restore` keep the backend.

```bash
llmctl run qwen3-8b --wait                  # block until /health answers (vllm takes ~1 min)
llmctl run qwen3-8b --gpu-mem 0.5           # cap at 50% of *total* VRAM (vllm default 0.9)
llmctl run qwen3-8b --tool-parser hermes    # OpenAI tool calling (--enable-auto-tool-choice)
llmctl run qwen3-8b --reasoning-parser qwen3   # split thinking into reasoning_content
llmctl run qwen3-8b --kv-quant q8_0         # → --kv-cache-dtype fp8 (no 4-bit KV in vllm)
llmctl run qwen3-8b --n-parallel 16         # → --max-num-seqs
llmctl run qwen3-8b --kv-cache-mem 6G       # → --kv-cache-memory-bytes (vllm only)
llmctl run qwen3-8b -X '--max-num-batched-tokens 8192'   # anything else, verbatim
llmctl run qwen3-8b -E VLLM_USE_FLASHINFER_SAMPLER=0     # env var for the server process
llmctl run qwen3-8b --backend sglang --wait               # the same model on SGLang
```

How the shared `run` flags map:

| `llmctl run`        | llama-server                  | vllm serve                         | sglang serve |
|---------------------|-------------------------------|------------------------------------|--------------|
| `--ctx-size N`      | `--ctx-size N`                | `--max-model-len N`                | `--context-length N` |
| `--kv-quant q8_0`   | `--cache-type-k/v q8_0`       | `--kv-cache-dtype fp8`             | `--kv-cache-dtype fp8_e4m3` |
| `--n-parallel N`    | `--parallel N`                | `--max-num-seqs N`                 | `--max-running-requests N` |
| `--gpu-mem F`       | —                             | `--gpu-memory-utilization F`       | `--mem-fraction-static F` |
| `--embeddings`      | `--embeddings --pooling ...`  | `--runner pooling`                 | `--is-embedding` |
| `--draft M`         | `--model-draft M`             | `--speculative-config {draft_model}` | `--speculative-algorithm STANDALONE` (`EAGLE`/`EAGLE3` with `--spec-type eagle`/`eagle3`) |
| `--spec-type mtp`   | —                             | `--speculative-config {mtp}`       | `--speculative-algorithm NEXTN` |
| `--spec-type ngram` | (llama types)                 | `--speculative-config {ngram}`     | `--speculative-algorithm NGRAM` |
| `--tool-parser P`   | —                             | `--enable-auto-tool-choice --tool-call-parser P` | `--tool-call-parser P` |
| `--name X`          | display name only             | also `--served-model-name X`       | also `--served-model-name X` |
| `-X ARGS` / `-E K=V`| appended / env                | appended / env                     | appended / env |

`--gpu-mem`, `--tool-parser`, `--reasoning-parser` and `--trust-remote-code`
are vllm/sglang-only; `--n-gpu-layers`, `--no-vision`, `--kv-unified`, `--cache-ram`,
`--reasoning` and the embedding batch flags are llama-only. Passing one to the
other side prints a warning and ignores it. SGLang always gets
`--enable-metrics` (the gateway reads its queue depth from `/metrics`), and
`SGLANG_ENABLE_HEALTH_ENDPOINT_GENERATION=0` so `/health` is an instant check
rather than a one-token generation that takes ~1 s.

On hybrid models (mamba / linear-attention layers, e.g. Qwen3.5 — detected from
`layer_types` in `config.json`) SGLang sizes the recurrent-state pool from
memory and then caps concurrency to fit it, whatever `--max-running-requests`
says: the log shows `max_running_requests is capped to 6 by the mamba state
cache (max_mamba_cache_size=34, 5 state slots per request)`. So with
`--n-parallel N` llmctl also passes `--max-mamba-cache-size` = N × 5 + 20%
(at least +4) — 60 for N = 10, which took p95 TTFT at 10 users from 3.6 s to
0.33 s for qwen3.5-9b on a 4090. Setting `--max-mamba-cache-size` or
`--mamba-full-memory-ratio` in `-X` turns this off.

`--kv-cache-mem SIZE` gives vLLM a fixed KV cache (`--kv-cache-memory-bytes`)
instead of sizing it from the memory free at start. Hybrid models (Qwen3.5,
Qwen3.6) need it: a warm restart frees less memory than a cold one expects
and can run out in `chunk_gated_delta_rule`. llmctl warns when a hybrid
model starts on vLLM without it. Take the size from a good start's log line
`Available KV cache memory: 4.73 GiB`.

`--gpu-mem` is how vllm and sglang share the card. vllm reserves that
fraction of **total** GPU memory up front (all of it, as KV cache — even for
a tiny model) and refuses to start if it isn't free. SGLang's fraction covers
weights and the KV pool; its activations and CUDA graphs come out of the
rest. When omitted, llmctl computes it from the VRAM free at launch (minus
~0.75 GB headroom capped at 0.9 for vllm, ~3 GB capped at 0.88 for sglang)
and prints the value it chose. So:

- vllm/sglang alone → it takes ~everything. Fine.
- llama model already running → the default fits in what's left. Fine.
- vllm/sglang first, llama model later → pass a deliberate `--gpu-mem 0.3` or
  the llama model won't fit.

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

### One port for all of them: the gateway

Writing a client? See the [gateway client guide](docs/gateway-client-guide.md).

The first `llmctl run` also starts a gateway on port 9000 that serves every
running model through one OpenAI-compatible base URL. Each model keeps its
own port; the gateway routes by the request's `model` field, the way the
OpenAI API does:

```bash
curl http://orin:9000/v1/models          # what's running (+ an "llmctl" block)
curl http://orin:9000/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model": "gemma-4-26b", "messages": [{"role": "user", "content": "hi"}]}'
```

- `GET /v1/models` lists the running models. `id` is the llmctl name, and
  `llmctl` adds backend, port, API (`openai` or `systemone`), context,
  vision, decision flag and status (`ready` / `loading` / `down`).
- `model` may be the exact name or a unique prefix (`"gemma-4-26b"` above),
  and may be left out when only one model is running. An unknown or
  ambiguous name gets an OpenAI-style `404 model_not_found` /
  `400 model_ambiguous` that lists the running ids.
- `POST /v1/systemone` is routed the same way among the decision endpoints
  (`llmctl decide`); `GET /v1/spec` takes the model in an `X-Model` header.
- `/m/<name>/<path>` reaches any other endpoint of one model, e.g.
  `/m/glm-ocr/props` or llama-server's native `/completion`.
- Streaming (SSE) passes through as it is generated; a client that hangs up
  stops the generation.

It stops with the last `llmctl stop`, is restored at boot with the models,
and shows in `llmctl ps`. `llmctl gateway` prints its URL and routes;
`llmctl gateway start|stop` control it by hand, `llmctl config gateway PORT`
moves it and `llmctl config gateway --off` keeps it from starting.
llama-server instances get `--alias <name>`, so their responses report the
same model id the gateway lists.

For a proxy or Compose health check, `GET /healthz` answers while the gateway
is up, and `GET /readyz` answers 200 only when every model in the autostart
manifest is running and ready (503 with each model's status otherwise). A
crashed model stays in the manifest, so it turns readiness off until the
gateway restarts it (see "Monitoring and operations"). A gateway with no
model declared and none running answers 503 too, because it has nothing
to serve.

### Who can reach the gateway: bind address and API keys

The gateway and every model listen on `0.0.0.0` by default.
`llmctl config bind 127.0.0.1` keeps them on loopback, reachable only through
a proxy on the same host; `llmctl restart` moves what is running.

To publish the gateway (for example through a reverse proxy at
`https://llmctl.example.net`), ask callers from outside for an API key:

```bash
llmctl gateway key add laptop        # prints the key once; only its hash is kept
llmctl gateway auth on               # outside callers now need a key
llmctl gateway key                   # the key names
llmctl gateway key rm laptop         # revoke (applies at once, no restart)
```

- A caller sends `Authorization: Bearer <key>` (what OpenAI clients send for
  their `api_key`) or `X-Api-Key: <key>`; without one it gets
  `401 invalid_api_key`.
- **Local callers need none:** loopback, the private ranges, Tailscale's
  100.64.0.0/10 and IPv6 ULA and link-local. `--local CIDR` replaces that
  list; `--local 127.0.0.0/8` wants keys from the LAN too.
- **Behind a reverse proxy**, a request from a proxy on a local network is
  judged by the client address in its `X-Forwarded-For`, so the proxy doesn't
  make the internet local. `--proxy ADDR` names the proxy, so that a request
  from it without a forwarded address counts as outside.
- Health checks, cluster hops and the registry (which check the cluster
  token) need no key. `llmctl gateway auth` prints the settings.

### Several machines as one: the registry

Gateways on different machines can share one model list. One always-on box
runs the registry; every machine (a *node*) reports its running models to it
every 5 s and gets the rest of the cluster back. Any gateway, the registry's
included, then lists and routes to every model in the cluster.

```bash
lab$  llmctl registry serve                     # prints the join command + token
lab$  llmctl autostart install                  # keep it up across reboots
rtx$  llmctl node set --name rtx --priority 0   # lower number = preferred
rtx$  llmctl registry set 192.168.8.213 --token <token>
orin$ llmctl node set --name orin --priority 1
orin$ llmctl registry set 192.168.8.213 --token <token>
any$  llmctl ps --all                           # every model, and where calls go
```

- **Routing.** A model running on several nodes goes to a `ready` instance
  first, then the lowest priority number, then this machine. A call for
  `quyet` made to the RTX's gateway goes to the Orin when only the Orin runs
  it; a model on both goes to the RTX. `"model": "quyet@orin"` (or
  `/m/quyet@orin/...`) pins a node.
- **Failover.** If the chosen instance is unreachable, answers 502/503, has
  just stopped, or refuses the prompt as longer than its context, the call
  goes to the next instance. Only refusals that come before any reply is
  sent count, so a generation is never repeated.
- **Groups.** Every model also answers to its *group*: by default the
  folder it sits in, which by the layout convention holds the quants of one
  model. So `"model": "gemma-4-26b-a4b-it"` reaches the Q8 on the RTX or the
  Q4 on the Orin, whichever routing prefers, and each gets the request under
  its own exact name. A prefix that matches several quants of one group
  (`"gemma-4-26b"`) means the group. `llmctl run MODEL --group NAME` puts a
  model in another group (e.g. two different models you treat as
  interchangeable), `--group none` in none. Several names, comma-separated,
  put it in each: `--group assistant,gemma-4-26b-a4b-it` (or `group:
  [assistant, gemma-4-26b-a4b-it]` in a serving file) lets one stable name
  reach whichever machine's model is up while the model keeps its usual
  group. `/v1/models` lists a group as a model of its own once it has two
  members (`llmctl.members`), and every model shows its `llmctl.group` (the
  first) and, with several, `llmctl.groups`. Exact ids still mean exactly
  that model. Nodes before 1.6 route by the first group only.
- **Spill-over.** Priority decides who gets a call, unless that instance is
  full. When every slot of the preferred instance is busy (llama-server's
  `/slots`; waiting requests on vLLM and SGLang; a decision endpoint counts as busy as
  the llama-server behind it), the call goes to the next instance that has a
  free slot right now. If every instance is busy, it queues on the
  preferred one as before, so work only moves to a slower machine when that
  machine is idle. The busy check happens on the machine running the model
  just before the request is sent (a ~1 ms localhost call); a busy remote
  answers at once, nothing is generated twice. `llmctl node set --spill off`
  makes a machine send its calls by priority only. With `--n-parallel 1`
  (one slot), one running request makes an instance busy.
- **Images** go only to instances that read them (a llama-server with its
  mmproj, or a decision endpoint in front of one); otherwise
  `400 vision_unavailable`.
- **Health.** The heartbeat is the health check: a node silent for 15 s is
  dropped, a gateway that stops says goodbye and is dropped at once. If the
  registry itself is down, nodes keep the last list for 2 minutes and
  failover covers machines that went away.
- **Token.** Heartbeats and gateway-to-gateway calls carry the cluster token
  (`llmctl registry token` prints it). Without it the registry refuses a node
  and a gateway refuses a forwarded call, so no one can register a fake
  model. Clients need no key unless the gateway asks for one (`gateway auth
  on`). The token travels in plain HTTP: fine on a LAN or Tailscale, not
  across the internet.
- **Versions.** Each heartbeat carries the llmctl version; differing ones
  show as a warning in `llmctl registry` and `llmctl ps --all`, and a node on
  an incompatible cluster protocol is refused with a message saying so.
- In a cluster the gateway stays up with no local models (the machine still
  routes to the others) and comes back with autostart.
- `llmctl registry` shows the nodes, `registry clear` leaves the cluster,
  `registry serve --off` stops hosting it. `node set --advertise URL` sets
  the address others use for this machine when the one the registry sees
  is wrong (NAT, Tailscale).
- `registry set` takes a host (`lab`, port 9000), `host:port`, or a full URL
  for a registry behind a reverse proxy: `https://llmctl.example.net` means
  port 443. Behind a proxy the registry sees the proxy's address, not the
  node's, so for a heartbeat that came through one (see `gateway auth
  --proxy`) it uses the address the node reports on its default route
  instead; a laptop's follows it from network to network. Set `node set
  --advertise` when that address isn't the one others can reach.
- **Away from home.** After each heartbeat a node checks that every other
  node answers, as itself, at the address the registry gave. One that
  doesn't (a laptop on another network can't reach LAN addresses) gets its
  calls through the registry's gateway instead, pinned to the node chosen,
  as an ordinary client call. If that gateway wants an API key
  (`gateway auth on`), give the node one with `llmctl gateway key set`
  (prompts for it), and the clients of its own gateway need none:
  `http://localhost:9000/v1` works at home and away.
- Each gateway port (9000) must be reachable from the other machines, e.g.
  `sudo ufw allow from 192.168.8.0/24 to any port 9000 proto tcp`.

## Production: containers, bundles and serving files

For a server that runs from pinned images, installs models from offline
bundles and is described by a file. Everything here works without network
access. The compatibility promise for these interfaces is in
[docs/compatibility.md](docs/compatibility.md).

### Run a backend in a container

```bash
llmctl images                               # the images this llmctl was tested with
docker pull vllm/vllm-openai@sha256:c291…   # or `llmctl images load FILE` offline
llmctl config runtime vllm docker           # every vllm start from now on
llmctl config runtime llama docker --image ghcr.io/ggml-org/llama.cpp@sha256:…
llmctl run qwen3-8b --runtime docker --gpu 1     # or per launch
llmctl config runtime vllm host             # back to host processes
```

vLLM, SGLang and llama.cpp can run in Docker; host processes stay the
default. What a container start does:

- The image must be pinned by digest (`repo@sha256:…`); `--allow-tag` accepts
  a tag for development. llmctl never pulls: a missing image is an error.
- GPUs come through CDI (`--device nvidia.com/gpu=N`) when the host has CDI
  specs (`nvidia-ctk cdi generate`), else `--gpus`. `--gpu 0,1` picks them;
  on the host runtime it sets `CUDA_VISIBLE_DEVICES`.
- The model's folder is mounted read-only at `/model`, `MODELS_DIR` at
  `/models`. Files outside `MODELS_DIR` aren't visible in the container.
- The port is published on the bind address only (`llmctl config bind`); the
  engine listens on `0.0.0.0` inside. The container joins the `bridge`
  network unless `config runtime … --network NAME` says otherwise. Docker
  can't publish a port from a container with no network, so block egress on
  the host: a `DOCKER-USER` iptables rule that drops new outbound
  connections from the Docker bridge.
- DNS: `config runtime … --dns IP[,IP]` (or `run --dns`, or `dns:` in a
  serving file) passes `--dns` to `docker run`; the default is Docker's own.
  On a user-defined network without egress, Docker's embedded resolver can
  still answer outside names when it forwards from the host (typically when
  the host's resolver is a local one, such as systemd-resolved), which
  leaves a DNS channel out. `--dns 127.0.0.1` closes it: container names
  still resolve, outside names don't. `run --dns ''` drops a configured list
  for one launch. The setting is kept with the launch, so restarts and
  `restore` keep it.
- It runs as your user, with the offline and telemetry variables set and
  no host compiler: the image brings its own.
- `docker run` stays attached, so `ps`, `logs`, `stop`, `--wait`, `restart`
  and `restore` work as for host processes. `stop` also runs `docker stop`.

The image and its digest are reported in `ps`, in `/v1/models` (the
`llmctl` object's `runtime`, `image`, `image_digest` and `image_id`, per node
too), in the first lines of the model's log, and in the gateway's log when it
starts routing to the model.

### Images on hosts without a registry

```bash
# where the registry is reachable (needs skopeo):
llmctl images save vllm -o vllm.tar          # or repo@sha256:…
# on the offline host:
llmctl images load vllm.tar --json
llmctl images                                # present, loaded as fba1a5021627
llmctl images --protect                      # tag images prune would delete
```

`docker save` of `repo@sha256:<index>` loses that name: after `docker load`
the image has no tag and no digest, so the pinned ref doesn't resolve. Its
image ID survives, and the registry links the two (index → platform
manifest → config). llmctl runs the pinned ref by that ID:

- `images.json` lists, per platform, the manifest and config digest of each
  image llmctl ships. On Docker's classic store (overlay2) a loaded image's
  ID is its config digest, so a plain `docker load` is enough there.
- `llmctl images load` checks an archive before loading it: against the
  index and manifest that `llmctl images save` put in it
  (`llmctl-image.json`), or, for a plain `docker save` archive, against
  `images.json`. After the load the ID must be the archive's config digest
  (classic store) or the digest of the manifest it carries (containerd
  store, the default on fresh Docker 29 installs). It records the ID, so the
  pinned ref resolves on either store. Use it on containerd-store hosts.
- The pinned ref stays the identity: `ps`, `apply` and the compile-cache
  rule `digest` see the index digest; `image_id` records what ran, and the
  model log says "run by ID". No tag is ever used to find an image.
- An image with no tag (pulled by digest, or loaded) is what `docker image
  prune` deletes. `images load` and `images --protect` tag it
  `llmctl/<repo>:<digest12>` to prevent that.

`llmctl images load --json` prints `{"ok", "ref", "image_id", "tag",
"platform", "config", "manifest", "verified_by", "host_platform"}`; `save
--json` prints `{"ok", "ref", "platform", "config", "manifest", "file",
"size"}`. Exit codes: 3 not a docker save archive or no chain names the
image (or, for `save`, not pinned), 4 a digest doesn't match, 6 docker,
skopeo or the registry unavailable, or (`save`) the image isn't on this host.

### Install a model bundle

A bundle is a directory with an `llmctl-manifest.json` and the files it
lists:

```json
{
  "schema": "llmctl-manifest/1",
  "name": "qwen3.5-9b",
  "model": {"id": "Qwen/Qwen3.5-9B", "revision": "4f1c…", "format": "hf",
            "license": "apache-2.0", "license_file": "LICENSE"},
  "files": [{"path": "config.json", "size": 1534, "sha256": "9b1e…"}, …],
  "caches": [{"kind": "vllm", "path": "cache/vllm", "engine": "vllm",
              "engine_version": "0.29.0", "image": "vllm/vllm-openai@sha256:…",
              "gpu_arch": "sm_89"}]
}
```

```bash
llmctl manifest /fast/ml/models/qwen3.5-9b --license apache-2.0 --revision 4f1c…
llmctl install /media/usb/qwen3.5-9b --name qwen3.5-9b-2026.10.1
llmctl install oci://zot.local:5000/models/qwen3.5-9b@sha256:… --plain-http
llmctl verify qwen3.5-9b-2026.10.1
```

`install` copies only the listed files, hashing each one, into a temporary
folder under `MODELS_DIR`, and renames it into place only when every size
and sha256 matches. Anything else (a changed byte, a missing file, a
symlink) refuses the whole install and leaves nothing behind. An existing
folder is never touched: install versions side by side under `--name`, or
pass `--replace` (refused while that folder is served). `oci://` refs must
be pinned by digest and are pulled with the `oras` CLI.

The installed folder keeps the manifest (`.llmctl-manifest.json`) and a
provenance sidecar (`.llmctl-source.json`: model id, revision, licence, each
file's sha256 and the manifest's own sha256). `llmctl list` shows the
revision and licence; `llmctl verify` re-hashes against the sidecar; `llmctl
update` refuses an installed model. llmctl doesn't check signatures: sign
the manifest bytes, and the per-file hashes cover the rest.

Manifest rules: paths are relative POSIX paths without `..`; `size` is in
bytes and `sha256` is lowercase hex; files under a `caches` path are listed
in `files` like any other; unknown top-level keys are ignored.

With `--json`, `install`, `verify` and `cache export` print one JSON object
on stdout (messages go to stderr) and exit with:

| Code | Meaning |
|---|---|
| 0 | done |
| 1 | anything else |
| 2 | usage error |
| 3 | bad manifest (unreadable, unknown schema, unsafe path, missing key) |
| 4 | checksum or size mismatch, or a listed file missing |
| 5 | name conflict (the folder exists, or `--replace` while it's served) |
| 6 | source unreachable (`oras` missing, or the pull failed) |

### Ship a warm compile cache

vLLM compiles the model with `torch.compile` and Triton on its first start
(17 s for Qwen3-0.6B on a 4090, minutes for big models), and a cold start
leaves less memory for the KV cache. Containers keep these caches per model
under `<state>/cache/<name>/<folder>-<digest>/`, mounted at `/cache`, so the
second start is warm. To ship a warm cache in a bundle:

```bash
llmctl run qwen3.5-9b --runtime docker --wait      # compiles once
llmctl cache export qwen3.5-9b /bundles/qwen3.5-9b --json
llmctl manifest /bundles/qwen3.5-9b --license apache-2.0 …   # lists cache/vllm
```

On the target, the first start of the installed model seeds its cache from
the bundle when the shipped cache's image digest and GPU architecture match
the launch (rule `digest`), and logs `cache: seeded … cache rule: digest`.
The host runtime may also use a cache built with the same vLLM version
(rule `engine_version`, logged as a dev fallback, since it doesn't pin torch
or CUDA). Otherwise the start compiles (`cache rule: none`). Because the
model is mounted at `/model`, a cache built under one folder name hits
under another; for Qwen3-0.6B the warm start took torch.compile from 17.4 s
to 0.2 s and left 0.5 GiB more for the KV cache.

### Serving files: `llmctl apply`

A YAML file says what runs on a host:

```yaml
version: 1
models:
  qwen35-9b:                          # the name clients call
    model: qwen3.5-9b-2026.10.1       # a model key or folder under MODELS_DIR
    group: qwen3.5-9b                 # a route name that stays across versions
    backend: vllm                     # default: from the model's format
    runtime: docker                   # default: `config runtime`
    port: 8101                        # required; ports are pinned
    gpu: 0
    ctx: 32768
    n_parallel: 8
    kv_cache_mem: 6G
    restart: on-failure               # or never; restart_max: 5 (in 10 min)
    args: ["--max-num-batched-tokens", "8192"]
    env: {VLLM_LOGGING_LEVEL: INFO}
```

```bash
llmctl apply host.yaml bundles/*.yaml --dry-run   # what would change
llmctl apply host.yaml bundles/*.yaml             # make it so
llmctl apply host.yaml --prune                    # and stop what isn't listed
```

Each entry is started as `llmctl run <model> --name <key> --port <port> …`
would start it, so it's recorded, restarted and restored like any other.
An entry running with the same arguments and runtime is left alone; a
changed one is restarted; a missing one is started. Stops happen before
starts, and each start waits for `/health` (`--no-wait` doesn't). Files
merge: a bundle can ship a one-model fragment next to the host file. The
same model name or port in two places is an error, and every error names
the file and line. `--json` prints the actions; exit 3 means invalid files,
1 that a start failed.

Rolling back is applying the previous version's file: `model` changes, the
name and group don't, and the old version's cache is already warm. On the
4090 that took 38 s for Qwen3-0.6B in a container.

Keys: `model`, `port`, `args`, `env`, and the `run` options `backend`,
`group` (`null` for none), `runtime`, `image`, `allow_tag`, `dns` (a list), `gpu`, `ctx`,
`n_parallel`, `gpu_mem`, `kv_cache_mem`, `kv_quant`, `tool_parser`,
`reasoning_parser`, `trust_remote_code`, `reasoning`, `embeddings`,
`no_vision`, `draft`, `no_draft`, `spec_type`, `spec_n_max`,
`n_gpu_layers`, `kv_unified`, `ctx_checkpoints`, `cache_ram`, `restart`,
`restart_max`, `owner`.

## Monitoring and operations

What a monitoring stack (Prometheus or VictoriaMetrics, Loki or
VictoriaLogs, an OpenTelemetry collector) reads from a node, and what
happens when a model dies. The names below are covered by
[docs/compatibility.md](docs/compatibility.md).

### Metrics

Every gateway serves Prometheus text format:

| Endpoint | What |
|---|---|
| `GET /metrics` | the gateway's own metrics (below) |
| `GET /metrics/sd` | [HTTP service discovery](https://prometheus.io/docs/prometheus/latest/http_sd/) JSON: one target per local engine, with `__metrics_path__` and `model`, `node`, `backend` labels |
| `GET /metrics/backends/<model>` | that engine's own `/metrics` (vLLM, SGLang, llama-server), with `model`, `node`, `backend` added to every sample |

Gateway metrics, labelled `model`, `node` (where the model ran) and
`backend`:

| Metric | Type |
|---|---|
| `llmctl_requests_total{…,status}` | counter, by HTTP status |
| `llmctl_request_errors_total{…,code}` | counter, by error code (`model_unavailable`, `model_not_found`, `busy`, `stream_aborted`, `client_left`, or the engine's own) |
| `llmctl_request_duration_seconds` | histogram, call to last byte |
| `llmctl_time_to_first_token_seconds` | histogram, streamed calls: call to first event (buckets 0.05…60 s) |
| `llmctl_tokens_total{…,direction="in"\|"out"}` | counter, from the replies' `usage` (streams only with `stream_options.include_usage`; the engines' own counters are under `/metrics/backends`) |
| `llmctl_requests_in_flight`, `llmctl_requests_queued` | gauges: being relayed; waiting for an engine slot |
| `llmctl_spills_total{model,from_node,to_node}`, `llmctl_failovers_total{model,from_node,reason}` | counters |
| `llmctl_node_up{node}` | 1 for nodes in the cluster view, 0 for ones that left |
| `llmctl_instance_state{…,state}` | 1 for each local model's state: `starting`, `ready`, `restarting`, `crashed`, `down` |
| `llmctl_instance_restarts_total{model,node}` | supervisor restarts |
| `llmctl_build_info{version,node}` | 1 |

A call is counted by the gateway the client called, not again by a node it
was handed to, so sums over all scraped gateways don't double count.
llama-server runs with `--metrics` (models started before 1.2 get it at
their next restart).

Scrapes follow the API-key rules of `/v1`: open from the local networks,
otherwise they need a key. `llmctl gateway metrics-token` makes a
scrape-only token instead, so the monitoring host holds nothing that can
run a model:

```yaml
# vmagent / Prometheus
scrape_configs:
  - job_name: llmctl
    bearer_token_file: /etc/vmagent/llmctl-token
    static_configs: [{targets: ["ws:9000", "orin:9000"]}]
  - job_name: llmctl-engines
    bearer_token_file: /etc/vmagent/llmctl-token
    http_sd_configs:
      - url: http://ws:9000/metrics/sd
        bearer_token_file: /etc/vmagent/llmctl-token
      - url: http://orin:9000/metrics/sd
        bearer_token_file: /etc/vmagent/llmctl-token
```

### When a model dies

The gateway supervises the models in the autostart manifest that it has
seen ready. One that crashes is restarted from its launch spec after 2, 4,
8 … s (capped at 60 s); after 5 restarts within 10 minutes it's left down
as `crashed`, and 10 minutes ready resets the count. `run --restart never`
leaves it down at once, `--restart-max N` changes the limit, and serving
files take `restart:` and `restart_max:`. `llmctl stop` takes a model out
of the manifest first, so a stop is never mistaken for a crash, and a
model that dies before it was ever ready (a failed load) isn't restarted:
it gets a `start_failed` event naming its log.

A planned restart is not a crash either. When `llmctl apply` restarts a
model (new files, new flags), it holds the model first: the supervisor
leaves it alone, and the model is supervised again once the new instance
is ready. Callers never see it vanish, because the gateway answers 503
until the new instance is ready.

What clients see:

- A call for a configured model that is down gets **503
  `model_unavailable`** with `Retry-After`, in the OpenAI error format.
  Other nodes' instances of the model are tried first. The same goes for
  a model that `apply` is restarting (`Retry-After: 30`) and one whose
  engine is still loading (`Retry-After: 10`), never a 404 in between.
- A stream whose engine dies ends with one last SSE event,
  `data: {"error": {"code": "model_unavailable", …}}`, and a clean end
  of the response, instead of a cut connection.

`llmctl ps` shows each model's STATE and RESTARTS (`--json` for scripts),
including down models the gateway is restarting; `/v1/models` lists down
models with `status: restarting` or `crashed`.

When several people or agent sessions share a machine, tag who started what:
`llmctl run MODEL --owner A1` (or `LLMCTL_OWNER=A1` in that session's
environment, or `owner:` in a serving file). The tag shows in `llmctl ps`
(OWNER), in `llmctl events` and their `owner` key, and in the `llmctl`
block of `/v1/models`. Restarts, `restore` and the supervisor keep it, and
a decision endpoint takes its model's. It's only a label: it grants
nothing, and a different tag is no reason for `apply` to restart a model.

`llmctl events [-f] [--json]` lists what happened to models, one JSON line
each in `<state>/events.jsonl`:

| event | when | by |
|---|---|---|
| `started` | a model was launched (pid, port, log) | `run`, `apply` |
| `ready` | it answered `/health` | `run --wait`, `apply --wait`, `supervisor` |
| `start_failed` | it exited, or wasn't ready in time, before it was ever ready (log) | `run --wait`, `apply --wait`, `supervisor` |
| `stopped` | it was stopped on purpose (`apply` gives the reason) | `stop`, `apply` |
| `restarting`, `restarted`, `restart_failed`, `crashed`, `starting`, `forgotten` | crash handling | `supervisor` |

Each line has `ts`, `model`, `event` and `detail`, plus `pid`, `port`,
`by`, `reason` and `log` when they apply. `llmctl logs NAME` shows the last
log of a model that isn't running, such as one that failed to start. `scripts/chaos-test.py MODEL` kills a model mid-stream and
checks the error arrives within 10 s and the model serves again within
120 s.

### Logs, request ids and traces

```bash
llmctl config gateway-log json                 # one JSON object per line
llmctl config gateway-otlp http://lab:4318     # export spans (off by default)
llmctl gateway restart                         # both take effect at a restart
```

Straight to [Phoenix](https://phoenix.arize.com), which takes OTLP protobuf
only, with a Phoenix project per API key:

```bash
llmctl config gateway-otlp http://lab:6006 --protocol protobuf \
    --header "Authorization=Bearer <phoenix api key>"   # when Phoenix has auth on
llmctl config gateway-otlp-project pa personal-assistant   # calls with key "pa"
llmctl config gateway-otlp-project '*' none                # export no other call
llmctl gateway restart
```

A project per key needs the client to send its key (`llmctl gateway key add
pa`), even from the LAN where the gateway doesn't ask for one.

- JSON lines have `ts`, `level`, `node`, `event` and `msg`. Each call
  logs `event: "request"` with `request_id`, `trace_id`, `method`, `path`,
  `model`, `backend`, `served_by`, `status`, `code`, `duration_ms`,
  `ttft_ms`, `tokens_in`, `tokens_out`, `bytes_out` and `client`.
  Supervisor, failover, spill, relay and cluster events have their own
  `event`. Health checks and scrapes aren't logged.
- No prompt or reply text is logged, in any mode. The engines don't log
  prompts either: llmctl never passes vLLM's `--enable-log-requests`,
  SGLang's `--log-requests` or llama-server's `--verbose`.
- `X-Request-Id` from the client is kept (up to 128 visible characters),
  otherwise one is made. It goes to the engine and comes back on every
  reply, errors included.
- The gateway continues the client's W3C `traceparent` (or starts a trace),
  and sends each engine a `traceparent` for its upstream attempt. With an
  OTLP endpoint it exports a server span per call and a client span per
  attempt, from a background queue that drops spans rather than slowing
  calls. Failover attempts, `/v1/systemone` and `/v1/embeddings` are
  covered the same way; `tracestate` passes through. vLLM exports its own
  spans when started with `args: ["--otlp-traces-endpoint",
  "http://lab:4317"]`; the pinned image includes the OpenTelemetry SDK.
- Spans carry metadata only, never prompts, replies, tool calls, decision
  state or questions, or embedding inputs. Resource: `service.name`
  (`llmctl-gateway`), `service.version`, `host.name`, and with a project
  `openinference.project.name`. The server span (`POST /v1/...`):
  `http.request.method`, `url.path` (no query), `llmctl.request_id`,
  `llmctl.node`, `llmctl.key_name` (the API key's name, when projects are
  set), `http.response.status_code`, `error.type`, `llmctl.model`,
  `llmctl.backend`, `llmctl.served_by`, `llmctl.ttft_ms`,
  `gen_ai.usage.input_tokens`, `gen_ai.usage.output_tokens`,
  `llmctl.failovers`, and from llama-server's `timings`
  `llmctl.prefill_ms`, `llmctl.prefill_tokens`, `llmctl.cached_tokens`,
  `llmctl.decode_ms`, `llmctl.decode_tokens_per_s`. Each attempt span
  (`upstream <model>`): `llmctl.model`, `llmctl.node`, `server.address`,
  `server.port`, `http.response.status_code`, `error.type`.
- `LLM_RUNNER_GATEWAY_LOG`, `LLM_RUNNER_GATEWAY_OTLP`,
  `LLM_RUNNER_GATEWAY_OTLP_PROTOCOL` and `LLM_RUNNER_GATEWAY_OTLP_HEADERS`
  (`Name=value,Name2=value2`) override the settings.

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

Other persisted settings:

| Command | Setting |
|---|---|
| `llmctl config vllm-bin <path>` (`--clear`) | The `vllm` CLI; `$LLM_RUNNER_VLLM_BIN` overrides it (see the vllm/sglang install section). |
| `llmctl config sglang-bin <path>` (`--clear`) | The `sglang` CLI; `$LLM_RUNNER_SGLANG_BIN` overrides it. |
| `llmctl config gateway PORT` (`--off`) | The gateway port, default 9000. |
| `llmctl config bind HOST` (`--reset`) | The address the gateway and models listen on, default `0.0.0.0`. |
| `llmctl config offline on\|off` | Offline mode (below); `$LLM_RUNNER_OFFLINE` overrides it. |
| `llmctl gateway auth`, `gateway key` | API keys (see the gateway section). |
| `llmctl node set`, `registry set` | Cluster membership (see the registry section). |

`llmctl config show` prints them all, with secrets masked.

### Offline hosts and telemetry

Every backend starts with telemetry off (`VLLM_NO_USAGE_STATS`,
`VLLM_DO_NOT_TRACK`, `DO_NOT_TRACK`, `HF_HUB_DISABLE_TELEMETRY`); `-E` can
still override them. On an air-gapped host, `llmctl config offline on` (or
`LLM_RUNNER_OFFLINE=1`) makes `download` and `update` refuse, and starts
backends with `HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1`, so they load
from disk or fail instead of trying the network.

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
- `cache/` — per-model compile caches for container (and bundle) starts.

Wipe state if it ever desyncs from reality:

```bash
rm -rf ~/.local/state/llm-runner
```

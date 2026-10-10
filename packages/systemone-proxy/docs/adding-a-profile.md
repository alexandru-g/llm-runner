# Adding a decision model (a "profile") to systemone-proxy

This guide is for a coding session (human or Claude) that adds support for a
new calibrated decision model. It covers how the pieces fit, the workflow
that worked for the first five models, the profile interface, and the traps
that cost time.

Read [`client-guide.md`](client-guide.md) first if you don't know what the API
looks like from the outside.

## How the pieces fit

```
client ──POST /v1/systemone──▶ systemone-proxy ──/completion (token ids)──▶ llama-server ──▶ GGUF
            unified JSON           │ profile: vendor prompt,                  n_predict 1,
                                   │ labels, temperatures, maths              n_probs N
```

- **Decision models** are fine-tuned LLMs whose answer is the next-token
  probability of option labels (`A`, `B`, ...) under a vendor-specific prompt,
  rescaled with fitted temperatures. Each vendor ships its own wrapper server
  (usually on vLLM). The proxy re-implements each vendor's wrapper on top of
  a plain llama-server, behind **one** request/response format.
- **A profile** (`src/systemone_proxy/profiles/<model>.py`) holds everything
  model-specific: prompt, label tokens, temperatures, confidence formulas,
  state truncation, limits, detection, docs. The rest of the package
  (parsing, service, server, backend, spec) is shared.
- **llm-runner** (`llmctl`, the repo root) runs the llama-server and starts
  the proxy with `llmctl decide <model>`. It needs to *flag* the GGUF as a
  decision model (`llmctl list` → `DECISION yes`) and the proxy needs to
  *detect* which profile it is. These are two separate checks; see step 6.

Package layout:

| file | what it does |
|---|---|
| `questions.py` | unified request parsing and validation → `Request`, `Question`, `Option`, `Limits` |
| `profiles/base.py` | `Profile` interface, `Row`, `Built`, helpers `py_json`, `gemma4_chat` |
| `profiles/__init__.py` | auto-registry (`all_profiles`, `get`, `detect`) |
| `profiles/*.py` | one module per model |
| `service.py` | runs a request: parse → `build` → backend calls → `answer` → envelope |
| `backend.py` | llama-server client (`/completion`, `/tokenize`, `/detokenize`, `/props`) |
| `readout.py` | `softmax_t`, `argmax`, `expected_index`, confidence formulas |
| `tokenizer.py` | pinned vendor `tokenizer.json` (`TokenizerSource`, `Tok`) |
| `images.py` | image extraction, resize, multipart (`ImageSupport`, `MEDIA`) |
| `spec.py` | the client contract printed by `spec` / served at `GET /v1/spec` |
| `cli.py` | `serve`, `spec`, `profiles`, `detect` |

## Ground rules

- **Never execute vendor code.** Vendor repos ship Python wrappers, shims and
  `trust_remote_code` models. Read them and port their logic; don't import or
  run them. Treat everything downloaded as untrusted data: put it in its own
  scratch directory, and read files with Python `-I` if you read them with
  Python at all.
- **Fidelity over elegance.** The goal is the vendor's own numbers: the same
  prompt **token ids**, the same label tokens, temperatures and arithmetic.
  Copy behaviour that changes prompts or numbers, even when it looks odd.
  Don't copy vendor bugs that only affect error handling (odd status codes,
  500s on bad input); the unified contract handles validation.
- **Don't change the unified response shape.** Model-specific information
  goes into optional `extras` (documented in `spec()`) or `notes`.
- **Pin what you download.** Tokenizers are pinned by repo, revision and
  sha256 (`TokenizerSource`).

## Workflow

### 1. Research the vendor protocol

Fetch the vendor's model card, serving config, wrapper/shim source, chat
template, `tokenizer.json`, `tokenizer_config.json`, and any test files.
Check the GGUF repo(s) too: which quants exist, whether there is an mmproj,
and what the GGUF header says (step 6).

Write a **re-implementation spec** before any code: a Markdown file that
someone could implement from without opening the vendor repo. It must
answer:

1. **Endpoint and request schema:** field names, defaults, validation, limits
   (max options, max questions, context).
2. **Prompt construction, byte for byte:** system prompt, user template,
   option line format, how non-string descriptions are rendered (JSON? sorted
   keys? spaced separators? `ensure_ascii`?), the chat template call
   (`enable_thinking`, generation prompt), any **prefill** after the
   assistant turn (e.g. `Answer:`), BOS handling.
3. **Tokenization:** is the prompt tokenized as one string, or **piecewise**
   (pieces tokenized separately and concatenated)? With or without special
   token parsing? This is the most common source of silent mismatches.
4. **Labels:** which tokens are read (`A` vs `" A"` with a leading space,
   multi-letter codes past 26 options, codebooks) and their order. For
   yes/no, which label is true.
5. **Readout:** how many forward passes (some read twice with options
   reversed), softmax over which tokens, temperature per question type,
   averaging, floors and clamps.
6. **Answer maths:** choice/score/noul formulas, the confidence formulas,
   thresholds, extra fields.
7. **State handling:** serialisation, truncation (head? tail? token or char
   based?) or rejection when too long.
8. **Usage accounting:** how the vendor counts `input_tokens`.
9. **Golden examples:** every worked example from the README, with the
   vendor's published response. You'll test against these.

Keep the spec in the session scratchpad; it is working material, not repo
documentation. The profile module's docstring and `spec()` carry the parts
a reader needs.

### 2. Write the profile

Create `src/systemone_proxy/profiles/<model>.py`. Copy the structure of the
closest existing profile:

| profile | what it shows |
|---|---|
| `quyet.py` | the simplest: Gemma 4 chat, letters A–J, per-type temperatures, state truncation |
| `h2o_lightning.py` | ChatML (Qwen) with prefill, labels read with a leading space, noul floor and thresholds, **images** |
| `bobcat.py` | **piecewise tokenization**, large identifier sets, context-margin rejection |
| `rune.py` | codebook beyond 26 options, one temperature for every type, filename fallback in `matches` |
| `xor.py` | **two backend calls per question** (options forward and reversed), averaging |

Any `Profile` subclass with an `id` in the `profiles` package is registered
automatically; there is nothing to add elsewhere in the proxy.

### 3. Write the tests

`tests/test_<model>.py`, offline only, with golden data in
`tests/fixtures/<model>_*.json`. The minimum:

- **Golden token ids:** the full id sequence of each README example prompt
  equals what the vendor's construction gives (build it from your spec with
  the vendor tokenizer). This catches piecewise vs whole-string bugs,
  whitespace, BOS and prefill errors.
- **Template parity:** if the vendor renders a chat template, render the
  fixture copy of the jinja template (`gemma4_chat_template.jinja`,
  `h2o_lightning_chat_template.jinja`) and compare it with your prompt
  builder (see `test_gemma4_template.py`).
- **Label ids:** each label is one token, distinct, and decodes as expected.
- **Answer maths:** fixed logprobs in, the vendor's numbers out (temperature,
  averaging, floors, confidence). Use `FakeBackend` from `conftest.py` for a
  service-level test.
- **Limits and state:** option-count limits, truncation or `too_long`.
- **`matches`:** true for every published GGUF of the model, false for its
  base model and for the other profiles' GGUFs.
- **`spec()`** works on an uninitialised instance and has the expected keys.

Run the whole suite from `packages/systemone-proxy`: `uv run pytest -q`.

### 4. Validate live

Unit tests prove you match your *reading* of the vendor. Only a live run
proves you match the *vendor*:

1. Download a Q8_0 (or f16) GGUF. Q4 quants shift the probabilities enough to
   hide small bugs; they're fine for checking that the right option wins, not
   for checking numbers.
2. `llmctl run <model> -c 16384`, then `llmctl decide <model>`. To test the
   repo copy of the proxy rather than the installed one, set
   `LLM_RUNNER_SYSTEMONE_PROXY=$PWD/packages/systemone-proxy/.venv/bin/systemone-proxy`.
3. The proxy's startup **self-test** must pass (it compares your tokenizer
   with llama-server's on `selftest_texts()`).
4. Send each README example and compare with the vendor's published
   response. Expect agreement to about 0.001 on Q8_0. Also compare
   `usage.input_tokens`: it is an exact check of prompt length.
5. If there's no published example, or numbers disagree, run a reference:
   the vendor's checkpoint on vLLM (bf16), queried directly with the same
   prompt and `logprobs`. Don't run the vendor's shim; reproduce its request
   to vLLM by hand. On the PC, vLLM lives in the conda env
   `~/miniconda3/envs/vllm` (if FlashInfer fails to compile, set
   `VLLM_USE_FLASHINFER_SAMPLER=0`). Stop the GGUF instance first, or vLLM
   won't find enough free VRAM.
6. Check `debug: true` output when a number is off: it returns each row's
   raw label logprobs and prompt sizes.

### 5. Document

- `spec()` is what client developers read. Write it for them: label order,
  temperatures, state handling, limits, extras, and notes on anything
  surprising (see [client-guide.md](client-guide.md) for how it's read).
- Add the model to the supported list in the repo `README.md` ("Decision
  endpoints with `llmctl decide`") and to the package `pyproject.toml`
  description.
- Add the GGUF repos to `gguf_repos`.

### 6. Make llmctl flag and detect it

Two separate checks must both work for the model's real GGUF files:

- **llmctl's flag** (`src/llm_runner/backends.py`, `is_decision_model`):
  true when the GGUF header's `general.tags` contain one of
  `_DECISION_TAGS`, or a sidecar file from `_DECISION_SIDECARS` sits beside
  it. When a header has no tags at all, the file name is matched against
  `_DECISION_FILENAMES`. Add the model's tag or file-name prefix there if
  neither covers it.
- **The proxy's detection** (`Profile.matches(meta, filename)`): decides which
  profile serves the file. Match on `general.basename` plus architecture,
  size or version, specific enough not to match the base model or other
  fine-tunes.

**Check every quant, not just one.** Livesport's own
`rune-26b-a4b-v3-Q8_0.gguf` has a stripped header (no basename, no tags,
`general.name: "Hf"`), while their IQ4_XS has the full metadata. Read the
header of each published file:

```bash
uv run python -c "from systemone_proxy.gguf import general_metadata as m; print(m('/path/to/file.gguf'))"
```

### 7. Definition of done

- [ ] Re-implementation spec written; open questions resolved or listed in `notes`.
- [ ] Profile module with docstring naming the upstream source and revision.
- [ ] Offline tests: golden ids, template parity, labels, maths, limits, `matches`, `spec`.
- [ ] Full suite passes.
- [ ] Live run on Q8_0: self-test passes, README examples match, `input_tokens` matches.
- [ ] `llmctl list` shows `DECISION yes` and `systemone-proxy detect` finds the profile, for every published quant.
- [ ] `spec()` complete; README supported-model list updated.

## The Profile interface

From `profiles/base.py`. Class attributes:

| attribute | meaning |
|---|---|
| `id` | stable profile id, e.g. `"rune-26b-a4b-v3"` (clients see it in `model`) |
| `name` | human name |
| `upstream` | where the protocol comes from (repo URL and file names) |
| `gguf_repos` | known GGUF repos, for docs |
| `tokenizer_source` | `TokenizerSource(repo, revision, sha256, filename="tokenizer.json")` |
| `temperatures` | `{"choice": T, "score": T, "noul": T}` |
| `limits` | `Limits(max_questions=64, max_choice=255, max_score=10)`: the vendor's limits |
| `image_support` | `None` (text only) or `ImageSupport(max_images, max_pixels, min_pixels, factor)` |

Methods:

| method | contract |
|---|---|
| `matches(meta, filename)` *(classmethod)* | `meta` is the GGUF `general.*` dict, `filename` the base name |
| `__init__(tok)` | call `super().__init__(tok)`; precompute label ids and verify each is one token (raise if not) |
| `build(req) -> Built` | turn a parsed `Request` into backend rows |
| `answer(q, logprobs, **kw) -> dict` | `logprobs[tag]` is aligned with that row's `label_ids`; return the unified answer |
| `selftest_texts()` | full golden prompt strings (special tokens as text) that must tokenize identically in your tokenizer and llama-server |
| `selftest_labels()` | `{token_id: text}` the GGUF must detokenize each label to (e.g. `" A"`) |
| `spec()` | model-specific docs; **must not use instance state** (it's called on an uninitialised instance) |

`self.n_ctx` holds llama-server's context size (set by the service before the
first `build`; `None` in offline tests). Use it when the vendor degrades
gracefully near the limit (Xor drops its reversed row; Bobcat rejects within
a margin).

### What the profile receives

`parse_request` has already validated everything generic:

- `req.state`: `str | dict | list` (with images, inline ones are already
  replaced by `"[image N]"`).
- `req.questions`: tuple of `Question(qid, type, instructions, options,
  threshold, raw)`.
  - **choice:** `options` in request order; `key` = label, `desc` = raw JSON
    value or `None`.
  - **score:** keys `"0".."n-1"`, `desc` = the raw level value.
  - **noul:** always `(Option("true", d), Option("false", d))` in that order,
    whatever your vendor's label order is. Map it yourself.
  - `raw` is the original question dict (for vendor aliases).
- `req.images`: fitted images (`png`, `width`, `height`) when the profile
  supports them.

### What `build` returns

```python
Built(
    rows=[Row(qid, ids, label_ids, tag="main")],  # one or more per question
    warnings=[],               # e.g. {"code": "state_truncated", ...}
    truncated=set(),           # qids whose state was truncated → answer gets "truncated": true
    input_tokens=None,         # None → sum of row lengths; set it to follow the vendor's accounting
    answer_kwargs={},          # passed to answer() for every question (e.g. {"image": True})
)
```

- `Row.ids` are the **exact prompt token ids**, including BOS if the vendor
  sends one. The backend sends them as-is; llama-server applies no template.
- `Row.label_ids` are read at the position after the prompt.
- Several rows per question use distinct `tag`s (`"forward"`, `"reverse"`).
- Image rows also set `text` (the prompt with one `images.MEDIA` per image),
  `images` (PNG bytes) and `extra_tokens` (image tokens beyond what `ids`
  counts). See `h2o_lightning.py`.

### The unified answers

Exact keys, nothing rounded:

```text
choice: {"type": "choice", "choice": label, "confidence": float, "probabilities": {label: p}}
score:  {"type": "score", "score": expected index (0-based float), "confidence": float,
         "legend": {"0": level, ...}, "probabilities": {"0": p, ...}}
noul:   {"type": "noul", "noul": P(true), "confidence": float}
```

Use the vendor's confidence formula when it has one (see `readout.py`);
otherwise `binary_confidence` for noul. Vendor-only fields that a client
needs (e.g. h2o's `threshold`/`decision`) are extra keys, listed in
`spec()["extras"]`.

### `spec()` keys

`confidence` (`{choice, score, noul}`), `labels`, `noul_order`, `state`,
`limits` (dict, merged into the generic limits), `calls`, `extras`
(`{field: meaning}`), `notes` (list), and optionally `images`.

## Traps (each of these cost real time)

1. **Piecewise tokenization.** Bobcat and Xor tokenize prompt pieces
   separately and concatenate the ids. Tokenizing the joined string gives
   different ids at the seams, with no error anywhere. Always build ids the
   vendor's way and test against golden ids.
2. **Special-token text.** `Tok.encode(text, special=True)` parses
   `<|turn>` etc. as special tokens; `special=False` treats them as plain
   text. Match what the vendor's tokenizer call does
   (`add_special_tokens`, `split_special_tokens`).
3. **Leading spaces on labels.** h2o reads `" A"` (one token, `ĠA`), not
   `"A"`. Derive label ids from the prompt context (encode `prefill + " A"`
   and take the last token), and check it adds exactly one token.
4. **BOS.** Gemma templates emit `<bos>` themselves; Qwen has none. A
   second BOS shifts everything; the backend checks `tokens_evaluated ==
   len(ids)` to catch it.
5. **The prompt cache changes answers.** On Gemma 4 (sliding-window
   attention), reusing llama-server's cached prefix changed results. The
   service sends `cache_prompt: false` by default; don't build anything that
   depends on caching.
6. **Stripped GGUF headers.** See step 6: check every quant's metadata.
7. **Same tokenizer, different model.** All Gemma 4 fine-tunes share one
   tokenizer (sha256 `cc8d3a0c…`), so the startup self-test can't tell them
   apart. `matches` is what distinguishes them; keep it specific.
8. **Labels outside the top-n.** llama-server returns the top `n_probs`
   logprobs only. The service asks for at least labels + 64 and bounds any
   missing label by the smallest returned logprob (warning
   `label_outside_top_n`). A profile with hundreds of labels relies on this.
9. **Quantisation.** Temperatures were fitted on bf16/fp8. Validate numbers
   on Q8_0; Q4 moves them by a few hundredths.
10. **JSON rendering.** Python's `json.dumps` defaults (`", "` separators,
    `ensure_ascii=True`, float repr) are part of the prompt. Use
    `base.py_json` with the vendor's exact options.
11. **The vendor's docs and code disagree.** Rune's card writes
    `json.dumps(state)` but the protocol owner's code keeps non-ASCII raw.
    Follow the code that served the published numbers, and say so in
    `notes`.

## llama-server facts the backend relies on

- `/completion` with `prompt` as a token-id array, `n_predict: 1`,
  `n_probs: N`, `post_sampling_probs: false` returns
  `completion_probabilities[0].top_logprobs`: a log-softmax of the **raw
  logits over the full vocabulary** (no temperature, no samplers). Softmax
  over the label logprobs divided by T therefore equals the vendor's softmax
  over label logits divided by T.
- `/tokenize` (`parse_special`) and `/detokenize` back the self-test.
- `/props` gives `model_path`, `default_generation_settings.n_ctx`,
  `modalities.vision` and the per-process random `media_marker`.
- With images, `prompt` is `{"prompt_string": ..., "multimodal_data":
  [base64, ...]}`; the marker in the string is replaced by each image's
  embeddings, wrapped in the model's image start/end tokens by mtmd.

# Known issues

Problems seen in real use that aren't fixed yet.

## Gemma 4 loses its answer after a stray `<channel|>`

**Status:** open, being watched. Found 2026-11-04.

**Seen on:** `gemma-4-31B-it-UD-Q4_K_XL.gguf` on llama-server b11805, launched with
`--reasoning-budget 0`. Other builds and Gemma sizes are untested.

**What you see:** the reply writes a plan, then stops mid-generation. A stream goes
silent while the model keeps generating, then ends normally. Nothing shows in the
logs, and the GPU stays busy until the reply ends.

**What happens:** the model sometimes thinks in visible text although thinking is
off, emits `<channel|>`, then writes the real answer. llama-server drops everything
after that token. Streaming and non-streaming replies both come back cut short, with
`finish_reason: stop`, so a client can't tell. This affects every client of the
model, not only `llmctl chat`.

**How often:** 1 of the 20 seeds tried, on one prompt. How often it happens on other
prompts is unknown.

### Reproducing it

Send this to the model, with default sampling:

```json
{"messages": [{"role": "user", "content": "hi"},
              {"role": "assistant", "content": "Hello! How can I help you today?"},
              {"role": "user", "content": "write a snake game in python"}],
 "seed": 219, "max_tokens": 4000, "logprobs": true, "top_logprobs": 1}
```

On the setup above it repeats every time: 2,163 tokens, with `<channel|>` (token 101)
at index 406. `message.content` has 1,454 characters, while the tokens listed in
`logprobs.content` add up to about 7,200, including the whole answer.
`"reasoning_format": "none"` doesn't bring the lost text back.

### Spotting it

- **A short reply for its tokens:** in the gateway's JSON log, a stream normally uses
  about 260 bytes per token (`bytes_out / tokens_out`). The stalled replies used about
  46, and such a reply also spends far more `tokens_out` than its text accounts for.
- **A long silence:** no stream data for many seconds while the slot stays busy, with
  `requests_processing` at 1 in the engine's metrics.

### Possible fixes (none tried)

1. See whether a newer llama.cpp fixes it. Not checked yet.
2. Patch llama-server to keep the text after a stray `<channel|>`, treating what came
   before it as reasoning.
3. Remove `--reasoning-budget 0`. Replies get slower, and it means restarting the model.

Replaying the request above shows whether a fix works.

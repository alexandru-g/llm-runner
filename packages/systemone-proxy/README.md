# systemone-proxy

One `POST /v1/systemone` endpoint for calibrated decision models, served on top
of a plain `llama-server`. Each supported model has a profile that reproduces
its vendor's exact prompt, label readout and temperatures, so the
probabilities match the vendor's own server; every model answers in the same
response format.

Usually started through `llmctl decide <model>` (see the llm-runner README);
it can also run on its own:

```bash
systemone-proxy profiles                         # supported models
systemone-proxy spec --profile rune-26b-a4b-v3   # the client contract for one model
systemone-proxy serve --backend http://127.0.0.1:8080 --port 8090   # profile detected from the GGUF
```

Install: `uv tool install ./packages/systemone-proxy` (or `pipx install`).
Tests: `uv run pytest` from this directory.

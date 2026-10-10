# systemone-proxy

Serves calibrated decision models behind one `POST /v1/systemone` API on top of
llama-server. One profile per model in `src/systemone_proxy/profiles/`.

- Adding a model: follow `docs/adding-a-profile.md` (workflow, interface, traps).
- The client-facing contract: `docs/client-guide.md`, and `systemone-proxy spec`.
- Tests: `uv run pytest -q` from this directory (offline; tokenizers are fetched
  once into the HF cache).
- Never run vendor code from model repos; port it.

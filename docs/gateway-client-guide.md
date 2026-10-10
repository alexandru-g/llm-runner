# Calling models through the llmctl gateway

This guide is for code (or an agent) that calls local models. It explains how
to move from one base URL per model (`http://host:8080/v1`,
`http://host:8081/v1`, ...) to the single gateway, which reaches every running
model on every machine in the cluster.

## The short version

- **One base URL:** `http://<machine>:9000/v1`. Any machine works:
  - `lab` (192.168.8.213) is always on and a good default;
  - `pc` (192.168.8.232) and `orin` (192.168.8.115) work too.
- **Discover models:** `GET /v1/models` lists every model running anywhere in
  the cluster.
- **Pick a model:** put its name in the request's `"model"` field. A model's
  name is its llmctl instance name, such as
  `"gemma-4-26b-a4b-it-ud-q4_k_m"`, or its group name, such as
  `"gemma-4-26b-a4b-it"`.
- **No API key** is needed.

The gateway chooses the machine and forwards the request. The response is the
model server's own response, streamed through unchanged.

```python
from openai import OpenAI

client = OpenAI(base_url="http://lab:9000/v1", api_key="unused")
print([m.id for m in client.models.list().data])
reply = client.chat.completions.create(
    model="gemma-4-26b-a4b-it",            # a group: the best quant on any machine
    messages=[{"role": "user", "content": "Hello"}],
)
```

## Migrating from per-model ports

| Before | After |
|---|---|
| `base_url = "http://orin:8080/v1"` for Gemma | `base_url = "http://lab:9000/v1"` |
| `model` ignored, or any string (llama-server serves only one model) | `model` **must** name the model, because several are running |
| One client per model | One client; change only `model` |
| A machine going down breaks the client | The gateway uses another copy if one is running |
| `http://orin:8082/completion` (llama-server native) | `http://lab:9000/m/quyet-1.0-large.q8_0/completion` |
| `http://orin:8090/v1/systemone` (decision endpoint) | `http://lab:9000/v1/systemone` with `"model": "quyet"` |

The per-model ports still work. Calling a port directly skips routing,
failover and spill-over.

## Finding models: `GET /v1/models`

This is the standard OpenAI list, plus an `llmctl` block on each entry:

```json
{"object": "list", "data": [
  {"id": "gemma-4-26b-a4b-it-ud-q4_k_m", "object": "model", "created": 1791646300,
   "owned_by": "llmctl",
   "llmctl": {
     "node": "rtx", "local": false, "status": "ready",
     "api": "openai", "backend": "llama", "group": "gemma-4-26b-a4b-it",
     "context": 4096, "vision": true, "embeddings": false, "decision": false,
     "parallel": 1, "path_prefix": "/m/gemma-4-26b-a4b-it-ud-q4_k_m",
     "nodes": [
       {"node": "rtx",  "priority": 0, "local": false, "status": "ready", "port": 8080, "context": 4096,  "vision": true},
       {"node": "orin", "priority": 1, "local": true,  "status": "ready", "port": 8080, "context": 32768, "vision": true}]}},
  {"id": "quyet-1.0-large.q8_0.decide", "llmctl": {"api": "systemone", "decision": true, "...": "..."}}
]}
```

| Field | Meaning |
|---|---|
| `id` | The model's exact name. Two machines running the same file report the same id. |
| `llmctl.node` | Where a call goes right now: the first entry of `nodes`. |
| `llmctl.local` | Whether the machine you asked also runs this model. |
| `llmctl.status` | `ready`, `loading` or `down`, for the best copy. |
| `llmctl.api` | `openai` (chat, completions, embeddings), or `systemone` (decision endpoints, see below). |
| `llmctl.group` | Another name that also reaches this model (see Groups). |
| `llmctl.context`, `vision`, `embeddings`, `decision`, `parallel` | What the best copy can do. Per-machine values are in `nodes`. |
| `llmctl.nodes` | Every copy, best first: ready copies, then lower priority number, then the machine you asked. |
| `llmctl.members` | Present only on a group entry; lists its model ids. |

A group gets its own entry in the list once two or more different models
belong to it. Every model entry carries its `group` either way.

`GET /v1/models/<id>` returns one entry.

## Naming the model

The `"model"` field accepts these forms:

| Form | Meaning |
|---|---|
| `"gemma-4-26b-a4b-it-ud-q4_k_m"` | Exactly this model, on whichever machine is best. |
| `"gemma-4-26b-a4b-it"` | The group: any of its members, typically every quant of one model. The best member answers. |
| `"gemma-4-26b"` | A prefix. Unique prefixes work. A prefix matching several quants of one group means the group. |
| `"quyet@orin"` | Pins a machine. Works with ids, groups and prefixes. |
| no `model` | Allowed only when exactly one model serves that kind of request. |

Recommendations:
- Use the **group name** when any quant will do. It keeps working when quants
  change or a machine is down.
- Use the **exact id** when the quant matters.
- Use **`@node`** only when the machine matters, for example to reuse
  llama-server's prompt cache across calls with a long shared prefix.

Each machine receives the request under its own exact id, so vLLM accepts it.
The response's `model` field is that machine's id. Models launched by llmctl
older than 0.8 report their GGUF path here until they are restarted.

## How a machine is chosen

The candidate copies are ordered as in `nodes`: ready first, then the lowest
priority number (the RTX is 0, the Orin 1), then the machine you asked. The
gateway then sends the request:

- **Spill-over.** If every slot of the first copy is busy, the request goes to
  the next copy that has a free slot. If every copy is busy, it waits on the
  first. Work moves to a slower machine only when that machine is idle.
- **Failover.** If a copy is unreachable, answers 502/503, has just stopped,
  or rejects the prompt as longer than its context, the next copy gets the
  request. This only happens before any of the reply has been sent, so a
  generation never runs twice. A long prompt can therefore succeed through
  the gateway on a machine with a larger context.
- **Images.** A request carrying images goes only to copies with `vision:
  true`. Images are OpenAI content parts with type `image_url` or
  `input_image`, a SystemOne `images` field, or llama-server's
  `multimodal_data`.
- **Prompt cache.** Two calls can land on different machines, so
  llama-server's prompt cache isn't guaranteed to be reused. Pin with `@node`
  if that matters.

## Streaming and cancelling

`"stream": true` works as usual: server-sent events pass through as they are
generated. Closing the connection cancels the generation on the model server.

## Endpoints

| Request | Routed by |
|---|---|
| `GET /v1/models`, `GET /v1/models/<id>` | Answered by the gateway itself. |
| `POST /v1/chat/completions`, `/v1/completions`, `/v1/embeddings`, `/v1/responses`, any other `/v1/...` with a JSON body | The body's `model`. |
| `POST /v1/systemone` (alias `/v1/decisions`) | The body's `model`, among decision endpoints only. |
| `GET /v1/spec` | The `X-Model` header, among decision endpoints. |
| `ANY /m/<name>/<path>` | `<name>`: an exact id, a group, or either with `@node`. `<path>` is passed through as is, for llama-server's `/completion`, `/tokenize`, `/detokenize`, `/props`, `/slots`, `/health`, or a decision endpoint's `/v1/spec`. |
| `GET /health` | The gateway itself: `{"status": "ok", "node": ..., "models": <local count>}`. |
| `GET /cluster/v1/status` | This machine's view of the cluster: its nodes, the registry state, warnings. |

A request without a JSON body (for example a multipart upload) takes its
model from the `X-Model` header. The body limit is 256 MiB.

## Decision endpoints (SystemOne)

Decision models (Quyet, Rune, Xor, Bobcat, h2o-lightning) are served behind
`llmctl decide` as SystemOne endpoints. They are listed with
`"api": "systemone"` and an id ending in `.decide`, e.g.
`quyet-1.0-large.q8_0.decide`.

- **Call:** `POST /v1/systemone` with the usual SystemOne body plus `"model"`.
  `"quyet"` works as a prefix.
- **Spec:** read it from `GET /m/<name>/v1/spec` before writing the client.
- **Request format:** see
  [the SystemOne client guide](../packages/systemone-proxy/docs/client-guide.md).

Don't call the plain chat endpoint of a decision model (the id without
`.decide`). Its chat output isn't meaningful.

## Errors

The gateway's own errors use the OpenAI shape,
`{"error": {"message", "type", "param", "code"}}`:

| Status | `code` | When |
|---|---|---|
| 400 | `model_required` | No `model`, and several models serve this kind of request. |
| 400 | `model_ambiguous` | The prefix matches several models; the message lists them. |
| 400 | `vision_unavailable` | Images sent, but no copy of that model reads them. |
| 400 | `invalid_json`, `invalid_model` | Malformed body, or `model` isn't a string. |
| 404 | `model_not_found` | No such model running, or not on the pinned node. The message lists what is running. |
| 413 | `body_too_large` | Body over 256 MiB. |
| 502 | `upstream_unavailable` | No copy answered. |
| 503 | `no_models` | Nothing of that kind is running in the cluster. |

Any other error is the model server's own response, passed through unchanged.

## Don'ts

- Don't send `X-Llmctl-Hop`, `X-Llmctl-Token` or `X-Llmctl-Spill`. They are for
  gateway-to-gateway calls only: a request with `X-Llmctl-Hop` is rejected
  (401) without the cluster token.
- Don't hardcode per-model ports or machine names. Use `/v1/models` and
  names.
- Don't assume the model list is fixed. Models start and stop; refresh
  `/v1/models` when you get `model_not_found`.

# Calling a decision endpoint: client guide

How to read a decision model's spec and write a client for it. Every model
served by systemone-proxy answers the same request and response format; the
spec tells you the few things that differ per model.

## The 30-second version

```bash
llmctl run <model>                 # the model on llama-server
llmctl decide <model>              # the decision endpoint, default port 8090
llmctl decide spec <model>         # read this before writing the client
```

```bash
curl -s localhost:8090/v1/systemone -H 'Content-Type: application/json' -d '{
  "state": {"message": "Please close my card, I lost it yesterday."},
  "questions": {
    "intent": {"type": "choice", "instructions": "What does the customer want?",
               "criteria": {"cancel": "close the card", "limit": "change the limit", "other": null}},
    "urgent": {"type": "noul", "instructions": "The request is urgent."},
    "mood":   {"type": "score", "instructions": "How upset is the customer?",
               "criteria": ["calm", "annoyed", "angry"]}}}'
```

```json
{"model": "rune-26b-a4b-v3",
 "answers": {
   "intent": {"type": "choice", "choice": "cancel", "confidence": 0.93,
              "probabilities": {"cancel": 0.95, "limit": 0.01, "other": 0.04}},
   "urgent": {"type": "noul", "noul": 0.83, "confidence": 0.83},
   "mood":   {"type": "score", "score": 0.71, "confidence": 0.52,
              "legend": {"0": "calm", "1": "annoyed", "2": "angry"},
              "probabilities": {"0": 0.38, "1": 0.53, "2": 0.09}}},
 "usage": {"input_tokens": 412, "output_tokens": 0, "backend_calls": 3},
 "warnings": [],
 "latency_ms": 240}
```

(The numbers are illustrative.)

Through the llmctl gateway (port 9000, started with the first model), the
same request goes to `http://<host>:9000/v1/systemone` with a `"model"`
field naming the decision endpoint (its llmctl name, e.g.
`"rune-26b-a4b-v3-iq4_xs-dyn.decide"`, or a unique prefix such as `"rune"`).
`model` may be left out when only one decision endpoint is running.
`GET http://<host>:9000/v1/models` lists them with `"llmctl": {"api":
"systemone", ...}`, and the spec is at `GET /m/<name>/v1/spec`.
In a cluster of machines (`llmctl registry`), any machine's gateway reaches
every node's decision endpoints the same way; `"model": "<name>@<node>"` pins
one, and a request with images only goes to an endpoint that reads them.

## Getting the spec

Three equivalent sources:

| where | form |
|---|---|
| `llmctl decide spec <model>` | readable text; `<model>` is a running instance, a model key from `llmctl list`, or a profile id |
| `llmctl decide spec <model> --json` | the same as JSON |
| `GET http://<host>:8090/v1/spec` | JSON, from the running endpoint (use this from code) |

`llmctl decide spec` with no model prints only the shared contract.
`llmctl decide profiles` lists the supported models.

The spec has four parts:

1. **Model specifics** (`model` in JSON): what differs for this model. Read it
   carefully; it's short.
2. **The shared contract** (`api`): endpoints, request, response, semantics,
   errors. The same for every model.
3. **Example request** and **example response** (illustrative numbers).

## Reading the model specifics

Each field, what it means and whether your client has to care:

| field | what it says | does the client care? |
|---|---|---|
| `id`, `name` | profile id (returned as `model` in every response) and human name | log `model` with each decision |
| `temperatures` | the vendor's calibration temperature per question type | **No.** Already applied; the probabilities you get are calibrated |
| `limits` | max questions per request, max choice options, max score levels, and sometimes a prompt-token limit | **Yes.** Exceeding them is a 422. Check them before you build questions |
| `confidence` | the formula behind `confidence` per question type | Only if you threshold on `confidence`; see [Using the answers](#using-the-answers) |
| `labels` | the tokens the model reads internally (`A`, `B`, ...) | **No.** You always use your own option names |
| `noul_order` | which internal label means true | **No.** `noul` is always P(true) |
| `state` | how the state is serialised, and what happens when it's too long: **truncated** (answers carry `truncated: true`) or **rejected** (422 `input_too_long`) | **Yes.** It decides how you size states and handle long records |
| `calls` | llama-server calls per question (Xor makes 2 per choice question) | For latency and capacity planning |
| `extras` | extra answer fields this model adds | Use them if you need them; other models won't send them |
| `images` | present only for image-capable models: limits, where images go, temperature | **Yes**, if you send images |
| `notes` | anything surprising: floors, rendering of non-string descriptions, unsupported vendor features | Read once |

### The five models at a glance

| | Quyet-1.0-Large | Rune 26B-A4B v3 | Xor 26B-A4B | Bobcat Flash 1.2 | H2O-Lightning-4B |
|---|---|---|---|---|---|
| profile id | `quyet-1.0-large` | `rune-26b-a4b-v3` | `xor-26b-a4b` | `bobcat-flash-1.2` | `h2o-lightning-4b` |
| choice options | 2–10 | 2–255 | 2–255 | 2–255 | 2–255 |
| score levels | 2–10 | 2–255 | 2–10 | 2–10 | 2–255 |
| questions/request | 64 | 64 | 64 | 128 | 64 |
| long state | truncated (6000 tokens) | rejected | rejected | rejected | truncated (32000 tokens, head + tail) |
| calls per question | 1 | 1 | 1 (2 for choice) | 1 | 1 |
| extras | — | — | — | — | `threshold`, `decision` (noul) |
| images | no | no | no | no | yes, up to 4 |
| noul quirk | — | — | — | — | floored: never between 0.199 and 0.801 |

Always check `llmctl decide spec` for the current values; this table is a
snapshot.

## Writing the request

```json
{
  "state": <string | object | array>,
  "questions": {"<question id>": <question>, ...},
  "images": [...],        // image models only
  "debug": false          // optional
}
```

- **`state`** is the record the questions are about: a ticket, a message, a
  transaction, a conversation. Objects are serialised as JSON with their key
  order kept, so put the most important fields first (models that truncate
  keep the head, except Quyet keeps an array's tail).
- **`questions`** is an object; each question is answered independently, and
  answers come back under the same ids in the same order. Ask several
  questions about one state in one request: it saves round trips and
  validation.
- Unknown top-level fields are ignored (e.g. `model`).

The three question types:

| type | `criteria` | the answer |
|---|---|---|
| `choice` | `{"label": "description", ...}` (description may be `null`), or `["label", ...]` | `choice`: the most likely label |
| `score` | `["lowest level", ..., "highest level"]` | `score`: expected level index, a float from 0 to n−1 |
| `noul` (yes/no) | `null`, or `{"true": "what yes means", "false": "what no means"}` | `noul`: P(true) |

Every question needs `type` and a non-empty `instructions` string. For noul,
write the instructions as a statement to be judged ("The request is urgent.")
rather than a question. `threshold` (a number in [0, 1]) is accepted on any
question, but only models listing it under `extras` use it.

Tips:

- Labels are your keys; descriptions are what the model reads. Short labels
  with clear descriptions work best. Descriptions may be any JSON value;
  non-strings are rendered as JSON text (see the model's `notes`).
- Score levels go lowest first; the answer's `legend` echoes them.
- Choice labels must be unique; option order is the order you give.

### Images (image models only)

If the spec has an `images` section:

- Put images in a top-level `images` list as data URIs
  (`data:image/png;base64,...`), bare base64, or image parts
  (`{"type": "image_url", "image_url": {"url": "<data URI>"}}`).
- Or inline anywhere in `state` as data URIs or image parts. Each inline
  image is replaced by `[image N]` in the state, so the record can refer to
  it (top-level images are numbered first).
- Or send `multipart/form-data`: the JSON in a field named `request`, each
  image as a file part:
  `curl -F 'request={...};type=application/json' -F 'img=@receipt.png' .../v1/systemone`
- URLs are not fetched. Send the bytes.
- Images are resized to the model's pixel budget; you don't need to.

Text-only models reject an `images` field with 422. If the server was started
without its vision projector, an image request gets 503 `vision_unavailable`.

## Reading the response

```text
model        profile id that answered
answers      {question id: answer}, in request order
usage        {input_tokens, output_tokens: 0, backend_calls, images?}
warnings     [{code, ...}]
latency_ms   time spent in the proxy
```

Answers by type:

```text
choice  {type, choice, confidence, probabilities: {label: p}}
score   {type, score, confidence, legend: {"0": level, ...}, probabilities: {"0": p, ...}}
noul    {type, noul, confidence}
```

Plus, when they apply:

- `truncated: true` on an answer when the model's protocol cut the state.
- The model's `extras` (e.g. `threshold` and `decision` on h2o noul answers).
- `debug` when you sent `"debug": true`: the raw label logprobs and prompt
  sizes per backend call. For troubleshooting, not for decisions.

Warning codes:

| code | meaning | what to do |
|---|---|---|
| `state_truncated` | the state was cut to fit (with the original and kept token counts) | the answer saw only part of the record; consider shortening or splitting it |
| `label_outside_top_n` | an option's probability was too small to be returned exactly; it was bounded from above | nothing; that option is very unlikely |
| `reverse_order_skipped` | Xor: the reversed-order pass didn't fit, so the choice was scored forward-only | slightly less reliable choice |
| `image_read_as_text` | a text-only model got image data inside the state | send images to an image model, or remove them |

Treat unknown warning codes as informational.

## Using the answers

- **Probabilities are calibrated** (that's what the vendor's temperatures
  are for): over many decisions, answers given 0.8 should be right about 80%
  of the time. They sum to 1 over the options you offered. Nothing is
  rounded.
- **Decide with probabilities, not just the top label.** For choice, act on
  `probabilities[label] >= your threshold` and route the rest to a human or a
  fallback. For noul, compare `noul` with your threshold; pick the threshold
  from the cost of each kind of error.
- **`score` is an expected value**, not a level: 0.71 on a 3-level scale
  means mostly between "calm" and "annoyed". Use it as a continuous signal,
  or take the argmax of `probabilities` if you need a level.
- **`confidence` is model-specific.** Each model uses its vendor's formula
  (see `confidence` in the spec): some report the top probability, some
  rescale it so a uniform answer is 0. Compare confidences within one model,
  never across models. When in doubt, use the probabilities.
- **h2o's noul is floored**: it is never between 0.199 and 0.801. A
  threshold inside that band acts like 0.5. Use its `decision` field with
  your own `threshold` instead.
- **Log `model`, the answer and `usage`** with every decision. Switching
  models changes the calibration, so re-tune thresholds when you switch.

## Errors

Every error is JSON: `{"error": {"code": "...", "message": "..."}}`.

| status | code | cause | retry? |
|---|---|---|---|
| 400 | `bad_request` | body isn't a JSON object, or invalid JSON | no: fix the client |
| 404 | `not_found` | wrong path | no |
| 413 | `body_too_large` | body over 128 MiB | no |
| 422 | `invalid_request` | a field breaks the contract (type, criteria, option count, too many images, ...) | no: fix the request |
| 422 | `input_too_long` | a question's prompt doesn't fit the model's context | no: shorten the state or use a truncating model |
| 500 | `prompt_mismatch` / `internal_error` | a proxy bug or an unexpected llama-server change | no; report it |
| 502 | `backend_error` | llama-server failed or is unreachable | yes, with backoff |
| 503 | `vision_unavailable` | image request, but no vision projector loaded | no: a deployment problem |

Before the endpoint is ready (while llama-server loads and the self-test
runs), connections are refused. Poll `GET /health` until it returns
`{"status": "ok", "model": "<profile id>"}`.

## Performance

- Each question is one forward pass over the whole prompt, state included
  (two for Xor choice questions). Latency grows with questions × state
  length. `usage.backend_calls` tells you how many passes ran.
- The prompt cache is off by default, so every question recomputes the
  state. That keeps answers exact on the Gemma 4 models, where reusing the
  cache changed answers. An operator can start the endpoint with
  `--cache request` for speed at that risk.
- Calls run in parallel up to the llama-server's `--parallel` slots.
- Nothing is generated (`output_tokens` is always 0), so long answers never
  slow you down; long states do.

## A minimal Python client

```python
import json
import time
import urllib.error
import urllib.request

BASE = "http://127.0.0.1:8090"


def call(path, body=None, timeout=600):
    data = None if body is None else json.dumps(body).encode()
    req = urllib.request.Request(BASE + path, data=data,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


def decide(state, questions, retries=3):
    for attempt in range(retries):
        try:
            return call("/v1/systemone", {"state": state, "questions": questions})
        except urllib.error.HTTPError as e:
            err = json.load(e)["error"]
            if e.code == 502 and attempt < retries - 1:
                time.sleep(2 ** attempt)
                continue
            raise RuntimeError(f"{e.code} {err['code']}: {err['message']}") from None


spec = call("/v1/spec")["model"]          # check limits once at startup
print(spec["id"], spec["limits"])

out = decide({"message": "Please close my card, I lost it yesterday."}, {
    "intent": {"type": "choice", "instructions": "What does the customer want?",
               "criteria": {"cancel": "close the card", "limit": "change the limit"}},
    "urgent": {"type": "noul", "instructions": "The request is urgent."},
})
intent = out["answers"]["intent"]
if intent["probabilities"][intent["choice"]] >= 0.8:
    print("route to", intent["choice"])
else:
    print("send to a human", intent["probabilities"])
print("urgent" if out["answers"]["urgent"]["noul"] >= 0.7 else "not urgent")
for w in out["warnings"]:
    print("warning:", w)
```

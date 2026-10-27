"""H2O-Lightning-4B (h2oai), Qwen3.5-4B fine-tune.

Source: ``h2o_lightning_shim.py`` and ``serve_config.json`` of
h2oai/h2o-lightning-4b v1.4.1 (cc18ee33). Since v1.2.3 the one-way prompt,
temperatures and yes/no floor are unchanged; v1.4.1 reports a score's
confidence as its most likely level's probability. The vendor's default
two-way reading needs its own vLLM build and is not ported. There is one
prompt per question:

    <|im_start|>system\\n{SYSTEM}<|im_end|>\\n<|im_start|>user\\n{USER}<|im_end|>\\n
    <|im_start|>assistant\\n<think>\\n\\n</think>\\n\\nAnswer:

The answer slot is the token after ``Answer:``. Each option is read as
``" " + label`` (``ĠA``, ``ĠB``, ...), which is how the shim finds its label
ids. The shim sends the finished string to vLLM's /v1/completions with
``add_special_tokens: false``. vLLM tokenizes it as one string with the
special-token text parsed and with no BOS (Qwen has none), so we encode the
whole string with ``special=True``.
"""
from __future__ import annotations

import json

from systemone_proxy.images import MEDIA, ImageSupport
from systemone_proxy.profiles.base import Built, Profile, Row, py_json
from systemone_proxy.questions import Limits, Option, Question, Request
from systemone_proxy.readout import (argmax, binary_confidence, expected_index,
                                     normalized_confidence, softmax_t)
from systemone_proxy.tokenizer import TokenizerSource

# serve_config.json, verbatim
SYSTEM = ("You are a decision engine. You read a record and answer one question about it by "
          "choosing exactly one option. Reply with a single letter.")
PREFILL = "Answer:"
DEFAULT_INSTRUCTIONS = "Answer the question below."   # Contract default (not in the config)
MAX_STATE_TOKENS = 32000
NOUL_FLOOR = 0.801
# serve_config.json "image" block; min_pixels and the 32-pixel grid (patch 16 ×
# merge 2) from preprocessor_config.json
IMAGE_SUPPORT = ImageSupport(max_images=4, max_pixels=1_048_576, min_pixels=65_536, factor=32)
IMAGE_TEMPERATURE = 0.65
# what Qwen3.5's template writes for one image part; vLLM expands the pad into
# the image's tokens, llama.cpp wraps its image embeddings in the same two tokens
VISION_SLOT = "<|vision_start|><|image_pad|><|vision_end|>"
LABELS = (
    list("ABCDEFGHIJKLMNOPQRSTUVWXYZ") + list("abcdefghijklmnopqrstuvwxyz")
    + list("αβγδεζηθικλμνξοπστυφχψω")
    + ["in", "er", "on", "re", "at", "st", "en", "or", "th", "le", "it", "an", "ar", "al", "the",
       "ou", "is", "ing", "es", "ion", "ed", "ic", "et", "ro", "as", "el", "ct", "nd", "ent", "id",
       "am", "to", "of", "om", "im", "il", "and", "ur", "se", "ex", "ad", "ch", "ut", "if", "em",
       "ol", "ig", "iv", "ce", "od", "ate", "ag", "ay", "ot", "us", "un", "ul", "ue", "ow", "ew",
       "for", "ab", "ort", "um", "ame", "pe", "tr", "ck", "ist", "he", "lo", "be", "ers", "con",
       "ap", "ub", "ass", "int", "ly", "urn", "av", "ir", "nt", "end", "de", "ith", "out", "our",
       "lic", "res", "pt", "wh", "ver", "age", "ht", "ext", "ess", "os", "ect", "ke", "rom", "qu",
       "iz", "op", "up", "get", "ile", "ata", "ore", "ri", "pro", "ter", "ain", "com", "art",
       "ack", "est", "ine", "ill", "ind", "ere", "elf", "orm", "ult", "str", "you", "pl", "new",
       "ld", "oc", "ant", "ase", "ave", "ang", "are", "ial", "act", "ive", "ode", "ost", "not",
       "og", "ord", "all", "ff", "ont", "ime", "pr", "ies", "ure", "by", "ire", "sh", "ice",
       "ast", "ok", "cl", "ard", "was", "ip", "The", "ide", "ace", "ib", "ac", "rou", "we", "ak",
       "ve", "ath", "oid", "ust", "que", "ans", "pp", "one", "ail", "ach", "div", "ell"]
)
assert len(LABELS) == 255 and len(set(LABELS)) == 255

# tokenizer.json is identical at v1.2.3, this revision and v1.4.1 (cc18ee33).
H2O_LIGHTNING_TOKENIZER = TokenizerSource(
    repo="h2oai/h2o-lightning-4b",
    revision="9030a28dac81fcffd328f927e41a5deb85a27e5f",
    sha256="5f9e4d4901a92b997e463c1f46055088b6cca5ca61a6522d1b9f64c4bb81cb42",
)


def desc_text(v) -> str:
    """shim.desc_text: a string is used as is; anything else, None included,
    is JSON (so a null score level becomes "null")."""
    return v if isinstance(v, str) else json.dumps(v, ensure_ascii=False)


def option_texts(q: Question) -> list[str]:
    """shim.options_of: the description of each option, in option order."""
    if q.type == "noul":
        t, f = q.options[0].desc, q.options[1].desc
        return [desc_text(t) if t not in (None, "") else "the statement holds",
                desc_text(f) if f not in (None, "") else "it does not"]
    if q.type == "score":
        return [desc_text(o.desc) for o in q.options]
    return [desc_text(o.desc) if o.desc is not None else o.key for o in q.options]


def user_turn(state_text: str, q: Question) -> str:
    """shim.user_turn, with layout "plain"."""
    body = "\n".join(f"{l}) {o.key}: {d}" for l, o, d in zip(LABELS, q.options, option_texts(q)))
    instr = q.instructions.lstrip() or DEFAULT_INSTRUCTIONS
    return f"record: {state_text}\nquestion: {instr}\noptions:\n{body}"


def render_chat(user: str) -> str:
    """shim.render_chat with chat_template=true and thinking off."""
    return ("<|im_start|>system\n" + SYSTEM.strip() + "<|im_end|>\n"
            + "<|im_start|>user\n" + user.strip() + "<|im_end|>\n"
            + "<|im_start|>assistant\n<think>\n\n</think>\n\n" + PREFILL)


def render_state(state) -> str:
    """shim.render_state with state_compact=true."""
    return state if isinstance(state, str) else py_json(state, compact=True)


def common_prefix_len(seqs: list[list[int]]) -> int:
    if len(seqs) < 2:
        return 0
    n = min(len(x) for x in seqs)
    for i in range(n):
        if any(x[i] != seqs[0][i] for x in seqs[1:]):
            return i
    return n


def noul_threshold(q: Question) -> float | None:
    """shim.noul_threshold. ``threshold`` has already been validated. If the
    ``threshold`` key is absent, the vendor reads the ``noul_threshold`` alias."""
    if q.type != "noul":
        return None
    if "threshold" in q.raw:
        return q.threshold
    t = q.raw.get("noul_threshold")
    if isinstance(t, (int, float)) and not isinstance(t, bool) and 0.0 <= float(t) <= 1.0:
        return float(t)
    return None


def commit_noul(p_yes: float, floor: float = NOUL_FLOOR) -> float:
    """shim.commit_noul: if max(p, 1-p) is below the floor, p moves to the
    floor on its own side (a tie goes to yes)."""
    if not floor or max(p_yes, 1.0 - p_yes) >= floor:
        return p_yes
    return floor if p_yes >= 0.5 else 1.0 - floor


class H2OLightning4B(Profile):
    id = "h2o-lightning-4b"
    name = "H2O-Lightning-4B"
    upstream = ("https://huggingface.co/h2oai/h2o-lightning-4b (h2o_lightning_shim.py, "
                "serve_config.json)")
    # holds pre-v1.4.0 weights; v1.4.1's own GGUF (h2oai) is served natively
    gguf_repos = ("mradermacher/h2o-lightning-4b-GGUF",)
    tokenizer_source = H2O_LIGHTNING_TOKENIZER
    temperatures = {"choice": 0.75, "noul": 0.8, "score": 0.65}
    limits = Limits(max_choice=len(LABELS), max_score=len(LABELS))
    image_support = IMAGE_SUPPORT

    @classmethod
    def matches(cls, meta: dict, filename: str) -> bool:
        arch = str(meta.get("general.architecture", "")).lower()
        base = str(meta.get("general.basename", "")).lower()
        size = str(meta.get("general.size_label", "")).lower()
        if arch != "qwen35":
            return False
        if base == "h2o-lightning" and size == "4b":
            return True
        # no basename in the header: fall back to the file name
        return not base and filename.lower().startswith("h2o-lightning-4b")

    def __init__(self, tok):
        super().__init__(tok)
        # shim.check_backend: after the prompt, " <label>" must add exactly one
        # token. Only the tail matters here, because the pre-tokenizer splits at ':'.
        head = tok.encode("<|im_start|>assistant\n<think>\n\n</think>\n\n" + PREFILL, special=True)
        self.label_ids: list[int] = []
        for label in LABELS:
            full = tok.encode("<|im_start|>assistant\n<think>\n\n</think>\n\n" + PREFILL + " " + label,
                              special=True)
            if len(full) != len(head) + 1 or full[:len(head)] != head:
                raise RuntimeError(f"label {label!r} is not one token at the answer slot: {full[len(head):]}")
            self.label_ids.append(full[-1])
        if len(set(self.label_ids)) != len(self.label_ids):
            raise RuntimeError("label tokens are not distinct")

    # -- prompt --------------------------------------------------------------
    def prompt_text(self, state_text: str, q: Question) -> str:
        return render_chat(user_turn(state_text, q))

    def truncate_state(self, text: str) -> tuple[str, int | None]:
        """shim.truncate_state: a state over 32000 tokens keeps its first
        16000 tokens, then "\\n...\\n", then its last 16000. Returns the text and
        the original token count, or (text, None) when nothing was cut."""
        if len(text.encode("utf-8", "ignore")) <= MAX_STATE_TOKENS:
            return text, None
        ids = self.tok.encode(text, special=True)       # vLLM /tokenize, add_special_tokens false
        if len(ids) <= MAX_STATE_TOKENS:
            return text, None
        half = MAX_STATE_TOKENS // 2
        return self.tok.decode(ids[:half]) + "\n...\n" + self.tok.decode(ids[-half:]), len(ids)

    def build(self, req: Request) -> Built:
        state_text, n_state = self.truncate_state(render_state(req.state))
        if req.images:
            # shim.image_prompt: the images ahead of the user text. Each one
            # adds its grid's tokens minus the one <|image_pad|> the text form has.
            n = len(req.images)
            pngs = tuple(im.png for im in req.images)
            extra = sum((im.width // IMAGE_SUPPORT.factor) * (im.height // IMAGE_SUPPORT.factor) - 1
                        for im in req.images)
            rows = [Row(q.qid,
                        self.tok.encode(render_chat(VISION_SLOT * n + user_turn(state_text, q)),
                                        special=True),
                        self.label_ids[:len(q.options)],
                        text=render_chat(MEDIA * n + user_turn(state_text, q)),
                        images=pngs, extra_tokens=extra)
                    for q in req.questions]
        else:
            extra = 0
            rows = [Row(q.qid, self.tok.encode(self.prompt_text(state_text, q), special=True),
                        self.label_ids[:len(q.options)])
                    for q in req.questions]
        # shim.respond: the head the questions share is counted once (the
        # images are in it)
        shared = common_prefix_len([r.ids for r in rows]) + (extra if len(rows) > 1 else 0)
        built = Built(rows=rows,
                      input_tokens=sum(len(r.ids) + r.extra_tokens for r in rows)
                      - shared * (len(rows) - 1),
                      answer_kwargs={"image": bool(req.images)})
        if n_state is not None:
            built.truncated = {q.qid for q in req.questions}
            built.warnings.append({"code": "state_truncated", "state_tokens": n_state,
                                   "kept_tokens": MAX_STATE_TOKENS, "kept": "head and tail"})
        return built

    # -- answer ----------------------------------------------------------------
    def answer(self, q: Question, logprobs: dict[str, list[float]], image: bool = False) -> dict:
        # shim.respond: an image request uses one temperature for every type
        p = softmax_t(logprobs["main"], IMAGE_TEMPERATURE if image else self.temperatures[q.type])
        keys = q.keys
        if q.type == "choice":
            i = argmax(p)
            return {"type": "choice", "choice": keys[i], "confidence": normalized_confidence(p),
                    "probabilities": dict(zip(keys, p))}
        if q.type == "score":
            # shim.format_answer since v1.4.1: the most likely level's probability
            return {"type": "score", "score": expected_index(p), "confidence": float(max(p)),
                    "legend": dict(zip(keys, option_texts(q))),
                    "probabilities": dict(zip(keys, p))}
        py = commit_noul(float(p[0]))
        out = {"type": "noul", "noul": py, "confidence": binary_confidence(py)}
        thr = noul_threshold(q)
        if thr is not None:
            out["threshold"] = thr
            out["decision"] = bool(py >= thr)
        return out

    # -- self-test -------------------------------------------------------------
    def selftest_texts(self) -> list[str]:
        """The README's priority prompt (digits, ':' and '=' runs) and a
        non-ASCII prompt with 60 options that ends at the slot of the label " α"."""
        readme = Question("priority", "choice", "Which priority does the severity guide assign?",
                          (Option("p1", "Revenue path fully down"), Option("p2", "Degraded but working"),
                           Option("p3", "Cosmetic")))
        wide = Question("w", "choice", "Welche Option passt? (naïve café, 日本語, ½ €)",
                        tuple(Option(f"o{i}", f"d{i}") for i in range(60)))
        return [self.prompt_text(README_STATE, readme),
                self.prompt_text(py_json({"x": 1.0, "ü": [1e-05, "Ⅳ"]}, compact=True), wide) + " α"]

    def selftest_labels(self) -> dict[int, str]:
        picks = ["A", "B", "C", "D", "a", "α", "in", "the"]
        return {self.label_ids[LABELS.index(l)]: " " + l for l in picks}

    # -- documentation ---------------------------------------------------------
    def spec(self) -> dict:
        return {
            "confidence": {"choice": "(max p - 1/n) / (1 - 1/n)",
                           "score": "max p (the most likely level's probability)",
                           "noul": "max(p, 1 - p) of the reported (floored) P(true)"},
            "labels": ("A..Z, a..z, 23 Greek letters, then common English word pieces "
                       "(255 in all), one per option in the order given; read as the token "
                       "\" <label>\" after the \"Answer:\" prefill"),
            "noul_order": "A = true, B = false",
            "state": (f"strings as is, other JSON as compact JSON. Over {MAX_STATE_TOKENS} "
                      "tokens the state is TRUNCATED, not rejected: the first 16000 and last "
                      "16000 tokens are kept around a \"...\" line. Answers then carry "
                      "\"truncated\": true and a state_truncated warning is added."),
            "limits": {"options": f"2..{len(LABELS)} for choice and score"},
            "calls": "one llama-server call per question",
            "images": {
                "max_images": IMAGE_SUPPORT.max_images,
                "max_pixels": IMAGE_SUPPORT.max_pixels,
                "temperature": IMAGE_TEMPERATURE,
                "where": "the top-level images list, or inline in state (data URIs, image "
                         "parts, bare base64 of PNG/JPEG/GIF/WebP); each inline image becomes "
                         "\"[image N]\" in the state, top-level images numbered first",
                "prompt": "all images go ahead of the question text, in that order",
                "resize": f"Qwen smart_resize: sides multiples of {IMAGE_SUPPORT.factor}, area "
                          f"{IMAGE_SUPPORT.min_pixels}..{IMAGE_SUPPORT.max_pixels} pixels; "
                          "transparency flattened onto white",
                "tokens": "each image adds (width/32)·(height/32) tokens after resizing",
            },
            "extras": {
                "threshold": "noul only, when the question gives a threshold (or the vendor's "
                             "noul_threshold alias): echoed back",
                "decision": "noul only, with a threshold: noul >= threshold (the floored noul)",
            },
            "notes": [
                "Temperatures are per question type, from serve_config.json: choice 0.75, "
                "noul 0.8, score 0.65. A request with images uses 0.65 for every type.",
                f"noul answers are floored as the vendor ships them: when max(p, 1 - p) < "
                f"{NOUL_FLOOR}, P(true) is reported as {NOUL_FLOOR} (or {1 - NOUL_FLOOR:.3f}) on "
                "its own side. The chosen side never changes, but there are no uncertain "
                "yes/no values (the vendor's evaluated configuration).",
                "Option descriptions that are not strings are rendered as JSON text; a missing "
                "choice description falls back to the label, and a missing noul description "
                "to \"the statement holds\" / \"it does not\". A null score level is "
                "rendered as \"null\".",
                "Images need a vision projector next to the GGUF. The GGUF repo ships none, "
                "and h2o's vision tower is unchanged from Qwen3.5-4B (its tensors are "
                "identical to unsloth/Qwen3.5-4B-GGUF mmproj-F16.gguf), so use that file. "
                "Without it, a request with images gets 503 vision_unavailable.",
                "Special-token text inside the state or questions (e.g. <|im_end|>) is parsed "
                "as special tokens, as vLLM does for the vendor's prompts.",
                "usage.input_tokens counts the prompt head that the questions share only once, "
                "as the vendor does.",
            ],
        }


README_STATE = ("Ticket 4471: since 09:10 the checkout page returns HTTP 500 for every customer; "
                "no orders have gone through in 40 minutes. Severity guide: P1 = a revenue path "
                "fully down; P2 = degraded but working; P3 = cosmetic.")

"""Images: finding them in a request, fitting them, and h2o-lightning's image path."""
import base64
import io
import math

import pytest
from PIL import Image as PILImage

from systemone_proxy import images as I
from systemone_proxy.errors import ApiError
from systemone_proxy.profiles import h2o_lightning as H
from systemone_proxy.profiles.h2o_lightning import H2OLightning4B
from systemone_proxy.questions import Limits, parse_request
from systemone_proxy.service import DecisionService

from conftest import FIXTURES, FakeBackend

SUPPORT = H.IMAGE_SUPPORT


def png(w, h, mode="RGB", color=(200, 30, 30)):
    buf = io.BytesIO()
    PILImage.new(mode, (w, h), color).save(buf, format="PNG")
    return buf.getvalue()


def b64(raw):
    return base64.b64encode(raw).decode()


QS = {"q": {"type": "noul", "instructions": "Is it red?"}}


# -- finding images --------------------------------------------------------------

def test_inline_data_uri_and_top_level_numbering():
    a, b = png(40, 40), png(50, 50)
    body = {"state": {"photo": "data:image/png;base64," + b64(a), "note": "see photo"},
            "images": [b64(b)], "questions": QS}
    state, found = I.extract(body)
    assert state == {"photo": "[image 2]", "note": "see photo"}
    assert found == [b, a]                     # top-level first


def test_parts_and_bare_base64_in_state():
    a, b = png(40, 40), png(41, 41)
    state = [{"type": "image_url", "image_url": {"url": "data:image/png;base64," + b64(a)}},
             b64(b), "plain text", {"type": "image", "image": "a sunset"}]
    out, found = I.extract({"state": state})
    assert out == ["[image 1]", "[image 2]", "plain text", {"type": "image", "image": "a sunset"}]
    assert found == [a, b]


def test_text_is_never_an_image():
    state = {"a": "iVBORw0KGgo but with spaces", "b": "https://example.com/x.png", "c": "x" * 100}
    out, found = I.extract({"state": state})
    assert out is state and found == []
    assert not I.has_inline_images(state)


def test_urls_are_not_fetched():
    with pytest.raises(ApiError) as e:
        I.extract({"state": "", "images": ["https://example.com/x.png"]})
    assert e.value.status == 422 and "not fetched" in e.value.message


def test_bad_data_uri_and_bad_entry():
    with pytest.raises(ApiError):
        I.extract({"state": "data:image/png;base64,!!!"})
    with pytest.raises(ApiError):
        I.extract({"state": "", "images": ["not an image at all"]})
    with pytest.raises(ApiError):
        I.extract({"state": "", "images": [42]})


def test_multipart():
    img = png(30, 30)
    boundary = "XyZ"
    raw = (f"--{boundary}\r\nContent-Disposition: form-data; name=\"request\"\r\n\r\n"
           '{"state": "s", "questions": {"q": {"type": "noul", "instructions": "x"}}}\r\n'
           f"--{boundary}\r\nContent-Disposition: form-data; name=\"f\"; filename=\"a.png\"\r\n"
           "Content-Type: image/png\r\n\r\n").encode() + img + f"\r\n--{boundary}--\r\n".encode()
    body = I.parse_multipart(f"multipart/form-data; boundary={boundary}", raw)
    assert body["state"] == "s" and base64.b64decode(body["images"][0]) == img


# -- fitting ----------------------------------------------------------------------

def hf_smart_resize(height, width, factor=32, min_pixels=65536, max_pixels=1048576):
    """transformers' Qwen2-VL smart_resize, verbatim logic."""
    h_bar = round(height / factor) * factor
    w_bar = round(width / factor) * factor
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = max(factor, math.floor(height / beta / factor) * factor)
        w_bar = max(factor, math.floor(width / beta / factor) * factor)
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = math.ceil(height * beta / factor) * factor
        w_bar = math.ceil(width * beta / factor) * factor
    return h_bar, w_bar


@pytest.mark.parametrize("h,w", [(600, 800), (1080, 1920), (4000, 3000), (50, 70), (32, 32),
                                 (1000, 1000), (1024, 1024), (333, 777)])
def test_smart_resize_matches_transformers(h, w):
    assert I.smart_resize(h, w, 32, 65536, 1048576) == hf_smart_resize(h, w)


def test_fit_resizes_into_the_budget_and_flattens_alpha():
    im = I.fit(png(3000, 2000, "RGBA", (0, 0, 0, 0)), SUPPORT)
    assert (im.height, im.width) == hf_smart_resize(2000, 3000)
    assert im.width * im.height <= SUPPORT.max_pixels
    out = PILImage.open(io.BytesIO(im.png))
    assert out.mode == "RGB" and out.getpixel((5, 5)) == (255, 255, 255)   # transparent → white


def test_fit_rejects_garbage_and_huge_files():
    with pytest.raises(ApiError):
        I.fit(b"\x89PNG\r\n\x1a\nnot really", SUPPORT)
    with pytest.raises(ApiError):
        I.fit(b"0" * (SUPPORT.max_bytes + 1), SUPPORT)


# -- request parsing -----------------------------------------------------------

def test_text_models_reject_the_images_field():
    with pytest.raises(ApiError) as e:
        parse_request({"state": "s", "questions": QS, "images": [b64(png(40, 40))]}, Limits())
    assert "does not read images" in e.value.message


def test_too_many_images():
    body = {"state": "s", "questions": QS, "images": [b64(png(40, 40))] * 5}
    with pytest.raises(ApiError) as e:
        parse_request(body, Limits(), SUPPORT)
    assert "at most 4" in e.value.message


# -- h2o-lightning image path -------------------------------------------------------

@pytest.fixture(scope="session")
def prof():
    from systemone_proxy import tokenizer
    try:
        return H2OLightning4B(tokenizer.load(H2OLightning4B.tokenizer_source))
    except Exception as e:  # offline
        pytest.skip(f"H2O-Lightning tokenizer unavailable: {e}")


@pytest.fixture(scope="session")
def qwen_chat():
    jinja2 = pytest.importorskip("jinja2")
    env = jinja2.Environment(loader=jinja2.FileSystemLoader(str(FIXTURES)),
                             extensions=["jinja2.ext.loopcontrols"])
    env.globals["raise_exception"] = lambda msg: (_ for _ in ()).throw(RuntimeError(msg))
    return env.get_template("h2o_lightning_chat_template.jinja")


RECEIPT_REQ = {
    "state": {"receipt": None, "claim": {"employee": "J. Ortiz", "category": "office supplies",
                                         "amount": "444.26"}},
    "questions": {
        "matches": {"type": "noul", "instructions": "The receipt total equals claim.amount.",
                    "criteria": {"true": "Same amount", "false": "Different amount"}},
        "items": {"type": "choice", "instructions": "How many item lines are on the receipt?",
                  "criteria": {"three": "3", "four": "4", "five": "5", "six": "6"}},
        "legible": {"type": "score", "instructions": "How legible is the receipt?",
                    "criteria": ["Unreadable", "Partly readable", "Fully readable"]}},
}


def receipt_body(w=480, h=640):
    body = {**RECEIPT_REQ, "state": dict(RECEIPT_REQ["state"])}
    body["state"]["receipt"] = "data:image/png;base64," + b64(png(w, h))
    return body


def test_image_prompt_matches_the_chat_template(prof, qwen_chat):
    """shim.check_chat: vLLM's chat rendering of [system, user{image, text},
    assistant "Answer:"] (continue_final_message) is the text prompt with one
    vision slot per image ahead of the user text."""
    r = parse_request(receipt_body(), prof.limits, prof.image_support)
    q = r.questions[0]
    user = H.user_turn(H.render_state(r.state), q)
    msgs = [{"role": "system", "content": H.SYSTEM},
            {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": user}]},
            {"role": "assistant", "content": H.PREFILL}]
    rendered = qwen_chat.render(messages=msgs, add_generation_prompt=False, enable_thinking=False)
    rendered = rendered[:rendered.rindex(H.PREFILL) + len(H.PREFILL)]   # continue_final_message
    assert rendered == H.render_chat(H.VISION_SLOT + user)
    assert '"receipt":"[image 1]"' in user


def test_image_rows_and_usage(prof):
    r = parse_request(receipt_body(480, 640), prof.limits, prof.image_support)
    built = prof.build(r)
    grid = (480 // 32) * (640 // 32)            # 15 x 20 = 300 image tokens
    for row in built.rows:
        assert row.images == (r.images[0].png,)
        assert row.text.count(I.MEDIA) == 1 and H.VISION_SLOT not in row.text
        assert row.text.replace(I.MEDIA, H.VISION_SLOT) == prof.tok.decode(row.ids)
        assert row.extra_tokens == grid - 1
    shared = H.common_prefix_len([x.ids for x in built.rows])
    assert built.input_tokens == sum(len(x.ids) for x in built.rows) + 3 * (grid - 1) \
        - 2 * (shared + grid - 1)
    assert built.answer_kwargs == {"image": True}


def test_image_answers_use_the_image_temperature(prof):
    lps = [-0.1, -2.0]
    q = parse_request({"state": "s", "questions": QS}, prof.limits).questions[0]
    text = prof.answer(q, {"main": lps})["noul"]
    image = prof.answer(q, {"main": lps}, image=True)["noul"]
    assert image == pytest.approx(1 / (1 + math.exp((-2.0 + 0.1) / 0.65)))
    assert text == pytest.approx(1 / (1 + math.exp((-2.0 + 0.1) / 0.8)))


def test_service_sends_images_and_reports_them(prof):
    fb = FakeBackend([{prof.label_ids[0]: -0.1, prof.label_ids[1]: -3.0}] * 3)
    svc = DecisionService(prof, fb, n_ctx=40960, vision=True)
    out = svc.decide(receipt_body())
    assert out["usage"]["images"] == 1 and out["usage"]["backend_calls"] == 3
    assert all(c["images"] and c["text"] for c in fb.calls)
    assert all(c["extra_tokens"] == 15 * 20 - 1 for c in fb.calls)


def test_service_without_vision_refuses_images(prof):
    svc = DecisionService(prof, FakeBackend([]), n_ctx=40960, vision=False)
    with pytest.raises(ApiError) as e:
        svc.decide(receipt_body())
    assert e.value.status == 503 and e.value.code == "vision_unavailable"


def test_text_request_is_unchanged_on_a_vision_server(prof):
    fb = FakeBackend([{prof.label_ids[0]: -0.1, prof.label_ids[1]: -3.0}])
    svc = DecisionService(prof, fb, n_ctx=40960, vision=True)
    out = svc.decide({"state": "s", "questions": QS})
    assert "images" not in out["usage"] and not fb.calls[0]["images"]


def test_text_model_warns_about_inline_images(gemma_tok):
    from systemone_proxy.profiles.rune import Rune  # any text-only profile
    prof = Rune(gemma_tok)
    fb = FakeBackend([{i: -1.0 for i in range(300000)}] * 1)
    svc = DecisionService(prof, fb, n_ctx=40960)
    body = {"state": {"photo": "data:image/png;base64," + b64(png(40, 40))}, "questions": QS}
    out = svc.decide(body)
    assert any(w["code"] == "image_read_as_text" for w in out["warnings"])

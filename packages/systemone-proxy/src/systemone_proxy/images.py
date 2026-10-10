"""Images in a request: finding them, decoding them, and fitting them to the
model's pixel budget.

Where images come from (the h2o-lightning shim's rules, which are the only
vendor rules there are):

* the top-level ``images`` list: data URIs, bare base64, or image part objects
  (``{"type": "image_url", "image_url": {"url"}}``, ``{"type": "image",
  "image"}``, ``{"type": "input_image", "image_url"}``, ``{"type": "base64",
  "media_type", "data"}``);
* anywhere inside ``state``: a ``data:image/...;base64,`` string, an image part
  object, or bare base64 of a PNG/JPEG/GIF/WebP (one unbroken string of 64+
  characters, so ordinary text never qualifies). Each one is replaced by
  ``[image N]`` in the state; the top-level images are numbered first.
* multipart/form-data: the JSON in a ``request`` field, every file part an
  image appended to ``images`` (see :func:`parse_multipart`).

URLs are not fetched: the proxy never makes outbound requests on a client's
behalf. Send the bytes.

Images are then resized with Qwen's ``smart_resize`` (sides multiples of the
patch grid, area within [min_pixels, max_pixels]) and composited onto white,
as vLLM's processor does, and sent to llama-server as PNG. llama.cpp's own
resize step then keeps that size (it uses the same rule with wider limits),
so both servers see the same pixels and the same number of image tokens.
"""
from __future__ import annotations

import base64
import binascii
import email.parser
import email.policy
import io
import json
import math
import re
from dataclasses import dataclass

from systemone_proxy.errors import ApiError, invalid

# where an image goes in Row.text; the backend swaps in llama-server's marker
MEDIA = "\x00systemone-image\x00"

_DATA_URI = re.compile(r"data:(image/[A-Za-z0-9.+-]+);base64,(.*)", re.S | re.I)
_B64_CHARS = re.compile(r"[A-Za-z0-9+/=\s]+")
_B64_URLSAFE = re.compile(r"[A-Za-z0-9_\-=\s]+")
_PART_TYPES = ("image_url", "image", "input_image")
# how base64 of a PNG, JPEG, GIF and WebP begins: the cheap test before a
# string inside `state` is decoded
_B64_HEADS = ("iVBORw0KGgo", "/9j/", "R0lGOD", "UklGR")


@dataclass(frozen=True)
class ImageSupport:
    """What a profile accepts. Pixel limits follow Qwen's processor."""
    max_images: int
    max_pixels: int
    min_pixels: int
    factor: int                 # patch size × spatial merge
    max_bytes: int = 20 * 1024 * 1024


@dataclass(frozen=True)
class Image:
    png: bytes                  # what llama-server gets
    width: int
    height: int
    source_bytes: int


def sniff(raw: bytes) -> str | None:
    if raw.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if raw.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    if raw.startswith((b"GIF87a", b"GIF89a")):
        return "image/gif"
    if raw[:4] == b"RIFF" and raw[8:12] == b"WEBP":
        return "image/webp"
    if raw.startswith(b"BM"):
        return "image/bmp"
    if raw[:4] in (b"II*\x00", b"MM\x00*"):
        return "image/tiff"
    return None


def _b64(s: str) -> bytes | None:
    t = re.sub(r"\s+", "", s)
    if not t:
        return None
    try:
        if _B64_CHARS.fullmatch(t):
            return base64.b64decode(t + "=" * (-len(t) % 4), validate=True)
        if _B64_URLSAFE.fullmatch(t):
            return base64.urlsafe_b64decode(t + "=" * (-len(t) % 4))
    except (binascii.Error, ValueError):
        return None
    return None


def _from_string(s: str, explicit: bool) -> bytes | None:
    """Image bytes in a string, or None when it isn't one. ``explicit``: the
    string sits where only an image may be (the images list, an image part)."""
    s = s.strip()
    m = _DATA_URI.match(s)
    if m:
        raw = _b64(m.group(2))
        if not raw:
            raise invalid("an image data URI does not carry valid base64")
        return raw
    if explicit:
        if re.match(r"https?://", s, re.I):
            raise invalid("image URLs are not fetched; send the image as a data URI or base64")
        raw = _b64(s) if len(s) >= 16 else None
        if raw and sniff(raw):
            return raw
        raise invalid("an image entry holds neither a data URI nor base64 image bytes")
    if len(s) < 64 or any(c.isspace() for c in s) or not s.startswith(_B64_HEADS) \
            or not _B64_CHARS.fullmatch(s):
        return None
    raw = _b64(s)
    return raw if raw and sniff(raw) in ("image/png", "image/jpeg", "image/gif", "image/webp") else None


def _from_part(obj, in_state: bool) -> bytes | None:
    """An image part object's bytes, or None when ``obj`` is not one. Inside
    ``state`` a part-shaped object whose value is not an image is record data."""
    if not isinstance(obj, dict):
        return None
    typ = obj.get("type")
    if typ == "base64" and isinstance(obj.get("data"), str):
        v = f"data:{obj.get('media_type') or 'image/png'};base64,{obj['data']}"
    elif typ in _PART_TYPES:
        v = obj.get(typ) if typ != "input_image" else obj.get("image_url", obj.get("image"))
        if isinstance(v, dict):
            v = v.get("url", v.get("data"))
        if not isinstance(v, str):
            if in_state:
                return None
            raise invalid(f"an image part of type {typ!r} has no url or data")
    else:
        return None
    if in_state and not (_DATA_URI.match(v.strip()) or re.match(r"https?://", v.strip(), re.I)):
        return _from_string(v, False)
    return _from_string(v, True)


def _explicit(v) -> bytes:
    if isinstance(v, str):
        return _from_string(v, True)
    if isinstance(v, dict):
        got = _from_part(v, in_state=False)
        if got is not None:
            return got
        for k in ("url", "image_url", "data", "image"):
            inner = v.get(k)
            if isinstance(inner, dict):
                inner = inner.get("url", inner.get("data"))
            if isinstance(inner, str):
                return _from_string(inner, True)
    raise invalid("an image entry must be a string or an object with a url or base64 data")


def _lift(state, found: list[bytes]):
    if isinstance(state, str):
        raw = _from_string(state, False)
        if raw is None:
            return state
        found.append(raw)
        return f"[image {len(found)}]"
    if isinstance(state, dict):
        raw = _from_part(state, in_state=True)
        if raw is not None:
            found.append(raw)
            return f"[image {len(found)}]"
        out, changed = {}, False
        for k, v in state.items():
            out[k] = _lift(v, found)
            changed = changed or out[k] is not v
        return out if changed else state
    if isinstance(state, list):
        out = [_lift(v, found) for v in state]
        return out if any(a is not b for a, b in zip(out, state)) else state
    return state


def extract(body: dict) -> tuple[object, list[bytes]]:
    """(state with ``[image N]`` placeholders, raw image bytes). The state is
    returned unchanged (the same object) when it holds no image."""
    found: list[bytes] = []
    top = body.get("images")
    if top is not None:
        for item in (top if isinstance(top, list) else [top]):
            found.append(_explicit(item))
    return _lift(body["state"], found), found


def has_inline_images(state) -> bool:
    """Whether ``state`` holds something :func:`extract` would lift (for a
    warning on models that read images as text)."""
    try:
        return _lift(state, []) is not state
    except ApiError:
        return True


def smart_resize(height: int, width: int, factor: int, min_pixels: int,
                 max_pixels: int) -> tuple[int, int]:
    """Qwen2-VL's ``smart_resize`` (transformers), returning (height, width)."""
    if max(height, width) / min(height, width) > 200:
        raise invalid(f"image aspect ratio over 200 ({width}x{height})")
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


def fit(raw: bytes, support: ImageSupport) -> Image:
    """Decode, flatten onto white, resize to the model's grid, encode PNG."""
    if len(raw) > support.max_bytes:
        raise invalid(f"an image of {len(raw)} bytes is over the {support.max_bytes}-byte limit")
    try:
        from PIL import Image as PILImage
    except ImportError:  # pragma: no cover - pillow is a dependency
        raise ApiError(500, "internal_error", "Pillow is not installed") from None
    try:
        img = PILImage.open(io.BytesIO(raw))
        img.load()
    except Exception as e:  # noqa: BLE001 - PIL raises many types
        raise invalid(f"an image could not be decoded: {e}") from None
    # vLLM: RGBA (and LA/P with transparency) composited onto white, then RGB.
    if img.mode in ("RGBA", "LA") or (img.mode == "P" and "transparency" in img.info):
        rgba = img.convert("RGBA")
        bg = PILImage.new("RGBA", rgba.size, (255, 255, 255, 255))
        img = PILImage.alpha_composite(bg, rgba).convert("RGB")
    elif img.mode != "RGB":
        img = img.convert("RGB")
    h, w = smart_resize(img.height, img.width, support.factor, support.min_pixels,
                        support.max_pixels)
    if (w, h) != img.size:
        img = img.resize((w, h), PILImage.Resampling.BICUBIC)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return Image(buf.getvalue(), w, h, len(raw))


def parse_multipart(content_type: str, raw: bytes) -> dict:
    """A multipart/form-data body as a request object: the JSON from the
    ``request`` (or ``body``/``json``/``data``) field, every file part as a
    base64 image appended to the top-level ``images`` list."""
    msg = email.parser.BytesParser(policy=email.policy.HTTP).parsebytes(
        b"Content-Type: " + content_type.encode("latin-1") + b"\r\n\r\n" + raw)
    if not msg.is_multipart():
        raise invalid("a multipart request without parts")
    body, files = None, []
    for part in msg.iter_parts():
        name = part.get_param("name", header="content-disposition")
        data = part.get_payload(decode=True) or b""
        if part.get_filename() is not None or part.get_content_maintype() == "image":
            if not sniff(data):
                raise invalid(f"file part {name!r} is not a recognised image")
            files.append(base64.b64encode(data).decode("ascii"))
        elif name in ("request", "body", "json", "data"):
            try:
                body = json.loads(data.decode("utf-8"))
            except ValueError as e:
                raise invalid(f"multipart field {name!r} is not JSON: {e}") from None
    if not isinstance(body, dict):
        raise invalid("a multipart request needs its JSON in a `request` field")
    if files:
        have = body.get("images")
        body["images"] = (have if isinstance(have, list) else ([] if have is None else [have])) + files
    return body

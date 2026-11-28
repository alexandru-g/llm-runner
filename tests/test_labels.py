"""Model labels (1.19): detected from the files, changed with `llmctl label`,
shown by list and ps, and routed to like a group."""
from __future__ import annotations

import json
import struct

import pytest
from click.testing import CliRunner

from llm_runner import chat, cli, control, labels, registry
from llm_runner import config as config_mod
from llm_runner import process as proc
from llm_runner.cluster import Instance, Unroutable, pick

from test_apply import host  # noqa: F401  (fixture)


def _gguf(path, template: str | None = None, size: int = 1 << 20):
    """A GGUF header with (optionally) a chat template, padded to ``size``."""
    kv = b""
    n = 0
    if template is not None:
        key = b"tokenizer.chat_template"
        val = template.encode()
        kv += struct.pack("<Q", len(key)) + key + struct.pack("<I", 8) \
            + struct.pack("<Q", len(val)) + val
        n = 1
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        f.write(b"GGUF" + struct.pack("<I", 3) + struct.pack("<QQ", 0, n) + kv)
        f.truncate(size)


QWEN = "{% if tools %}...{% endif %}{% if enable_thinking %}<think>{% endif %}"
PLAIN = "{% for m in messages %}{{ m.content }}{% endfor %}"


@pytest.fixture
def models(host):  # noqa: F811
    m = registry.MODELS_DIR
    _gguf(m / "qwen3.6-27b" / "Qwen3.6-27B-Q4_K_M.gguf", QWEN)
    _gguf(m / "qwen3.6-27b" / "Qwen3.6-27B-Q8_0.gguf", QWEN)
    _gguf(m / "qwen3.6-27b" / "mmproj-F16.gguf")
    _gguf(m / "glm-ocr" / "GLM-OCR-Q8_0.gguf", PLAIN)
    _gguf(m / "bge-m3" / "bge-m3-Q8_0.gguf")
    _gguf(m / "loose-Q4_K_M.gguf", PLAIN)
    from test_omni import _flux
    _flux(m, "flux.1-dev", dit=1024, t5=1024)
    return host


def _labels():
    return {k: labels.of_spec(s) for k, s in registry.discover_models(registry.MODELS_DIR).items()}


def test_detected_labels(models):
    got = _labels()
    assert got["qwen3.6-27b-q4_k_m"] == ["chat", "vision", "reasoning", "tools"]
    assert got["glm-ocr-q8_0"] == ["chat", "ocr"]
    assert got["bge-m3-q8_0"] == ["embed"]
    assert got["flux.1-dev"] == ["image"]
    assert got["qwen3-v1"] == ["chat"]                  # no template files
    assert got["loose-q4_k_m"] == ["chat"]


def test_template_features_are_cached_until_the_file_changes(models, monkeypatch):
    path = registry.MODELS_DIR / "glm-ocr" / "GLM-OCR-Q8_0.gguf"
    assert labels.template_features(path) == {"reasoning": False, "tools": False}
    assert str(path) in json.loads((proc.STATE_DIR / labels.CACHE).read_text())
    labels._template_cache.clear()
    with monkeypatch.context() as m:        # never monkeypatch.undo(): it drops the fixtures' too
        m.setattr(labels, "_template_text", lambda p: pytest.fail("read again"))
        labels.template_features(path)                   # from the state dir's cache
    _gguf(path, QWEN, size=2 << 20)
    assert labels.template_features(path) == {"reasoning": True, "tools": True}


def test_add_rm_reset_are_kept_in_the_folder_for_every_quant(models):
    r = CliRunner()
    out = r.invoke(cli.cli, ["label", "add", "qwen3.6-27b-q4_k_m", "code"]).output
    assert "qwen3.6-27b-q4_k_m: chat, vision, reasoning, tools, code" in out
    assert "also for qwen3.6-27b-q8_0" in out
    r.invoke(cli.cli, ["label", "rm", "qwen3.6-27b-q8_0", "tools"])
    side = registry.MODELS_DIR / "qwen3.6-27b" / labels.FILE
    assert json.loads(side.read_text()) == {"add": ["code"], "remove": ["tools"]}
    assert _labels()["qwen3.6-27b-q4_k_m"] == ["chat", "vision", "reasoning", "code"]
    show = r.invoke(cli.cli, ["label", "show", "qwen3.6-27b-q4_k_m"]).output
    assert "added:    code" in show and "removed:  tools" in show
    r.invoke(cli.cli, ["label", "reset", "qwen3.6-27b-q4_k_m"])
    assert not side.exists() and "tools" in _labels()["qwen3.6-27b-q8_0"]
    assert r.invoke(cli.cli, ["label", "add", "qwen3.6-27b-q4_k_m", "Bad Name"]).exit_code == 2


def test_a_loose_gguf_has_its_own_file(models):
    CliRunner().invoke(cli.cli, ["label", "add", "loose-q4_k_m", "code"])
    assert (registry.MODELS_DIR / f".loose-Q4_K_M.gguf{labels.FILE}").is_file()
    assert "code" not in _labels()["glm-ocr-q8_0"]


def test_create_delete_and_the_label_list(models):
    r = CliRunner()
    assert r.invoke(cli.cli, ["label", "create", "chat"]).exit_code != 0     # built in
    r.invoke(cli.cli, ["label", "create", "code", "--about", "good at code"])
    out = r.invoke(cli.cli, ["label"]).output
    assert "code" in out and "yours" in out and "good at code" in out
    assert [ln.split()[1] for ln in out.splitlines() if ln.startswith("image ")] == ["1"]
    assert r.invoke(cli.cli, ["label", "delete", "code"]).exit_code == 0
    assert "code" not in labels.created()


def test_list_shows_and_filters_by_labels(models):
    r = CliRunner()
    out = r.invoke(cli.cli, ["list"]).output
    head = out.splitlines()[2].split()
    assert "LABELS" in head and "VISION" not in head and "DECISION" not in head
    assert "chat,vision,reasoning,tools" in out
    out = r.invoke(cli.cli, ["list", "--label", "chat", "--label", "vision"]).output
    keys = [ln.split()[0] for ln in out.splitlines()[4:] if ln.strip()]
    assert keys == ["qwen3.6-27b-q4_k_m", "qwen3.6-27b-q8_0"]
    js = json.loads(r.invoke(cli.cli, ["list", "--json", "--label", "image"]).stdout)
    assert [(m["key"], m["labels"]) for m in js["models"]] == [("flux.1-dev", ["image"])]


def test_run_a_label_starts_its_only_model_or_asks(models, monkeypatch):
    r = CliRunner()
    res = r.invoke(cli.cli, ["run", "embed", "--no-menu", "--port", "8140"])
    assert res.exit_code == 0, res.output
    assert proc.find("bge-m3-q8_0") is not None
    res = r.invoke(cli.cli, ["run", "vision", "--no-menu", "--port", "8141"])
    assert res.exit_code == 2 and "Several models are labelled vision" in res.output
    monkeypatch.setattr(cli, "_interactive", lambda: True)
    monkeypatch.setattr(cli, "_pick_model_interactively",
                        lambda specs, prompt="": sorted(specs)[-1])
    res = r.invoke(cli.cli, ["run", "vision", "--no-menu", "--port", "8141"])
    assert res.exit_code == 0, res.output
    assert proc.autostart_find("qwen3.6-27b-q8_0").run_args[0] == "qwen3.6-27b-q8_0"


def test_download_labels(models, monkeypatch):
    folder = registry.MODELS_DIR / "glm-ocr"
    cli._label_download(folder, ["handwriting"])
    assert _labels()["glm-ocr-q8_0"] == ["chat", "ocr", "handwriting"]
    monkeypatch.setattr(cli, "_interactive", lambda: True)
    monkeypatch.setattr(cli, "_ask_labels", lambda *a: pytest.fail("asked twice"))
    cli._label_download(folder, [])                     # labelled already: no checklist
    asked = []
    monkeypatch.setattr(cli, "_ask_labels", lambda title, cur, det, used: asked.append(
        (title, cur, det)) or ["image"])
    cli._label_download(registry.MODELS_DIR / "bge-m3", [])
    assert asked == [("Labels for bge-m3-q8_0:", ["embed"], ["embed"])]
    assert _labels()["bge-m3-q8_0"] == ["image"]


def test_ps_lists_groups_and_labels_last(models):
    r = CliRunner()
    assert r.invoke(cli.cli, ["run", "qwen3.6-27b-q4_k_m", "--no-menu", "--port", "8142",
                              "--group", "assistant,qwen"]).exit_code == 0
    out = r.invoke(cli.cli, ["ps"]).output
    head, _, row = out.splitlines()[:3]
    assert head.split()[-3:] == ["RUNTIME", "GROUPS", "LABELS"]
    assert row.split()[-2:] == ["assistant,qwen", "chat,vision,reasoning,tools"]
    import re
    assert re.search(r"qwen3.6-27b-q4_k_m .*(ready|starting|-)", row)   # what scripts grep
    js = json.loads(r.invoke(cli.cli, ["ps", "--json"]).stdout)
    [m] = [m for m in js["models"] if m["name"] == "qwen3.6-27b-q4_k_m"]
    assert m["labels"] == ["chat", "vision", "reasoning", "tools"]


def test_a_launch_without_its_projector_is_not_vision(models):
    path = str(registry.MODELS_DIR / "qwen3.6-27b" / "Qwen3.6-27B-Q4_K_M.gguf")
    assert "vision" not in labels.of_launch("q", "llama", path, ["llama-server", "-m", path])
    assert "vision" in labels.of_launch("q", "llama", path, ["llama-server", "--mmproj", "x"])
    assert labels.of_launch("q", "llama", path, ["--embeddings"])[0] == "embed"


# ---------------------------------------------------------------- routing --

def _inst(id_, node="ws", prio=0, labels_=(), group=None, status="ready"):
    return Instance(id=id_, node=node, priority=prio, local=node == "ws",
                    info={"labels": list(labels_), "group": group, "status": status})


POOL = [_inst("gemma-31b", labels_=["chat", "vision"], group="gemma"),
        _inst("qwen-2b", node="orin", prio=1, labels_=["chat"]),
        _inst("bge", node="orin", prio=1, labels_=["embed"])]


def test_a_label_routes_to_its_models():
    assert [i.id for i in pick(POOL, "chat", "openai")] == ["gemma-31b", "qwen-2b"]
    assert [i.id for i in pick(POOL, "embed", "openai")] == ["bge"]
    assert [i.id for i in pick(POOL, "chat@orin", "openai")] == ["qwen-2b"]
    assert [i.id for i in pick(POOL, "chat", "openai", needs_vision=False)][0] == "gemma-31b"


def test_a_preferred_order_comes_first():
    prefer = {"chat": ["qwen-2b"]}
    assert [i.id for i in pick(POOL, "chat", "openai", prefer=prefer)] == ["qwen-2b", "gemma-31b"]
    prefer = {"chat": ["nothing", "gemma"]}               # a group name counts
    assert pick(POOL, "chat", "openai", prefer=prefer)[0].id == "gemma-31b"
    down = [_inst("qwen-2b", labels_=["chat"], status="loading"), POOL[0]]
    assert pick(down, "chat", "openai", prefer={"chat": ["qwen-2b"]})[0].id == "gemma-31b"


def test_models_and_groups_win_and_labels_take_no_prefix():
    pool = POOL + [_inst("chat", labels_=["embed"])]
    assert [i.id for i in pick(pool, "chat", "openai")] == ["chat"]
    pool = POOL + [_inst("other", group="chat")]
    assert [i.id for i in pick(pool, "chat", "openai")] == ["other"]
    with pytest.raises(Unroutable) as e:
        pick(POOL, "emb", "openai")
    assert e.value.code == "model_not_found"


def test_chat_resolves_a_label_to_one_model():
    def entry(id_, labels_, node, local, status="ready"):
        return {"id": id_, "llmctl": {"labels": labels_, "status": status, "local": local,
                                      "node": node, "api": "openai",
                                      "nodes": [{"node": node, "status": status}]}}
    entries = [entry("qwen-2b", ["chat"], "orin", False), entry("gemma", ["chat"], "ws", True)]
    assert chat.resolve(entries, "chat").ref == "gemma"
    config_mod.update(label_prefer={"chat": ["qwen-2b"]})
    assert chat.resolve(entries, "chat").ref == "qwen-2b"


def test_label_commands_run_remotely():
    for argv in (["label"], ["label", "add", "m", "code"], ["label", "prefer", "chat"]):
        control.check_argv(argv)


def test_label_prefer_command():
    r = CliRunner()
    assert "no preferred order" in r.invoke(cli.cli, ["label", "prefer", "chat"]).output
    r.invoke(cli.cli, ["label", "prefer", "chat", "gemma-4-31b-it", "inkling-2-mini"])
    assert labels.preferences() == {"chat": ["gemma-4-31b-it", "inkling-2-mini"]}
    r.invoke(cli.cli, ["label", "prefer", "chat", "--clear"])
    assert labels.preferences() == {}


def test_the_gateway_reports_labels(models):
    from llm_runner import gateway as gw
    assert CliRunner().invoke(cli.cli, ["run", "qwen3.6-27b-q4_k_m", "--no-menu", "--port",
                                        "8143", "--no-vision"]).exit_code == 0
    [m] = gw.LocalModels(proc.STATE_FILE).models()
    assert gw.LocalModels(proc.STATE_FILE).describe(m)["llmctl"]["labels"] == \
        ["chat", "reasoning", "tools"]                  # started without its projector

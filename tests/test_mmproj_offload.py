"""`run --no-mmproj-offload` and the menu's Image encoder row: the vision
encoder on the CPU, for a vision model whose images don't fit in VRAM."""
from __future__ import annotations

from pathlib import Path

from llm_runner import apply, cli, launch

from test_launch import _ctx, _spec, state  # noqa: F401  (fixture)


def _cmd(tmp_path, **kw):
    mmproj = tmp_path / "mmproj.gguf"
    mmproj.touch()
    return cli._build_llama_cmd(Path("/m/x.gguf"), mmproj, 8080, 4096, False, **kw)


def test_the_flag_follows_the_mmproj(tmp_path):
    cmd = _cmd(tmp_path, mmproj_offload=False)
    assert cmd[cmd.index("--mmproj") + 2] == "--no-mmproj-offload"
    assert "--no-mmproj-offload" not in _cmd(tmp_path)
    # Without a projector (--no-vision, a text model) there's nothing to place.
    text = cli._build_llama_cmd(Path("/m/x.gguf"), None, 8080, 4096, False,
                                mmproj_offload=False)
    assert "--no-mmproj-offload" not in text


def test_run_option_is_recorded_for_restarts():
    ctx = _ctx("--no-mmproj-offload")
    assert ctx.params["no_mmproj_offload"] is True
    assert "--no-mmproj-offload" in cli._run_option_args(ctx)
    assert "--no-mmproj-offload" not in cli._run_option_args(_ctx())


def test_menu_row_only_for_vision_models_on_llama(state):  # noqa: F811
    seeing = launch._Menu(_ctx(), launch.Target(_spec(state), "llama"))
    row = next(s for s in seeing.settings if s.param == "no_mmproj_offload")
    assert row.label == "Image encoder"
    assert [v for v, _ in row.values(seeing.target)] == [False, True]   # GPU is the default
    assert seeing.defaults["no_mmproj_offload"] is False
    blind = launch._Menu(_ctx(), launch.Target(_spec(state, "qwen3-8b", vision=False), "llama"))
    assert "no_mmproj_offload" not in [s.param for s in blind.settings]


def test_menu_choice_becomes_a_typed_flag(state):  # noqa: F811
    ctx = _ctx()
    m = launch._Menu(ctx, launch.Target(_spec(state), "llama"))
    m.commit({**m.defaults, "no_mmproj_offload": True})
    assert "--no-mmproj-offload" in cli._run_option_args(ctx)
    assert launch._load_choices()[m.target.spec.key] == {"no_mmproj_offload": True}


def test_menu_warns_when_vision_is_off(state):  # noqa: F811
    m = launch._Menu(_ctx(), launch.Target(_spec(state, draft=False), "llama"))
    assert "does nothing" in m.warning({**m.defaults, "no_vision": True,
                                        "no_mmproj_offload": True})
    assert not m.warning({**m.defaults, "no_mmproj_offload": True})


def test_apply_key():
    e = apply.Entry(name="g", model="gemma", port=8081, options={"no_mmproj_offload": True})
    assert "--no-mmproj-offload" in e.run_argv("gemma")

"""The Group row of the launch menu."""
from pathlib import Path

from llm_runner import launch
from llm_runner.registry import ModelSpec


def test_group_values_offer_auto_none_and_running_groups(tmp_path, monkeypatch):
    folder = tmp_path / "gemma-4-26b-a4b-it"
    folder.mkdir()
    gguf = folder / "gemma-4-26b-a4b-it-Q8_0.gguf"
    gguf.touch()
    monkeypatch.setattr("llm_runner.registry.MODELS_DIR", tmp_path)
    monkeypatch.setattr(launch, "_cluster_groups",
                        lambda: ["gemma-4-26b-a4b-it", "qwen-small"])
    target = launch.Target(ModelSpec("gemma-4-26b-a4b-it-q8_0", gguf, None, 0), "llama")
    values = launch._group_values(target)
    assert values == [(None, "auto (gemma-4-26b-a4b-it)"), ("none", "none"),
                      ("qwen-small", "qwen-small  (running)")]
    group = next(s for s in launch.SETTINGS if s.param == "group_name")
    assert group.custom and group.applies(target)


def test_cli_registry_module_is_not_shadowed():
    # The `registry` click group once hid the registry module from `run`.
    from llm_runner import cli, registry
    assert cli.registry is registry
    assert "registry" in cli.cli.commands


def test_group_of_records_from_before_groups(tmp_path, monkeypatch):
    from llm_runner.registry import group_of
    folder = tmp_path / "quyet-1.0-large"
    folder.mkdir()
    gguf = str(folder / "Quyet-1.0-Large.Q8_0.gguf")
    monkeypatch.setattr("llm_runner.registry.MODELS_DIR", tmp_path)
    assert group_of("quyet-1.0-large.q8_0", "llama", gguf, None) == "quyet-1.0-large"
    assert group_of("quyet-1.0-large.q8_0.decide", "decide", gguf, None) == "quyet-1.0-large.decide"
    assert group_of("quyet-1.0-large.q8_0", "llama", gguf, "") == ""          # --group none
    assert group_of("quyet-1.0-large.q8_0", "llama", gguf, "mine") == "mine"


# ------------------------------------------------------------------ profiles --

import click
import pytest


@pytest.fixture
def state(tmp_path, monkeypatch):
    monkeypatch.setattr(launch, "CHOICES_FILE", tmp_path / "launch.json")
    monkeypatch.setattr(launch, "PROFILES_FILE", tmp_path / "profiles.json")
    monkeypatch.setattr(launch, "_cluster_groups", lambda: [])
    return tmp_path


def _spec(tmp_path, name="gemma-4-31b-it-q4_k_m", draft=True, vision=True, decision=False):
    folder = tmp_path / "models" / name
    folder.mkdir(parents=True, exist_ok=True)
    gguf = folder / f"{name}.gguf"
    gguf.touch()
    return ModelSpec(name, gguf, folder / "mmproj.gguf" if vision else None, 0,
                     draft_path=folder / "mtp.gguf" if draft else None, decision=decision)


def _ctx(*args):
    from llm_runner import cli
    return cli.run.make_context("run", ["m", *args])


def _names(ctx, target):
    return [p.name for p in launch._Menu(ctx, target).profiles()]


def test_profiles_offered_by_model_kind(state):
    drafted = launch.Target(_spec(state), "llama")
    assert _names(_ctx(), drafted) == ["default", "fast", "long", "multi", "serve", "lean"]
    plain = launch.Target(_spec(state, "qwen3-8b", draft=False), "llama")
    assert _names(_ctx(), plain) == ["default", "long", "multi", "serve", "lean"]
    decision = launch.Target(_spec(state, "quyet-1.0-large", draft=False, decision=True), "llama")
    assert _names(_ctx(), decision) == ["default"]
    embedder = launch.Target(_spec(state, "bge-m3-q8_0", draft=False, vision=False), "llama")
    assert _names(_ctx(), embedder) == ["default"]
    assert _names(_ctx("--no-embeddings"), embedder)[1:] == ["long", "multi", "serve", "lean"]


def test_apply_profile_typed_flags_win_and_are_recorded(state):
    from llm_runner import cli
    target = launch.Target(_spec(state), "llama")
    ctx = _ctx("--profile", "long", "--ctx-size", "65536", "--group", "mine")
    launch.apply_profile(ctx, target, "long")
    assert ctx.params["ctx_size"] == 65536            # typed beats the profile's 131072
    assert ctx.params["kv_quant"] == "q8_0" and ctx.params["no_draft"] is True
    assert ctx.params["group_name"] == "mine"
    args = cli._run_option_args(ctx)
    assert "--profile" not in args
    assert args[args.index("--kv-quant") + 1] == "q8_0" and "--no-draft" in args
    remembered = launch._load_choices()[target.spec.key]
    assert remembered["kv_quant"] == "q8_0" and remembered["group_name"] == "mine"


def test_switching_profiles_resets_other_rows_but_keeps_the_group(state):
    target = launch.Target(_spec(state), "llama")
    m = launch._Menu(_ctx(), target)
    lean = m.resolve(m.find("lean"), {**m.defaults, "group_name": "mine"})
    assert lean["no_vision"] is True and lean["group_name"] == "mine"
    serve = m.resolve(m.find("serve"), lean)
    assert serve["no_vision"] is False and serve["n_parallel"] == 4
    assert serve["group_name"] == "mine"
    assert m.resolve(m.find("default"), serve) == {**m.defaults, "group_name": "mine"}


def test_last_restores_the_previous_run_including_the_group(state):
    target = launch.Target(_spec(state), "llama")
    launch._save_choices(target.spec.key, {"ctx_size": 8192, "group_name": "mine"})
    m = launch._Menu(_ctx(), target)
    assert [p.name for p in m.profiles()][:2] == ["last", "default"]
    last = m.resolve(m.find("last"), m.defaults)
    assert last["ctx_size"] == 8192 and last["group_name"] == "mine"


def test_user_profiles_only_where_they_fit(state):
    launch.save_profile("ocr", "llama", {"no_vision": False, "ctx_size": 16384})
    launch.save_profile("textonly", "llama", {"no_vision": True})
    launch.save_profile("big", "vllm", {"ctx_size": 65536})
    seeing = launch.Target(_spec(state), "llama")
    assert _names(_ctx(), seeing)[-3:] == ["lean", "ocr", "textonly"]
    blind = launch.Target(_spec(state, "qwen3-8b", vision=False), "llama")
    assert "ocr" not in _names(_ctx(), blind)          # no Vision row to set
    assert launch.user_profiles()["big"]["backend"] == "vllm"
    assert "big" not in _names(_ctx(), seeing)
    assert launch.delete_profile("ocr") and not launch.delete_profile("ocr")


def test_unknown_profile_names_the_choices(state):
    target = launch.Target(_spec(state, draft=False), "llama")
    with pytest.raises(click.BadParameter, match="default, long, multi, serve, lean"):
        launch.apply_profile(_ctx(), target, "fast")


def test_menu_warns_on_quantized_kv_with_mtp(state):
    m = launch._Menu(_ctx(), launch.Target(_spec(state), "llama"))
    assert m.warning({**m.defaults, "kv_quant": "q8_0"})
    assert not m.warning({**m.defaults, "kv_quant": "q8_0", "no_draft": True})
    assert not m.warning(m.defaults)


def test_multi_is_llama_only(state):
    vllm = launch.Target(_spec(state, "qwen3-8b", draft=False, vision=False), "vllm")
    assert "multi" not in _names(_ctx(), vllm)
    m = launch._Menu(_ctx(), launch.Target(_spec(state), "llama"))
    multi = m.resolve(m.find("multi"), m.defaults)
    assert multi["cache_ram"] == 8192 and multi["n_parallel"] == 1
    assert m.resolve(m.find("lean"), multi)["cache_ram"] == 0

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

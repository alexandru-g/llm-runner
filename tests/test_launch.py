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

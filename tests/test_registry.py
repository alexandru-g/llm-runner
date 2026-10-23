"""Model discovery: the embedding-mode guess."""
import pytest

from llm_runner import registry


@pytest.mark.parametrize("name,emb", [
    ("qwen3.5-4b-imgtest", False), ("ministral-3-14b-fp8-0.1.0", False),
    ("qwen3-v2.5-e5m2", False), ("bge-m3", True), ("multilingual-e5-large", True),
    ("gte-qwen2-7b", True), ("all-MiniLM-L6-v2", True), ("nomic-embed-text-v1.5", True),
    ("Qwen3-Embedding-0.6B", True), ("embeddinggemma-300m", True)])
def test_embedding_hints_match_whole_tokens(name, emb):
    assert registry.looks_like_embedding_model(name) is emb


def test_a_generative_config_overrides_the_name(tmp_path):
    d = tmp_path / "gte-chat"
    d.mkdir()
    (d / "config.json").write_text('{"architectures": ["Qwen3_5ForConditionalGeneration"]}')
    assert not registry.looks_like_embedding_model(d.name, d)
    (d / "config.json").write_text('{"architectures": ["Qwen3Model"]}')
    assert registry.looks_like_embedding_model(d.name, d)

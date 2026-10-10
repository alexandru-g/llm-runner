"""The llama-server readout call."""
import math

from systemone_proxy.backend import LABEL_BIAS, LlamaServer


def fake(response):
    seen = {}

    def call(self, method, path, body=None, timeout=None):
        seen.update(method=method, path=path, body=body)
        return response
    return seen, call


def test_labels_are_biased_and_read_after_sampling(monkeypatch):
    seen, call = fake({"tokens_evaluated": 3, "completion_probabilities": [{"top_probs": [
        {"id": 10, "prob": 0.7}, {"id": 11, "prob": 0.3}, {"id": 12, "prob": 0.0}]}]})
    monkeypatch.setattr(LlamaServer, "_call", call)
    out = LlamaServer("http://x").next_token_logprobs([1, 2, 3], n_probs=67, cache_prompt=False,
                                                      label_ids=[10, 11, 12])
    b = seen["body"]
    assert b["logit_bias"] == [[10, LABEL_BIAS], [11, LABEL_BIAS], [12, LABEL_BIAS]]
    assert b["post_sampling_probs"] is True and b["samplers"] == ["temperature"]
    assert b["temperature"] == 1.0 and b["prompt"] == [1, 2, 3]
    # an underflowed label is left out (the service bounds it and warns)
    assert out == {10: math.log(0.7), 11: math.log(0.3)}


def test_without_labels_reads_raw_top_logprobs(monkeypatch):
    seen, call = fake({"tokens_evaluated": 2, "completion_probabilities": [{"top_logprobs": [
        {"id": 5, "logprob": -0.1}]}]})
    monkeypatch.setattr(LlamaServer, "_call", call)
    out = LlamaServer("http://x").next_token_logprobs([1, 2], n_probs=8, cache_prompt=False)
    assert seen["body"]["post_sampling_probs"] is False and "logit_bias" not in seen["body"]
    assert out == {5: -0.1}

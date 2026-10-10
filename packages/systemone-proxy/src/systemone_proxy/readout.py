"""Probability arithmetic shared by the profiles.

All of the supported models answer with the next-token distribution over the
option labels: ``p = softmax(logprob / T)`` over the offered labels only.
llama-server's ``n_probs`` returns log-softmax values over the full vocabulary,
which differ from the raw logits by one constant per row, so the renormalised
result is identical.
"""
from __future__ import annotations

import math


def softmax_t(logprobs: list[float], t: float) -> list[float]:
    t = max(float(t), 1e-3)
    z = [v / t for v in logprobs]
    m = max(z)
    e = [math.exp(v - m) for v in z]
    s = sum(e)
    return [v / s for v in e]


def argmax(p: list[float]) -> int:
    """First index of the maximum (request order breaks ties)."""
    return max(range(len(p)), key=p.__getitem__)


def expected_index(p: list[float]) -> float:
    total = 0.0
    for i, v in enumerate(p):  # sequential, in option order
        total += float(i) * v
    return total


def top_confidence(p: list[float]) -> float:
    return max(p)


def normalized_confidence(p: list[float]) -> float:
    """TypeSafe choice confidence: how far the top option is above uniform."""
    n = len(p)
    if n == 1:
        return 1.0
    u = 1.0 / n
    return max(0.0, min(1.0, (max(p) - u) / (1.0 - u)))


def score_spread_confidence(p: list[float]) -> float:
    """TypeSafe score confidence: 1 − mean distance from the mode, relative to
    the same distance under a uniform distribution."""
    n = len(p)
    if n == 1:
        return 1.0
    mode = argmax(p)
    distance = 0.0
    for i, v in enumerate(p):
        distance += v * abs(float(i) - float(mode))
    center = (n - 1) / 2.0
    mad = sum(abs(float(i) - center) for i in range(n)) / n
    return max(0.0, 1.0 - distance / mad)


def binary_confidence(p_true: float) -> float:
    return max(p_true, 1.0 - p_true)

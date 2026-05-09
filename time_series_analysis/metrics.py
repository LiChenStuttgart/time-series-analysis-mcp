"""Forecast metric helpers."""

from __future__ import annotations

import math


def mae(actual: list[float], predicted: list[float]) -> float | None:
    pairs = _pairs(actual, predicted)
    if not pairs:
        return None
    return sum(abs(a - p) for a, p in pairs) / len(pairs)


def rmse(actual: list[float], predicted: list[float]) -> float | None:
    pairs = _pairs(actual, predicted)
    if not pairs:
        return None
    return math.sqrt(sum((a - p) ** 2 for a, p in pairs) / len(pairs))


def mape(actual: list[float], predicted: list[float]) -> float | None:
    pairs = [(a, p) for a, p in _pairs(actual, predicted) if abs(a) > 1e-12]
    if not pairs:
        return None
    return 100.0 * sum(abs((a - p) / a) for a, p in pairs) / len(pairs)


def _pairs(actual: list[float], predicted: list[float]) -> list[tuple[float, float]]:
    out = []
    for a, p in zip(actual, predicted, strict=False):
        if math.isfinite(a) and math.isfinite(p):
            out.append((float(a), float(p)))
    return out

"""Structural-break diagnostics."""

from __future__ import annotations

import math

from time_series_analysis_mcp.errors import ValidationError

from .schemas import SeriesCollection, StructuralBreakResult, StructuralBreakSeriesResult
from .utils import clean_values, mean, variance


def run_structural_break_tests(
    collection: SeriesCollection,
    *,
    test_type: str = "auto",
    break_point: int | str | None = None,
    max_breaks: int = 5,
    trim_fraction: float = 0.15,
    alpha: float = 0.05,
) -> StructuralBreakResult:
    selected = test_type.lower()
    results = []
    for item in collection.series:
        if any(value is None for value in item.values):
            raise ValidationError(
                "structural-break tests require complete series; run missing-data handler first",
                series_id=item.series_id,
            )
        values = clean_values(item.values)
        if len(values) < 10:
            results.append(
                StructuralBreakSeriesResult(
                    series_id=item.series_id,
                    test_type=selected,
                    n_obs=len(values),
                    significant=None,
                    warnings=["at least 10 observations are required"],
                )
            )
            continue
        resolved_type = selected
        resolved_break = _resolve_break_point(break_point, collection.time_index)
        if selected == "auto":
            resolved_type = "chow" if resolved_break is not None else "sup_f"
        if resolved_type == "chow":
            if resolved_break is None:
                resolved_break = len(values) // 2
            stat, p_value = _chow(values, resolved_break)
            breaks = [resolved_break]
        elif resolved_type in {"sup_f", "quandt_andrews"}:
            stat, p_value, bp = _sup_f(values, trim_fraction)
            breaks = [bp]
            resolved_type = "sup_f"
        elif resolved_type in {"bai_perron", "ruptures"}:
            stat, p_value, breaks = _multiple_breaks(values, max_breaks, trim_fraction, alpha)
            if selected == "ruptures":
                resolved_type = "bai_perron_fallback"
        elif resolved_type == "variance":
            stat, p_value, bp = _variance_break(values, trim_fraction)
            breaks = [bp]
        else:
            raise ValidationError("unsupported structural break test", test_type=test_type)
        significant = p_value is not None and p_value < alpha
        if not significant and resolved_type == "chow":
            breaks = []
        results.append(
            StructuralBreakSeriesResult(
                series_id=item.series_id,
                test_type=resolved_type,
                n_obs=len(values),
                test_statistic=stat,
                p_value=p_value,
                break_points=breaks,
                break_timestamps=[collection.time_index[idx] for idx in breaks if idx < len(collection.time_index)],
                n_breaks=len(breaks),
                significant=significant,
                warnings=[] if selected != "ruptures" else ["ruptures not installed; used pure-Python fallback"],
            )
        )
    return StructuralBreakResult(structural_break_results=results)


def _resolve_break_point(break_point: int | str | None, time_index: list[str]) -> int | None:
    if break_point is None:
        return None
    if isinstance(break_point, int):
        return break_point
    if break_point in time_index:
        return time_index.index(break_point)
    raise ValidationError("break_point timestamp not found in time_index", break_point=break_point)


def _ssr_mean(values: list[float]) -> float:
    if not values:
        return 0.0
    center = mean(values)
    return sum((value - center) ** 2 for value in values)


def _chow(values: list[float], break_point: int) -> tuple[float, float]:
    n = len(values)
    if break_point <= 1 or break_point >= n - 1:
        raise ValidationError("break_point must leave at least two observations on each side")
    restricted = _ssr_mean(values)
    unrestricted = _ssr_mean(values[:break_point]) + _ssr_mean(values[break_point:])
    denom = unrestricted / max(n - 2, 1)
    if denom <= 1e-12:
        return float("inf"), 0.0
    stat = max((restricted - unrestricted) / denom, 0.0)
    p_value = math.exp(-0.5 * stat)
    return float(stat), float(max(0.0, min(1.0, p_value)))


def _sup_f(values: list[float], trim_fraction: float) -> tuple[float, float, int]:
    n = len(values)
    trim = max(2, int(n * trim_fraction))
    best = (-1.0, 1.0, trim)
    for bp in range(trim, n - trim):
        stat, p_value = _chow(values, bp)
        if stat > best[0]:
            best = (stat, p_value, bp)
    return best


def _multiple_breaks(
    values: list[float],
    max_breaks: int,
    trim_fraction: float,
    alpha: float,
) -> tuple[float, float, list[int]]:
    breaks: list[int] = []
    last_stat = 0.0
    last_p = 1.0
    for _ in range(max_breaks):
        candidates = []
        cuts = [0, *breaks, len(values)]
        for start, stop in zip(cuts[:-1], cuts[1:], strict=True):
            segment = values[start:stop]
            if len(segment) < 10:
                continue
            stat, p_value, local_bp = _sup_f(segment, trim_fraction)
            candidates.append((stat, p_value, start + local_bp))
        if not candidates:
            break
        candidates.sort(reverse=True)
        stat, p_value, bp = candidates[0]
        if p_value >= alpha:
            last_stat, last_p = stat, p_value
            break
        breaks.append(bp)
        breaks.sort()
        last_stat, last_p = stat, p_value
    return last_stat, last_p, breaks


def _variance_break(values: list[float], trim_fraction: float) -> tuple[float, float, int]:
    n = len(values)
    trim = max(2, int(n * trim_fraction))
    best = (-1.0, 1.0, trim)
    for bp in range(trim, n - trim):
        left = values[:bp]
        right = values[bp:]
        v_left = variance(left, sample=True)
        v_right = variance(right, sample=True)
        if min(v_left, v_right) <= 1e-12:
            continue
        ratio = max(v_left, v_right) / min(v_left, v_right)
        p_value = math.exp(-0.5 * ratio)
        if ratio > best[0]:
            best = (ratio, p_value, bp)
    return best

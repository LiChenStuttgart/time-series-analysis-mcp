"""Seasonality diagnostics."""

from __future__ import annotations

import math

from .schemas import SeasonalityResult, SeasonalitySeriesResult, SeriesCollection
from .utils import autocorrelation, clean_values, mean, variance


def run_seasonality_tests(
    collection: SeriesCollection,
    *,
    seasonal_period: int | None = None,
    methods: list[str] | None = None,
    alpha: float = 0.05,
) -> SeasonalityResult:
    requested = [method.lower() for method in (methods or ["acf", "stl_strength", "seasonal_anova"])]
    period = seasonal_period or infer_period(collection.schema_.frequency)
    results: list[SeasonalitySeriesResult] = []
    for item in collection.series:
        if any(value is None for value in item.values):
            results.append(
                SeasonalitySeriesResult(
                    series_id=item.series_id,
                    n_obs=len(item.values),
                    seasonal=None,
                    seasonal_period=period,
                    decision="inconclusive",
                    warnings=["seasonality test requires complete data"],
                )
            )
            continue
        values = clean_values(item.values)
        if not period or len(values) < period * 2:
            results.append(
                SeasonalitySeriesResult(
                    series_id=item.series_id,
                    n_obs=len(values),
                    seasonal=None,
                    seasonal_period=period,
                    decision="inconclusive",
                    warnings=["at least two full seasonal periods are required"],
                )
            )
            continue
        method_results = {}
        support = 0
        if "acf" in requested:
            acf_value = autocorrelation(values, period)
            method_results["acf"] = {"lag": period, "value": acf_value}
            if abs(acf_value) > max(0.3, 2 / math.sqrt(len(values))):
                support += 1
        strength = None
        if "stl_strength" in requested:
            strength = _seasonal_strength(values, period)
            method_results["stl_strength"] = {"value": strength}
            if strength > 0.45:
                support += 1
        if "seasonal_anova" in requested:
            stat = _seasonal_anova_stat(values, period)
            method_results["seasonal_anova"] = {"statistic": stat, "alpha": alpha}
            if stat > 1.8:
                support += 1
        if "periodogram" in requested:
            power = _periodogram_power(values, period)
            method_results["periodogram"] = {"relative_power": power}
            if power > 0.25:
                support += 1
        seasonal = support >= 2 or (strength is not None and strength > 0.7)
        results.append(
            SeasonalitySeriesResult(
                series_id=item.series_id,
                n_obs=len(values),
                seasonal=seasonal,
                seasonal_period=period,
                seasonal_strength=strength,
                method_results=method_results,
                decision="seasonal" if seasonal else "not_seasonal",
            )
        )
    return SeasonalityResult(seasonality_results=results)


def infer_period(frequency: str) -> int | None:
    freq = frequency.upper()
    if freq == "D":
        return 7
    if freq == "M":
        return 12
    if freq == "Q":
        return 4
    return None


def _seasonal_strength(values: list[float], period: int) -> float:
    seasonal_means = []
    overall = mean(values)
    for pos in range(period):
        group = [value for idx, value in enumerate(values) if idx % period == pos]
        seasonal_means.append(mean(group) if group else overall)
    fitted = [seasonal_means[idx % period] for idx in range(len(values))]
    residual = [value - fit for value, fit in zip(values, fitted, strict=True)]
    raw_var = variance(values)
    resid_var = variance(residual)
    if raw_var <= 1e-12:
        return 0.0
    return max(0.0, min(1.0, 1 - resid_var / raw_var))


def _seasonal_anova_stat(values: list[float], period: int) -> float:
    overall = mean(values)
    groups = [[value for idx, value in enumerate(values) if idx % period == pos] for pos in range(period)]
    between = sum(len(group) * (mean(group) - overall) ** 2 for group in groups if group)
    within = sum(sum((value - mean(group)) ** 2 for value in group) for group in groups if group)
    df_between = max(sum(1 for group in groups if group) - 1, 1)
    df_within = max(len(values) - len(groups), 1)
    return (between / df_between) / max(within / df_within, 1e-12)


def _periodogram_power(values: list[float], period: int) -> float:
    n = len(values)
    centered = [value - mean(values) for value in values]
    target_k = max(1, round(n / period))
    powers = []
    for k in range(1, n // 2 + 1):
        real = sum(centered[t] * math.cos(2 * math.pi * k * t / n) for t in range(n))
        imag = sum(centered[t] * math.sin(2 * math.pi * k * t / n) for t in range(n))
        powers.append((k, real * real + imag * imag))
    total = sum(power for _k, power in powers) or 1.0
    target = sum(power for k, power in powers if abs(k - target_k) <= 1)
    return target / total

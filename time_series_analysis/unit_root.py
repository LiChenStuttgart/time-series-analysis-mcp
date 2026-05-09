"""Stationarity and unit-root diagnostics."""

from __future__ import annotations

from time_series_analysis_mcp.errors import ValidationError

from .schemas import SeriesCollection, SingleTestResult, UnitRootResult, UnitRootSeriesResult
from .utils import clean_values, normal_left_tail_p, ols, variance


SUPPORTED_TESTS = {"adf", "kpss", "pp"}


def run_unit_root_tests(
    collection: SeriesCollection,
    *,
    tests: list[str] | None = None,
    regression: str = "c",
    max_lags: int | None = None,
    alpha: float = 0.05,
    difference_recommendation: bool = True,
) -> UnitRootResult:
    requested = [test.lower() for test in (tests or ["adf", "kpss"])]
    unknown = [test for test in requested if test not in SUPPORTED_TESTS]
    if unknown:
        raise ValidationError("unsupported unit-root test", tests=unknown)
    results: list[UnitRootSeriesResult] = []
    for item in collection.series:
        if any(value is None for value in item.values):
            raise ValidationError(
                "unit-root tests require complete series; run time_series_missing_data_handler first",
                series_id=item.series_id,
            )
        values = clean_values(item.values)
        if len(values) < 8:
            results.append(
                UnitRootSeriesResult(
                    series_id=item.series_id,
                    n_obs=len(values),
                    tests=[],
                    recommendation="inconclusive",
                    warnings=["unit-root diagnostics require at least 8 observations"],
                )
            )
            continue
        test_results = []
        for test in requested:
            if test == "adf":
                test_results.append(_adf(values, regression, max_lags, alpha))
            elif test == "kpss":
                test_results.append(_kpss(values, regression, alpha))
            elif test == "pp":
                pp = _adf(values, regression, max_lags, alpha)
                pp.test_name = "Phillips-Perron Test (ADF fallback)"
                pp.warnings.append(
                    "pure-Python fallback approximates PP with an ADF-style regression"
                )
                test_results.append(pp)
        recommendation = _recommend(test_results, difference_recommendation)
        results.append(
            UnitRootSeriesResult(
                series_id=item.series_id,
                n_obs=len(values),
                tests=test_results,
                recommendation=recommendation,
            )
        )
    return UnitRootResult(unit_root_results=results)


def _adf(
    values: list[float],
    regression: str,
    max_lags: int | None,
    alpha: float,
) -> SingleTestResult:
    if len(values) < 4:
        raise ValidationError("ADF test requires at least 4 observations")
    lagged = values[:-1]
    delta = [b - a for a, b in zip(values[:-1], values[1:], strict=True)]
    rows = []
    targets = []
    lags = max_lags or 0
    start = lags
    for idx in range(start, len(delta)):
        row = []
        if regression in {"c", "ct"}:
            row.append(1.0)
        if regression == "ct":
            row.append(float(idx + 1))
        row.append(lagged[idx])
        for lag in range(1, lags + 1):
            row.append(delta[idx - lag])
        rows.append(row)
        targets.append(delta[idx])
    beta, se, _sse = ols(targets, rows)
    coeff_idx = (1 if regression in {"c", "ct"} else 0) + (1 if regression == "ct" else 0)
    stat = beta[coeff_idx] / se[coeff_idx] if se[coeff_idx] > 1e-12 else 0.0
    critical = {"1%": -3.43, "5%": -2.86, "10%": -2.57}
    p_value = normal_left_tail_p(stat + 1.65)
    reject = stat < critical["5%"] if alpha <= 0.05 else stat < critical["10%"]
    interpretation = "stationary" if reject else "unit root not rejected"
    return SingleTestResult(
        test_name="Augmented Dickey-Fuller Test",
        null_hypothesis="series has a unit root (non-stationary)",
        test_statistic=float(stat),
        p_value=float(p_value),
        critical_values=critical,
        lags=lags,
        reject_null=reject,
        stationary_interpretation=interpretation,
    )


def _kpss(values: list[float], regression: str, alpha: float) -> SingleTestResult:
    n = len(values)
    if regression == "ct":
        rows = [[1.0, float(idx)] for idx in range(n)]
    else:
        rows = [[1.0] for _ in range(n)]
    beta, _se, _sse = ols(values, rows)
    residuals = [y - sum(b * x for b, x in zip(beta, row, strict=True)) for y, row in zip(values, rows, strict=True)]
    sigma2 = max(variance(residuals), 1e-12)
    cumulative = []
    total = 0.0
    for resid in residuals:
        total += resid
        cumulative.append(total)
    stat = sum(value * value for value in cumulative) / (n * n * sigma2)
    critical = {"10%": 0.347, "5%": 0.463, "2.5%": 0.574, "1%": 0.739}
    if regression == "ct":
        critical = {"10%": 0.119, "5%": 0.146, "2.5%": 0.176, "1%": 0.216}
    reject = stat > critical["5%"] if alpha <= 0.05 else stat > critical["10%"]
    p_value = 0.01 if stat > critical["1%"] else 0.05 if stat > critical["5%"] else 0.1
    interpretation = "non-stationary" if reject else "stationarity not rejected"
    return SingleTestResult(
        test_name="KPSS Test",
        null_hypothesis="series is stationary",
        test_statistic=float(stat),
        p_value=float(p_value),
        critical_values=critical,
        lags=None,
        reject_null=reject,
        stationary_interpretation=interpretation,
    )


def _recommend(results: list[SingleTestResult], difference_recommendation: bool) -> str:
    adf_stationary = any(
        result.test_name.startswith("Augmented") and result.reject_null is True for result in results
    )
    kpss_stationary = any(
        result.test_name.startswith("KPSS") and result.reject_null is False for result in results
    )
    kpss_nonstationary = any(
        result.test_name.startswith("KPSS") and result.reject_null is True for result in results
    )
    if adf_stationary and kpss_stationary:
        return "stationary"
    if not adf_stationary and kpss_nonstationary and difference_recommendation:
        return "difference_once"
    if adf_stationary and kpss_nonstationary:
        return "trend_or_regime_change_candidate"
    return "inconclusive"

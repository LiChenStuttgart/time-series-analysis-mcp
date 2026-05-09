"""Forecasting models."""

from __future__ import annotations

import math

from time_series_analysis_mcp.errors import EstimationError, ValidationError

from .metrics import mae, mape, rmse
from .schemas import ForecastMetrics, ForecastResult, ForecastSeriesResult, SeriesCollection
from .utils import clean_values, difference, integrate_forecast, next_index_values, ols, stddev


def forecast_arima(
    collection: SeriesCollection,
    *,
    order: list[int] | tuple[int, int, int] | None = None,
    seasonal_order: list[int] | tuple[int, int, int, int] | None = None,
    auto_order: bool = False,
    max_p: int = 3,
    max_d: int = 2,
    max_q: int = 3,
    forecast_steps: int = 12,
    confidence_level: float = 0.95,
    holdout_size: int = 0,
) -> ForecastResult:
    if forecast_steps <= 0:
        raise ValidationError("forecast_steps must be positive", forecast_steps=forecast_steps)
    results = []
    for item in collection.series:
        if any(value is None for value in item.values):
            raise ValidationError(
                "ARIMA requires complete series; run time_series_missing_data_handler first",
                series_id=item.series_id,
            )
        values = clean_values(item.values)
        if len(values) < 8:
            raise ValidationError("ARIMA requires at least 8 observations", series_id=item.series_id)
        train, holdout = _split_holdout(values, holdout_size)
        selected_order = _select_order(train, max_p=max_p, max_d=max_d, max_q=max_q) if auto_order else _coerce_order(order)
        fit = _fit_ar_model(train, selected_order)
        forecast = _forecast_ar(train, fit, forecast_steps)
        lower, upper = _intervals(forecast, fit["residual_std"], confidence_level)
        metrics = ForecastMetrics(
            in_sample_mae=mae(fit["actual"], fit["fitted"]),
            in_sample_rmse=rmse(fit["actual"], fit["fitted"]),
            in_sample_mape=mape(fit["actual"], fit["fitted"]),
        )
        if holdout:
            holdout_forecast = _forecast_ar(train, fit, len(holdout))
            metrics.holdout_mae = mae(holdout, holdout_forecast)
            metrics.holdout_rmse = rmse(holdout, holdout_forecast)
            metrics.holdout_mape = mape(holdout, holdout_forecast)
        warnings = []
        if seasonal_order is not None:
            warnings.append("seasonal_order recorded but pure-Python ARIMA fallback ignores seasonal terms")
        if selected_order[2] != 0:
            warnings.append("MA terms are approximated by an autoregressive fallback")
        results.append(
            ForecastSeriesResult(
                series_id=item.series_id,
                model="ARIMA",
                order=list(selected_order),
                seasonal_order=list(seasonal_order) if seasonal_order is not None else None,
                parameters=fit["parameters"],
                aic=fit["aic"],
                bic=fit["bic"],
                forecast_index=next_index_values(
                    collection.time_index,
                    collection.schema_.frequency,
                    forecast_steps,
                ),
                forecast=forecast,
                lower=lower,
                upper=upper,
                metrics=metrics,
                fit_warnings=warnings + fit["warnings"],
            )
        )
    return ForecastResult(forecast_results=results)


def forecast_exponential_smoothing(
    collection: SeriesCollection,
    *,
    trend: str | None = "add",
    seasonal: str | None = None,
    seasonal_periods: int | None = None,
    damped_trend: bool = False,
    initialization_method: str = "estimated",
    forecast_steps: int = 12,
    confidence_level: float = 0.95,
    holdout_size: int = 0,
) -> ForecastResult:
    if forecast_steps <= 0:
        raise ValidationError("forecast_steps must be positive", forecast_steps=forecast_steps)
    if trend not in {None, "add", "mul"}:
        raise ValidationError("trend must be null, 'add', or 'mul'", trend=trend)
    if seasonal not in {None, "add", "mul"}:
        raise ValidationError("seasonal must be null, 'add', or 'mul'", seasonal=seasonal)
    results = []
    for item in collection.series:
        if any(value is None for value in item.values):
            raise ValidationError(
                "exponential smoothing requires complete series; run missing-data handler first",
                series_id=item.series_id,
            )
        values = clean_values(item.values)
        if len(values) < 4:
            raise ValidationError(
                "exponential smoothing requires at least 4 observations",
                series_id=item.series_id,
            )
        if (trend == "mul" or seasonal == "mul") and any(value <= 0 for value in values):
            raise ValidationError("multiplicative smoothing requires strictly positive data")
        period = seasonal_periods or _infer_period(collection.schema_.frequency) if seasonal else None
        if seasonal and period and len(values) < period * 2:
            raise ValidationError("seasonal smoothing requires at least two full seasons")
        train, holdout = _split_holdout(values, holdout_size)
        fit = _fit_exp_smoothing(
            train,
            trend=trend,
            seasonal=seasonal,
            seasonal_periods=period,
            damped_trend=damped_trend,
        )
        forecast = _forecast_exp(fit, forecast_steps)
        lower, upper = _intervals(forecast, fit["residual_std"], confidence_level)
        metrics = ForecastMetrics(
            in_sample_mae=mae(fit["actual"], fit["fitted"]),
            in_sample_rmse=rmse(fit["actual"], fit["fitted"]),
            in_sample_mape=mape(fit["actual"], fit["fitted"]),
        )
        if holdout:
            holdout_forecast = _forecast_exp(fit, len(holdout))
            metrics.holdout_mae = mae(holdout, holdout_forecast)
            metrics.holdout_rmse = rmse(holdout, holdout_forecast)
            metrics.holdout_mape = mape(holdout, holdout_forecast)
        results.append(
            ForecastSeriesResult(
                series_id=item.series_id,
                model="ExponentialSmoothing",
                order=None,
                seasonal_order=[0, 0, 0, period] if period else None,
                parameters=fit["parameters"],
                aic=fit["aic"],
                bic=fit["bic"],
                forecast_index=next_index_values(
                    collection.time_index,
                    collection.schema_.frequency,
                    forecast_steps,
                ),
                forecast=forecast,
                lower=lower,
                upper=upper,
                metrics=metrics,
                fit_warnings=fit["warnings"]
                + ([] if initialization_method == "estimated" else ["initialization_method recorded but pure-Python fallback estimates initial state"]),
            )
        )
    return ForecastResult(forecast_results=results)


def _coerce_order(order: list[int] | tuple[int, int, int] | None) -> tuple[int, int, int]:
    if order is None:
        return (1, 0, 0)
    if len(order) != 3:
        raise ValidationError("order must have three integers [p, d, q]", order=list(order))
    p, d, q = [int(value) for value in order]
    if min(p, d, q) < 0:
        raise ValidationError("order values must be non-negative", order=[p, d, q])
    return p, d, q


def _split_holdout(values: list[float], holdout_size: int) -> tuple[list[float], list[float]]:
    if holdout_size <= 0:
        return values, []
    if holdout_size >= len(values) - 6:
        raise ValidationError("holdout_size leaves too few training observations")
    return values[:-holdout_size], values[-holdout_size:]


def _select_order(values: list[float], *, max_p: int, max_d: int, max_q: int) -> tuple[int, int, int]:
    best_order = (1, 0, 0)
    best_aic = float("inf")
    for d in range(max_d + 1):
        for p in range(max_p + 1):
            try:
                fit = _fit_ar_model(values, (p, d, 0))
            except Exception:
                continue
            if fit["aic"] is not None and fit["aic"] < best_aic:
                best_aic = fit["aic"]
                best_order = (p, d, 0)
    if max_q > 0:
        # q is kept at zero in the pure-Python fallback, but max_q remains in
        # the public API so a statsmodels implementation can fill it later.
        pass
    return best_order


def _fit_ar_model(values: list[float], order: tuple[int, int, int]) -> dict:
    p, d, q = order
    transformed = difference(values, d)
    if len(transformed) <= max(p + 2, 3):
        raise EstimationError("not enough observations for requested ARIMA order", order=list(order))
    warnings: list[str] = []
    if p == 0:
        level = sum(transformed) / len(transformed)
        fitted = [level for _ in transformed]
        residuals = [value - level for value in transformed]
        parameters = {"const": level}
        actual = transformed
    else:
        rows = []
        targets = []
        for idx in range(p, len(transformed)):
            rows.append([1.0, *[transformed[idx - lag] for lag in range(1, p + 1)]])
            targets.append(transformed[idx])
        try:
            beta, _se, _sse = ols(targets, rows)
        except Exception as exc:  # noqa: BLE001
            warnings.append(f"OLS fit failed ({exc}); using persistence fallback")
            beta = [0.0] + [1.0] + [0.0] * (p - 1)
        fitted = [sum(beta[j] * row[j] for j in range(len(beta))) for row in rows]
        residuals = [target - pred for target, pred in zip(targets, fitted, strict=True)]
        parameters = {"const": beta[0], **{f"ar.L{i}": beta[i] for i in range(1, len(beta))}}
        actual = targets
    sse = sum(resid * resid for resid in residuals)
    n = max(len(actual), 1)
    k = len(parameters)
    sigma = math.sqrt(max(sse / n, 1e-12))
    aic = n * math.log(max(sse / n, 1e-12)) + 2 * k
    bic = n * math.log(max(sse / n, 1e-12)) + k * math.log(n)
    return {
        "order": order,
        "transformed": transformed,
        "parameters": {key: float(value) for key, value in parameters.items()},
        "fitted": fitted,
        "actual": actual,
        "residual_std": sigma,
        "aic": float(aic),
        "bic": float(bic),
        "warnings": warnings,
    }


def _forecast_ar(values: list[float], fit: dict, steps: int) -> list[float]:
    p, d, _q = fit["order"]
    transformed = fit["transformed"][:]
    params = fit["parameters"]
    diff_forecast = []
    for _ in range(steps):
        if p == 0:
            pred = params["const"]
        else:
            pred = params.get("const", 0.0)
            for lag in range(1, p + 1):
                pred += params.get(f"ar.L{lag}", 0.0) * transformed[-lag]
        transformed.append(pred)
        diff_forecast.append(pred)
    return [float(value) for value in integrate_forecast(values, diff_forecast, d)]


def _fit_exp_smoothing(
    values: list[float],
    *,
    trend: str | None,
    seasonal: str | None,
    seasonal_periods: int | None,
    damped_trend: bool,
) -> dict:
    alpha = 0.4
    beta = 0.2 if trend else 0.0
    gamma = 0.2 if seasonal and seasonal_periods else 0.0
    phi = 0.9 if damped_trend else 1.0
    level = values[0]
    trend_value = values[1] - values[0] if len(values) > 1 and trend else 0.0
    seasonals = [0.0] * (seasonal_periods or 1)
    warnings = []
    if seasonal and seasonal_periods:
        baseline = sum(values[:seasonal_periods]) / seasonal_periods
        for idx in range(seasonal_periods):
            same = [values[j] for j in range(idx, len(values), seasonal_periods)]
            seasonals[idx] = sum(same) / len(same) - baseline
        if seasonal == "mul":
            warnings.append("multiplicative seasonality approximated by additive fallback")
    if trend == "mul":
        warnings.append("multiplicative trend approximated by additive fallback")
    fitted = []
    for idx, value in enumerate(values):
        seasonal_component = seasonals[idx % len(seasonals)] if seasonal else 0.0
        prediction = level + phi * trend_value + seasonal_component
        fitted.append(prediction)
        prev_level = level
        level = alpha * (value - seasonal_component) + (1 - alpha) * (level + phi * trend_value)
        if trend:
            trend_value = beta * (level - prev_level) + (1 - beta) * phi * trend_value
        if seasonal and seasonal_periods:
            seasonals[idx % seasonal_periods] = gamma * (value - level) + (1 - gamma) * seasonal_component
    residuals = [actual - pred for actual, pred in zip(values, fitted, strict=True)]
    sse = sum(resid * resid for resid in residuals)
    n = len(values)
    k = 2 + (seasonal_periods or 0)
    aic = n * math.log(max(sse / n, 1e-12)) + 2 * k
    bic = n * math.log(max(sse / n, 1e-12)) + k * math.log(n)
    return {
        "level": float(level),
        "trend": float(trend_value),
        "seasonals": [float(value) for value in seasonals],
        "seasonal_periods": seasonal_periods,
        "parameters": {
            "smoothing_level": alpha,
            "smoothing_trend": beta,
            "smoothing_seasonal": gamma,
            "damping_trend": phi,
        },
        "actual": values,
        "fitted": fitted,
        "residual_std": stddev(residuals),
        "aic": float(aic),
        "bic": float(bic),
        "warnings": warnings,
    }


def _forecast_exp(fit: dict, steps: int) -> list[float]:
    out = []
    period = fit["seasonal_periods"] or 1
    seasonals = fit["seasonals"]
    for h in range(1, steps + 1):
        seasonal_component = seasonals[(h - 1) % period] if seasonals else 0.0
        out.append(float(fit["level"] + h * fit["trend"] + seasonal_component))
    return out


def _intervals(values: list[float], residual_std: float, confidence_level: float) -> tuple[list[float], list[float]]:
    if confidence_level >= 0.99:
        z = 2.58
    elif confidence_level >= 0.95:
        z = 1.96
    elif confidence_level >= 0.9:
        z = 1.64
    else:
        z = 1.28
    width = z * residual_std
    return [float(value - width) for value in values], [float(value + width) for value in values]


def _infer_period(frequency: str) -> int | None:
    freq = frequency.upper()
    if freq == "D":
        return 7
    if freq == "M":
        return 12
    if freq == "Q":
        return 4
    return None

"""Classical statistical forecasting models."""

from __future__ import annotations

import math
import warnings
from itertools import product
from typing import Any

import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing

from time_series_analysis_mcp.errors import EstimationError, ValidationError

from .covariates import build_covariate_bundles
from .metrics import mae, mape, rmse
from .schemas import (
    ForecastMetrics,
    ForecastResult,
    ForecastSeriesResult,
    SeriesCollection,
    TimeSeriesItem,
)
from .utils import clean_values, next_index_values


def forecast_arima(
    collection: SeriesCollection,
    *,
    target_metrics: list[str] | None = None,
    order: list[int] | tuple[int, int, int] | None = None,
    seasonal_order: list[int] | tuple[int, int, int, int] | None = None,
    auto_order: bool = False,
    max_p: int = 3,
    max_d: int = 2,
    max_q: int = 3,
    seasonal_period: int | None = None,
    max_P: int = 1,
    max_D: int = 1,
    max_Q: int = 1,
    information_criterion: str = "aic",
    forecast_steps: int = 12,
    confidence_level: float = 0.95,
    holdout_size: int = 0,
    known_future_covariate_metrics: list[str] | None = None,
    past_covariate_metrics: list[str] | None = None,
    future_covariates: dict[str, Any] | None = None,
    calendar_covariates: list[str] | None = None,
) -> ForecastResult:
    """Fit ARIMA/SARIMA models with maximum-likelihood parameter estimation."""
    _validate_forecast_request(forecast_steps, confidence_level)
    _validate_order_bounds(max_p, max_d, max_q, max_P, max_D, max_Q)
    criterion = _coerce_information_criterion(information_criterion)
    requested_order = _coerce_order(order)
    requested_seasonal = _coerce_seasonal_order(seasonal_order)
    excluded_metrics = set(known_future_covariate_metrics or []) | set(past_covariate_metrics or [])
    selected_items = _select_forecast_series(collection, target_metrics, excluded_metrics)
    covariates = build_covariate_bundles(
        collection,
        selected_items,
        forecast_steps=forecast_steps,
        known_future_covariate_metrics=known_future_covariate_metrics,
        past_covariate_metrics=past_covariate_metrics,
        future_covariates=future_covariates,
        calendar_covariates=calendar_covariates,
        allow_past_only=False,
    )
    results = []
    for item in selected_items:
        if any(value is None for value in item.values):
            raise ValidationError(
                "ARIMA requires complete series; run time_series_missing_data_handler first",
                series_id=item.series_id,
            )
        values = clean_values(item.values)
        if len(values) < 8:
            raise ValidationError("ARIMA requires at least 8 observations", series_id=item.series_id)
        bundle = covariates[item.series_id]
        exog = _exog_array(bundle.history_rows(bundle.known_future_names))
        future_exog = _exog_array(
            bundle.future_rows(history_end=len(values), prediction_length=forecast_steps)
        )
        train, holdout = _split_holdout(values, holdout_size)
        train_exog = exog[: len(train)] if exog is not None else None
        holdout_exog = exog[len(train) :] if exog is not None and holdout else None
        fit_warnings: list[str] = []

        if auto_order:
            search_period = None
            fixed_seasonal_order = requested_seasonal if seasonal_order is not None else None
            if fixed_seasonal_order is None:
                search_period = (
                    seasonal_period
                    if seasonal_period is not None
                    else _infer_period(collection.schema_.frequency)
                )
                if search_period is not None:
                    _validate_seasonal_period(search_period)
                    if len(train) < 2 * search_period:
                        fit_warnings.append(
                            "seasonal order selection skipped because training data contains fewer than two seasonal periods"
                        )
                        search_period = None
            selected_order, selected_seasonal = _select_sarima_spec(
                train,
                fixed_seasonal_order=fixed_seasonal_order,
                seasonal_period=search_period,
                exog=train_exog,
                max_p=max_p,
                max_d=max_d,
                max_q=max_q,
                max_P=max_P,
                max_D=max_D,
                max_Q=max_Q,
                information_criterion=criterion,
            )
            report_seasonal = seasonal_order is not None or search_period is not None
        else:
            selected_order = requested_order
            selected_seasonal = requested_seasonal
            report_seasonal = seasonal_order is not None

        _validate_sarima_history(train, selected_seasonal)
        if holdout:
            training_fit, training_warnings = _fit_sarima(
                train, selected_order, selected_seasonal, exog=train_exog
            )
            holdout_forecast, _lower, _upper = _sarima_forecast(
                training_fit, len(holdout), confidence_level, exog=holdout_exog
            )
            holdout_metrics = _holdout_metrics(holdout, holdout_forecast)
            fit_warnings.extend(f"holdout fit: {message}" for message in training_warnings)
        else:
            holdout_metrics = {}

        _validate_sarima_history(values, selected_seasonal)
        fitted, final_warnings = _fit_sarima(
            values, selected_order, selected_seasonal, exog=exog
        )
        forecast, lower, upper = _sarima_forecast(
            fitted, forecast_steps, confidence_level, exog=future_exog
        )
        metrics = _in_sample_metrics(values, fitted.fittedvalues)
        _assign_metrics(metrics, holdout_metrics)
        fit_warnings.extend(final_warnings)
        results.append(
            ForecastSeriesResult(
                series_id=item.series_id,
                model="ARIMA",
                order=list(selected_order),
                seasonal_order=list(selected_seasonal) if report_seasonal else None,
                parameters=_sarima_parameters(fitted),
                aic=_optional_finite_float(fitted.aic),
                bic=_optional_finite_float(fitted.bic),
                forecast_index=next_index_values(
                    collection.time_index,
                    collection.schema_.frequency,
                    forecast_steps,
                ),
                forecast=forecast,
                lower=lower,
                upper=upper,
                metrics=metrics,
                covariates=bundle.metadata(),
                fit_warnings=_unique_warnings(fit_warnings),
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
    interval_simulations: int = 1000,
    random_seed: int | None = None,
) -> ForecastResult:
    """Fit optimized Holt-Winters exponential smoothing models."""
    _validate_forecast_request(forecast_steps, confidence_level)
    if trend not in {None, "add", "mul"}:
        raise ValidationError("trend must be null, 'add', or 'mul'", trend=trend)
    if seasonal not in {None, "add", "mul"}:
        raise ValidationError("seasonal must be null, 'add', or 'mul'", seasonal=seasonal)
    if initialization_method not in {"estimated", "heuristic", "legacy-heuristic"}:
        raise ValidationError(
            "initialization_method must be 'estimated', 'heuristic', or 'legacy-heuristic'",
            initialization_method=initialization_method,
        )
    if interval_simulations <= 0:
        raise ValidationError(
            "interval_simulations must be positive", interval_simulations=interval_simulations
        )
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
        period = None
        if seasonal:
            period = (
                seasonal_periods
                if seasonal_periods is not None
                else _infer_period(collection.schema_.frequency)
            )
        if seasonal:
            if period is None:
                raise ValidationError(
                    "seasonal_periods is required when seasonality cannot be inferred from frequency"
                )
            _validate_seasonal_period(period)
        train, holdout = _split_holdout(values, holdout_size)
        _validate_ets_history(train, seasonal, period)
        fit_warnings: list[str] = []
        if holdout:
            training_fit, training_warnings = _fit_exponential_smoothing(
                train,
                trend=trend,
                seasonal=seasonal,
                seasonal_periods=period,
                damped_trend=damped_trend,
                initialization_method=initialization_method,
            )
            holdout_forecast = _float_list(training_fit.forecast(len(holdout)), "ETS holdout forecast")
            holdout_metrics = _holdout_metrics(holdout, holdout_forecast)
            fit_warnings.extend(f"holdout fit: {message}" for message in training_warnings)
        else:
            holdout_metrics = {}

        _validate_ets_history(values, seasonal, period)
        fitted, final_warnings = _fit_exponential_smoothing(
            values,
            trend=trend,
            seasonal=seasonal,
            seasonal_periods=period,
            damped_trend=damped_trend,
            initialization_method=initialization_method,
        )
        forecast = _float_list(fitted.forecast(forecast_steps), "ETS forecast")
        lower, upper = _ets_intervals(
            fitted,
            forecast_steps,
            confidence_level,
            interval_simulations,
            random_seed,
        )
        metrics = _in_sample_metrics(values, fitted.fittedvalues)
        _assign_metrics(metrics, holdout_metrics)
        fit_warnings.extend(final_warnings)
        results.append(
            ForecastSeriesResult(
                series_id=item.series_id,
                model="ExponentialSmoothing",
                order=None,
                seasonal_order=[0, 0, 0, period] if period else None,
                parameters=_ets_parameters(fitted.params),
                aic=_optional_finite_float(fitted.aic),
                bic=_optional_finite_float(fitted.bic),
                forecast_index=next_index_values(
                    collection.time_index,
                    collection.schema_.frequency,
                    forecast_steps,
                ),
                forecast=forecast,
                lower=lower,
                upper=upper,
                metrics=metrics,
                fit_warnings=_unique_warnings(fit_warnings),
            )
        )
    return ForecastResult(forecast_results=results)


def _validate_forecast_request(forecast_steps: int, confidence_level: float) -> None:
    if forecast_steps <= 0:
        raise ValidationError("forecast_steps must be positive", forecast_steps=forecast_steps)
    if not 0 < confidence_level < 1:
        raise ValidationError(
            "confidence_level must be between 0 and 1", confidence_level=confidence_level
        )


def _coerce_order(order: list[int] | tuple[int, int, int] | None) -> tuple[int, int, int]:
    if order is None:
        return (1, 0, 0)
    if len(order) != 3:
        raise ValidationError("order must have three integers [p, d, q]", order=list(order))
    p, d, q = [int(value) for value in order]
    if min(p, d, q) < 0:
        raise ValidationError("order values must be non-negative", order=[p, d, q])
    return p, d, q


def _coerce_seasonal_order(
    seasonal_order: list[int] | tuple[int, int, int, int] | None,
) -> tuple[int, int, int, int]:
    if seasonal_order is None:
        return (0, 0, 0, 0)
    if len(seasonal_order) != 4:
        raise ValidationError(
            "seasonal_order must have four integers [P, D, Q, s]",
            seasonal_order=list(seasonal_order),
        )
    P, D, Q, period = [int(value) for value in seasonal_order]
    if min(P, D, Q, period) < 0:
        raise ValidationError(
            "seasonal_order values must be non-negative",
            seasonal_order=[P, D, Q, period],
        )
    if P + D + Q > 0:
        _validate_seasonal_period(period)
    elif period == 1:
        raise ValidationError("seasonal period must be at least 2", seasonal_period=period)
    return P, D, Q, period


def _validate_order_bounds(
    max_p: int, max_d: int, max_q: int, max_P: int, max_D: int, max_Q: int
) -> None:
    bounds = {
        "max_p": max_p,
        "max_d": max_d,
        "max_q": max_q,
        "max_P": max_P,
        "max_D": max_D,
        "max_Q": max_Q,
    }
    invalid = {name: value for name, value in bounds.items() if value < 0}
    if invalid:
        raise ValidationError("automatic order limits must be non-negative", **invalid)


def _coerce_information_criterion(value: str) -> str:
    criterion = value.lower()
    if criterion not in {"aic", "aicc", "bic"}:
        raise ValidationError(
            "information_criterion must be 'aic', 'aicc', or 'bic'",
            information_criterion=value,
        )
    return criterion


def _validate_seasonal_period(period: int) -> None:
    if period < 2:
        raise ValidationError("seasonal period must be at least 2", seasonal_period=period)


def _split_holdout(values: list[float], holdout_size: int) -> tuple[list[float], list[float]]:
    if holdout_size < 0:
        raise ValidationError("holdout_size must be non-negative", holdout_size=holdout_size)
    if holdout_size <= 0:
        return values, []
    if holdout_size >= len(values) - 6:
        raise ValidationError("holdout_size leaves too few training observations")
    return values[:-holdout_size], values[-holdout_size:]


def _select_forecast_series(
    collection: SeriesCollection,
    target_metrics: list[str] | None,
    excluded_metrics: set[str],
) -> list[TimeSeriesItem]:
    requested = {metric for metric in (target_metrics or []) if metric}
    overlap = requested & excluded_metrics
    if overlap:
        raise ValidationError(
            "target_metrics cannot also be covariate metrics",
            metrics=sorted(overlap),
        )
    selected = [
        item
        for item in collection.series
        if (not requested or item.metric in requested)
        and (requested or item.metric not in excluded_metrics)
    ]
    if not selected:
        raise ValidationError("no series matched target_metrics", target_metrics=target_metrics)
    return selected


def _validate_sarima_history(values: list[float], seasonal_order: tuple[int, int, int, int]) -> None:
    P, D, Q, period = seasonal_order
    if P + D + Q > 0 and len(values) < 2 * period:
        raise ValidationError(
            "seasonal ARIMA requires at least two full seasonal periods",
            n_obs=len(values),
            seasonal_order=list(seasonal_order),
        )


def _validate_ets_history(values: list[float], seasonal: str | None, period: int | None) -> None:
    if seasonal and period and len(values) < period * 2:
        raise ValidationError(
            "seasonal smoothing requires at least two full seasons",
            n_obs=len(values),
            seasonal_periods=period,
        )


def _select_sarima_spec(
    values: list[float],
    *,
    fixed_seasonal_order: tuple[int, int, int, int] | None,
    seasonal_period: int | None,
    exog: Any | None,
    max_p: int,
    max_d: int,
    max_q: int,
    max_P: int,
    max_D: int,
    max_Q: int,
    information_criterion: str,
) -> tuple[tuple[int, int, int], tuple[int, int, int, int]]:
    order_candidates = list(
        product(range(max_p + 1), range(max_d + 1), range(max_q + 1))
    )
    if fixed_seasonal_order is not None:
        _validate_sarima_history(values, fixed_seasonal_order)
        return _best_sarima_candidate(
            values, order_candidates, [fixed_seasonal_order], information_criterion, exog=exog
        )

    selected_order, _no_seasonality = _best_sarima_candidate(
        values, order_candidates, [(0, 0, 0, 0)], information_criterion, exog=exog
    )
    if seasonal_period is None:
        return selected_order, (0, 0, 0, 0)

    seasonal_candidates = [
        (P, D, Q, seasonal_period)
        for P, D, Q in product(range(max_P + 1), range(max_D + 1), range(max_Q + 1))
    ]
    return _best_sarima_candidate(
        values, [selected_order], seasonal_candidates, information_criterion, exog=exog
    )


def _best_sarima_candidate(
    values: list[float],
    order_candidates: list[tuple[int, int, int]],
    seasonal_candidates: list[tuple[int, int, int, int]],
    information_criterion: str,
    *,
    exog: Any | None,
) -> tuple[tuple[int, int, int], tuple[int, int, int, int]]:
    best: tuple[tuple[int, int, int], tuple[int, int, int, int]] | None = None
    best_score = float("inf")
    for candidate_order in order_candidates:
        for candidate_seasonal in seasonal_candidates:
            try:
                fitted, _warnings = _fit_sarima(
                    values,
                    candidate_order,
                    candidate_seasonal,
                    exog=exog,
                    retain_warnings=False,
                )
                score = _optional_finite_float(getattr(fitted, information_criterion))
            except EstimationError:
                continue
            if score is not None and score < best_score:
                best = candidate_order, candidate_seasonal
                best_score = score
    if best is None:
        raise EstimationError(
            "automatic SARIMA order selection did not produce a valid fitted model",
            information_criterion=information_criterion,
        )
    return best


def _fit_sarima(
    values: list[float],
    order: tuple[int, int, int],
    seasonal_order: tuple[int, int, int, int],
    *,
    exog: Any | None = None,
    retain_warnings: bool = True,
) -> tuple[Any, list[str]]:
    try:
        with warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter("always")
            fitted = ARIMA(values, exog=exog, order=order, seasonal_order=seasonal_order).fit()
    except Exception as exc:  # noqa: BLE001
        raise EstimationError(
            "SARIMA model estimation failed",
            order=list(order),
            seasonal_order=list(seasonal_order),
            reason=str(exc),
        ) from exc
    if not retain_warnings:
        return fitted, []
    return fitted, _captured_warnings(captured)


def _sarima_forecast(
    fitted: Any, steps: int, confidence_level: float, *, exog: Any | None = None
) -> tuple[list[float], list[float], list[float]]:
    try:
        prediction = fitted.get_forecast(steps=steps, exog=exog)
        forecast = _float_list(prediction.predicted_mean, "SARIMA forecast")
        confidence = np.asarray(
            prediction.conf_int(alpha=1.0 - confidence_level), dtype=float
        )
        lower = _float_list(confidence[:, 0], "SARIMA lower interval")
        upper = _float_list(confidence[:, 1], "SARIMA upper interval")
    except Exception as exc:  # noqa: BLE001
        raise EstimationError("SARIMA forecasting failed", reason=str(exc)) from exc
    return forecast, lower, upper


def _exog_array(rows: list[list[float]]) -> Any | None:
    if not rows:
        return None
    return np.asarray(rows, dtype=float)


def _sarima_parameters(fitted: Any) -> dict[str, float]:
    return {
        str(name): float(value)
        for name, value in zip(fitted.param_names, fitted.params, strict=True)
        if math.isfinite(float(value))
    }


def _fit_exponential_smoothing(
    values: list[float],
    *,
    trend: str | None,
    seasonal: str | None,
    seasonal_periods: int | None,
    damped_trend: bool,
    initialization_method: str,
) -> tuple[Any, list[str]]:
    try:
        with warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter("always")
            fitted = ExponentialSmoothing(
                values,
                trend=trend,
                damped_trend=damped_trend,
                seasonal=seasonal,
                seasonal_periods=seasonal_periods,
                initialization_method=initialization_method,
            ).fit(optimized=True)
    except Exception as exc:  # noqa: BLE001
        raise EstimationError("exponential smoothing estimation failed", reason=str(exc)) from exc
    return fitted, _captured_warnings(captured)


def _ets_intervals(
    fitted: Any,
    steps: int,
    confidence_level: float,
    interval_simulations: int,
    random_seed: int | None,
) -> tuple[list[float], list[float]]:
    try:
        simulated = np.asarray(
            fitted.simulate(
                steps,
                anchor="end",
                repetitions=interval_simulations,
                random_state=random_seed,
            ),
            dtype=float,
        )
        if simulated.ndim == 1:
            simulated = simulated[:, np.newaxis]
        alpha = 1.0 - confidence_level
        lower = np.quantile(simulated, alpha / 2.0, axis=1)
        upper = np.quantile(simulated, 1.0 - alpha / 2.0, axis=1)
    except Exception as exc:  # noqa: BLE001
        raise EstimationError(
            "exponential smoothing interval simulation failed", reason=str(exc)
        ) from exc
    return _float_list(lower, "ETS lower interval"), _float_list(upper, "ETS upper interval")


def _ets_parameters(raw: dict[str, Any]) -> dict[str, float]:
    parameters: dict[str, float] = {}
    for name in (
        "smoothing_level",
        "smoothing_trend",
        "smoothing_seasonal",
        "damping_trend",
        "initial_level",
        "initial_trend",
    ):
        value = raw.get(name)
        if value is not None and math.isfinite(float(value)):
            parameters[name] = float(value)
    initial_seasons = raw.get("initial_seasons")
    if initial_seasons is not None:
        for index, value in enumerate(initial_seasons):
            if math.isfinite(float(value)):
                parameters[f"initial_seasonal.{index}"] = float(value)
    return parameters


def _in_sample_metrics(actual: list[float], fitted: Any) -> ForecastMetrics:
    predicted = [float(value) for value in fitted]
    return ForecastMetrics(
        in_sample_mae=mae(actual, predicted),
        in_sample_rmse=rmse(actual, predicted),
        in_sample_mape=mape(actual, predicted),
    )


def _holdout_metrics(actual: list[float], forecast: list[float]) -> dict[str, float | None]:
    return {
        "holdout_mae": mae(actual, forecast),
        "holdout_rmse": rmse(actual, forecast),
        "holdout_mape": mape(actual, forecast),
    }


def _assign_metrics(metrics: ForecastMetrics, holdout: dict[str, float | None]) -> None:
    for name, value in holdout.items():
        setattr(metrics, name, value)


def _captured_warnings(captured: list[Any]) -> list[str]:
    return _unique_warnings([str(item.message) for item in captured])


def _unique_warnings(messages: list[str]) -> list[str]:
    return list(dict.fromkeys(message for message in messages if message))


def _float_list(values: Any, description: str) -> list[float]:
    result = [float(value) for value in values]
    if not all(math.isfinite(value) for value in result):
        raise EstimationError(f"{description} contains non-finite values")
    return result


def _optional_finite_float(value: Any) -> float | None:
    if value is None:
        return None
    converted = float(value)
    return converted if math.isfinite(converted) else None


def _infer_period(frequency: str) -> int | None:
    freq = frequency.upper()
    if freq == "D":
        return 7
    if freq == "M":
        return 12
    if freq == "Q":
        return 4
    return None

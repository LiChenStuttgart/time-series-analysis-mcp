"""DeSeaTS decomposition and forecasting orchestration."""

from __future__ import annotations

import math
from typing import Any

from time_series_analysis_mcp.config import LIMITS
from time_series_analysis_mcp.errors import EstimationError, ValidationError

from .deseats_bridge import DeseatsBridge
from .schemas import (
    DeseatsDecompositionResult,
    DeseatsDecompositionSeriesResult,
    DeseatsForecastResult,
    DeseatsForecastSeriesResult,
    DeseatsOptions,
    ForecastIntervalBand,
    SeriesCollection,
)
from .utils import next_index_values


_KERNELS = {"uniform", "epanechnikov", "bisquare", "triweight"}
_BOUNDARY_METHODS = {"shorten", "extend"}
_ERROR_MODELS = {"free", "ARMA"}
_INFORMATION_CRITERIA = {"bic", "aic"}
_INTERVAL_METHODS = {"norm", "boot"}
_BOOTSTRAP_METHODS = {"simple", "advanced"}


def decompose_deseats(
    collection: SeriesCollection,
    *,
    seasonal_period: int | None = None,
    order_poly: int = 3,
    kernel_fun: str = "epanechnikov",
    bwidth: float | None = None,
    boundary_method: str = "shorten",
    bwidth_start: float | None = None,
    inflation_rate: str | None = None,
    correction_factor: bool = False,
    autocor: bool = True,
    drop: float | None = None,
    error_model: str = "free",
    nar_lim: list[int] | None = None,
    nma_lim: list[int] | None = None,
    arma_mean: bool = False,
    inf_criterion: str = "bic",
    include_weights: bool = False,
    bridge: DeseatsBridge | None = None,
) -> DeseatsDecompositionResult:
    period = _validate_common(
        collection,
        seasonal_period=seasonal_period,
        order_poly=order_poly,
        kernel_fun=kernel_fun,
        bwidth=bwidth,
        boundary_method=boundary_method,
        bwidth_start=bwidth_start,
        inflation_rate=inflation_rate,
        drop=drop,
        error_model=error_model,
        nar_lim=nar_lim,
        nma_lim=nma_lim,
        inf_criterion=inf_criterion,
    )
    series_payload = _series_payloads(collection, period, order_poly)
    options = _common_options(
        seasonal_period=period,
        order_poly=order_poly,
        kernel_fun=kernel_fun,
        bwidth=bwidth,
        boundary_method=boundary_method,
        bwidth_start=bwidth_start,
        inflation_rate=inflation_rate,
        correction_factor=correction_factor,
        autocor=autocor,
        drop=drop,
        error_model=error_model,
        nar_lim=nar_lim,
        nma_lim=nma_lim,
        arma_mean=arma_mean,
        inf_criterion=inf_criterion,
    )
    raw = (bridge or DeseatsBridge()).run(
        {"operation": "decompose", "options": options, "series": series_payload}
    )
    results = []
    raw_results = _extract_results(raw)
    _validate_result_count(raw_results, collection)
    for item, raw_result in zip(collection.series, raw_results, strict=True):
        _raise_series_error(raw_result)
        selected_bandwidth = _required_float(raw_result, "selected_bandwidth")
        warnings = _string_list(raw_result.get("warnings"))
        if include_weights:
            warnings.append("include_weights was requested but the first DeSeaTS MCP bridge omits weight matrices")
        results.append(
            DeseatsDecompositionSeriesResult(
                series_id=item.series_id,
                n_obs=int(raw_result.get("n_obs", len(item.values))),
                frequency=int(raw_result.get("frequency", period)),
                time_index=list(collection.time_index),
                observations=_float_list(raw_result.get("observations")),
                trend=_float_list(raw_result.get("trend")),
                seasonality=_float_list(raw_result.get("seasonality")),
                fitted=_float_list(raw_result.get("fitted")),
                residuals=_float_list(raw_result.get("residuals")),
                deseasonalized=_float_list(raw_result.get("deseasonalized")),
                detrended=_float_list(raw_result.get("detrended")),
                selected_bandwidth=selected_bandwidth,
                sum_autocov=_optional_float(raw_result.get("sum_autocov")),
                options=_options_model(
                    options,
                    selected_bandwidth=selected_bandwidth,
                    autocor=autocor,
                    bwidth=bwidth,
                ),
                warnings=warnings,
            )
        )
    return DeseatsDecompositionResult(decomposition_results=results)


def forecast_deseats(
    collection: SeriesCollection,
    *,
    seasonal_period: int | None = None,
    forecast_steps: int = 12,
    confidence_levels: list[float] | None = None,
    interval_method: str = "norm",
    bootstrap_method: str = "simple",
    npaths: int = 1000,
    quant_type: int = 8,
    expo: bool = False,
    adjust_bias: bool = True,
    order_poly: int = 3,
    kernel_fun: str = "epanechnikov",
    bwidth: float | None = None,
    boundary_method: str = "shorten",
    bwidth_start: float | None = None,
    inflation_rate: str | None = None,
    correction_factor: bool = False,
    drop: float | None = None,
    error_model: str = "free",
    ar_order: int | None = None,
    ma_order: int | None = None,
    nar_lim: list[int] | None = None,
    nma_lim: list[int] | None = None,
    arma_mean: bool = False,
    inf_criterion: str = "bic",
    seed: int | None = None,
    bridge: DeseatsBridge | None = None,
) -> DeseatsForecastResult:
    if forecast_steps <= 0:
        raise ValidationError("forecast_steps must be positive", forecast_steps=forecast_steps)
    if interval_method not in _INTERVAL_METHODS:
        raise ValidationError("unsupported interval_method", supported=sorted(_INTERVAL_METHODS))
    if bootstrap_method not in _BOOTSTRAP_METHODS:
        raise ValidationError("unsupported bootstrap_method", supported=sorted(_BOOTSTRAP_METHODS))
    if npaths <= 0:
        raise ValidationError("npaths must be positive", npaths=npaths)
    if quant_type <= 0:
        raise ValidationError("quant_type must be positive", quant_type=quant_type)
    if ar_order is not None and ar_order < 0:
        raise ValidationError("ar_order must be non-negative", ar_order=ar_order)
    if ma_order is not None and ma_order < 0:
        raise ValidationError("ma_order must be non-negative", ma_order=ma_order)
    levels = confidence_levels or [0.95]
    if any(level <= 0 or level >= 1 for level in levels):
        raise ValidationError("confidence_levels must be between 0 and 1", confidence_levels=levels)

    period = _validate_common(
        collection,
        seasonal_period=seasonal_period,
        order_poly=order_poly,
        kernel_fun=kernel_fun,
        bwidth=bwidth,
        boundary_method=boundary_method,
        bwidth_start=bwidth_start,
        inflation_rate=inflation_rate,
        drop=drop,
        error_model=error_model,
        nar_lim=nar_lim,
        nma_lim=nma_lim,
        inf_criterion=inf_criterion,
    )
    series_payload = _series_payloads(collection, period, order_poly)
    options = _common_options(
        seasonal_period=period,
        order_poly=order_poly,
        kernel_fun=kernel_fun,
        bwidth=bwidth,
        boundary_method=boundary_method,
        bwidth_start=bwidth_start,
        inflation_rate=inflation_rate,
        correction_factor=correction_factor,
        autocor=True,
        drop=drop,
        error_model=error_model,
        nar_lim=nar_lim,
        nma_lim=nma_lim,
        arma_mean=arma_mean,
        inf_criterion=inf_criterion,
    )
    options.update(
        {
            "forecast_steps": int(forecast_steps),
            "confidence_levels": [float(level) for level in levels],
            "interval_method": interval_method,
            "bootstrap_method": bootstrap_method,
            "npaths": int(npaths),
            "quant_type": int(quant_type),
            "expo": bool(expo),
            "adjust_bias": bool(adjust_bias),
            "ar_order": ar_order,
            "ma_order": ma_order,
            "seed": seed,
        }
    )
    raw = (bridge or DeseatsBridge()).run(
        {"operation": "forecast", "options": options, "series": series_payload}
    )
    forecast_index = next_index_values(collection.time_index, collection.schema_.frequency, forecast_steps)
    results = []
    raw_results = _extract_results(raw)
    _validate_result_count(raw_results, collection)
    for item, raw_result in zip(collection.series, raw_results, strict=True):
        _raise_series_error(raw_result)
        selected_bandwidth = _required_float(raw_result, "selected_bandwidth")
        bands = [
            ForecastIntervalBand(
                confidence_level=float(band["confidence_level"]),
                lower=_float_list(band.get("lower")),
                upper=_float_list(band.get("upper")),
            )
            for band in _ensure_list(raw_result.get("intervals"))
        ]
        first_band = bands[0] if bands else None
        warnings = _string_list(raw_result.get("warnings"))
        warnings.append(
            "DeSeaTS Semi-ARMA forecasts are intended for short horizons because trend and seasonal components are extrapolated"
        )
        results.append(
            DeseatsForecastSeriesResult(
                series_id=item.series_id,
                n_obs=int(raw_result.get("n_obs", len(item.values))),
                frequency=int(raw_result.get("frequency", period)),
                forecast_index=forecast_index,
                forecast=_float_list(raw_result.get("forecast")),
                intervals=bands,
                lower=first_band.lower if first_band else [],
                upper=first_band.upper if first_band else [],
                selected_bandwidth=selected_bandwidth,
                arma_order=[int(value) for value in raw_result.get("arma_order", [])]
                if raw_result.get("arma_order") is not None
                else None,
                decomposition_tail={
                    str(key): _float_list(value)
                    for key, value in (raw_result.get("decomposition_tail") or {}).items()
                },
                options=dict(options),
                fit_warnings=warnings,
            )
        )
    return DeseatsForecastResult(forecast_results=results)


def _validate_common(
    collection: SeriesCollection,
    *,
    seasonal_period: int | None,
    order_poly: int,
    kernel_fun: str,
    bwidth: float | None,
    boundary_method: str,
    bwidth_start: float | None,
    inflation_rate: str | None,
    drop: float | None,
    error_model: str,
    nar_lim: list[int] | None,
    nma_lim: list[int] | None,
    inf_criterion: str,
) -> int:
    if order_poly not in {1, 3}:
        raise ValidationError("order_poly must be 1 or 3", order_poly=order_poly)
    if kernel_fun not in _KERNELS:
        raise ValidationError("unsupported kernel_fun", kernel_fun=kernel_fun, supported=sorted(_KERNELS))
    if boundary_method not in _BOUNDARY_METHODS:
        raise ValidationError("unsupported boundary_method", boundary_method=boundary_method)
    if bwidth is not None and not (0 < bwidth < 0.5):
        raise ValidationError("bwidth must be between 0 and 0.5", bwidth=bwidth)
    if bwidth_start is not None and not (0 < bwidth_start < 0.5):
        raise ValidationError("bwidth_start must be between 0 and 0.5", bwidth_start=bwidth_start)
    if inflation_rate is not None and inflation_rate not in {"optimal", "naive"}:
        raise ValidationError("inflation_rate must be 'optimal' or 'naive'", inflation_rate=inflation_rate)
    if drop is not None and not (0 <= drop <= 0.25):
        raise ValidationError("drop must be between 0 and 0.25", drop=drop)
    if error_model not in _ERROR_MODELS:
        raise ValidationError("unsupported error_model", error_model=error_model, supported=sorted(_ERROR_MODELS))
    if inf_criterion not in _INFORMATION_CRITERIA:
        raise ValidationError(
            "unsupported inf_criterion",
            inf_criterion=inf_criterion,
            supported=sorted(_INFORMATION_CRITERIA),
        )
    _validate_pair(nar_lim, "nar_lim")
    _validate_pair(nma_lim, "nma_lim")
    if len(collection.series) > LIMITS.deseats_max_series:
        raise ValidationError(
            "DeSeaTS series limit exceeded",
            series=len(collection.series),
            max_series=LIMITS.deseats_max_series,
        )
    return _infer_period(collection.schema_.frequency, seasonal_period)


def _validate_pair(value: list[int] | None, name: str) -> None:
    if value is None:
        return
    if len(value) != 2:
        raise ValidationError(f"{name} must contain two values", **{name: value})
    if value[0] < 0 or value[1] < 0 or value[0] > value[1]:
        raise ValidationError(f"{name} must be a non-negative ascending range", **{name: value})


def _infer_period(frequency: str, explicit: int | None) -> int:
    if explicit is not None:
        if explicit <= 0:
            raise ValidationError("seasonal_period must be positive", seasonal_period=explicit)
        return int(explicit)
    mapping = {"D": 7, "M": 12, "Q": 4, "Y": 1}
    freq = frequency.upper()
    if freq not in mapping:
        raise ValidationError("seasonal_period is required for this frequency", frequency=frequency)
    return mapping[freq]


def _series_payloads(collection: SeriesCollection, period: int, order_poly: int) -> list[dict[str, Any]]:
    out = []
    for item in collection.series:
        values = []
        if len(item.values) > LIMITS.deseats_max_obs:
            raise ValidationError(
                "DeSeaTS observation limit exceeded",
                series_id=item.series_id,
                observations=len(item.values),
                max_observations=LIMITS.deseats_max_obs,
            )
        for value in item.values:
            if value is None:
                raise ValidationError(
                    "DeSeaTS requires complete series; run time_series_missing_data_handler first",
                    series_id=item.series_id,
                )
            value_float = float(value)
            if not math.isfinite(value_float):
                raise ValidationError("DeSeaTS values must be finite", series_id=item.series_id)
            values.append(value_float)
        min_obs = max(8, 2 * period if period > 1 else order_poly + 6)
        if len(values) < min_obs:
            raise ValidationError(
                "DeSeaTS requires more observations for the requested seasonal period",
                series_id=item.series_id,
                observations=len(values),
                minimum=min_obs,
                seasonal_period=period,
            )
        out.append({"series_id": item.series_id, "values": values, "time_index": collection.time_index})
    return out


def _common_options(
    *,
    seasonal_period: int,
    order_poly: int,
    kernel_fun: str,
    bwidth: float | None,
    boundary_method: str,
    bwidth_start: float | None,
    inflation_rate: str | None,
    correction_factor: bool,
    autocor: bool,
    drop: float | None,
    error_model: str,
    nar_lim: list[int] | None,
    nma_lim: list[int] | None,
    arma_mean: bool,
    inf_criterion: str,
) -> dict[str, Any]:
    return {
        "seasonal_period": int(seasonal_period),
        "order_poly": int(order_poly),
        "kernel_fun": kernel_fun,
        "bwidth": bwidth,
        "boundary_method": boundary_method,
        "bwidth_start": bwidth_start,
        "inflation_rate": inflation_rate,
        "correction_factor": bool(correction_factor),
        "autocor": bool(autocor),
        "drop": drop,
        "error_model": error_model,
        "nar_lim": nar_lim or [0, 3],
        "nma_lim": nma_lim or [0, 3],
        "arma_mean": bool(arma_mean),
        "inf_criterion": inf_criterion,
    }


def _options_model(
    options: dict[str, Any],
    *,
    selected_bandwidth: float,
    autocor: bool | None,
    bwidth: float | None,
) -> DeseatsOptions:
    return DeseatsOptions(
        seasonal_period=int(options["seasonal_period"]),
        order_poly=int(options["order_poly"]),
        kernel_fun=str(options["kernel_fun"]),
        bwidth=bwidth,
        selected_bwidth=selected_bandwidth,
        boundary_method=str(options["boundary_method"]),
        autocor=autocor,
        error_model=str(options["error_model"]),
        inf_criterion=str(options["inf_criterion"]),
    )


def _extract_results(raw: dict[str, Any]) -> list[dict[str, Any]]:
    if not raw.get("ok", False):
        raise EstimationError("DeSeaTS bridge returned an error", raw=raw)
    results = raw.get("results")
    if not isinstance(results, list):
        raise EstimationError("DeSeaTS bridge returned malformed results", raw=raw)
    return results


def _validate_result_count(raw_results: list[dict[str, Any]], collection: SeriesCollection) -> None:
    if len(raw_results) == len(collection.series):
        return
    raise EstimationError(
        "DeSeaTS bridge returned the wrong number of series results",
        expected=len(collection.series),
        actual=len(raw_results),
    )


def _raise_series_error(raw_result: dict[str, Any]) -> None:
    if raw_result.get("ok", True):
        return
    raise EstimationError(
        "DeSeaTS estimation failed for a series",
        series_id=raw_result.get("series_id"),
        error=raw_result.get("error"),
    )


def _required_float(raw: dict[str, Any], key: str) -> float:
    value = _optional_float(raw.get(key))
    if value is None:
        raise EstimationError("DeSeaTS bridge omitted required numeric field", field=key)
    return value


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    parsed = float(value)
    if not math.isfinite(parsed):
        return None
    return parsed


def _float_list(value: Any) -> list[float]:
    if value is None:
        return []
    values = _ensure_list(value)
    return [float(item) for item in values]


def _string_list(value: Any) -> list[str]:
    return [str(item) for item in _ensure_list(value)]


def _ensure_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]

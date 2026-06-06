"""Amazon Chronos-2 zero-shot forecast integration."""

from __future__ import annotations

import math
from functools import cache
from typing import Any

from time_series_analysis_mcp.config import LIMITS
from time_series_analysis_mcp.errors import DependencyError, EstimationError, ValidationError
from time_series_analysis_mcp.model_assets import (
    injected_model_asset,
    resolve_model_asset,
)

from ..covariates import build_covariate_bundles
from ..metrics import mae, mape, rmse
from ..schemas import (
    ForecastMetrics,
    ProbabilisticForecastResult,
    ProbabilisticForecastSeriesResult,
    SeriesCollection,
    TimeSeriesItem,
)
from ..utils import next_index_values

_DEFAULT_QUANTILES = [0.1, 0.5, 0.9]
_MODEL_MAX_CONTEXT = 8192
_MODEL_MAX_PREDICTION_LENGTH = 1024


def forecast_chronos2(
    collection: SeriesCollection,
    *,
    target_metrics: list[str] | None = None,
    forecast_steps: int = 12,
    quantile_levels: list[float] | None = None,
    model_id: str | None = None,
    device: str | None = None,
    context_length: int | None = None,
    holdout_size: int = 0,
    known_future_covariate_metrics: list[str] | None = None,
    past_covariate_metrics: list[str] | None = None,
    future_covariates: dict[str, Any] | None = None,
    calendar_covariates: list[str] | None = None,
    pipeline: Any | None = None,
) -> ProbabilisticForecastResult:
    """Generate Chronos-2 forecasts for each selected series as a target item."""
    levels = _validate_request(
        collection,
        forecast_steps=forecast_steps,
        quantile_levels=quantile_levels,
        context_length=context_length,
        holdout_size=holdout_size,
    )
    excluded_metrics = set(known_future_covariate_metrics or []) | set(past_covariate_metrics or [])
    selected = _select_series(collection, target_metrics, excluded_metrics)
    if len(selected) > LIMITS.chronos2_max_series:
        raise ValidationError(
            "Chronos-2 series limit exceeded",
            series=len(selected),
            max_series=LIMITS.chronos2_max_series,
        )
    covariates = build_covariate_bundles(
        collection,
        selected,
        forecast_steps=forecast_steps,
        known_future_covariate_metrics=known_future_covariate_metrics,
        past_covariate_metrics=past_covariate_metrics,
        future_covariates=future_covariates,
        calendar_covariates=calendar_covariates,
    )

    chosen_model = model_id or LIMITS.chronos2_model_id
    chosen_device = device or LIMITS.model_device
    asset = (
        injected_model_asset(chosen_model)
        if pipeline is not None
        else resolve_model_asset(
            chosen_model,
            weights_dir=LIMITS.model_weights_dir,
            allow_remote=LIMITS.allow_remote_models and not LIMITS.model_offline,
        )
    )
    provider = pipeline or _load_pipeline(
        asset.load_path,
        chosen_device,
        LIMITS.model_cache_dir,
        asset.local_files_only,
    )
    effective_context = _effective_context_length(len(collection.time_index), context_length)
    shared_warnings = _context_warnings(len(collection.time_index), effective_context)
    forecast_index = next_index_values(
        collection.time_index,
        collection.schema_.frequency,
        forecast_steps,
    )
    main_predictions = _predict(
        provider,
        selected,
        collection.time_index,
        prediction_length=forecast_steps,
        quantile_levels=levels,
        context_length=effective_context,
        covariates=covariates,
        future_index=forecast_index,
        history_end=len(collection.time_index),
    )

    holdout_predictions: dict[str, dict[str, Any]] = {}
    if holdout_size:
        training_index = collection.time_index[:-holdout_size]
        training_context = _effective_context_length(len(training_index), context_length)
        holdout_predictions = _predict(
            provider,
            selected,
            training_index,
            prediction_length=holdout_size,
            quantile_levels=levels,
            context_length=training_context,
            values_end=-holdout_size,
            covariates=covariates,
            future_index=collection.time_index[-holdout_size:],
            history_end=len(training_index),
        )

    results: list[ProbabilisticForecastSeriesResult] = []
    for item in selected:
        prediction = main_predictions[item.series_id]
        metrics = ForecastMetrics()
        if holdout_size:
            actual = [float(value) for value in item.values[-holdout_size:] if value is not None]
            holdout = holdout_predictions[item.series_id]["forecast"]
            metrics.holdout_mae = mae(actual, holdout)
            metrics.holdout_rmse = rmse(actual, holdout)
            metrics.holdout_mape = mape(actual, holdout)
        results.append(
            ProbabilisticForecastSeriesResult(
                series_id=item.series_id,
                model=chosen_model,
                forecast_index=forecast_index,
                forecast=prediction["forecast"],
                quantiles=prediction["quantiles"],
                metrics=metrics,
                covariates=covariates[item.series_id].metadata(),
                inference_metadata={
                    "provider": "chronos-forecasting",
                    **asset.inference_metadata(),
                    "device": chosen_device,
                    "prediction_length": forecast_steps,
                    "context_length": effective_context,
                    "quantile_levels": levels,
                    "zero_shot": True,
                    "covariates": covariates[item.series_id].metadata(),
                },
                fit_warnings=list(shared_warnings),
            )
        )
    return ProbabilisticForecastResult(forecast_results=results)


def _validate_request(
    collection: SeriesCollection,
    *,
    forecast_steps: int,
    quantile_levels: list[float] | None,
    context_length: int | None,
    holdout_size: int,
) -> list[float]:
    if forecast_steps <= 0:
        raise ValidationError("forecast_steps must be positive", forecast_steps=forecast_steps)
    max_prediction_length = _max_prediction_length()
    max_context = _max_context()
    if forecast_steps > max_prediction_length:
        raise ValidationError(
            "forecast_steps exceeds the configured Chronos-2 prediction limit",
            forecast_steps=forecast_steps,
            max_prediction_length=max_prediction_length,
        )
    if context_length is not None and context_length <= 0:
        raise ValidationError("context_length must be positive", context_length=context_length)
    if context_length is not None and context_length > max_context:
        raise ValidationError(
            "context_length exceeds the configured Chronos-2 context limit",
            context_length=context_length,
            max_context=max_context,
        )
    if holdout_size < 0:
        raise ValidationError("holdout_size must be non-negative", holdout_size=holdout_size)
    if holdout_size and holdout_size >= len(collection.time_index) - 1:
        raise ValidationError("holdout_size leaves too few context observations", holdout_size=holdout_size)
    try:
        levels = sorted({float(level) for level in (quantile_levels or _DEFAULT_QUANTILES)})
    except (TypeError, ValueError) as exc:
        raise ValidationError("quantile_levels must contain numeric values") from exc
    if not levels or any(not math.isfinite(level) or level <= 0 or level >= 1 for level in levels):
        raise ValidationError("quantile_levels must contain values strictly between 0 and 1")
    if 0.5 not in levels:
        raise ValidationError("quantile_levels must include 0.5 for the point forecast")
    return [float(level) for level in levels]


def _select_series(
    collection: SeriesCollection,
    target_metrics: list[str] | None,
    excluded_metrics: set[str] | None = None,
) -> list[TimeSeriesItem]:
    requested = set(target_metrics or [])
    excluded = excluded_metrics or set()
    overlap = requested & excluded
    if overlap:
        raise ValidationError(
            "target_metrics cannot also be covariate metrics",
            metrics=sorted(overlap),
        )
    selected = [
        item
        for item in collection.series
        if (not requested or item.metric in requested)
        and (requested or item.metric not in excluded)
    ]
    if not selected:
        raise ValidationError("no series matched target_metrics", target_metrics=target_metrics)
    for item in selected:
        if any(value is None for value in item.values):
            raise ValidationError(
                "Chronos-2 integration requires complete target series; run missing-data handler first",
                series_id=item.series_id,
            )
    return selected


def _effective_context_length(observations: int, requested: int | None) -> int:
    return min(observations, requested or _max_context())


def _max_context() -> int:
    return min(max(1, LIMITS.chronos2_max_context), _MODEL_MAX_CONTEXT)


def _max_prediction_length() -> int:
    return min(max(1, LIMITS.chronos2_max_prediction_length), _MODEL_MAX_PREDICTION_LENGTH)


def _context_warnings(observations: int, context_length: int) -> list[str]:
    if observations > context_length:
        return [f"input history truncated to the most recent {context_length} observations"]
    return []


def _predict(
    pipeline: Any,
    series: list[TimeSeriesItem],
    time_index: list[str],
    *,
    prediction_length: int,
    quantile_levels: list[float],
    context_length: int,
    values_end: int | None = None,
    covariates: dict[str, Any] | None = None,
    future_index: list[str] | None = None,
    history_end: int | None = None,
) -> dict[str, dict[str, Any]]:
    context_df = _context_frame(
        series,
        time_index,
        context_length,
        values_end=values_end,
        covariates=covariates,
    )
    future_df = _future_frame(
        series,
        future_index or [],
        prediction_length=prediction_length,
        covariates=covariates,
        history_end=history_end if history_end is not None else len(time_index),
    )
    predict_kwargs = {
        "prediction_length": prediction_length,
        "quantile_levels": quantile_levels,
        "id_column": "item_id",
        "timestamp_column": "timestamp",
        "target": "target",
    }
    if future_df is not None:
        predict_kwargs["future_df"] = future_df
    try:
        prediction_df = pipeline.predict_df(context_df, **predict_kwargs)
    except Exception as exc:  # noqa: BLE001
        raise EstimationError(
            "Chronos-2 inference failed",
            error_type=type(exc).__name__,
            reason=str(exc),
        ) from exc
    return _parse_predictions(prediction_df, series, prediction_length, quantile_levels)


def _context_frame(
    series: list[TimeSeriesItem],
    time_index: list[str],
    context_length: int,
    *,
    values_end: int | None,
    covariates: dict[str, Any] | None,
):
    pd = _import_pandas()
    end = len(time_index)
    start = max(0, end - context_length)
    rows = []
    for item in series:
        values = item.values if values_end is None else item.values[:values_end]
        bundle = covariates.get(item.series_id) if covariates else None
        for idx in range(start, end):
            stamp = time_index[idx]
            value = values[idx]
            row = {
                "item_id": item.series_id,
                "timestamp": pd.Timestamp(stamp),
                "target": float(value),
            }
            if bundle is not None:
                for name in bundle.all_names:
                    row[name] = bundle.history[name][idx]
            rows.append(row)
    return pd.DataFrame(rows)


def _future_frame(
    series: list[TimeSeriesItem],
    future_index: list[str],
    *,
    prediction_length: int,
    covariates: dict[str, Any] | None,
    history_end: int,
):
    if not covariates:
        return None
    rows = []
    pd = _import_pandas()
    for item in series:
        bundle = covariates.get(item.series_id)
        if bundle is None or not bundle.known_future_names:
            continue
        values = bundle.future_values(history_end=history_end, prediction_length=prediction_length)
        for idx, stamp in enumerate(future_index):
            row = {
                "item_id": item.series_id,
                "timestamp": pd.Timestamp(stamp),
            }
            for name in bundle.known_future_names:
                row[name] = values[name][idx]
            rows.append(row)
    if not rows:
        return None
    return pd.DataFrame(rows)


def _parse_predictions(
    prediction_df: Any,
    series: list[TimeSeriesItem],
    prediction_length: int,
    quantile_levels: list[float],
) -> dict[str, dict[str, Any]]:
    columns = set(getattr(prediction_df, "columns", []))
    if "item_id" not in columns:
        raise EstimationError("Chronos-2 response does not include item_id")
    out: dict[str, dict[str, Any]] = {}
    for item in series:
        rows = prediction_df[prediction_df["item_id"] == item.series_id]
        if "timestamp" in columns:
            rows = rows.sort_values("timestamp")
        if len(rows) != prediction_length:
            raise EstimationError(
                "Chronos-2 returned an unexpected number of forecasts",
                series_id=item.series_id,
                expected=prediction_length,
                actual=len(rows),
            )
        quantiles = {
            _level_key(level): _column_values(rows, _quantile_column(rows, level))
            for level in quantile_levels
        }
        point_column = "predictions" if "predictions" in columns else _quantile_column(rows, 0.5)
        out[item.series_id] = {
            "forecast": _column_values(rows, point_column),
            "quantiles": quantiles,
        }
    return out


def _quantile_column(frame: Any, level: float) -> Any:
    key = _level_key(level)
    columns = set(getattr(frame, "columns", []))
    if key in columns:
        return key
    if level in columns:
        return level
    raise EstimationError("Chronos-2 response does not include requested quantile", quantile=level)


def _column_values(frame: Any, column: Any) -> list[float]:
    values = [float(value) for value in frame[column].tolist()]
    if any(not math.isfinite(value) for value in values):
        raise EstimationError("Chronos-2 response contains non-finite forecast values")
    return values


def _level_key(level: float) -> str:
    return f"{level:g}"


def _import_pandas():
    try:
        import pandas as pd  # type: ignore[import-not-found]
    except ImportError as exc:
        raise DependencyError(
            "Chronos-2 forecasting requires pandas; install the chronos2 extra",
            dependency="pandas",
            install_command="pip install -e '.[chronos2]'",
        ) from exc
    return pd


@cache
def _load_pipeline(
    model_path: str,
    device: str,
    cache_dir: str | None,
    local_files_only: bool,
) -> Any:
    try:
        from chronos import Chronos2Pipeline  # type: ignore[import-not-found]
    except ImportError as exc:
        raise DependencyError(
            "Chronos-2 forecasting requires chronos-forecasting; install the chronos2 extra",
            dependency="chronos-forecasting",
            install_command="pip install -e '.[chronos2]'",
        ) from exc
    load_options: dict[str, Any] = {
        "device_map": device,
        "local_files_only": local_files_only,
    }
    if cache_dir:
        load_options["cache_dir"] = cache_dir
    try:
        return Chronos2Pipeline.from_pretrained(model_path, **load_options)
    except Exception as exc:  # noqa: BLE001
        raise DependencyError(
            "could not load the Chronos-2 model",
            model_path=model_path,
            device=device,
            local_files_only=local_files_only,
            error_type=type(exc).__name__,
            reason=str(exc),
        ) from exc

"""AutoGluon TimeSeries model-selection and probabilistic forecast integration."""

from __future__ import annotations

import math
import sys
import tempfile
from contextlib import nullcontext
from typing import Any

from time_series_analysis_mcp.config import LIMITS
from time_series_analysis_mcp.errors import DependencyError, EstimationError, ValidationError

from ..covariates import build_covariate_bundles, covariate_metadata
from ..schemas import (
    AutoMLForecastResult,
    ProbabilisticForecastSeriesResult,
    SeriesCollection,
    TimeSeriesItem,
)
from ..utils import next_index_values

_DEFAULT_QUANTILES = [0.1, 0.5, 0.9]


def forecast_automl(
    collection: SeriesCollection,
    *,
    target_metrics: list[str] | None = None,
    forecast_steps: int = 12,
    quantile_levels: list[float] | None = None,
    presets: str = "medium_quality",
    eval_metric: str = "MASE",
    model_names: list[str] | None = None,
    num_val_windows: int = 1,
    enable_ensemble: bool = True,
    time_limit_seconds: int | None = None,
    model_path: str | None = None,
    known_future_covariate_metrics: list[str] | None = None,
    past_covariate_metrics: list[str] | None = None,
    future_covariates: dict[str, Any] | None = None,
    calendar_covariates: list[str] | None = None,
    predictor_factory: Any | None = None,
    timeseries_dataframe_cls: Any | None = None,
) -> AutoMLForecastResult:
    """Train/evaluate AutoGluon forecasting candidates and forecast selected series."""
    levels = _validate_request(
        collection,
        forecast_steps=forecast_steps,
        quantile_levels=quantile_levels,
        num_val_windows=num_val_windows,
        time_limit_seconds=time_limit_seconds,
        model_path=model_path,
    )
    excluded_metrics = set(known_future_covariate_metrics or []) | set(past_covariate_metrics or [])
    selected = _select_series(collection, target_metrics, excluded_metrics)
    if len(selected) > LIMITS.automl_max_series:
        raise ValidationError(
            "AutoML series limit exceeded",
            series=len(selected),
            max_series=LIMITS.automl_max_series,
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
    known_covariate_names = _known_covariate_names(covariates)
    resolved_limit = (
        LIMITS.automl_time_limit_seconds if time_limit_seconds is None else time_limit_seconds
    )
    if predictor_factory is None or timeseries_dataframe_cls is None:
        predictor_factory, timeseries_dataframe_cls = _load_autogluon()
    data = timeseries_dataframe_cls.from_data_frame(
        _context_frame(selected, collection.time_index, covariates=covariates),
        id_column="item_id",
        timestamp_column="timestamp",
    )
    known_covariates = None
    if known_covariate_names:
        known_covariates = timeseries_dataframe_cls.from_data_frame(
            _future_frame(
                selected,
                next_index_values(collection.time_index, collection.schema_.frequency, forecast_steps),
                prediction_length=forecast_steps,
                covariates=covariates,
                history_end=len(collection.time_index),
            ),
            id_column="item_id",
            timestamp_column="timestamp",
        )
    fit_kwargs: dict[str, Any] = {
        "train_data": data,
        "presets": presets,
        "num_val_windows": num_val_windows,
        "enable_ensemble": enable_ensemble,
        "time_limit": resolved_limit,
    }
    if model_names:
        fit_kwargs["hyperparameters"] = {name: {} for name in model_names}
    temporary_directory = None
    try:
        if model_path is None:
            temporary_directory = tempfile.TemporaryDirectory(prefix="time_series_automl_")
            storage_path = temporary_directory.name
        else:
            storage_path = model_path
        predictor_init_kwargs: dict[str, Any] = {
            "prediction_length": forecast_steps,
            "target": "target",
            "quantile_levels": levels,
            "eval_metric": eval_metric,
            "path": storage_path,
        }
        if known_covariate_names:
            predictor_init_kwargs["known_covariates_names"] = known_covariate_names
        predictor = predictor_factory(**predictor_init_kwargs)
        with _joblib_forecasting_context():
            predictor.fit(**fit_kwargs)
            if known_covariates is not None:
                predictions = predictor.predict(data, known_covariates=known_covariates)
            else:
                predictions = predictor.predict(data)
            leaderboard = _leaderboard_records(predictor.leaderboard(silent=True))
        best_model = _best_model(predictor, leaderboard)
    except Exception as exc:  # noqa: BLE001
        raise EstimationError(
            "AutoGluon forecasting failed",
            error_type=type(exc).__name__,
            reason=str(exc),
        ) from exc
    finally:
        if temporary_directory is not None:
            temporary_directory.cleanup()
    forecast_index = next_index_values(
        collection.time_index,
        collection.schema_.frequency,
        forecast_steps,
    )
    results = [
        _parse_prediction(
            predictions,
            item,
            forecast_index=forecast_index,
            levels=levels,
            best_model=best_model,
            eval_metric=eval_metric,
            covariates=covariates[item.series_id].metadata(),
        )
        for item in selected
    ]
    return AutoMLForecastResult(
        forecast_results=results,
        best_model=best_model,
        leaderboard=leaderboard,
        configuration={
            "provider": "autogluon.timeseries",
            "prediction_length": forecast_steps,
            "quantile_levels": levels,
            "presets": presets,
            "eval_metric": eval_metric,
            "model_names": model_names,
            "num_val_windows": num_val_windows,
            "enable_ensemble": enable_ensemble,
            "time_limit_seconds": resolved_limit,
            "model_path": model_path,
            "covariates": covariate_metadata(covariates),
        },
    )


def _validate_request(
    collection: SeriesCollection,
    *,
    forecast_steps: int,
    quantile_levels: list[float] | None,
    num_val_windows: int,
    time_limit_seconds: int | None,
    model_path: str | None,
) -> list[float]:
    if forecast_steps <= 0:
        raise ValidationError("forecast_steps must be positive", forecast_steps=forecast_steps)
    if num_val_windows <= 0:
        raise ValidationError("num_val_windows must be positive", num_val_windows=num_val_windows)
    if time_limit_seconds is not None and time_limit_seconds <= 0:
        raise ValidationError("time_limit_seconds must be positive", time_limit_seconds=time_limit_seconds)
    if model_path is not None and not model_path.strip():
        raise ValidationError("model_path must not be empty when supplied")
    if len(collection.time_index) <= forecast_steps * num_val_windows:
        raise ValidationError(
            "AutoML validation windows leave too few training observations",
            observations=len(collection.time_index),
            forecast_steps=forecast_steps,
            num_val_windows=num_val_windows,
        )
    try:
        levels = sorted({float(level) for level in (quantile_levels or _DEFAULT_QUANTILES)})
    except (TypeError, ValueError) as exc:
        raise ValidationError("quantile_levels must contain numeric values") from exc
    if not levels or any(not math.isfinite(level) or level <= 0 or level >= 1 for level in levels):
        raise ValidationError("quantile_levels must contain values strictly between 0 and 1")
    return levels


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
                "AutoML integration requires complete target series; run missing-data handler first",
                series_id=item.series_id,
            )
    return selected


def _context_frame(
    series: list[TimeSeriesItem],
    time_index: list[str],
    *,
    covariates: dict[str, Any] | None,
):
    pd = _import_pandas()
    rows = []
    for item in series:
        bundle = covariates.get(item.series_id) if covariates else None
        for idx, (stamp, value) in enumerate(zip(time_index, item.values, strict=True)):
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
    covariates: dict[str, Any],
    history_end: int,
):
    pd = _import_pandas()
    rows = []
    for item in series:
        bundle = covariates[item.series_id]
        if not bundle.known_future_names:
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
    return pd.DataFrame(rows)


def _parse_prediction(
    predictions: Any,
    item: TimeSeriesItem,
    *,
    forecast_index: list[str],
    levels: list[float],
    best_model: str | None,
    eval_metric: str,
    covariates: dict[str, Any],
) -> ProbabilisticForecastSeriesResult:
    try:
        rows = predictions.loc[item.series_id]
        if hasattr(rows, "sort_index"):
            rows = rows.sort_index()
    except Exception as exc:  # noqa: BLE001
        raise EstimationError(
            "AutoGluon response does not include forecasted series",
            series_id=item.series_id,
        ) from exc
    if len(rows) != len(forecast_index):
        raise EstimationError(
            "AutoGluon returned an unexpected number of forecasts",
            series_id=item.series_id,
            expected=len(forecast_index),
            actual=len(rows),
        )
    quantiles = {
        f"{level:g}": _column_values(rows, _quantile_column(rows, level)) for level in levels
    }
    point_column = "mean" if "mean" in set(rows.columns) else _quantile_column(rows, 0.5)
    return ProbabilisticForecastSeriesResult(
        series_id=item.series_id,
        model=best_model or "AutoGluon",
        forecast_index=forecast_index,
        forecast=_column_values(rows, point_column),
        quantiles=quantiles,
        covariates=covariates,
        inference_metadata={
            "provider": "autogluon.timeseries",
            "best_model": best_model,
            "eval_metric": eval_metric,
            "point_forecast_column": str(point_column),
            "covariates": covariates,
        },
    )


def _known_covariate_names(covariates: dict[str, Any]) -> list[str]:
    names: list[str] = []
    for bundle in covariates.values():
        for name in bundle.known_future_names:
            if name not in names:
                names.append(name)
    return names


def _quantile_column(frame: Any, level: float) -> Any:
    key = f"{level:g}"
    columns = set(frame.columns)
    if key in columns:
        return key
    if level in columns:
        return level
    raise EstimationError("AutoGluon response does not include requested quantile", quantile=level)


def _column_values(frame: Any, column: Any) -> list[float]:
    values = [float(value) for value in frame[column].tolist()]
    if any(not math.isfinite(value) for value in values):
        raise EstimationError("AutoGluon response contains non-finite forecast values")
    return values


def _leaderboard_records(frame: Any) -> list[dict[str, Any]]:
    if frame is None:
        return []
    try:
        records = frame.to_dict(orient="records")
    except Exception as exc:  # noqa: BLE001
        raise EstimationError("AutoGluon leaderboard response is not tabular") from exc
    return [{str(key): _clean_value(value) for key, value in record.items()} for record in records]


def _clean_value(value: Any) -> Any:
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _best_model(predictor: Any, leaderboard: list[dict[str, Any]]) -> str | None:
    model_name = getattr(predictor, "model_best", None)
    if model_name:
        return str(model_name)
    if leaderboard:
        for name in ("model", "Model"):
            if name in leaderboard[0]:
                return str(leaderboard[0][name])
    return None


def _joblib_forecasting_context() -> Any:
    try:
        from joblib import parallel_config  # type: ignore[import-not-found]
    except ImportError:
        return nullcontext()
    return parallel_config(backend="threading", n_jobs=1)


def _import_pandas() -> Any:
    try:
        import pandas as pd  # type: ignore[import-not-found]
    except ImportError as exc:
        raise DependencyError(
            "AutoML forecasting requires pandas and AutoGluon; install the automl extra",
            dependency="pandas",
            python_executable=sys.executable,
            install_command=f"{sys.executable} -m pip install -e '.[automl]'",
        ) from exc
    return pd


def _load_autogluon() -> tuple[Any, Any]:
    try:
        from autogluon.timeseries import (  # type: ignore[import-not-found]
            TimeSeriesDataFrame,
            TimeSeriesPredictor,
        )
    except ImportError as exc:
        raise DependencyError(
            "AutoML forecasting requires autogluon.timeseries in the MCP server interpreter; install the automl extra",
            dependency="autogluon.timeseries",
            python_executable=sys.executable,
            install_command=f"{sys.executable} -m pip install -e '.[automl]'",
        ) from exc
    return TimeSeriesPredictor, TimeSeriesDataFrame

"""Datadog Toto 2.0 zero-shot forecast integration."""

from __future__ import annotations

import math
import os
import sys
import tempfile
from collections import defaultdict
from functools import cache
from pathlib import Path
from typing import Any

from time_series_analysis_mcp.config import LIMITS
from time_series_analysis_mcp.errors import DependencyError, EstimationError, ValidationError

from ..metrics import mae, mape, rmse
from ..schemas import (
    ForecastMetrics,
    ProbabilisticForecastResult,
    ProbabilisticForecastSeriesResult,
    SeriesCollection,
    TimeSeriesItem,
)
from ..utils import next_index_values

_FIXED_QUANTILES = tuple(round(index / 10, 1) for index in range(1, 10))
_DEFAULT_QUANTILES = [0.1, 0.5, 0.9]


def forecast_toto2(
    collection: SeriesCollection,
    *,
    target_metrics: list[str] | None = None,
    group_by_dimensions: list[str] | None = None,
    forecast_steps: int = 12,
    quantile_levels: list[float] | None = None,
    model_id: str | None = None,
    device: str | None = None,
    context_length: int | None = None,
    decode_block_size: int | None = None,
    local_files_only: bool | None = None,
    holdout_size: int = 0,
    model: Any | None = None,
    torch_module: Any | None = None,
) -> ProbabilisticForecastResult:
    """Generate Toto 2.0 forecasts, optionally modeling grouped series together."""
    levels = _validate_request(
        collection,
        forecast_steps=forecast_steps,
        quantile_levels=quantile_levels,
        context_length=context_length,
        decode_block_size=decode_block_size,
        holdout_size=holdout_size,
    )
    selected = _select_series(collection, target_metrics)
    if len(selected) > LIMITS.toto2_max_series:
        raise ValidationError(
            "Toto 2.0 series limit exceeded",
            series=len(selected),
            max_series=LIMITS.toto2_max_series,
        )
    groups = _group_series(collection, selected, group_by_dimensions)
    chosen_model = model_id or LIMITS.toto2_model_id
    chosen_device = device or LIMITS.model_device
    offline = LIMITS.model_offline if local_files_only is None else local_files_only
    torch = torch_module or _import_torch()
    device_value = torch.device(chosen_device)
    provider = model or _load_model(
        chosen_model,
        chosen_device,
        LIMITS.model_cache_dir,
        offline,
    )
    effective_context = min(len(collection.time_index), context_length or _max_context())

    predictions: dict[str, dict[str, Any]] = {}
    holdout_predictions: dict[str, dict[str, Any]] = {}
    for group_id, items in groups:
        predictions.update(
            _predict_group(
                provider,
                torch,
                device_value,
                group_id,
                items,
                prediction_length=forecast_steps,
                quantile_levels=levels,
                context_length=effective_context,
                decode_block_size=decode_block_size,
            )
        )
        if holdout_size:
            holdout_context = min(
                len(collection.time_index) - holdout_size,
                context_length or _max_context(),
            )
            holdout_predictions.update(
                _predict_group(
                    provider,
                    torch,
                    device_value,
                    group_id,
                    items,
                    prediction_length=holdout_size,
                    quantile_levels=levels,
                    context_length=holdout_context,
                    decode_block_size=decode_block_size,
                    values_end=-holdout_size,
                )
            )

    forecast_index = next_index_values(
        collection.time_index,
        collection.schema_.frequency,
        forecast_steps,
    )
    results = []
    for item in selected:
        prediction = predictions[item.series_id]
        metrics = ForecastMetrics()
        if holdout_size:
            actual, predicted = _observed_holdout(
                item.values[-holdout_size:],
                holdout_predictions[item.series_id]["forecast"],
            )
            metrics.holdout_mae = mae(actual, predicted)
            metrics.holdout_rmse = rmse(actual, predicted)
            metrics.holdout_mape = mape(actual, predicted)
        results.append(
            ProbabilisticForecastSeriesResult(
                series_id=item.series_id,
                model=chosen_model,
                forecast_index=forecast_index,
                forecast=prediction["forecast"],
                quantiles=prediction["quantiles"],
                metrics=metrics,
                inference_metadata={
                    "provider": "toto2",
                    "model_id": chosen_model,
                    "device": chosen_device,
                    "prediction_length": forecast_steps,
                    "context_length": effective_context,
                    "quantile_levels": levels,
                    "decode_block_size": decode_block_size,
                    "local_files_only": offline,
                    "group_id": prediction["group_id"],
                    "group_variates": prediction["group_variates"],
                    "zero_shot": True,
                },
                fit_warnings=prediction["warnings"],
            )
        )
    return ProbabilisticForecastResult(forecast_results=results)


def _validate_request(
    collection: SeriesCollection,
    *,
    forecast_steps: int,
    quantile_levels: list[float] | None,
    context_length: int | None,
    decode_block_size: int | None,
    holdout_size: int,
) -> list[float]:
    if forecast_steps <= 0:
        raise ValidationError("forecast_steps must be positive", forecast_steps=forecast_steps)
    if forecast_steps > LIMITS.toto2_max_prediction_length:
        raise ValidationError(
            "forecast_steps exceeds the configured Toto 2.0 prediction limit",
            forecast_steps=forecast_steps,
            max_prediction_length=LIMITS.toto2_max_prediction_length,
        )
    if context_length is not None and context_length <= 0:
        raise ValidationError("context_length must be positive", context_length=context_length)
    if context_length is not None and context_length > _max_context():
        raise ValidationError(
            "context_length exceeds the configured Toto 2.0 context limit",
            context_length=context_length,
            max_context=_max_context(),
        )
    if decode_block_size is not None and decode_block_size <= 0:
        raise ValidationError("decode_block_size must be positive or null")
    if holdout_size < 0:
        raise ValidationError("holdout_size must be non-negative", holdout_size=holdout_size)
    if holdout_size and holdout_size >= len(collection.time_index) - 1:
        raise ValidationError("holdout_size leaves too few context observations", holdout_size=holdout_size)
    return _quantile_levels(quantile_levels)


def _quantile_levels(quantile_levels: list[float] | None) -> list[float]:
    try:
        levels = sorted({float(level) for level in (quantile_levels or _DEFAULT_QUANTILES)})
    except (TypeError, ValueError) as exc:
        raise ValidationError("quantile_levels must contain numeric values") from exc
    if not levels or any(not math.isfinite(level) for level in levels):
        raise ValidationError("quantile_levels must contain finite values")
    unsupported = [level for level in levels if not any(abs(level - allowed) < 1e-9 for allowed in _FIXED_QUANTILES)]
    if unsupported:
        raise ValidationError(
            "Toto 2.0 exposes only fixed quantile levels from 0.1 through 0.9",
            unsupported=unsupported,
        )
    if not any(abs(level - 0.5) < 1e-9 for level in levels):
        raise ValidationError("quantile_levels must include 0.5 for the point forecast")
    return [round(level, 1) for level in levels]


def _max_context() -> int:
    return max(1, LIMITS.toto2_max_context)


def _select_series(
    collection: SeriesCollection,
    target_metrics: list[str] | None,
) -> list[TimeSeriesItem]:
    requested = set(target_metrics or [])
    selected = [
        item for item in collection.series if not requested or item.metric in requested
    ]
    if not selected:
        raise ValidationError("no series matched target_metrics", target_metrics=target_metrics)
    return selected


def _group_series(
    collection: SeriesCollection,
    selected: list[TimeSeriesItem],
    group_by_dimensions: list[str] | None,
) -> list[tuple[str, list[TimeSeriesItem]]]:
    if group_by_dimensions is None:
        return [(item.series_id, [item]) for item in selected]
    invalid = sorted(set(group_by_dimensions) - set(collection.schema_.dimension_columns))
    if invalid:
        raise ValidationError(
            "group_by_dimensions contains unknown dimensions",
            invalid_dimensions=invalid,
        )
    grouped: dict[tuple[tuple[str, str], ...], list[TimeSeriesItem]] = defaultdict(list)
    for item in selected:
        key = tuple((dimension, item.dimensions.get(dimension, "")) for dimension in group_by_dimensions)
        grouped[key].append(item)
    out = []
    for key, items in sorted(grouped.items()):
        group_id = "|".join(f"{name}={value}" for name, value in key) if key else "all"
        out.append((group_id, sorted(items, key=lambda item: item.series_id)))
    return out


def _predict_group(
    model: Any,
    torch: Any,
    device: Any,
    group_id: str,
    items: list[TimeSeriesItem],
    *,
    prediction_length: int,
    quantile_levels: list[float],
    context_length: int,
    decode_block_size: int | None,
    values_end: int | None = None,
) -> dict[str, dict[str, Any]]:
    histories = [item.values if values_end is None else item.values[:values_end] for item in items]
    histories = [history[-context_length:] for history in histories]
    patch_size = _patch_size(model)
    padding = (-len(histories[0])) % patch_size
    if decode_block_size is not None and decode_block_size % patch_size:
        raise ValidationError(
            "decode_block_size must be divisible by the Toto 2.0 model patch size",
            decode_block_size=decode_block_size,
            patch_size=patch_size,
        )
    target_values = [
        [0.0] * padding + [0.0 if value is None else float(value) for value in history]
        for history in histories
    ]
    mask_values = [
        [False] * padding + [value is not None for value in history] for history in histories
    ]
    has_missing_observations = any(value is None for history in histories for value in history)
    has_missing = has_missing_observations or padding > 0
    input_payload = {
        "target": torch.tensor([target_values], dtype=torch.float32, device=device),
        "target_mask": torch.tensor([mask_values], dtype=torch.bool, device=device),
        "series_ids": torch.zeros((1, len(items)), dtype=torch.long, device=device),
    }
    try:
        raw_quantiles = model.forecast(
            input_payload,
            horizon=prediction_length,
            decode_block_size=decode_block_size,
            has_missing_values=has_missing,
        )
        output = _to_list(raw_quantiles)
    except Exception as exc:  # noqa: BLE001
        raise EstimationError(
            "Toto 2.0 inference failed",
            group_id=group_id,
            error_type=type(exc).__name__,
            reason=str(exc),
        ) from exc
    results: dict[str, dict[str, Any]] = {}
    for variate, item in enumerate(items):
        try:
            quantiles = {
                f"{level:g}": [float(value) for value in output[_fixed_index(level)][0][variate]]
                for level in quantile_levels
            }
        except (IndexError, TypeError, ValueError) as exc:
            raise EstimationError(
                "Toto 2.0 returned an unexpected forecast tensor shape",
                group_id=group_id,
            ) from exc
        if any(len(values) != prediction_length for values in quantiles.values()):
            raise EstimationError(
                "Toto 2.0 returned an unexpected number of forecasts",
                series_id=item.series_id,
            )
        point = quantiles["0.5"]
        if any(not math.isfinite(value) for values in quantiles.values() for value in values):
            raise EstimationError("Toto 2.0 response contains non-finite forecast values")
        warnings = []
        if has_missing_observations:
            warnings.append("Toto 2.0 used target masks for missing values in the input context")
        if padding:
            warnings.append(
                f"Toto 2.0 left-padded the input context by {padding} masked values to align patch_size={patch_size}"
            )
        results[item.series_id] = {
            "forecast": point,
            "quantiles": quantiles,
            "group_id": group_id,
            "group_variates": len(items),
            "warnings": warnings,
        }
    return results


def _fixed_index(level: float) -> int:
    return min(range(len(_FIXED_QUANTILES)), key=lambda index: abs(_FIXED_QUANTILES[index] - level))


def _observed_holdout(
    actual: list[float | None],
    predicted: list[float],
) -> tuple[list[float], list[float]]:
    pairs = [
        (float(value), prediction)
        for value, prediction in zip(actual, predicted, strict=True)
        if value is not None
    ]
    return [value for value, _prediction in pairs], [prediction for _value, prediction in pairs]


def _to_list(tensor: Any) -> Any:
    value = tensor
    for method_name in ("detach", "cpu"):
        method = getattr(value, method_name, None)
        if method is not None:
            value = method()
    method = getattr(value, "tolist", None)
    return method() if method is not None else value


def _patch_size(model: Any) -> int:
    value = getattr(getattr(model, "config", None), "patch_size", 1)
    try:
        patch_size = int(value)
    except (TypeError, ValueError) as exc:
        raise EstimationError("Toto 2.0 model exposes an invalid patch size") from exc
    if patch_size <= 0:
        raise EstimationError("Toto 2.0 model exposes an invalid patch size")
    return patch_size


def _import_torch() -> Any:
    try:
        import torch  # type: ignore[import-not-found]
    except ImportError as exc:
        raise DependencyError(
            "Toto 2.0 forecasting requires PyTorch and toto2; install the toto2 extra",
            dependency="torch",
            python_executable=sys.executable,
            install_command=f"{sys.executable} -m pip install -e '.[toto2]'",
        ) from exc
    return torch


@cache
def _load_model(
    model_id: str,
    device: str,
    cache_dir: str | None,
    local_files_only: bool,
) -> Any:
    _configure_matplotlib_cache()
    torch = _import_torch()
    try:
        from toto2 import Toto2Model  # type: ignore[import-not-found]
    except ImportError as exc:
        raise DependencyError(
            "Toto 2.0 forecasting requires Datadog's toto2 package; install the toto2 extra",
            dependency="toto2",
            python_executable=sys.executable,
            install_command=f"{sys.executable} -m pip install -e '.[toto2]'",
        ) from exc
    load_options = {"cache_dir": cache_dir, "local_files_only": True}
    try:
        provider = Toto2Model.from_pretrained(model_id, **load_options)
    except Exception as local_exc:  # noqa: BLE001
        if local_files_only:
            raise DependencyError(
                "Toto 2.0 model weights are not available in the configured local cache",
                model_id=model_id,
                cache_dir=cache_dir,
                local_files_only=True,
                reason=str(local_exc),
            ) from local_exc
        try:
            provider = Toto2Model.from_pretrained(
                model_id,
                cache_dir=cache_dir,
                local_files_only=False,
            )
        except Exception as exc:  # noqa: BLE001
            raise DependencyError(
                "could not load the Toto 2.0 model; download weights outside a network-restricted MCP process or set TS_MCP_MODEL_CACHE_DIR",
                model_id=model_id,
                device=device,
                cache_dir=cache_dir,
                local_files_only=False,
                error_type=type(exc).__name__,
                reason=str(exc),
            ) from exc
    try:
        return provider.to(torch.device(device)).eval()
    except Exception as exc:  # noqa: BLE001
        raise DependencyError(
            "could not initialize the Toto 2.0 model on the selected device",
            model_id=model_id,
            device=device,
            error_type=type(exc).__name__,
            reason=str(exc),
        ) from exc


def _configure_matplotlib_cache() -> None:
    if "MPLCONFIGDIR" in os.environ:
        return
    path = Path(tempfile.gettempdir()) / "time-series-analysis-mcp-matplotlib"
    path.mkdir(parents=True, exist_ok=True)
    os.environ["MPLCONFIGDIR"] = str(path)

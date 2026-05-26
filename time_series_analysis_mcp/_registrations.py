"""Central manifest for MCP tool registration."""

from __future__ import annotations

import importlib
import json
import logging
from collections.abc import Callable
from functools import cache
from typing import Any

from .registry import REGISTRY, register

log = logging.getLogger(__name__)


@cache
def _import_module(module_name: str) -> Any:
    try:
        return importlib.import_module(module_name)
    except Exception as exc:  # noqa: BLE001
        log.warning("failed to import %s: %s", module_name, exc)
        return exc


def _resolve(target: str) -> Callable[..., Any]:
    module_name, _, func_name = target.partition(":")
    module = _import_module(module_name)
    if isinstance(module, Exception):
        return _unavailable_stub(target, module)
    return getattr(module, func_name)


def _unavailable_stub(target: str, cause: Exception) -> Callable[..., str]:
    message = f"tool {target!r} unavailable: {type(cause).__name__}: {cause}"

    def stub(*args: Any, **kwargs: Any) -> str:
        return json.dumps(
            {"ok": False, "error": {"code": "tool_unavailable", "message": message}},
            indent=2,
            ensure_ascii=False,
        )

    stub.__name__ = f"unavailable_{target.rpartition(':')[2] or 'tool'}"
    stub.__doc__ = message
    return stub


_MANIFEST: tuple[tuple[str, str, str, str], ...] = (
    (
        "list_tools",
        "server",
        "tools.time_series_adapter:list_tools_adapter",
        "List all MCP tools with descriptions and guidance on when to use each tool.",
    ),
    (
        "time_series_loader",
        "data_preparation",
        "tools.time_series_adapter:loader_adapter",
        "Load structured files and build regular multi-series time-series collections.",
    ),
    (
        "time_series_missing_data_handler",
        "data_quality",
        "tools.time_series_adapter:missing_data_handler_adapter",
        "Detect and handle missing values and calendar gaps in time-series collections.",
    ),
    (
        "time_series_outlier_handler",
        "data_quality",
        "tools.time_series_adapter:outlier_handler_adapter",
        "Detect and handle point outliers in time-series collections.",
    ),
    (
        "time_series_unit_root_test",
        "diagnostics",
        "tools.time_series_adapter:unit_root_test_adapter",
        "Run stationarity and unit-root diagnostics per time series.",
    ),
    (
        "time_series_seasonality_test",
        "diagnostics",
        "tools.time_series_adapter:seasonality_test_adapter",
        "Detect and score seasonality per time series.",
    ),
    (
        "time_series_structural_break_test",
        "diagnostics",
        "tools.time_series_adapter:structural_break_test_adapter",
        "Detect structural breaks and change points per time series.",
    ),
    (
        "time_series_deseats_decompose",
        "decomposition",
        "tools.time_series_adapter:deseats_decompose_adapter",
        "Run DeSeaTS data-driven seasonal decomposition per time series.",
    ),
    (
        "time_series_arima",
        "forecasting",
        "tools.time_series_adapter:arima_adapter",
        "Fit optimized ARIMA or seasonal ARIMA models and forecast selected time series.",
    ),
    (
        "time_series_exponential_smoothing",
        "forecasting",
        "tools.time_series_adapter:exponential_smoothing_adapter",
        "Fit optimized Holt-Winters exponential smoothing models and forecast selected time series.",
    ),
    (
        "time_series_deseats_forecast",
        "forecasting",
        "tools.time_series_adapter:deseats_forecast_adapter",
        "Fit DeSeaTS Seasonal Semi-ARMA models and forecast selected time series.",
    ),
    (
        "time_series_chronos2_forecast",
        "forecasting",
        "tools.time_series_adapter:chronos2_forecast_adapter",
        "Generate zero-shot probabilistic forecasts with the optional Amazon Chronos-2 model.",
    ),
    (
        "time_series_toto2_forecast",
        "forecasting",
        "tools.time_series_adapter:toto2_forecast_adapter",
        "Generate zero-shot probabilistic forecasts with the optional Datadog Toto 2.0 model.",
    ),
    (
        "time_series_automl_forecast",
        "forecasting",
        "tools.time_series_adapter:automl_forecast_adapter",
        "Select and evaluate forecasting models with optional AutoGluon TimeSeries.",
    ),
)


def load_all() -> None:
    existing = set(REGISTRY.tools)
    for name, group, target, description in _MANIFEST:
        if name in existing:
            continue
        register(name, _resolve(target), description, group=group)

"""MCP-facing adapters for the eight time-series tools."""

from __future__ import annotations

from typing import Any

from time_series_analysis.forecasting import forecast_arima, forecast_exponential_smoothing
from time_series_analysis.loading import (
    collection_from_single_series,
    load_time_series_file,
    parse_series_collection,
    save_text,
)
from time_series_analysis.missing_data import handle_missing_data
from time_series_analysis.outliers import handle_outliers
from time_series_analysis.seasonality import run_seasonality_tests
from time_series_analysis.structural_breaks import run_structural_break_tests
from time_series_analysis.unit_root import run_unit_root_tests
from time_series_analysis_mcp.errors import ValidationError

from .output_formatter import render


def loader_adapter(
    file_path: str,
    time_column: str,
    value_columns: list[str] | None = None,
    dimension_columns: list[str] | None = None,
    frequency: str = "month",
    aggregation: str = "sum",
    timezone: str | None = None,
    date_format: str | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
    fill_calendar: bool = True,
    missing_tokens: list[str] | None = None,
    output_format: str = "json",
    save_path: str | None = None,
) -> str:
    result = load_time_series_file(
        file_path,
        time_column=time_column,
        value_columns=value_columns,
        dimension_columns=dimension_columns,
        frequency=frequency,
        aggregation=aggregation,
        timezone=timezone,
        date_format=date_format,
        start_date=start_date,
        end_date=end_date,
        fill_calendar=fill_calendar,
        missing_tokens=missing_tokens,
    )
    return _render_and_save(result, output_format, save_path)


def missing_data_handler_adapter(
    series_collection: dict[str, Any] | None = None,
    data: list[float] | None = None,
    file_path: str | None = None,
    time_column: str | None = None,
    value_columns: list[str] | None = None,
    dimension_columns: list[str] | None = None,
    frequency: str = "month",
    strategy: str = "auto",
    max_interpolation_gap: int = 3,
    seasonal_period: int | None = None,
    preserve_flags: bool = True,
    output_format: str = "json",
    save_path: str | None = None,
) -> str:
    collection = _collection_from_inputs(
        series_collection=series_collection,
        data=data,
        file_path=file_path,
        time_column=time_column,
        value_columns=value_columns,
        dimension_columns=dimension_columns,
        frequency=frequency,
    )
    result = handle_missing_data(
        collection,
        strategy=strategy,
        max_interpolation_gap=max_interpolation_gap,
        seasonal_period=seasonal_period,
        preserve_flags=preserve_flags,
    )
    return _render_and_save(result, output_format, save_path)


def outlier_handler_adapter(
    series_collection: dict[str, Any] | None = None,
    data: list[float] | None = None,
    file_path: str | None = None,
    time_column: str | None = None,
    value_columns: list[str] | None = None,
    dimension_columns: list[str] | None = None,
    frequency: str = "month",
    detection_method: str = "auto",
    handling_method: str = "flag",
    z_threshold: float = 3.5,
    iqr_multiplier: float = 1.5,
    seasonal_period: int | None = None,
    preserve_flags: bool = True,
    output_format: str = "json",
    save_path: str | None = None,
) -> str:
    collection = _collection_from_inputs(
        series_collection=series_collection,
        data=data,
        file_path=file_path,
        time_column=time_column,
        value_columns=value_columns,
        dimension_columns=dimension_columns,
        frequency=frequency,
    )
    result = handle_outliers(
        collection,
        detection_method=detection_method,
        handling_method=handling_method,
        z_threshold=z_threshold,
        iqr_multiplier=iqr_multiplier,
        seasonal_period=seasonal_period,
        preserve_flags=preserve_flags,
    )
    return _render_and_save(result, output_format, save_path)


def unit_root_test_adapter(
    series_collection: dict[str, Any] | None = None,
    data: list[float] | None = None,
    file_path: str | None = None,
    time_column: str | None = None,
    value_columns: list[str] | None = None,
    dimension_columns: list[str] | None = None,
    frequency: str = "month",
    tests: list[str] | None = None,
    regression: str = "c",
    max_lags: int | None = None,
    alpha: float = 0.05,
    difference_recommendation: bool = True,
    output_format: str = "json",
    save_path: str | None = None,
) -> str:
    collection = _collection_from_inputs(
        series_collection=series_collection,
        data=data,
        file_path=file_path,
        time_column=time_column,
        value_columns=value_columns,
        dimension_columns=dimension_columns,
        frequency=frequency,
    )
    result = run_unit_root_tests(
        collection,
        tests=tests,
        regression=regression,
        max_lags=max_lags,
        alpha=alpha,
        difference_recommendation=difference_recommendation,
    )
    return _render_and_save(result, output_format, save_path)


def seasonality_test_adapter(
    series_collection: dict[str, Any] | None = None,
    data: list[float] | None = None,
    file_path: str | None = None,
    time_column: str | None = None,
    value_columns: list[str] | None = None,
    dimension_columns: list[str] | None = None,
    frequency: str = "month",
    seasonal_period: int | None = None,
    methods: list[str] | None = None,
    alpha: float = 0.05,
    output_format: str = "json",
    save_path: str | None = None,
) -> str:
    collection = _collection_from_inputs(
        series_collection=series_collection,
        data=data,
        file_path=file_path,
        time_column=time_column,
        value_columns=value_columns,
        dimension_columns=dimension_columns,
        frequency=frequency,
    )
    result = run_seasonality_tests(
        collection,
        seasonal_period=seasonal_period,
        methods=methods,
        alpha=alpha,
    )
    return _render_and_save(result, output_format, save_path)


def structural_break_test_adapter(
    series_collection: dict[str, Any] | None = None,
    data: list[float] | None = None,
    file_path: str | None = None,
    time_column: str | None = None,
    value_columns: list[str] | None = None,
    dimension_columns: list[str] | None = None,
    frequency: str = "month",
    test_type: str = "auto",
    break_point: int | str | None = None,
    max_breaks: int = 5,
    trim_fraction: float = 0.15,
    alpha: float = 0.05,
    output_format: str = "json",
    save_path: str | None = None,
) -> str:
    collection = _collection_from_inputs(
        series_collection=series_collection,
        data=data,
        file_path=file_path,
        time_column=time_column,
        value_columns=value_columns,
        dimension_columns=dimension_columns,
        frequency=frequency,
    )
    result = run_structural_break_tests(
        collection,
        test_type=test_type,
        break_point=break_point,
        max_breaks=max_breaks,
        trim_fraction=trim_fraction,
        alpha=alpha,
    )
    return _render_and_save(result, output_format, save_path)


def arima_adapter(
    series_collection: dict[str, Any] | None = None,
    data: list[float] | None = None,
    file_path: str | None = None,
    time_column: str | None = None,
    value_columns: list[str] | None = None,
    dimension_columns: list[str] | None = None,
    frequency: str = "month",
    order: list[int] | None = None,
    seasonal_order: list[int] | None = None,
    auto_order: bool = False,
    max_p: int = 3,
    max_d: int = 2,
    max_q: int = 3,
    forecast_steps: int = 12,
    confidence_level: float = 0.95,
    holdout_size: int = 0,
    output_format: str = "json",
    save_path: str | None = None,
) -> str:
    collection = _collection_from_inputs(
        series_collection=series_collection,
        data=data,
        file_path=file_path,
        time_column=time_column,
        value_columns=value_columns,
        dimension_columns=dimension_columns,
        frequency=frequency,
    )
    result = forecast_arima(
        collection,
        order=order,
        seasonal_order=seasonal_order,
        auto_order=auto_order,
        max_p=max_p,
        max_d=max_d,
        max_q=max_q,
        forecast_steps=forecast_steps,
        confidence_level=confidence_level,
        holdout_size=holdout_size,
    )
    return _render_and_save(result, output_format, save_path)


def exponential_smoothing_adapter(
    series_collection: dict[str, Any] | None = None,
    data: list[float] | None = None,
    file_path: str | None = None,
    time_column: str | None = None,
    value_columns: list[str] | None = None,
    dimension_columns: list[str] | None = None,
    frequency: str = "month",
    trend: str | None = "add",
    seasonal: str | None = None,
    seasonal_periods: int | None = None,
    damped_trend: bool = False,
    initialization_method: str = "estimated",
    forecast_steps: int = 12,
    confidence_level: float = 0.95,
    holdout_size: int = 0,
    output_format: str = "json",
    save_path: str | None = None,
) -> str:
    collection = _collection_from_inputs(
        series_collection=series_collection,
        data=data,
        file_path=file_path,
        time_column=time_column,
        value_columns=value_columns,
        dimension_columns=dimension_columns,
        frequency=frequency,
    )
    result = forecast_exponential_smoothing(
        collection,
        trend=trend,
        seasonal=seasonal,
        seasonal_periods=seasonal_periods,
        damped_trend=damped_trend,
        initialization_method=initialization_method,
        forecast_steps=forecast_steps,
        confidence_level=confidence_level,
        holdout_size=holdout_size,
    )
    return _render_and_save(result, output_format, save_path)


def _collection_from_inputs(
    *,
    series_collection: dict[str, Any] | None = None,
    data: list[float] | None = None,
    file_path: str | None = None,
    time_column: str | None = None,
    value_columns: list[str] | None = None,
    dimension_columns: list[str] | None = None,
    frequency: str = "month",
):
    if series_collection is not None:
        return parse_series_collection(series_collection)
    if data is not None:
        return collection_from_single_series(data, frequency=frequency)
    if file_path is not None:
        if not time_column:
            raise ValidationError("time_column is required when file_path is supplied")
        return load_time_series_file(
            file_path,
            time_column=time_column,
            value_columns=value_columns,
            dimension_columns=dimension_columns,
            frequency=frequency,
        ).dataset
    raise ValidationError("provide series_collection, data, or file_path")


def _render_and_save(result: Any, output_format: str, save_path: str | None) -> str:
    body = render(result, output_format)
    if save_path:
        save_text(body, save_path)
    return body

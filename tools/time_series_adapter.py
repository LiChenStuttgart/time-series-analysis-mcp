"""MCP-facing adapters for the time-series tools."""

from __future__ import annotations

from typing import Any

from time_series_analysis.deseats import decompose_deseats, forecast_deseats
from time_series_analysis.forecasting import forecast_arima, forecast_exponential_smoothing
from time_series_analysis.foundation import forecast_automl, forecast_chronos2, forecast_toto2
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

_WHEN_TO_USE: dict[str, str] = {
    "list_tools": "Use when you need to discover the available MCP tools, compare their purpose, or decide the next step in a time-series workflow.",
    "time_series_loader": "Use first for enterprise tabular data from CSV, TXT, JSON, XLSX, or XLS when you need a regular multi-series SeriesCollection grouped by dimensions.",
    "time_series_missing_data_handler": "Use after loading data and before diagnostics or forecasting when series contain nulls, NaNs, or calendar gaps.",
    "time_series_outlier_handler": "Use after loading or missing-data handling when spikes or anomalous points may distort diagnostics or forecasts.",
    "time_series_unit_root_test": "Use when deciding whether a series is stationary or should be differenced before ARIMA-style modeling.",
    "time_series_seasonality_test": "Use when checking whether daily, monthly, quarterly, or custom-period series have meaningful seasonal structure.",
    "time_series_structural_break_test": "Use when a series may have a level, trend, or variance shift caused by regime changes, policy changes, or business events.",
    "time_series_deseats_decompose": "Use for data-driven local-regression seasonal decomposition with DeSeaTS when you need trend, seasonal, fitted, residual, detrended, or seasonally adjusted components.",
    "time_series_arima": "Use for fitted ARIMA or seasonal ARIMA forecasting when autocorrelation, moving-average effects, and seasonal dependence are central to the modeling problem.",
    "time_series_exponential_smoothing": "Use for optimized Holt-Winters level, trend, and seasonal forecasts where recent observations should receive more weight.",
    "time_series_deseats_forecast": "Use for short-horizon forecasts based on DeSeaTS seasonal decomposition plus ARMA residual modeling.",
    "time_series_chronos2_forecast": "Use for zero-shot probabilistic forecasting with Amazon Chronos-2 when model-backed forecasts and native quantiles are needed.",
    "time_series_toto2_forecast": "Use for zero-shot probabilistic forecasts with Toto 2.0, especially when aligned series should be modeled as multivariate targets.",
    "time_series_automl_forecast": "Use for automated model comparison, validation scoring, ensembles, and model-selected probabilistic forecasts through AutoGluon.",
}


def list_tools_adapter(
    group: str | None = None,
    output_format: str = "json",
    save_path: str | None = None,
) -> str:
    """List registered tools with purpose and usage guidance."""
    from time_series_analysis_mcp._registrations import load_all
    from time_series_analysis_mcp.registry import REGISTRY

    if len(REGISTRY) == 0:
        load_all()

    group_filter = group.lower() if group else None
    tools = []
    for spec in sorted(REGISTRY, key=lambda item: item.name):
        if group_filter and spec.group.lower() != group_filter:
            continue
        tools.append(
            {
                "name": spec.name,
                "group": spec.group,
                "description": spec.description,
                "when_to_use": _WHEN_TO_USE.get(
                    spec.name,
                    "Use when this tool's description matches the current analysis task.",
                ),
            }
        )
    result = {
        "ok": True,
        "tool_count": len(tools),
        "group": group,
        "tools": tools,
    }
    return _render_and_save(result, output_format, save_path)


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


def deseats_decompose_adapter(
    series_collection: dict[str, Any] | None = None,
    data: list[float] | None = None,
    file_path: str | None = None,
    time_column: str | None = None,
    value_columns: list[str] | None = None,
    dimension_columns: list[str] | None = None,
    frequency: str = "month",
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
    result = decompose_deseats(
        collection,
        seasonal_period=seasonal_period,
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
        include_weights=include_weights,
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
    seasonal_period: int | None = None,
    max_seasonal_p: int = 1,
    max_seasonal_d: int = 1,
    max_seasonal_q: int = 1,
    information_criterion: str = "aic",
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
        seasonal_period=seasonal_period,
        max_P=max_seasonal_p,
        max_D=max_seasonal_d,
        max_Q=max_seasonal_q,
        information_criterion=information_criterion,
        forecast_steps=forecast_steps,
        confidence_level=confidence_level,
        holdout_size=holdout_size,
    )
    return _render_and_save(result, output_format, save_path)


def deseats_forecast_adapter(
    series_collection: dict[str, Any] | None = None,
    data: list[float] | None = None,
    file_path: str | None = None,
    time_column: str | None = None,
    value_columns: list[str] | None = None,
    dimension_columns: list[str] | None = None,
    frequency: str = "month",
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
    result = forecast_deseats(
        collection,
        seasonal_period=seasonal_period,
        forecast_steps=forecast_steps,
        confidence_levels=confidence_levels,
        interval_method=interval_method,
        bootstrap_method=bootstrap_method,
        npaths=npaths,
        quant_type=quant_type,
        expo=expo,
        adjust_bias=adjust_bias,
        order_poly=order_poly,
        kernel_fun=kernel_fun,
        bwidth=bwidth,
        boundary_method=boundary_method,
        bwidth_start=bwidth_start,
        inflation_rate=inflation_rate,
        correction_factor=correction_factor,
        drop=drop,
        error_model=error_model,
        ar_order=ar_order,
        ma_order=ma_order,
        nar_lim=nar_lim,
        nma_lim=nma_lim,
        arma_mean=arma_mean,
        inf_criterion=inf_criterion,
        seed=seed,
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
    interval_simulations: int = 1000,
    random_seed: int | None = None,
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
        interval_simulations=interval_simulations,
        random_seed=random_seed,
    )
    return _render_and_save(result, output_format, save_path)


def chronos2_forecast_adapter(
    series_collection: dict[str, Any] | None = None,
    data: list[float] | None = None,
    file_path: str | None = None,
    time_column: str | None = None,
    value_columns: list[str] | None = None,
    dimension_columns: list[str] | None = None,
    frequency: str = "month",
    target_metrics: list[str] | None = None,
    forecast_steps: int = 12,
    quantile_levels: list[float] | None = None,
    model_id: str | None = None,
    device: str | None = None,
    context_length: int | None = None,
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
    result = forecast_chronos2(
        collection,
        target_metrics=target_metrics,
        forecast_steps=forecast_steps,
        quantile_levels=quantile_levels,
        model_id=model_id,
        device=device,
        context_length=context_length,
        holdout_size=holdout_size,
    )
    return _render_and_save(result, output_format, save_path)


def toto2_forecast_adapter(
    series_collection: dict[str, Any] | None = None,
    data: list[float] | None = None,
    file_path: str | None = None,
    time_column: str | None = None,
    value_columns: list[str] | None = None,
    dimension_columns: list[str] | None = None,
    frequency: str = "month",
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
    result = forecast_toto2(
        collection,
        target_metrics=target_metrics,
        group_by_dimensions=group_by_dimensions,
        forecast_steps=forecast_steps,
        quantile_levels=quantile_levels,
        model_id=model_id,
        device=device,
        context_length=context_length,
        decode_block_size=decode_block_size,
        local_files_only=local_files_only,
        holdout_size=holdout_size,
    )
    return _render_and_save(result, output_format, save_path)


def automl_forecast_adapter(
    series_collection: dict[str, Any] | None = None,
    data: list[float] | None = None,
    file_path: str | None = None,
    time_column: str | None = None,
    value_columns: list[str] | None = None,
    dimension_columns: list[str] | None = None,
    frequency: str = "month",
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
    result = forecast_automl(
        collection,
        target_metrics=target_metrics,
        forecast_steps=forecast_steps,
        quantile_levels=quantile_levels,
        presets=presets,
        eval_metric=eval_metric,
        model_names=model_names,
        num_val_windows=num_val_windows,
        enable_ensemble=enable_ensemble,
        time_limit_seconds=time_limit_seconds,
        model_path=model_path,
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

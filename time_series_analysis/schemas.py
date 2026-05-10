"""Canonical Pydantic result and data-contract models."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class SourceInfo(BaseModel):
    file_path: str | None = Field(None, description="Source file path when loaded from disk.")
    file_type: str | None = Field(None, description="Input file extension without the leading dot.")
    loaded_at: str | None = Field(None, description="UTC timestamp when the dataset was loaded.")


class DatasetSchema(BaseModel):
    time_column: str = Field(..., description="Column used as the time axis.")
    value_columns: list[str] = Field(..., description="Metric/value columns converted to series.")
    dimension_columns: list[str] = Field(
        default_factory=list,
        description="Columns whose combinations define individual series.",
    )
    frequency: str = Field(..., description="Normalized frequency code.")
    requested_frequency: str = Field(..., description="Original requested frequency.")
    aggregation: str = Field(..., description="Aggregation used inside each time bucket.")


class SeriesQuality(BaseModel):
    missing_count: int = 0
    missing_rate: float = 0.0
    calendar_gap_count: int = 0
    outlier_count: int = 0


class TimeSeriesItem(BaseModel):
    series_id: str
    metric: str
    dimensions: dict[str, str] = Field(default_factory=dict)
    values: list[float | None]
    quality: SeriesQuality = Field(default_factory=SeriesQuality)
    metadata: dict[str, Any] = Field(default_factory=dict)


class DatasetQuality(BaseModel):
    n_rows_loaded: int = 0
    n_rows_after_cleaning: int = 0
    n_series: int = 0
    n_timestamps: int = 0
    warnings: list[str] = Field(default_factory=list)


class SeriesCollection(BaseModel):
    dataset_id: str | None = None
    source: SourceInfo = Field(default_factory=SourceInfo)
    schema_: DatasetSchema = Field(..., alias="schema")
    time_index: list[str]
    series: list[TimeSeriesItem]
    quality: DatasetQuality = Field(default_factory=DatasetQuality)

    model_config = {"populate_by_name": True}


class ToolWarning(BaseModel):
    code: str
    message: str
    details: dict[str, Any] = Field(default_factory=dict)


class LoaderResult(BaseModel):
    ok: bool = True
    dataset: SeriesCollection


class MissingDataDecision(BaseModel):
    series_id: str
    strategy: str
    reason: str
    missing_before: int
    missing_after: int
    filled_positions: list[int] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class MissingDataReport(BaseModel):
    total_missing_before: int
    total_missing_after: int
    series_decisions: list[MissingDataDecision]


class MissingDataResult(BaseModel):
    ok: bool = True
    dataset: SeriesCollection
    missing_data_report: MissingDataReport


class OutlierDecision(BaseModel):
    series_id: str
    detection_method: str
    handling_method: str
    outlier_positions: list[int] = Field(default_factory=list)
    outlier_timestamps: list[str] = Field(default_factory=list)
    possible_break_positions: list[int] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class OutlierReport(BaseModel):
    total_outliers: int
    series_decisions: list[OutlierDecision]


class OutlierResult(BaseModel):
    ok: bool = True
    dataset: SeriesCollection
    outlier_report: OutlierReport


class SingleTestResult(BaseModel):
    test_name: str
    null_hypothesis: str
    test_statistic: float | None = None
    p_value: float | None = None
    critical_values: dict[str, float] = Field(default_factory=dict)
    lags: int | None = None
    reject_null: bool | None = None
    stationary_interpretation: str
    warnings: list[str] = Field(default_factory=list)


class UnitRootSeriesResult(BaseModel):
    series_id: str
    n_obs: int
    tests: list[SingleTestResult]
    recommendation: str
    warnings: list[str] = Field(default_factory=list)


class UnitRootResult(BaseModel):
    ok: bool = True
    unit_root_results: list[UnitRootSeriesResult]


class SeasonalitySeriesResult(BaseModel):
    series_id: str
    n_obs: int
    seasonal: bool | None
    seasonal_period: int | None
    seasonal_strength: float | None = None
    method_results: dict[str, Any] = Field(default_factory=dict)
    decision: str
    warnings: list[str] = Field(default_factory=list)


class SeasonalityResult(BaseModel):
    ok: bool = True
    seasonality_results: list[SeasonalitySeriesResult]


class StructuralBreakSeriesResult(BaseModel):
    series_id: str
    test_type: str
    n_obs: int
    test_statistic: float | None = None
    p_value: float | None = None
    break_points: list[int] = Field(default_factory=list)
    break_timestamps: list[str] = Field(default_factory=list)
    n_breaks: int = 0
    significant: bool | None = None
    warnings: list[str] = Field(default_factory=list)


class StructuralBreakResult(BaseModel):
    ok: bool = True
    structural_break_results: list[StructuralBreakSeriesResult]


class ForecastMetrics(BaseModel):
    holdout_mae: float | None = None
    holdout_rmse: float | None = None
    holdout_mape: float | None = None
    in_sample_mae: float | None = None
    in_sample_rmse: float | None = None
    in_sample_mape: float | None = None


class ForecastSeriesResult(BaseModel):
    series_id: str
    model: str
    order: list[int] | None = None
    seasonal_order: list[int] | None = None
    parameters: dict[str, float] = Field(default_factory=dict)
    aic: float | None = None
    bic: float | None = None
    forecast_index: list[str]
    forecast: list[float]
    lower: list[float]
    upper: list[float]
    metrics: ForecastMetrics = Field(default_factory=ForecastMetrics)
    fit_warnings: list[str] = Field(default_factory=list)


class ForecastResult(BaseModel):
    ok: bool = True
    forecast_results: list[ForecastSeriesResult]


class DeseatsOptions(BaseModel):
    seasonal_period: int
    order_poly: int
    kernel_fun: str
    bwidth: float | None = None
    selected_bwidth: float | None = None
    boundary_method: str
    autocor: bool | None = None
    error_model: str
    inf_criterion: str


class DeseatsDecompositionSeriesResult(BaseModel):
    series_id: str
    n_obs: int
    frequency: int
    time_index: list[str]
    observations: list[float]
    trend: list[float]
    seasonality: list[float]
    fitted: list[float]
    residuals: list[float]
    deseasonalized: list[float]
    detrended: list[float]
    selected_bandwidth: float
    sum_autocov: float | None = None
    options: DeseatsOptions
    warnings: list[str] = Field(default_factory=list)


class DeseatsDecompositionResult(BaseModel):
    ok: bool = True
    decomposition_results: list[DeseatsDecompositionSeriesResult]


class ForecastIntervalBand(BaseModel):
    confidence_level: float
    lower: list[float]
    upper: list[float]


class DeseatsForecastSeriesResult(BaseModel):
    series_id: str
    model: str = "DeSeaTS-SemiARMA"
    n_obs: int
    frequency: int
    forecast_index: list[str]
    forecast: list[float]
    intervals: list[ForecastIntervalBand] = Field(default_factory=list)
    lower: list[float] = Field(default_factory=list)
    upper: list[float] = Field(default_factory=list)
    selected_bandwidth: float
    arma_order: list[int] | None = None
    decomposition_tail: dict[str, list[float]] = Field(default_factory=dict)
    options: dict[str, Any] = Field(default_factory=dict)
    fit_warnings: list[str] = Field(default_factory=list)


class DeseatsForecastResult(BaseModel):
    ok: bool = True
    forecast_results: list[DeseatsForecastSeriesResult]

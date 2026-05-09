# Time Series Analysis MCP Server Implementation Plan

## 1. Goal

Build a production-oriented Python MCP server for enterprise time-series analysis. The server must expose these MCP tools:

1. `time_series_loader`
2. `time_series_missing_data_handler`
3. `time_series_outlier_handler`
4. `time_series_unit_root_test`
5. `time_series_seasonality_test`
6. `time_series_structural_break_test`
7. `time_series_arima`
8. `time_series_exponential_smoothing`

The server should load multi-dimensional tabular files, construct regular time-series collections per non-time dimension combination, perform quality handling and statistical diagnostics, and forecast each selected series.

This plan is based on a review of the sibling `aigroup-econ-mcp` project. No code in `aigroup-econ-mcp` should be changed.

## 2. Reference Project Review: `aigroup-econ-mcp`

### 2.1 Architecture Pattern To Reuse

The reference project has a clean three-layer architecture:

- `aigroup_econ_mcp/`: MCP server, CLI, tool registry, registration manifest, error handling.
- `tools/`: adapter layer for MCP-facing function signatures, file loading, argument coercion, and output formatting.
- `econometrics/`: pure statistical implementations returning Pydantic result models.

The key implementation choices worth reusing are:

- `server.py` builds a `FastMCP` server, calls the registration loader, and registers every tool from a central registry.
- `registry.py` wraps every handler with a uniform JSON error payload instead of letting Python exceptions leak as transport failures.
- `_registrations.py` is the single manifest for tool names, groups, handler targets, and descriptions.
- Adapter functions expose explicit Python signatures so FastMCP can generate usable input schemas.
- Result objects are Pydantic models, serialized to JSON for MCP responses.
- `guide://...` MCP resources are generated from the live registry to avoid stale documentation.
- Tests cover registry shape, manifest resolution, smoke calls, mathematical correctness, and real MCP stdio protocol.

### 2.2 Reference Project Limitations For This Use Case

The time-series tools in `aigroup-econ-mcp` are useful examples, but the new server should not directly copy their assumptions:

- Existing ARIMA, exponential smoothing, unit-root, and structural-break adapters are mostly univariate.
- The current file loader is regression-centric: first column is `y_data`, remaining columns are `x_data`, or a flat single numeric column.
- There is no canonical multi-series collection schema for product-region-store-account style enterprise panels.
- Missing data support is generic matrix imputation, not time-aware calendar-gap handling.
- There is no dedicated outlier handler.
- Seasonality exists indirectly through decomposition, not as a dedicated seasonality test tool.
- The reference KPSS stationarity boolean is conceptually risky because KPSS has a stationary null hypothesis; the new implementation must report each test's null and decision explicitly.
- Structural-break logic in the reference project was corrected from earlier stubs, but it remains simple and should be expanded for enterprise diagnostics.

### 2.3 Recommended Adaptation

Use the reference project's server/registry/test architecture, but design a new domain model around `SeriesCollection` instead of plain `list[float]`. Every tool should accept either:

- a `series_collection` object returned by `time_series_loader`, or
- direct inline series data for small one-off workflows, or
- a `file_path` plus loader parameters where appropriate.

For enterprise workflows, `time_series_loader` should be the normal first step.

## 3. Proposed Repository Structure

Create this structure in `time-series-analysis-mcp`:

```text
time-series-analysis-mcp/
|-- pyproject.toml
|-- README.md
|-- LICENSE
|-- CHANGELOG.md
|-- time_series_analysis_mcp/
|   |-- __init__.py
|   |-- __main__.py
|   |-- cli.py
|   |-- server.py
|   |-- registry.py
|   |-- _registrations.py
|   |-- errors.py
|   |-- config.py
|   `-- py.typed
|-- tools/
|   |-- __init__.py
|   |-- time_series_adapter.py
|   |-- data_loader.py
|   |-- output_formatter.py
|   `-- validation.py
|-- time_series_analysis/
|   |-- __init__.py
|   |-- schemas.py
|   |-- loading.py
|   |-- missing_data.py
|   |-- outliers.py
|   |-- unit_root.py
|   |-- seasonality.py
|   |-- structural_breaks.py
|   |-- forecasting.py
|   `-- metrics.py
|-- resources/
|   |-- DATA_CONTRACTS.md
|   `-- TOOL_GUIDE.md
|-- docs/
|   |-- ARCHITECTURE.md
|   |-- TESTING.md
|   `-- ENTERPRISE_OPERATIONS.md
`-- tests/
    |-- conftest.py
    |-- test_registry.py
    |-- test_manifest.py
    |-- test_all_tools_smoke.py
    |-- test_correctness.py
    |-- test_loader_contract.py
    |-- test_mcp_protocol.py
    `-- test_startup_perf.py
```

Layering rule:

- `time_series_analysis_mcp/` can import `tools`.
- `tools/` can import `time_series_analysis`.
- `time_series_analysis/` must not import `tools` or `time_series_analysis_mcp`.

## 4. Dependencies

Use Python `>=3.10`.

Core dependencies:

- `mcp>=1.0,<2`
- `pydantic>=2,<3`
- `pandas>=2,<3`
- `numpy>=1.24,<3`
- `scipy>=1.10,<2`
- `statsmodels>=0.14,<1`
- `scikit-learn>=1.2,<2`
- `openpyxl>=3,<4`
- `xlrd>=2,<3` for `.xls` reads
- `click>=8,<9` or `argparse` for CLI

Recommended optional dependencies:

- `arch>=6,<8` for Phillips-Perron unit-root tests.
- `ruptures>=1.1,<2` for structural-break and change-point detection.
- `pyarrow>=14,<20` if Parquet support is later added.

Do not add heavy optional dependencies unless the tool can degrade cleanly when they are absent.

## 5. MCP Server Design

### 5.1 Server

`time_series_analysis_mcp/server.py` should:

- configure UTF-8 stdio on Windows;
- call `load_all()` from `_registrations.py`;
- create `FastMCP("time-series-analysis-mcp")`;
- register every `ToolSpec` from `REGISTRY`;
- expose resources:
  - `guide://time-series-analysis`
  - `guide://data-contracts`
- run with stdio transport in `main()`.

### 5.2 Registry

Mirror the reference project's registry pattern:

- `ToolSpec(name, description, group, handler)`
- `REGISTRY.register(...)`
- `REGISTRY.groups()`
- uniform error wrapper around handlers
- debug traceback controlled by `TIME_SERIES_ANALYSIS_MCP_DEBUG=1`

Error payload:

```json
{
  "ok": false,
  "error": {
    "code": "validation_error",
    "message": "time_column is required",
    "details": {}
  }
}
```

### 5.3 Manifest

Register exactly the eight requested tools initially:

```python
_MANIFEST = (
    ("time_series_loader", "data_preparation", "tools.time_series_adapter:loader_adapter", "Load structured files and build regular multi-series time-series collections."),
    ("time_series_missing_data_handler", "data_quality", "tools.time_series_adapter:missing_data_handler_adapter", "Detect and handle missing values and calendar gaps in time-series collections."),
    ("time_series_outlier_handler", "data_quality", "tools.time_series_adapter:outlier_handler_adapter", "Detect and handle point outliers in time-series collections."),
    ("time_series_unit_root_test", "diagnostics", "tools.time_series_adapter:unit_root_test_adapter", "Run stationarity and unit-root diagnostics per time series."),
    ("time_series_seasonality_test", "diagnostics", "tools.time_series_adapter:seasonality_test_adapter", "Detect and score seasonality per time series."),
    ("time_series_structural_break_test", "diagnostics", "tools.time_series_adapter:structural_break_test_adapter", "Detect structural breaks and change points per time series."),
    ("time_series_arima", "forecasting", "tools.time_series_adapter:arima_adapter", "Fit ARIMA or SARIMAX-style models and forecast selected time series."),
    ("time_series_exponential_smoothing", "forecasting", "tools.time_series_adapter:exponential_smoothing_adapter", "Fit exponential smoothing models and forecast selected time series."),
)
```

Use import-tolerant resolution like the reference project only if optional dependencies are isolated enough to fail per tool.

## 6. Canonical Data Contract

### 6.1 `SeriesCollection`

All tools should internally operate on a canonical `SeriesCollection` Pydantic model. This contract should be documented in `resources/DATA_CONTRACTS.md`.

Recommended JSON shape:

```json
{
  "ok": true,
  "dataset": {
    "dataset_id": "optional-stable-id",
    "source": {
      "file_path": "sales.csv",
      "file_type": "csv",
      "loaded_at": "2026-05-09T00:00:00Z"
    },
    "schema": {
      "time_column": "date",
      "value_columns": ["sales"],
      "dimension_columns": ["product", "region"],
      "frequency": "M",
      "aggregation": "sum"
    },
    "time_index": ["2024-01-31", "2024-02-29", "2024-03-31"],
    "series": [
      {
        "series_id": "sales|product=A|region=US",
        "metric": "sales",
        "dimensions": {
          "product": "A",
          "region": "US"
        },
        "values": [120.0, null, 128.5],
        "quality": {
          "missing_count": 1,
          "missing_rate": 0.3333,
          "calendar_gap_count": 1,
          "outlier_count": 0
        }
      }
    ],
    "quality": {
      "n_rows_loaded": 1000,
      "n_rows_after_cleaning": 1000,
      "n_series": 1,
      "n_timestamps": 3,
      "warnings": []
    }
  }
}
```

Rationale:

- `time_index` is shared by all series, which keeps payloads smaller.
- Each `series` holds a dense list of values aligned to `time_index`.
- Missing values are represented as JSON `null`.
- Each series has explicit dimension metadata.
- The loader can support multiple value columns by creating one series per value column and dimension combination.

### 6.2 Frequency Vocabulary

Support these user-facing values:

- `day`, `daily`, `D`
- `month`, `monthly`, `M`
- `quarter`, `quarterly`, `quartal`, `Q`
- `year`, `annual`, `yearly`, `Y`

Normalize internally to pandas-compatible values:

- `D`
- `M` or `ME` depending on pandas version conventions
- `Q` or `QE`
- `Y` or `YE`

The tool response should include both `requested_frequency` and `normalized_frequency`.

### 6.3 Aggregation Semantics

For tabular input, the loader should:

1. parse the time column;
2. infer or accept value columns;
3. infer or accept dimension columns;
4. group by `dimension_columns + normalized_time_bucket`;
5. aggregate duplicate rows inside each bucket;
6. construct one regular series per `dimension_columns` combination and value column.

Accepted aggregation methods:

- `sum`
- `mean`
- `median`
- `min`
- `max`
- `count`
- `first`
- `last`

Default should be `sum` for enterprise metric data such as revenue, sales, volume, or demand. The user can override this for rates and prices.

## 7. Tool Specifications

### 7.1 `time_series_loader`

Purpose:

Load multi-dimensional structured data from `.txt`, `.csv`, `.xlsx`, `.xls`, or `.json`, aggregate into regular time buckets, and build a canonical `SeriesCollection`.

MCP signature:

```python
def time_series_loader(
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
```

Implementation details:

- Use `pandas.read_csv`, `read_excel`, `read_json`, and a controlled TXT parser.
- For TXT, support delimiter inference via pandas when possible; otherwise accept whitespace-delimited data.
- Validate the file path and extension.
- Reject empty data frames.
- Parse `time_column` with `pd.to_datetime`.
- Apply `date_format` when provided for predictable parsing.
- Drop or report rows with unparsable dates based on a `strict` mode in a later version. Initial default should fail fast if any time values cannot be parsed.
- If `value_columns` is omitted, infer numeric columns excluding the time and dimension columns.
- If `dimension_columns` is omitted, infer non-numeric/non-time columns as dimensions.
- Bucket timestamps to the requested frequency.
- Aggregate duplicate records per bucket and dimension combination.
- Build a complete regular `time_index` when `fill_calendar=True`.
- Preserve missing calendar buckets as `null`.
- Generate stable `series_id` values from metric plus sorted dimension key-value pairs.
- Return row counts, series counts, calendar range, and warnings.

Output:

- JSON `SeriesCollection` by default.
- Markdown summary if `output_format="markdown"`.
- Optional file save through `save_path`, respecting workspace and path safety rules.

Correctness tests:

- CSV with `date,product,region,sales` produces one series per product-region.
- Duplicate rows in the same month aggregate correctly.
- Missing months are inserted as `null`.
- `quartal` normalizes to quarterly frequency.
- Multiple value columns create separate metric series.

### 7.2 `time_series_missing_data_handler`

Purpose:

Detect missing values and missing calendar buckets across all series, choose a handling strategy per series, and apply it.

MCP signature:

```python
def time_series_missing_data_handler(
    series_collection: dict | None = None,
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
```

Input behavior:

- If `series_collection` is supplied, operate on it.
- If `file_path` is supplied, call loader logic internally using the loader parameters.
- If neither is supplied, return a validation error.

Missingness types:

- explicit null/NaN values from input files;
- calendar gaps inserted by regularization;
- leading missing block;
- trailing missing block;
- internal missing block;
- fully missing or near-empty series.

Auto strategy decision:

- no missing values: `none`;
- very short internal gaps: time interpolation;
- short leading/trailing gaps: forward fill/back fill only if allowed and well documented;
- seasonal data with repeated seasonal positions: seasonal median or seasonal interpolation;
- higher missing rate but enough correlated series: iterative imputation on aligned series matrix;
- too sparse: leave unchanged and flag as unusable for modeling.

Supported strategies:

- `none`
- `drop_series`
- `forward_fill`
- `backward_fill`
- `linear_interpolation`
- `time_interpolation`
- `seasonal_median`
- `rolling_median`
- `kalman`
- `iterative`
- `auto`

Implementation details:

- Use pandas interpolation for linear/time methods.
- Use statsmodels state-space/Kalman smoothing if available for `kalman`.
- Use scikit-learn `IterativeImputer` for cross-series imputation if the aligned matrix has enough complete data.
- Never silently overwrite data. Add `imputation_flags` or a parallel mask if `preserve_flags=True`.
- Return decisions per series, before/after missing counts, and warnings.

Output additions:

```json
{
  "missing_data_report": {
    "total_missing_before": 10,
    "total_missing_after": 0,
    "series_decisions": [
      {
        "series_id": "sales|product=A|region=US",
        "strategy": "time_interpolation",
        "reason": "single internal gap",
        "missing_before": 1,
        "missing_after": 0
      }
    ]
  }
}
```

Correctness tests:

- one missing internal point in a linear trend is interpolated.
- leading missing values are not filled unless the selected strategy supports it.
- sparse series is flagged instead of producing misleading values.
- seasonal median uses the same seasonal position.

### 7.3 `time_series_outlier_handler`

Purpose:

Detect outliers in each series, distinguish point outliers from potential level shifts, choose a handling strategy, and apply it if requested.

MCP signature:

```python
def time_series_outlier_handler(
    series_collection: dict | None = None,
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
```

Detection methods:

- `mad`: robust modified z-score using median absolute deviation.
- `iqr`: interquartile range rule.
- `rolling_hampel`: rolling median and MAD, good for local anomalies.
- `stl_residual`: STL decomposition residual outliers for seasonal data.
- `isolation_forest`: optional cross-series anomaly detection.
- `auto`: choose based on length and seasonality.

Handling methods:

- `flag`: do not alter values, only annotate.
- `winsorize`: cap to robust bounds.
- `interpolate`: replace point outliers by local interpolation.
- `seasonal_median`: replace by same seasonal position median.
- `null`: set detected outliers to null so missing-data handler can process them.

Auto decision rules:

- If the series is too short, only run conservative MAD/IQR and flag.
- If seasonal_period is available and there are at least two full seasons, prefer STL residual.
- If consecutive flagged values exceed a configured run length, report a possible structural break instead of treating them as simple outliers.

Output additions:

```json
{
  "outlier_report": {
    "total_outliers": 3,
    "series_decisions": [
      {
        "series_id": "sales|product=A|region=US",
        "detection_method": "rolling_hampel",
        "handling_method": "interpolate",
        "outlier_positions": [12],
        "outlier_timestamps": ["2025-01-31"]
      }
    ]
  }
}
```

Correctness tests:

- injected spike is detected by MAD/Hampel.
- trend-only data does not create false positives.
- consecutive level shift is reported as possible break, not blindly winsorized.

### 7.4 `time_series_unit_root_test`

Purpose:

Run stationarity/unit-root diagnostics across selected series.

MCP signature:

```python
def time_series_unit_root_test(
    series_collection: dict | None = None,
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
```

Supported tests:

- `adf`: Augmented Dickey-Fuller, null is unit root/non-stationary.
- `kpss`: KPSS, null is stationary.
- `pp`: Phillips-Perron, null is unit root/non-stationary. Prefer `arch.unitroot.PhillipsPerron`.

Default `tests` should be `["adf", "kpss"]`; include `pp` when `arch` is installed or return a clear unavailable warning for PP only.

Implementation details:

- Drop or reject missing values according to a `missing_policy` later. Initial implementation should require no missing values and instruct callers to run missing-data handler first.
- For each result, return:
  - test statistic;
  - p-value;
  - critical values;
  - lags used;
  - null hypothesis;
  - decision at `alpha`;
  - stationarity interpretation.
- Report a combined recommendation:
  - stationary;
  - difference once;
  - trend-stationary candidate;
  - inconclusive.

Important KPSS rule:

- KPSS p-value below alpha means reject stationarity. Do not use the same boolean rule as ADF.

Correctness tests:

- stationary AR(1) should be identified as stationary by ADF and mostly not rejected by KPSS.
- random walk should be identified as non-stationary by ADF and rejected by KPSS.
- PP unavailable path is structured and non-fatal when `arch` is absent.

### 7.5 `time_series_seasonality_test`

Purpose:

Determine whether each series has meaningful seasonality and estimate the relevant period strength.

MCP signature:

```python
def time_series_seasonality_test(
    series_collection: dict | None = None,
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
```

Default period inference:

- daily: 7, and optionally 365 for long data;
- monthly: 12;
- quarterly/quartal: 4;
- yearly: no default seasonality period unless explicitly supplied.

Supported methods:

- `acf`: autocorrelation at seasonal lag.
- `stl_strength`: seasonal strength from STL decomposition.
- `seasonal_anova`: ANOVA or Kruskal-Wallis by seasonal position.
- `periodogram`: spectral peak around candidate period.

Default `methods` should be `["acf", "stl_strength", "seasonal_anova"]`.

Decision rule:

- Mark seasonal if at least two independent methods support seasonality, or if STL strength is very high and sample size is adequate.
- Return an `inconclusive` status for short series.

Output:

```json
{
  "seasonality_results": [
    {
      "series_id": "sales|product=A|region=US",
      "seasonal": true,
      "seasonal_period": 12,
      "seasonal_strength": 0.82,
      "method_results": {
        "acf": {"lag": 12, "value": 0.71},
        "stl_strength": {"value": 0.82},
        "seasonal_anova": {"p_value": 0.001}
      }
    }
  ]
}
```

Correctness tests:

- sine-wave seasonal monthly series is detected.
- random noise is not detected as seasonal.
- too-short series returns inconclusive with a warning.

### 7.6 `time_series_structural_break_test`

Purpose:

Detect level, trend, or variance breaks in each series.

MCP signature:

```python
def time_series_structural_break_test(
    series_collection: dict | None = None,
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
```

Supported tests:

- `chow`: known break point, mean/trend regression.
- `sup_f`: Quandt-Andrews style unknown single break scan.
- `bai_perron`: multiple break approximation.
- `ruptures`: change-point detection using `ruptures` if installed.
- `variance`: variance break test using Levene/Brown-Forsythe over candidate splits.
- `auto`: use `chow` if break_point is supplied, otherwise use `sup_f` plus optional `ruptures`.

Implementation details:

- Support integer `break_point` index or timestamp string.
- Use trimming to avoid invalid near-boundary splits.
- Return break positions as both indexes and timestamps.
- Distinguish possible point outliers from persistent breaks.
- Add `fit_warnings` when using fallback methods.

Correctness tests:

- series with a mean shift is detected around the true break.
- no-break AR(1) series does not report significant breaks at alpha.
- timestamp break point is resolved correctly.

### 7.7 `time_series_arima`

Purpose:

Fit ARIMA models and forecast selected series.

MCP signature:

```python
def time_series_arima(
    series_collection: dict | None = None,
    data: list[float] | None = None,
    file_path: str | None = None,
    time_column: str | None = None,
    value_columns: list[str] | None = None,
    dimension_columns: list[str] | None = None,
    frequency: str = "month",
    order: list[int] | tuple[int, int, int] | None = None,
    seasonal_order: list[int] | tuple[int, int, int, int] | None = None,
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
```

Implementation details:

- Use `statsmodels.tsa.arima.model.ARIMA` for non-seasonal models.
- Use `statsmodels.tsa.statespace.SARIMAX` when `seasonal_order` is supplied.
- If `auto_order=True`, implement a conservative AIC grid search over bounded p/d/q values. Avoid bringing in `pmdarima` as a required dependency.
- Validate:
  - no missing values;
  - enough observations for requested order;
  - `forecast_steps > 0`;
  - order elements are non-negative integers.
- Use `fit_warnings` for convergence warnings and fallback choices.
- Return fitted parameters, information criteria, residual diagnostics, forecast, forecast intervals, and holdout metrics when requested.

Output:

```json
{
  "forecast_results": [
    {
      "series_id": "sales|product=A|region=US",
      "model": "ARIMA",
      "order": [1, 1, 1],
      "seasonal_order": null,
      "aic": 123.4,
      "bic": 130.2,
      "forecast_index": ["2026-06-30", "2026-07-31"],
      "forecast": [142.1, 144.0],
      "lower": [130.0, 128.5],
      "upper": [154.2, 159.5],
      "metrics": {
        "holdout_mae": null,
        "holdout_rmse": null,
        "holdout_mape": null
      },
      "fit_warnings": []
    }
  ]
}
```

Correctness tests:

- AR(1) generated series estimates AR coefficient in a reasonable range.
- `forecast_steps=0` is rejected with a validation error.
- `auto_order=True` returns an order within configured bounds.
- missing values are rejected with a clear message.

### 7.8 `time_series_exponential_smoothing`

Purpose:

Fit simple, Holt, or Holt-Winters exponential smoothing models and forecast selected series.

MCP signature:

```python
def time_series_exponential_smoothing(
    series_collection: dict | None = None,
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
```

Implementation details:

- Use `statsmodels.tsa.holtwinters.ExponentialSmoothing`.
- Accept `trend` as `None`, `add`, or `mul`.
- Accept `seasonal` as `None`, `add`, or `mul`.
- Infer `seasonal_periods` from frequency if `seasonal` is supplied and the user does not set it.
- Validate positive data for multiplicative trend/seasonality.
- Compute in-sample metrics: SSE, MSE, RMSE, MAE, MAPE when valid.
- Forecast intervals can be approximated by residual bootstrap in phase 2 if statsmodels does not provide native intervals for this model class.

Correctness tests:

- linear trend series forecasts in the expected direction.
- seasonal synthetic data produces a valid Holt-Winters fit.
- multiplicative model rejects non-positive data.

## 8. Adapter Layer

`tools/time_series_adapter.py` should stay thin:

- parse MCP inputs;
- call loader when `file_path` is supplied;
- convert dict payloads into Pydantic models;
- call the pure algorithm functions;
- serialize result models;
- delegate markdown/text rendering to `OutputFormatter`.

Do not put statistical logic in adapters.

Common adapter helper functions:

- `_load_or_parse_collection(...)`
- `_parse_single_series_or_collection(...)`
- `_serialize_result(...)`
- `_save_if_requested(...)`
- `_normalize_output_format(...)`

## 9. Algorithm Layer

### 9.1 `schemas.py`

Define Pydantic models:

- `SourceInfo`
- `DatasetSchema`
- `SeriesQuality`
- `TimeSeriesItem`
- `DatasetQuality`
- `SeriesCollection`
- `ToolWarning`
- `MissingDataDecision`
- `MissingDataReport`
- `OutlierDecision`
- `OutlierReport`
- `UnitRootTestResult`
- `SeasonalityResult`
- `StructuralBreakResult`
- `ForecastResult`
- `ForecastMetrics`

Use typed fields and descriptive metadata. Keep model names stable because client code may depend on them.

### 9.2 `loading.py`

Responsibilities:

- file validation;
- tabular parsing;
- frequency normalization;
- time bucketing;
- dimension inference;
- value column inference;
- aggregation;
- dense calendar construction;
- series ID generation;
- quality report generation.

### 9.3 `missing_data.py`

Responsibilities:

- missingness scan;
- auto strategy selection;
- imputation execution;
- masks and flags.

### 9.4 `outliers.py`

Responsibilities:

- robust detection methods;
- auto method selection;
- treatment methods;
- avoid confusing outlier runs with structural breaks.

### 9.5 `unit_root.py`

Responsibilities:

- ADF/KPSS/PP wrappers;
- consistent interpretation model;
- combined stationarity recommendation.

### 9.6 `seasonality.py`

Responsibilities:

- period inference;
- ACF method;
- STL strength method;
- seasonal group tests;
- periodogram method;
- combined seasonality decision.

### 9.7 `structural_breaks.py`

Responsibilities:

- Chow known break;
- sup-F unknown break scan;
- multiple break approximation;
- optional `ruptures` integration;
- timestamp/index break conversion.

### 9.8 `forecasting.py`

Responsibilities:

- ARIMA/SARIMAX fitting;
- bounded auto-order search;
- exponential smoothing fitting;
- forecast index generation;
- holdout metrics;
- warnings and convergence metadata.

### 9.9 `metrics.py`

Reusable metrics:

- MAE
- RMSE
- MAPE with zero-safe handling
- sMAPE
- MASE where enough history exists

## 10. Enterprise Requirements

### 10.1 File Safety

Initial local MCP implementation should:

- validate file extensions;
- reject missing files;
- reject directories;
- enforce optional max file size;
- avoid remote URL reads;
- avoid shell execution;
- only write to `save_path` when explicitly requested.

Future enterprise deployment should support an allowlist base directory and audit file access.

### 10.2 Resource Limits

Configurable environment variables:

- `TS_MCP_MAX_FILE_MB`
- `TS_MCP_MAX_ROWS`
- `TS_MCP_MAX_SERIES`
- `TS_MCP_MAX_TIMESTAMPS`
- `TS_MCP_DEBUG`

Behavior:

- fail fast with structured validation errors when limits are exceeded;
- include limit details in the error payload;
- offer guidance to filter dimensions or date ranges.

### 10.3 Determinism

Tools should be deterministic by default:

- fixed random seed where stochastic methods are used;
- return seed in result metadata;
- sort dimensions, timestamps, and series IDs consistently;
- do not rely on system locale.

### 10.4 Observability

Use stderr logging only; stdout is reserved for MCP protocol.

Log:

- server startup;
- tool registration counts;
- tool invocation start/end at debug level;
- validation failures;
- model convergence warnings.

Do not log raw enterprise data values by default.

### 10.5 Governance

Every statistical result should expose:

- assumptions;
- warnings;
- data quality flags;
- sample size used;
- rows or series dropped;
- method selected when `auto` is used.

This is essential for enterprise auditability.

## 11. Testing Plan

Mirror and tighten the reference project test pyramid.

### 11.1 Registry and Manifest Tests

Files:

- `tests/test_registry.py`
- `tests/test_manifest.py`

Assertions:

- exactly 8 initial tools registered;
- tool names unique and snake_case;
- groups are from known set;
- every manifest target resolves;
- every tool has a non-empty description;
- every handler returns structured JSON on bad input.

### 11.2 Smoke Tests

File:

- `tests/test_all_tools_smoke.py`

Add a minimal fixture for every tool:

- loader fixture using temporary CSV;
- missing-data fixture with one null;
- outlier fixture with one spike;
- unit-root fixture with AR(1) data;
- seasonality fixture with synthetic monthly seasonality;
- structural-break fixture with mean shift;
- ARIMA fixture;
- exponential smoothing fixture.

Smoke test passes if the tool returns parseable JSON and does not raise uncaught exceptions.

### 11.3 Correctness Tests

File:

- `tests/test_correctness.py`

Known data-generating processes:

- loader:
  - known product-region monthly aggregation;
  - expected dense time index.
- missing data:
  - linear series with one missing internal value;
  - expected interpolation.
- outliers:
  - one injected spike;
  - expected flagged timestamp.
- unit root:
  - stationary AR(1);
  - random walk.
- seasonality:
  - sine wave with period 12;
  - white noise negative control.
- structural break:
  - mean shift at known index.
- ARIMA:
  - AR(1) process with known coefficient range.
- exponential smoothing:
  - trend plus seasonal data with expected forecast direction.

Use fixed RNG seeds. Assert properties and tolerances, not exact floating-point equality.

### 11.4 Loader Contract Tests

File:

- `tests/test_loader_contract.py`

Cover:

- CSV, JSON, TXT, XLSX, XLS where feasible;
- date format parsing;
- frequency normalization;
- dimension inference;
- value inference;
- aggregation methods;
- missing calendar fill;
- invalid input errors.

### 11.5 MCP Protocol Tests

File:

- `tests/test_mcp_protocol.py`

Spawn:

```bash
python -m time_series_analysis_mcp
```

Validate:

- initialize handshake;
- `tools/list` returns 8 tools with input schemas;
- `tools/call` for loader works over stdio;
- bad input returns structured error.

### 11.6 Startup Performance Tests

File:

- `tests/test_startup_perf.py`

Goals:

- `build_server()` under 3 seconds on warm developer machine;
- subprocess initialize under 10 seconds;
- avoid eager imports of statsmodels/sklearn in server startup where practical.

## 12. Documentation Plan

### 12.1 `README.md`

Include:

- overview;
- tool list;
- installation;
- MCP client configuration;
- quick examples:
  - load CSV;
  - handle missing data;
  - run seasonality test;
  - forecast with ARIMA.

### 12.2 `docs/ARCHITECTURE.md`

Document:

- layers;
- dependency direction;
- adding a new tool;
- error handling;
- result model conventions.

### 12.3 `resources/DATA_CONTRACTS.md`

Document:

- `SeriesCollection` schema;
- frequency vocabulary;
- loader aggregation semantics;
- missing/outlier flags;
- forecast output schema.

### 12.4 Dynamic MCP Guide

`guide://time-series-analysis` should render from `REGISTRY`, including:

- version;
- tool count;
- tools by group;
- input/output conventions;
- error payload.

## 13. Implementation Phases

### Phase 0: Project Scaffold

Deliverables:

- `pyproject.toml`
- package directories;
- CLI entrypoint;
- basic server;
- registry;
- errors;
- empty docs/resources placeholders.

Acceptance:

- `python -m time_series_analysis_mcp --version` works.
- `python -m time_series_analysis_mcp` starts MCP server.
- tests import package.

### Phase 1: Tool Registration and Data Contracts

Deliverables:

- `_registrations.py` with eight tools;
- Pydantic schemas;
- output formatter;
- dynamic guide resource.

Acceptance:

- registry tests pass;
- `tools/list` returns 8 tools with schemas.

### Phase 2: Loader

Deliverables:

- file readers;
- frequency normalization;
- time bucketing;
- multi-dimensional aggregation;
- canonical `SeriesCollection` output.

Acceptance:

- loader contract tests pass;
- loader smoke test passes through MCP call.

### Phase 3: Missing and Outlier Handling

Deliverables:

- missing scanner;
- auto missing strategy;
- imputation methods;
- outlier detection and handling;
- data quality reports.

Acceptance:

- correctness tests for interpolation and outlier injection pass.
- all mutations include flags/reports.

### Phase 4: Diagnostics

Deliverables:

- unit-root tests;
- seasonality tests;
- structural-break tests.

Acceptance:

- stationary/non-stationary tests pass.
- sine-wave seasonality test passes.
- mean-shift structural-break test passes.

### Phase 5: Forecasting

Deliverables:

- ARIMA/SARIMAX wrapper;
- bounded auto-order search;
- exponential smoothing wrapper;
- forecast index generation;
- forecast metrics.

Acceptance:

- ARIMA and exponential smoothing correctness tests pass.
- missing values are rejected with actionable errors.

### Phase 6: Enterprise Hardening

Deliverables:

- resource limits;
- strict file safety;
- debug logging;
- fit warnings;
- docs for operations.

Acceptance:

- startup performance tests pass.
- large/invalid input tests return structured validation errors.

### Phase 7: Packaging and Release Readiness

Deliverables:

- README quick start;
- LICENSE;
- CHANGELOG;
- build configuration;
- PyPI metadata.

Acceptance:

- `python -m build` succeeds.
- `twine check dist/*` succeeds.
- full test suite passes.

## 14. Initial Acceptance Criteria

The first complete release is ready when:

- all eight requested tools are available over MCP;
- loader can read CSV, JSON, TXT, XLSX, and XLS;
- loader builds one regular series per metric and dimension combination;
- missing-data handler reports and handles calendar gaps;
- outlier handler detects an injected spike and can flag or replace it;
- unit-root test correctly distinguishes AR(1) from random walk in tests;
- seasonality test detects period-12 synthetic seasonality;
- structural-break test detects a known mean shift;
- ARIMA and exponential smoothing forecast selected series;
- every tool returns valid JSON on success and structured JSON on failure;
- MCP stdio initialize, tools/list, and tools/call work in tests;
- no code in `aigroup-econ-mcp` is modified.

## 15. Risks and Mitigations

| Risk | Mitigation |
|------|------------|
| Large enterprise files produce huge JSON payloads | Add max rows/series/timestamps, `save_path`, and future artifact references |
| Auto methods make opaque decisions | Return `strategy`, `reason`, assumptions, warnings, and quality metrics per series |
| Missing data and outlier handling distort forecasts | Preserve flags and report all modified points |
| Different statistical tests disagree | Return per-test null hypotheses and a combined recommendation, not a single unexplained boolean |
| Optional dependency missing | Return per-tool or per-method structured unavailable warnings |
| Short series produce unreliable tests | Return `inconclusive` with minimum sample-size warning |
| Multiplicative models receive zero/negative data | Validate and return actionable error |
| MCP startup gets slow | Keep heavy stats imports inside algorithm functions where feasible and test startup time |

## 16. Suggested First Engineering Task

Start with Phase 0 through Phase 2:

1. scaffold the package using the reference project's registry pattern;
2. implement the `SeriesCollection` schema;
3. implement `time_series_loader`;
4. add loader contract tests and MCP protocol smoke tests.

This establishes the data contract that every other tool depends on. Forecasting and diagnostics should not start until the loader contract is stable.

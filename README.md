# time-series-analysis-mcp

`time-series-analysis-mcp` is a Model Context Protocol (MCP) server for
reproducible time-series analysis and forecasting. It exposes a structured
tool catalogue for loading tabular time-series data, repairing data-quality
issues, running diagnostics, decomposing seasonal components, and producing
forecasts with statistical, decomposition-based, foundation-model, and AutoML
methods.

The server was developed as a scientific-paper implementation artifact. This
README is written so it can be used as an appendix describing the implemented
MCP server, its public interface, and representative forecasting tasks.

## Project Metadata

- Public repository: <https://github.com/LiChenStuttgart/time-series-analysis-mcp>
- Author: LiChenStuttgart, 3380836@gmail.com
- Package name: `time-series-analysis-mcp`
- Current package version: `0.1.0`
- License: MIT
- Python requirement: `>=3.10`
- MCP transport: stdio through `mcp.server.fastmcp.FastMCP`

## Overall Summary

The MCP server acts as a bridge between an LLM-capable MCP client and a
time-series analysis library. It registers 14 public tools and 2 MCP resources.
The main workflow is:

1. Discover tools with `list_tools` or the `guide://time-series-analysis`
   resource.
2. Load tabular data into the canonical `SeriesCollection` contract with
   `time_series_loader`.
3. Handle missing values and outliers while preserving audit metadata.
4. Run stationarity, seasonality, or structural-break diagnostics.
5. Decompose series with DeSeaTS when seasonal components are part of the
   analysis.
6. Forecast with ARIMA/SARIMA, Holt-Winters exponential smoothing, DeSeaTS
   Semi-ARMA, Chronos-2, Toto 2.0, or AutoGluon TimeSeries.

Downstream tools accept one of three input styles:

- `series_collection`: the `dataset` object returned by `time_series_loader` or
  by a previous quality-handling tool.
- `data`: a direct list of numeric values, which is converted into a synthetic
  single-series collection.
- `file_path` plus loader arguments such as `time_column`, `value_columns`,
  `dimension_columns`, and `frequency`.

Tool outputs are JSON strings by default. Most tools also accept
`output_format` as `json`, `markdown`, or `text`, and `save_path` to persist a
rendered result.

## MCP Resources

- `guide://time-series-analysis`: live registry-generated tool catalogue,
  grouped by server, data preparation, data quality, diagnostics,
  decomposition, and forecasting.
- `guide://data-contracts`: canonical input and output schema documentation for
  `SeriesCollection`, quality flags, and forecast result objects.

## Tool Catalogue

| Group | Tool | Brief description |
| --- | --- | --- |
| server | `list_tools` | Lists registered MCP tools, groups, descriptions, and usage guidance. |
| data preparation | `time_series_loader` | Loads CSV, TXT, JSON, XLSX, or XLS files and builds regular multi-series `SeriesCollection` objects. |
| data quality | `time_series_missing_data_handler` | Detects and handles nulls, NaNs, and calendar gaps with strategies such as auto, interpolation, seasonal median, rolling median, forward fill, backward fill, or drop series. |
| data quality | `time_series_outlier_handler` | Detects point outliers with MAD, IQR, rolling Hampel, or auto selection, then flags, winsorizes, interpolates, nulls, or seasonally replaces them. |
| diagnostics | `time_series_unit_root_test` | Runs stationarity and unit-root diagnostics and can return differencing recommendations. |
| diagnostics | `time_series_seasonality_test` | Scores and tests seasonal structure for daily, monthly, quarterly, yearly, or explicit seasonal periods. |
| diagnostics | `time_series_structural_break_test` | Detects structural breaks and change points, including possible level or regime shifts. |
| decomposition | `time_series_deseats_decompose` | Calls the R DeSeaTS package to produce trend, seasonality, fitted values, residuals, deseasonalized values, and detrended values. |
| forecasting | `time_series_arima` | Fits optimized ARIMA, SARIMA, or ARIMAX-style models with optional auto order selection, holdout metrics, prediction intervals, and known-future covariates. |
| forecasting | `time_series_exponential_smoothing` | Fits optimized Holt-Winters exponential smoothing models with level, trend, optional seasonality, simulation-based intervals, and holdout metrics. |
| forecasting | `time_series_deseats_forecast` | Calls DeSeaTS Seasonal Semi-ARMA forecasting for short-horizon seasonal forecasts and interval bands. |
| forecasting | `time_series_chronos2_forecast` | Generates zero-shot probabilistic forecasts with optional Amazon Chronos-2 weights, quantiles, covariates, and holdout metrics. |
| forecasting | `time_series_toto2_forecast` | Generates zero-shot probabilistic forecasts with optional Datadog Toto 2.0 weights and can model aligned series together using `group_by_dimensions`. |
| forecasting | `time_series_automl_forecast` | Uses AutoGluon TimeSeries for candidate model fitting, validation scoring, model selection, optional ensembling, leaderboard output, and probabilistic forecasts. |

## Data Contract Notes

`time_series_loader` returns a top-level object with `ok: true` and `dataset`.
The `dataset` is a `SeriesCollection` with:

- `source`: file path, file type, and load timestamp.
- `schema`: time column, value columns, dimension columns, normalized frequency,
  requested frequency, and aggregation.
- `time_index`: regular timestamps generated from the requested frequency.
- `series`: one item per metric and dimension combination.
- `quality`: row counts, series count, timestamp count, and warnings.

Accepted frequency aliases include daily (`day`, `daily`, `D`), monthly
(`month`, `monthly`, `M`), quarterly (`quarter`, `quarterly`, `Q`), and yearly
(`year`, `annual`, `yearly`, `Y`). They are normalized to `D`, `M`, `Q`, or
`Y`.

For wide scientific tables where columns are separate metrics rather than
dimensions, pass `dimension_columns: []` explicitly. If `dimension_columns` is
omitted, the loader infers non-time, non-value columns as dimensions.

Quality handlers preserve audit fields in `series[].metadata`, including
`imputation_flags`, `imputation_strategy`, `outlier_flags`,
`outlier_detection_method`, and `outlier_handling_method`.

## Running The Server

Create or activate a Python environment, then install the package. The
`scientific` extra is recommended for full local data loading and diagnostics:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[scientific]'
python -m time_series_analysis_mcp
```

The console script is also available after installation:

```bash
time-series-analysis-mcp --version
time-series-analysis-mcp
```

Example MCP client configuration:

```json
{
  "mcpServers": {
    "time-series-analysis-mcp": {
      "command": "/Users/lchen/github_phd/time-series-analysis-mcp/.venv/bin/python",
      "args": ["-m", "time_series_analysis_mcp"]
    }
  }
}
```

Use an absolute virtualenv Python path for MCP clients. Most clients launch
servers without activating the user's shell environment, so `python -m
time_series_analysis_mcp` can fail outside this repository if the package is
not installed in that interpreter.

## Optional Runtime Dependencies

Some tools are always registered but require optional runtime dependencies.
If a dependency is missing, the tool returns a structured
`tool_unavailable` or dependency error payload.

### DeSeaTS

`time_series_deseats_decompose` and `time_series_deseats_forecast` call the R
`deseats` package through `Rscript`.

```bash
Rscript -e 'install.packages("jsonlite", repos="https://cloud.r-project.org")'
R CMD INSTALL /path/to/deseats
```

Set `TS_MCP_RSCRIPT` if `Rscript` is not on `PATH`. Other relevant environment
variables include `TS_MCP_R_LIBS_USER`, `TS_MCP_DESEATS_TIMEOUT_SECONDS`,
`TS_MCP_DESEATS_MAX_SERIES`, `TS_MCP_DESEATS_MAX_OBS`, and
`TS_MCP_DESEATS_BOOT_CORES`.

### Chronos-2

`time_series_chronos2_forecast` uses Amazon Chronos-2 for zero-shot point and
quantile forecasts.

```bash
python -m pip install -e '.[chronos2]'
python scripts/manage_modelweights.py download
python scripts/manage_modelweights.py verify
```

The default `amazon/chronos-2` ID resolves to
`modelweights/chronos-2`. Runtime inference does not automatically download
the canonical model. Set `TS_MCP_MODEL_DEVICE` to `cpu`, `cuda`, or another
supported device value. Set `TS_MCP_MODEL_WEIGHTS_DIR` to use a different
local model root.

### Toto 2.0

`time_series_toto2_forecast` uses Datadog Toto 2.0 for zero-shot fixed-quantile
forecasts and multivariate grouped inference.

```bash
python -m pip install -e '.[toto2]'
python scripts/manage_modelweights.py download
python scripts/manage_modelweights.py verify
```

Toto 2.0 requires Python 3.12 or newer. The default
`Datadog/Toto-2.0-22m` ID resolves to `modelweights/toto-2.0-22m`.
Supported quantile outputs are fixed levels from `0.1` through `0.9`, and
`0.5` must be included because it is used as the point forecast.

### AutoML

`time_series_automl_forecast` uses AutoGluon TimeSeries:

```bash
python -m pip install -e '.[automl]'
```

Install this extra into the same Python environment used by the MCP server.
AutoML accepts past-only covariates, known-future covariates, generated
calendar covariates, and optional model lists such as `SeasonalNaive`, `ETS`,
or `Chronos2`. Temporary AutoGluon artifacts are cleaned up unless `model_path`
is supplied.

## Example Forecasting Tasks

The following examples are phrased as user prompts for an MCP client. The JSON
snippets show representative tool calls and abbreviated possible results from
the current implementation. Numeric values can vary across dependency versions,
random seeds, optional model weights, and hardware.

### Example 1: Monthly car registrations with seasonal ARIMA

Prompt:

```text
Load docs/First registration of brand new passenger cars - selected4SAC_columns.csv
as monthly time-series data. The first column is the month, and Belgium,
France, Germany, Greece, and United Kingdom are separate value series. Forecast
Germany for the next 3 months with automatic seasonal ARIMA and report MAPE.
```

Tool calls:

```json
{
  "name": "time_series_loader",
  "arguments": {
    "file_path": "docs/First registration of brand new passenger cars - selected4SAC_columns.csv",
    "time_column": "Country",
    "value_columns": ["Belgium", "France", "Germany", "Greece", "United Kingdom"],
    "dimension_columns": [],
    "frequency": "month"
  }
}
```

```json
{
  "name": "time_series_arima",
  "arguments": {
    "series_collection": "<dataset from time_series_loader>",
    "target_metrics": ["Germany"],
    "auto_order": true,
    "max_p": 2,
    "max_d": 1,
    "max_q": 2,
    "max_seasonal_p": 1,
    "max_seasonal_d": 1,
    "max_seasonal_q": 1,
    "seasonal_period": 12,
    "forecast_steps": 3,
    "holdout_size": 12
  }
}
```

Possible result excerpt:

```json
{
  "ok": true,
  "forecast_results": [
    {
      "series_id": "Germany",
      "model": "ARIMA",
      "order": [0, 1, 2],
      "seasonal_order": [0, 1, 1, 12],
      "forecast_index": ["2025-01-31", "2025-02-28", "2025-03-31"],
      "forecast": [199588.43, 214373.91, 278359.04],
      "metrics": {
        "holdout_mape": 10.5686,
        "in_sample_mape": 8.6666
      }
    }
  ]
}
```

### Example 2: Yearly electric-car stock with ARIMA and Holt-Winters

Prompt:

```text
Load docs/Yearly Global Electric Car Stock.csv. Forecast Total (millions) for
the next 3 years using ARIMA(1,1,1) and Holt-Winters exponential smoothing.
Use a 3-observation holdout and compare the validation MAPE.
```

Tool calls:

```json
{
  "name": "time_series_loader",
  "arguments": {
    "file_path": "docs/Yearly Global Electric Car Stock.csv",
    "time_column": "Year",
    "value_columns": ["Total (millions)"],
    "dimension_columns": [],
    "frequency": "year"
  }
}
```

```json
{
  "name": "time_series_arima",
  "arguments": {
    "series_collection": "<dataset from time_series_loader>",
    "order": [1, 1, 1],
    "forecast_steps": 3,
    "holdout_size": 3
  }
}
```

```json
{
  "name": "time_series_exponential_smoothing",
  "arguments": {
    "series_collection": "<dataset from time_series_loader>",
    "trend": "add",
    "seasonal": null,
    "forecast_steps": 3,
    "holdout_size": 3,
    "random_seed": 42
  }
}
```

Possible result excerpt:

```json
{
  "arima": {
    "series_id": "Total (millions)",
    "forecast_index": ["2026-12-31", "2027-12-31", "2028-12-31"],
    "forecast": [36.2602, 40.9644, 45.6134],
    "holdout_mape": 4.7453
  },
  "holt_winters": {
    "series_id": "Total (millions)",
    "forecast_index": ["2026-12-31", "2027-12-31", "2028-12-31"],
    "forecast": [36.3, 41.1, 45.9],
    "holdout_mape": 4.3264
  }
}
```

### Example 3: Probabilistic zero-shot forecasting with Chronos-2

Prompt:

```text
Load the Rossmann daily sales file, build daily sales series grouped by Store,
repair missing calendar gaps, and forecast the Sales series for the next
14 days with Chronos-2. Return 0.1, 0.5, and 0.9 quantiles, and show the
Store 1 result from the returned forecast list.
```

Tool calls:

```json
{
  "name": "time_series_loader",
  "arguments": {
    "file_path": "docs/Rossmann Store Daily Sales.csv",
    "time_column": "Date",
    "value_columns": ["Sales"],
    "dimension_columns": ["Store"],
    "frequency": "day",
    "aggregation": "sum"
  }
}
```

```json
{
  "name": "time_series_missing_data_handler",
  "arguments": {
    "series_collection": "<dataset from time_series_loader>",
    "strategy": "auto"
  }
}
```

```json
{
  "name": "time_series_chronos2_forecast",
  "arguments": {
    "series_collection": "<dataset from time_series_missing_data_handler>",
    "target_metrics": ["Sales"],
    "forecast_steps": 14,
    "quantile_levels": [0.1, 0.5, 0.9],
    "context_length": 365
  }
}
```

Possible result shape:

```json
{
  "ok": true,
  "forecast_results": [
    {
      "series_id": "Sales|Store=1",
      "model": "amazon/chronos-2",
      "forecast_index": ["2015-08-01", "2015-08-02", "..."],
      "forecast": ["<median forecast values>"],
      "quantiles": {
        "0.1": ["<lower quantile values>"],
        "0.5": ["<median quantile values>"],
        "0.9": ["<upper quantile values>"]
      },
      "inference_metadata": {
        "provider": "chronos-forecasting",
        "zero_shot": true,
        "prediction_length": 14,
        "context_length": 365
      }
    }
  ]
}
```

This example requires the `chronos2` optional dependency and verified local
Chronos-2 weights. `target_metrics` selects metric names, not dimensions; for
dimension-specific inference, pass a prefiltered `series_collection` or filter
the returned `forecast_results` by `series_id`.

## Reproducibility And Operational Notes

- The server returns structured errors. Failures use the shape
  `{"ok": false, "error": {"code": "...", "message": "...", "details": {}}}`.
  Set `TIME_SERIES_ANALYSIS_MCP_DEBUG=1` or `TS_MCP_DEBUG=1` to include
  tracebacks.
- Statistical forecasting tools require complete target series. Run
  `time_series_missing_data_handler` before ARIMA, Holt-Winters, DeSeaTS,
  Chronos-2, or AutoML when missing values are present. Toto 2.0 can pass
  missing histories through target masks, but preprocessing is still useful for
  comparability.
- `holdout_size` computes validation metrics on a training fit, then refits on
  the full series for the returned future forecast.
- ARIMA supports known-future covariates through
  `known_future_covariate_metrics`, `future_covariates`, and
  `calendar_covariates`. AutoML and Chronos-2 also support past-only,
  known-future, and generated calendar covariates where the upstream model can
  use them.
- Canonical Chronos-2 and Toto 2.0 model weights are intentionally local and
  verified with `resources/MODEL_WEIGHTS_MANIFEST.json`. The canonical models
  do not silently fall back to remote downloads. For development only, set
  `TS_MCP_ALLOW_REMOTE_MODELS=1` and explicitly request a non-vendored model
  ID.
- Runtime limits are configurable with environment variables such as
  `TS_MCP_MAX_FILE_MB`, `TS_MCP_MAX_ROWS`, `TS_MCP_MAX_SERIES`,
  `TS_MCP_MAX_TIMESTAMPS`, `TS_MCP_CHRONOS2_MAX_CONTEXT`,
  `TS_MCP_TOTO2_MAX_CONTEXT`, and `TS_MCP_AUTOML_TIME_LIMIT_SECONDS`.
- The implementation is organized in three layers:
  `time_series_analysis_mcp/` for MCP server, CLI, registry, and errors;
  `tools/` for MCP-facing adapters and rendering; and `time_series_analysis/`
  for loading, diagnostics, quality handling, decomposition, and forecasting
  logic.

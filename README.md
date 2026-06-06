# time-series-analysis-mcp

Enterprise-oriented MCP server for multi-dimensional time-series loading, quality handling, diagnostics, and forecasting.

## Tools

- `list_tools`
- `time_series_loader`
- `time_series_missing_data_handler`
- `time_series_outlier_handler`
- `time_series_unit_root_test`
- `time_series_seasonality_test`
- `time_series_structural_break_test`
- `time_series_deseats_decompose`
- `time_series_arima`
- `time_series_exponential_smoothing`
- `time_series_deseats_forecast`
- `time_series_chronos2_forecast`
- `time_series_toto2_forecast`
- `time_series_automl_forecast`

## Statistical Forecasting Tools

`time_series_arima` fits optimized ARIMA or seasonal ARIMA models through
`statsmodels`. Supply `seasonal_order=[P, D, Q, s]` for an explicit seasonal
model, or set `auto_order=true` to perform bounded information-criterion
selection in nonseasonal and seasonal stages; monthly, quarterly, and daily
inputs infer periods of 12, 4, and 7 when enough history is available. Use
`target_metrics`, `known_future_covariate_metrics`, `future_covariates`, and
optional `calendar_covariates` for ARIMAX-style forecasts with known future
influence factors.

`time_series_exponential_smoothing` fits optimized Holt-Winters models through
`statsmodels`, including native additive or multiplicative trend and seasonal
components. Its intervals are obtained by forecast simulation.

## Run

```bash
source .venv/bin/activate
python -m time_series_analysis_mcp
```

MCP client configuration:

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

To inspect available tools from an MCP client, call:

```json
{
  "name": "list_tools",
  "arguments": {}
}
```

Use the absolute virtualenv Python path for MCP clients. Most clients launch
servers without activating your shell environment, so `python -m
time_series_analysis_mcp` can fail outside this repository with `No module
named time_series_analysis_mcp`.

## Optional DeSeaTS Tools

`time_series_deseats_decompose` and `time_series_deseats_forecast` call the R
`deseats` package through `Rscript`. Install the R runtime dependencies before
using those tools:

```bash
Rscript -e 'install.packages("jsonlite", repos="https://cloud.r-project.org")'
R CMD INSTALL /Users/lchen/github_phd/deseats
```

Set `TS_MCP_RSCRIPT` if `Rscript` is not on `PATH`.

## Local Foundation-Model Weights

The direct Chronos-2 and Toto 2.0 tools load their pinned weights from the
repository-root `modelweights/` directory by default. Runtime inference does
not automatically download either canonical model.

Prepare and verify the ignored local weight files while online:

```bash
python scripts/manage_modelweights.py download
python scripts/manage_modelweights.py verify
```

The tracked `resources/MODEL_WEIGHTS_MANIFEST.json` pins upstream revisions,
file sizes, and SHA-256 checksums. Each model is verified before its first load
in a server process. Set `TS_MCP_MODEL_WEIGHTS_DIR` to use another local root.
Because `modelweights/` is gitignored, every checkout or deployment must
prepare the files separately.

Use `python scripts/manage_modelweights.py status` to inspect local files and
`python scripts/manage_modelweights.py check-latest` to check for upstream
revision changes. See `resources/MODEL_WEIGHTS.md` for the update procedure.

## Optional Chronos-2 Tool

`time_series_chronos2_forecast` runs Amazon Chronos-2 zero-shot forecasts and
returns point forecasts plus requested quantiles. Install its optional Python
dependencies before invoking the tool:

```bash
python -m pip install -e '.[chronos2]'
```

The default `amazon/chronos-2` ID resolves to
`modelweights/chronos-2`. Set `TS_MCP_MODEL_DEVICE` (`cpu`, `cuda`, or another
supported `device_map` value) to select the inference device. This integration
treats selected `SeriesCollection` series as target items and supports
`past_covariate_metrics`, `known_future_covariate_metrics`,
`future_covariates`, and generated `calendar_covariates`.

## Optional Toto 2.0 Tool

`time_series_toto2_forecast` uses Datadog Toto 2.0 directly for zero-shot
quantile forecasts. It supports masked missing inputs and can model aligned
series together using `group_by_dimensions`.

```bash
python -m pip install -e '.[toto2]'
```

Toto 2.0 requires Python 3.12 or newer. The default
`Datadog/Toto-2.0-22m` ID resolves to `modelweights/toto-2.0-22m`. Set
`TS_MCP_MODEL_DEVICE`, `TS_MCP_TOTO2_MAX_CONTEXT`, or
`TS_MCP_TOTO2_MAX_PREDICTION_LENGTH` to configure runtime behavior. Input
contexts are left-padded with masked values when required by Toto's patch
size. Toto 2.0 currently exposes fixed quantiles from `0.1` through `0.9`;
exogenous variables and fine-tuning are not supported by the Toto 2.0
upstream API yet.

Non-vendored model IDs are disabled by default. For development only, set
`TS_MCP_ALLOW_REMOTE_MODELS=1` and explicitly request a non-vendored model ID.
The two canonical pinned models never fall back to a remote download.

## Optional AutoML Tool

`time_series_automl_forecast` uses AutoGluon TimeSeries for model fitting,
validation scoring, optional ensembling, model selection, and probabilistic
forecasts.

```bash
python -m pip install -e '.[automl]'
```

Install this extra with the same Python executable configured for the MCP
server; installing into another environment leaves the tool unavailable.
Use `model_names` to provide AutoGluon model names such as `SeasonalNaive`,
`ETS`, or `Chronos2`. AutoGluon's documented `TotoModel` targets Toto 1.0;
use `time_series_toto2_forecast` for Toto 2.0. Fitted AutoGluon artifacts are
temporary by default; provide `model_path` when a fitted predictor must be
retained after the tool call. AutoGluon's own Chronos model management is
separate from the direct tool's repository-local `modelweights/` directory.
AutoML accepts past-only covariates, known-future covariates, and generated
calendar covariates; AutoGluon decides which candidate models use them.

## Example

```json
{
  "name": "time_series_loader",
  "arguments": {
    "file_path": "sales.csv",
    "time_column": "date",
    "value_columns": ["sales"],
    "dimension_columns": ["product", "region"],
    "frequency": "month",
    "aggregation": "sum"
  }
}
```

Forecast loaded or inline values with Chronos-2:

```json
{
  "name": "time_series_chronos2_forecast",
  "arguments": {
    "data": [15, 16, 14, 18, 20, 21, 22, 24],
    "frequency": "month",
    "forecast_steps": 3,
    "quantile_levels": [0.1, 0.5, 0.9]
  }
}
```

Run an AutoGluon model comparison:

```json
{
  "name": "time_series_automl_forecast",
  "arguments": {
    "data": [15, 16, 14, 18, 20, 21, 22, 24, 26, 27, 29, 30],
    "forecast_steps": 2,
    "model_names": ["SeasonalNaive", "Chronos2"],
    "time_limit_seconds": 120
  }
}
```

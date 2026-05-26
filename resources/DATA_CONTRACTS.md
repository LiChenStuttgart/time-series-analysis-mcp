# Data Contracts

## SeriesCollection

`time_series_loader` returns a JSON object with this top-level shape:

```json
{
  "ok": true,
  "dataset": {
    "dataset_id": "...",
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
      "requested_frequency": "month",
      "aggregation": "sum"
    },
    "time_index": ["2025-01-31", "2025-02-28"],
    "series": [
      {
        "series_id": "sales|product=A|region=US",
        "metric": "sales",
        "dimensions": {"product": "A", "region": "US"},
        "values": [10.0, null],
        "quality": {
          "missing_count": 1,
          "missing_rate": 0.5,
          "calendar_gap_count": 1,
          "outlier_count": 0
        },
        "metadata": {}
      }
    ],
    "quality": {
      "n_rows_loaded": 1,
      "n_rows_after_cleaning": 1,
      "n_series": 1,
      "n_timestamps": 2,
      "warnings": []
    }
  }
}
```

## Frequency Values

Accepted values include:

- `day`, `daily`, `D`
- `month`, `monthly`, `M`
- `quarter`, `quarterly`, `quartal`, `Q`
- `year`, `annual`, `yearly`, `Y`

The server normalizes these to `D`, `M`, `Q`, or `Y`.

## Missing And Outlier Flags

Handlers preserve audit flags in `series[].metadata`:

- `imputation_flags`
- `imputation_strategy`
- `outlier_flags`
- `outlier_detection_method`
- `outlier_handling_method`

## Statistical Forecast Results

`time_series_arima` returns fitted ARIMA or seasonal ARIMA results with
estimated AR, MA, seasonal AR, and seasonal MA parameters where requested.
`order` and `seasonal_order` report the fitted specification, and `lower` and
`upper` are model prediction intervals.

`time_series_exponential_smoothing` returns optimized Holt-Winters smoothing
and initial-state parameters. Its `lower` and `upper` forecast intervals are
estimated from simulated future paths.

For either tool, `holdout_size` computes validation metrics from a training
fit, followed by a full-data refit for the returned future forecast.

## DeSeaTS Results

`time_series_deseats_decompose` returns `decomposition_results[]` with
`observations`, `trend`, `seasonality`, `fitted`, `residuals`,
`deseasonalized`, and `detrended` vectors aligned to the input `time_index`.

`time_series_deseats_forecast` returns `forecast_results[]` with
`forecast_index`, point `forecast`, `intervals`, selected DeSeaTS
bandwidth, and the fitted residual ARMA order. These tools require `Rscript`
plus the R packages `deseats` and `jsonlite`.

## Chronos-2 Results

`time_series_chronos2_forecast` returns `forecast_results[]` with:

- `forecast_index`: future timestamps aligned to the dataset frequency.
- `forecast`: the Chronos-2 point forecast.
- `quantiles`: probability-level keys such as `"0.1"`, `"0.5"`, and `"0.9"`.
- `metrics`: optional trailing holdout metrics when `holdout_size` is requested.
- `inference_metadata`: model identifier, device, context length, horizon, and quantile levels.
- `fit_warnings`: input truncation or inference warnings.

The initial Chronos-2 integration accepts complete target series from the
existing `SeriesCollection` contract. It does not yet expose past/future
covariate roles or explicitly grouped multivariate targets.

## Toto 2.0 Results

`time_series_toto2_forecast` returns the probabilistic forecast contract used
for Chronos-2. Toto 2.0 quantile levels must be selected from the upstream
model's fixed `0.1` through `0.9` outputs and include `0.5` for the point
forecast. `inference_metadata` records the multivariate `group_id` and
`group_variates` used for each result.

By default each series is forecast independently. When
`group_by_dimensions` is supplied, series with matching requested dimension
values are placed in the same multivariate Toto 2.0 inference call. Missing
history values are supplied through Toto 2.0 target masks and reported in
`fit_warnings`; short contexts may also be left-padded with masked values to
align to the loaded model's patch size. Cached model weights are used before
remote loading is attempted.

## AutoML Results

`time_series_automl_forecast` returns:

- `forecast_results`: point and quantile predictions from AutoGluon's selected model.
- `best_model`: the selected model name.
- `leaderboard`: AutoGluon validation leaderboard rows.
- `configuration`: presets, validation windows, ensemble and time-limit settings.

This initial AutoML contract accepts complete target series. It does not
provide known-future covariate fields yet. The tool cleans up AutoGluon model
artifacts after inference unless an explicit `model_path` is supplied.

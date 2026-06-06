# Tool Guide

Use `guide://time-series-analysis` for a live registry-generated tool catalogue.

Recommended workflow:

1. Run `list_tools` when you need the tool catalogue from inside an MCP client.
2. Run `time_series_loader`.
3. Run `time_series_missing_data_handler`.
4. Run `time_series_outlier_handler`.
5. Run diagnostics.
6. Run `time_series_deseats_decompose` when you need trend, seasonal, fitted, residual, detrended, or seasonally adjusted components.
7. Run a forecasting tool.

`time_series_arima` uses fitted ARIMA or seasonal ARIMA models. Supply an
explicit `seasonal_order`, or use `auto_order` with bounded seasonal order
limits for staged information-criterion selection. It also supports
ARIMAX-style known-future covariates through
`known_future_covariate_metrics`, `future_covariates`, and generated
`calendar_covariates`.

`time_series_exponential_smoothing` uses optimized Holt-Winters estimation for
level, trend, and optional seasonal components.

`time_series_chronos2_forecast` requires the `chronos2` optional dependency
extra and verified weights under repository-root `modelweights/chronos-2`. It
provides zero-shot point and quantile forecasts for selected complete target
series, with optional past-only, known-future, and generated calendar
covariates.

`time_series_toto2_forecast` requires the `toto2` optional dependency extra.
It provides direct Toto 2.0 zero-shot fixed-quantile forecasts and supports
multivariate target grouping through `group_by_dimensions`. It loads verified
weights from repository-root `modelweights/toto-2.0-22m`.

Prepare both ignored local model directories with
`python scripts/manage_modelweights.py download`, then validate them with
`python scripts/manage_modelweights.py verify`. Runtime inference never
downloads either canonical model.

`time_series_automl_forecast` requires the `automl` optional dependency extra.
It evaluates AutoGluon forecasting candidates, returns a leaderboard and
generates predictions from its selected model. Install the extra into the
same Python environment used to launch the MCP server. It passes
known-future covariates to AutoGluon and keeps past-only covariates in the
training frame.

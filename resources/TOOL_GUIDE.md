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
limits for staged information-criterion selection.

`time_series_exponential_smoothing` uses optimized Holt-Winters estimation for
level, trend, and optional seasonal components.

`time_series_chronos2_forecast` requires the `chronos2` optional dependency
extra and model weights on first use. It provides zero-shot point and
quantile forecasts for selected complete target series.

`time_series_toto2_forecast` requires the `toto2` optional dependency extra.
It provides direct Toto 2.0 zero-shot fixed-quantile forecasts and supports
multivariate target grouping through `group_by_dimensions`. It checks local
cached model weights before attempting a network fetch; set
`TS_MCP_MODEL_OFFLINE=1` for a strictly offline MCP process.

`time_series_automl_forecast` requires the `automl` optional dependency extra.
It evaluates AutoGluon forecasting candidates, returns a leaderboard and
generates predictions from its selected model. Install the extra into the
same Python environment used to launch the MCP server.

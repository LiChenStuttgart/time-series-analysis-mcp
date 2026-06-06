# Changelog

## 0.1.0

- Initial enterprise time-series MCP server implementation.
- Adds loader, missing-data handling, outlier handling, diagnostics, ARIMA-style forecasting, and exponential smoothing tools.
- Adds an optional Amazon Chronos-2 zero-shot probabilistic forecasting tool and model-backed forecast contract.
- Adds optional Datadog Toto 2.0 zero-shot forecasting and AutoGluon model-selection tools.
- Loads direct Chronos-2 and Toto 2.0 forecasts from manifest-verified,
  repository-local `modelweights/` directories by default.
- Adds covariate-aware forecasting for ARIMA, Chronos-2, and AutoML, including
  known-future metric covariates and generated calendar covariates.

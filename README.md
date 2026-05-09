# time-series-analysis-mcp

Enterprise-oriented MCP server for multi-dimensional time-series loading, quality handling, diagnostics, and forecasting.

## Tools

- `time_series_loader`
- `time_series_missing_data_handler`
- `time_series_outlier_handler`
- `time_series_unit_root_test`
- `time_series_seasonality_test`
- `time_series_structural_break_test`
- `time_series_arima`
- `time_series_exponential_smoothing`

## Run

```bash
python -m time_series_analysis_mcp
```

MCP client configuration:

```json
{
  "mcpServers": {
    "time-series-analysis-mcp": {
      "command": "python",
      "args": ["-m", "time_series_analysis_mcp"]
    }
  }
}
```

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

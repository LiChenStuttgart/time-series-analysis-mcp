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

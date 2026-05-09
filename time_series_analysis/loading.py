"""File loading and multi-dimensional series construction."""

from __future__ import annotations

import csv
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Any

from time_series_analysis_mcp.config import LIMITS
from time_series_analysis_mcp.errors import DependencyError, ValidationError

from .schemas import (
    DatasetQuality,
    DatasetSchema,
    LoaderResult,
    SeriesCollection,
    SeriesQuality,
    SourceInfo,
    TimeSeriesItem,
)
from .utils import (
    bucket_datetime,
    make_regular_index,
    normalize_frequency,
    parse_datetime,
    parse_float,
    stable_hash,
    utc_now_iso,
)


SUPPORTED_SUFFIXES = {".txt", ".csv", ".json", ".xlsx", ".xls"}
AGGREGATIONS = {"sum", "mean", "median", "min", "max", "count", "first", "last"}


def load_time_series_file(
    file_path: str,
    *,
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
) -> LoaderResult:
    path = _validate_path(file_path)
    rows = _read_rows(path)
    if len(rows) > LIMITS.max_rows:
        raise ValidationError("row limit exceeded", rows=len(rows), max_rows=LIMITS.max_rows)
    if not rows:
        raise ValidationError("input file contains no rows")

    requested_frequency = frequency
    normalized_frequency = normalize_frequency(frequency)
    aggregation = aggregation.lower()
    if aggregation not in AGGREGATIONS:
        raise ValidationError(
            "unsupported aggregation",
            aggregation=aggregation,
            supported=sorted(AGGREGATIONS),
        )
    if timezone:
        # The pure-Python loader parses timestamps but does not convert zones.
        # Keep the value visible rather than pretending conversion occurred.
        timezone_warning = f"timezone {timezone!r} recorded but not converted by pure-Python loader"
    else:
        timezone_warning = ""

    headers = list(rows[0])
    if time_column not in headers:
        raise ValidationError("time_column not found", time_column=time_column, columns=headers)

    if start_date:
        start_dt = parse_datetime(start_date, date_format=date_format).date()
    else:
        start_dt = None
    if end_date:
        end_dt = parse_datetime(end_date, date_format=date_format).date()
    else:
        end_dt = None

    inferred_value_columns = value_columns or _infer_value_columns(rows, time_column, dimension_columns)
    if not inferred_value_columns:
        raise ValidationError("no value columns supplied or inferred")
    missing_values = [col for col in inferred_value_columns if col not in headers]
    if missing_values:
        raise ValidationError("value column not found", missing_columns=missing_values)

    inferred_dimension_columns = (
        dimension_columns
        if dimension_columns is not None
        else _infer_dimension_columns(rows, time_column, inferred_value_columns)
    )
    missing_dims = [col for col in inferred_dimension_columns if col not in headers]
    if missing_dims:
        raise ValidationError("dimension column not found", missing_columns=missing_dims)

    grouped: dict[tuple[str, str, tuple[tuple[str, str], ...]], list[float | None]] = defaultdict(list)
    all_buckets: set[str] = set()
    n_after_cleaning = 0
    warnings: list[str] = []
    if timezone_warning:
        warnings.append(timezone_warning)

    for raw in rows:
        dt = parse_datetime(raw.get(time_column), date_format=date_format)
        bucket = bucket_datetime(dt, normalized_frequency)
        if start_dt and bucket < start_dt:
            continue
        if end_dt and bucket > end_dt:
            continue
        bucket_key = bucket.isoformat()
        all_buckets.add(bucket_key)
        dims = tuple(
            (column, "" if raw.get(column) is None else str(raw.get(column)).strip())
            for column in inferred_dimension_columns
        )
        for metric in inferred_value_columns:
            value = parse_float(raw.get(metric), missing_tokens=missing_tokens)
            grouped[(metric, bucket_key, dims)].append(value)
        n_after_cleaning += 1

    if not all_buckets:
        raise ValidationError("no rows remained after date filtering")

    if fill_calendar:
        start_bucket = min(all_buckets)
        end_bucket = max(all_buckets)
        time_index = make_regular_index(
            parse_datetime(start_bucket).date(),
            parse_datetime(end_bucket).date(),
            normalized_frequency,
        )
    else:
        time_index = sorted(all_buckets)

    if len(time_index) > LIMITS.max_timestamps:
        raise ValidationError(
            "timestamp limit exceeded",
            timestamps=len(time_index),
            max_timestamps=LIMITS.max_timestamps,
        )

    series_keys = sorted({(metric, dims) for metric, _bucket, dims in grouped})
    if len(series_keys) > LIMITS.max_series:
        raise ValidationError(
            "series limit exceeded",
            series=len(series_keys),
            max_series=LIMITS.max_series,
        )

    index_lookup = {stamp: i for i, stamp in enumerate(time_index)}
    series_items: list[TimeSeriesItem] = []
    for metric, dims in series_keys:
        values: list[float | None] = [None] * len(time_index)
        calendar_gaps = 0
        for stamp in time_index:
            raw_values = grouped.get((metric, stamp, dims))
            if not raw_values:
                calendar_gaps += 1
                continue
            values[index_lookup[stamp]] = _aggregate(raw_values, aggregation)
        missing_count = sum(1 for value in values if value is None)
        dim_dict = dict(dims)
        series_id = _make_series_id(metric, dim_dict)
        series_items.append(
            TimeSeriesItem(
                series_id=series_id,
                metric=metric,
                dimensions=dim_dict,
                values=values,
                quality=SeriesQuality(
                    missing_count=missing_count,
                    missing_rate=missing_count / len(values) if values else 0.0,
                    calendar_gap_count=calendar_gaps,
                    outlier_count=0,
                ),
            )
        )

    collection = SeriesCollection(
        dataset_id=stable_hash(f"{file_path}|{time_column}|{inferred_value_columns}|{inferred_dimension_columns}"),
        source=SourceInfo(
            file_path=str(path),
            file_type=path.suffix.lower().lstrip("."),
            loaded_at=utc_now_iso(),
        ),
        schema=DatasetSchema(
            time_column=time_column,
            value_columns=inferred_value_columns,
            dimension_columns=inferred_dimension_columns,
            frequency=normalized_frequency,
            requested_frequency=requested_frequency,
            aggregation=aggregation,
        ),
        time_index=time_index,
        series=series_items,
        quality=DatasetQuality(
            n_rows_loaded=len(rows),
            n_rows_after_cleaning=n_after_cleaning,
            n_series=len(series_items),
            n_timestamps=len(time_index),
            warnings=warnings,
        ),
    )
    return LoaderResult(dataset=collection)


def parse_series_collection(payload: dict[str, Any]) -> SeriesCollection:
    data = payload.get("dataset", payload)
    return SeriesCollection.model_validate(data)


def collection_from_single_series(
    data: list[float | None],
    *,
    frequency: str = "month",
    metric: str = "value",
) -> SeriesCollection:
    normalized = normalize_frequency(frequency)
    values = [parse_float(value) for value in data]
    if not values:
        raise ValidationError("data must not be empty")
    # Use a synthetic index anchored at month-end 2000-01 for direct calls.
    from datetime import date

    start = date(2000, 1, 31)
    time_index = make_regular_index(start, _advance_for_length(start, normalized, len(values)), normalized)
    time_index = time_index[: len(values)]
    missing_count = sum(1 for value in values if value is None)
    item = TimeSeriesItem(
        series_id=metric,
        metric=metric,
        dimensions={},
        values=values,
        quality=SeriesQuality(
            missing_count=missing_count,
            missing_rate=missing_count / len(values),
            calendar_gap_count=missing_count,
        ),
    )
    return SeriesCollection(
        dataset_id=stable_hash(f"direct|{metric}|{len(values)}"),
        source=SourceInfo(file_path=None, file_type="inline", loaded_at=utc_now_iso()),
        schema=DatasetSchema(
            time_column="time",
            value_columns=[metric],
            dimension_columns=[],
            frequency=normalized,
            requested_frequency=frequency,
            aggregation="none",
        ),
        time_index=time_index,
        series=[item],
        quality=DatasetQuality(
            n_rows_loaded=len(values),
            n_rows_after_cleaning=len(values),
            n_series=1,
            n_timestamps=len(time_index),
            warnings=[],
        ),
    )


def _advance_for_length(start, frequency: str, length: int):
    from .utils import add_period

    current = start
    for _ in range(max(length - 1, 0)):
        current = add_period(current, frequency)
    return current


def _validate_path(file_path: str) -> Path:
    path = Path(file_path)
    if not path.exists():
        raise ValidationError("file not found", file_path=file_path)
    if path.is_dir():
        raise ValidationError("file_path points to a directory", file_path=file_path)
    suffix = path.suffix.lower()
    if suffix not in SUPPORTED_SUFFIXES:
        raise ValidationError(
            "unsupported file format",
            suffix=suffix,
            supported=sorted(SUPPORTED_SUFFIXES),
        )
    max_bytes = LIMITS.max_file_mb * 1024 * 1024
    size = path.stat().st_size
    if size > max_bytes:
        raise ValidationError("file size limit exceeded", bytes=size, max_bytes=max_bytes)
    return path


def _read_rows(path: Path) -> list[dict[str, Any]]:
    suffix = path.suffix.lower()
    if suffix in {".csv", ".txt"}:
        return _read_delimited(path)
    if suffix == ".json":
        return _read_json(path)
    if suffix == ".xlsx":
        return _read_xlsx(path)
    if suffix == ".xls":
        return _read_xls(path)
    raise AssertionError(suffix)


def _read_delimited(path: Path) -> list[dict[str, Any]]:
    text = path.read_text(encoding="utf-8-sig")
    sample = text[:4096]
    try:
        dialect = csv.Sniffer().sniff(sample)
    except csv.Error:
        dialect = csv.excel_tab if path.suffix.lower() == ".txt" else csv.excel
    reader = csv.DictReader(text.splitlines(), dialect=dialect)
    if not reader.fieldnames:
        raise ValidationError("delimited file has no header row")
    return [dict(row) for row in reader]


def _read_json(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        loaded = json.load(handle)
    if isinstance(loaded, dict):
        if isinstance(loaded.get("data"), list):
            loaded = loaded["data"]
        elif isinstance(loaded.get("rows"), list):
            loaded = loaded["rows"]
        else:
            raise ValidationError("JSON object must contain a list under 'data' or 'rows'")
    if not isinstance(loaded, list):
        raise ValidationError("JSON input must be a list of objects")
    if any(not isinstance(row, dict) for row in loaded):
        raise ValidationError("JSON input must be a list of objects")
    return [dict(row) for row in loaded]


def _read_xlsx(path: Path) -> list[dict[str, Any]]:
    try:
        import openpyxl  # type: ignore[import-not-found]
    except ImportError as exc:
        raise DependencyError(
            "reading .xlsx files requires openpyxl; install the scientific extra",
            dependency="openpyxl",
        ) from exc
    workbook = openpyxl.load_workbook(path, read_only=True, data_only=True)
    sheet = workbook.active
    rows = list(sheet.iter_rows(values_only=True))
    if not rows:
        return []
    headers = [str(cell).strip() if cell is not None else "" for cell in rows[0]]
    return [
        {headers[i]: value for i, value in enumerate(row) if i < len(headers) and headers[i]}
        for row in rows[1:]
    ]


def _read_xls(path: Path) -> list[dict[str, Any]]:
    try:
        import xlrd  # type: ignore[import-not-found]
    except ImportError as exc:
        raise DependencyError(
            "reading .xls files requires xlrd; install the scientific extra",
            dependency="xlrd",
        ) from exc
    workbook = xlrd.open_workbook(str(path))
    sheet = workbook.sheet_by_index(0)
    if sheet.nrows == 0:
        return []
    headers = [str(sheet.cell_value(0, col)).strip() for col in range(sheet.ncols)]
    out = []
    for row_idx in range(1, sheet.nrows):
        out.append({headers[col]: sheet.cell_value(row_idx, col) for col in range(sheet.ncols)})
    return out


def _infer_value_columns(
    rows: list[dict[str, Any]],
    time_column: str,
    dimension_columns: list[str] | None,
) -> list[str]:
    excluded = {time_column, *(dimension_columns or [])}
    candidates: list[str] = []
    for column in rows[0]:
        if column in excluded:
            continue
        seen = 0
        numeric = 0
        for row in rows[:100]:
            value = row.get(column)
            if value in (None, ""):
                continue
            seen += 1
            try:
                parse_float(value)
                numeric += 1
            except ValidationError:
                pass
        if seen and numeric == seen:
            candidates.append(column)
    return candidates


def _infer_dimension_columns(
    rows: list[dict[str, Any]],
    time_column: str,
    value_columns: list[str],
) -> list[str]:
    excluded = {time_column, *value_columns}
    return [column for column in rows[0] if column not in excluded]


def _aggregate(values: list[float | None], method: str) -> float | None:
    non_missing = [value for value in values if value is not None]
    if method == "count":
        return float(len(non_missing))
    if not non_missing:
        return None
    if method == "sum":
        return float(sum(non_missing))
    if method == "mean":
        return float(sum(non_missing) / len(non_missing))
    if method == "median":
        vals = sorted(non_missing)
        mid = len(vals) // 2
        if len(vals) % 2:
            return float(vals[mid])
        return float((vals[mid - 1] + vals[mid]) / 2)
    if method == "min":
        return float(min(non_missing))
    if method == "max":
        return float(max(non_missing))
    if method == "first":
        for value in values:
            if value is not None:
                return float(value)
    if method == "last":
        for value in reversed(values):
            if value is not None:
                return float(value)
    raise AssertionError(method)


def _make_series_id(metric: str, dimensions: dict[str, str]) -> str:
    if not dimensions:
        return metric
    parts = [metric, *(f"{key}={value}" for key, value in sorted(dimensions.items()))]
    return "|".join(parts)


def save_text(content: str, save_path: str) -> str:
    path = Path(save_path)
    parent = path.parent
    if parent and not parent.exists():
        os.makedirs(parent, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return str(path)

"""Adapter validation helpers."""

from __future__ import annotations

from time_series_analysis_mcp.errors import ValidationError


def require_time_column(file_path: str | None, time_column: str | None) -> None:
    if file_path and not time_column:
        raise ValidationError("time_column is required when file_path is supplied")

"""Runtime configuration helpers."""

from __future__ import annotations

import os
from dataclasses import dataclass


def _int_env(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _bool_env(name: str) -> bool:
    return os.environ.get(name, "").lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class RuntimeLimits:
    max_file_mb: int = _int_env("TS_MCP_MAX_FILE_MB", 100)
    max_rows: int = _int_env("TS_MCP_MAX_ROWS", 1_000_000)
    max_series: int = _int_env("TS_MCP_MAX_SERIES", 20_000)
    max_timestamps: int = _int_env("TS_MCP_MAX_TIMESTAMPS", 10_000)
    debug: bool = _bool_env("TIME_SERIES_ANALYSIS_MCP_DEBUG") or _bool_env("TS_MCP_DEBUG")


LIMITS = RuntimeLimits()

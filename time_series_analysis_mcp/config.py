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
    max_rows: int = _int_env("TS_MCP_MAX_ROWS", 2_000_000)
    max_series: int = _int_env("TS_MCP_MAX_SERIES", 50_000)
    max_timestamps: int = _int_env("TS_MCP_MAX_TIMESTAMPS", 10_000)
    debug: bool = _bool_env("TIME_SERIES_ANALYSIS_MCP_DEBUG") or _bool_env("TS_MCP_DEBUG")
    rscript_path: str = os.environ.get("TS_MCP_RSCRIPT", "Rscript")
    r_libs_user: str | None = os.environ.get("TS_MCP_R_LIBS_USER") or None
    deseats_timeout_seconds: int = _int_env("TS_MCP_DESEATS_TIMEOUT_SECONDS", 60)
    deseats_max_series: int = _int_env(
        "TS_MCP_DESEATS_MAX_SERIES",
        _int_env("TS_MCP_MAX_SERIES", 50_000),
    )
    deseats_max_obs: int = _int_env(
        "TS_MCP_DESEATS_MAX_OBS",
        _int_env("TS_MCP_MAX_TIMESTAMPS", 10_000),
    )
    deseats_boot_cores: int = _int_env("TS_MCP_DESEATS_BOOT_CORES", 1)
    model_device: str = os.environ.get("TS_MCP_MODEL_DEVICE", "cpu")
    model_cache_dir: str | None = os.environ.get("TS_MCP_MODEL_CACHE_DIR") or None
    model_weights_dir: str | None = os.environ.get("TS_MCP_MODEL_WEIGHTS_DIR") or None
    model_offline: bool = _bool_env("TS_MCP_MODEL_OFFLINE")
    allow_remote_models: bool = _bool_env("TS_MCP_ALLOW_REMOTE_MODELS")
    chronos2_model_id: str = os.environ.get("TS_MCP_CHRONOS2_MODEL_ID", "amazon/chronos-2")
    chronos2_max_series: int = _int_env(
        "TS_MCP_CHRONOS2_MAX_SERIES",
        _int_env("TS_MCP_MAX_SERIES", 50_000),
    )
    chronos2_max_context: int = _int_env("TS_MCP_CHRONOS2_MAX_CONTEXT", 8192)
    chronos2_max_prediction_length: int = _int_env(
        "TS_MCP_CHRONOS2_MAX_PREDICTION_LENGTH",
        1024,
    )
    toto2_model_id: str = os.environ.get("TS_MCP_TOTO2_MODEL_ID", "Datadog/Toto-2.0-22m")
    toto2_max_series: int = _int_env(
        "TS_MCP_TOTO2_MAX_SERIES",
        _int_env("TS_MCP_MAX_SERIES", 50_000),
    )
    toto2_max_context: int = _int_env("TS_MCP_TOTO2_MAX_CONTEXT", 4096)
    toto2_max_prediction_length: int = _int_env(
        "TS_MCP_TOTO2_MAX_PREDICTION_LENGTH",
        1024,
    )
    automl_max_series: int = _int_env(
        "TS_MCP_AUTOML_MAX_SERIES",
        _int_env("TS_MCP_MAX_SERIES", 50_000),
    )
    automl_time_limit_seconds: int = _int_env("TS_MCP_AUTOML_TIME_LIMIT_SECONDS", 300)


LIMITS = RuntimeLimits()

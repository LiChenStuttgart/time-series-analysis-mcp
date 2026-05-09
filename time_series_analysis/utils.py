"""Shared pure-Python utilities."""

from __future__ import annotations

import calendar
import hashlib
import math
from datetime import date, datetime, timedelta, timezone
from typing import Iterable

from time_series_analysis_mcp.errors import ValidationError


MISSING_STRINGS = {"", "na", "n/a", "nan", "null", "none", "."}


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def stable_hash(value: str, length: int = 12) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:length]


def parse_float(value: object, *, missing_tokens: Iterable[str] | None = None) -> float | None:
    tokens = set(MISSING_STRINGS)
    if missing_tokens:
        tokens.update(str(item).strip().lower() for item in missing_tokens)
    if value is None:
        return None
    if isinstance(value, (int, float)):
        if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
            return None
        return float(value)
    text = str(value).strip()
    if text.lower() in tokens:
        return None
    try:
        parsed = float(text.replace(",", ""))
    except ValueError as exc:
        raise ValidationError(f"value {text!r} is not numeric") from exc
    if math.isnan(parsed) or math.isinf(parsed):
        return None
    return parsed


def parse_datetime(value: object, *, date_format: str | None = None) -> datetime:
    if isinstance(value, datetime):
        return value
    if isinstance(value, date):
        return datetime(value.year, value.month, value.day)
    text = str(value).strip()
    if not text:
        raise ValidationError("empty timestamp value")
    if date_format:
        try:
            return datetime.strptime(text, date_format)
        except ValueError as exc:
            raise ValidationError(
                f"timestamp {text!r} does not match date_format {date_format!r}"
            ) from exc
    normalized = text.replace("Z", "+00:00")
    try:
        return datetime.fromisoformat(normalized)
    except ValueError:
        pass
    formats = (
        "%Y-%m-%d",
        "%Y/%m/%d",
        "%m/%d/%Y",
        "%d/%m/%Y",
        "%Y-%m",
        "%Y/%m",
        "%Y",
    )
    for fmt in formats:
        try:
            return datetime.strptime(text, fmt)
        except ValueError:
            continue
    raise ValidationError(f"could not parse timestamp {text!r}")


def normalize_frequency(frequency: str) -> str:
    key = frequency.strip().lower()
    mapping = {
        "d": "D",
        "day": "D",
        "daily": "D",
        "m": "M",
        "me": "M",
        "month": "M",
        "monthly": "M",
        "q": "Q",
        "qe": "Q",
        "quarter": "Q",
        "quarterly": "Q",
        "quartal": "Q",
        "y": "Y",
        "ye": "Y",
        "a": "Y",
        "annual": "Y",
        "year": "Y",
        "yearly": "Y",
    }
    if key not in mapping:
        raise ValidationError(
            "unsupported frequency",
            frequency=frequency,
            supported=["day", "month", "quarter", "quartal", "year"],
        )
    return mapping[key]


def bucket_datetime(dt: datetime, frequency: str) -> date:
    freq = normalize_frequency(frequency)
    local = dt.date()
    if freq == "D":
        return local
    if freq == "M":
        return date(local.year, local.month, calendar.monthrange(local.year, local.month)[1])
    if freq == "Q":
        quarter_end_month = ((local.month - 1) // 3 + 1) * 3
        return date(
            local.year,
            quarter_end_month,
            calendar.monthrange(local.year, quarter_end_month)[1],
        )
    if freq == "Y":
        return date(local.year, 12, 31)
    raise AssertionError(freq)


def add_period(d: date, frequency: str, steps: int = 1) -> date:
    freq = normalize_frequency(frequency)
    if steps < 0:
        raise ValidationError("steps must be non-negative")
    out = d
    for _ in range(steps):
        if freq == "D":
            out = out + timedelta(days=1)
        elif freq == "M":
            month = out.month + 1
            year = out.year + (month - 1) // 12
            month = ((month - 1) % 12) + 1
            out = date(year, month, calendar.monthrange(year, month)[1])
        elif freq == "Q":
            month = out.month + 3
            year = out.year + (month - 1) // 12
            month = ((month - 1) % 12) + 1
            quarter_end_month = ((month - 1) // 3 + 1) * 3
            out = date(
                year,
                quarter_end_month,
                calendar.monthrange(year, quarter_end_month)[1],
            )
        elif freq == "Y":
            out = date(out.year + 1, 12, 31)
    return out


def make_regular_index(start: date, end: date, frequency: str) -> list[str]:
    if end < start:
        return []
    current = bucket_datetime(datetime(start.year, start.month, start.day), frequency)
    final = bucket_datetime(datetime(end.year, end.month, end.day), frequency)
    out: list[str] = []
    while current <= final:
        out.append(current.isoformat())
        current = add_period(current, frequency)
    return out


def next_index_values(time_index: list[str], frequency: str, steps: int) -> list[str]:
    if not time_index:
        return []
    current = date.fromisoformat(time_index[-1])
    out = []
    for _ in range(steps):
        current = add_period(current, frequency)
        out.append(current.isoformat())
    return out


def clean_values(values: Iterable[float | None]) -> list[float]:
    return [float(v) for v in values if v is not None and math.isfinite(float(v))]


def mean(values: Iterable[float]) -> float:
    vals = list(values)
    if not vals:
        raise ValidationError("cannot compute mean of empty data")
    return sum(vals) / len(vals)


def median(values: Iterable[float]) -> float:
    vals = sorted(values)
    if not vals:
        raise ValidationError("cannot compute median of empty data")
    mid = len(vals) // 2
    if len(vals) % 2:
        return vals[mid]
    return (vals[mid - 1] + vals[mid]) / 2


def variance(values: Iterable[float], *, sample: bool = False) -> float:
    vals = list(values)
    if len(vals) < 2:
        return 0.0
    m = mean(vals)
    denom = len(vals) - 1 if sample and len(vals) > 1 else len(vals)
    return sum((v - m) ** 2 for v in vals) / denom


def stddev(values: Iterable[float], *, sample: bool = False) -> float:
    return math.sqrt(max(variance(values, sample=sample), 0.0))


def normal_two_sided_p(z_value: float) -> float:
    return max(0.0, min(1.0, math.erfc(abs(z_value) / math.sqrt(2.0))))


def normal_left_tail_p(z_value: float) -> float:
    return max(0.0, min(1.0, 0.5 * math.erfc(-z_value / math.sqrt(2.0))))


def autocorrelation(values: list[float], lag: int) -> float:
    if lag <= 0 or lag >= len(values):
        return 0.0
    x = values[:-lag]
    y = values[lag:]
    mx = mean(x)
    my = mean(y)
    denom = math.sqrt(sum((a - mx) ** 2 for a in x) * sum((b - my) ** 2 for b in y))
    if denom == 0:
        return 0.0
    return sum((a - mx) * (b - my) for a, b in zip(x, y, strict=True)) / denom


def ols(y: list[float], x: list[list[float]]) -> tuple[list[float], list[float], float]:
    """Small normal-equation OLS with intercept supplied by caller in x."""
    if not y or len(y) != len(x):
        raise ValidationError("OLS requires x and y with matching non-zero lengths")
    n = len(y)
    k = len(x[0])
    if any(len(row) != k for row in x):
        raise ValidationError("all OLS rows must have the same number of columns")
    xtx = [[0.0 for _ in range(k)] for _ in range(k)]
    xty = [0.0 for _ in range(k)]
    for row, target in zip(x, y, strict=True):
        for i in range(k):
            xty[i] += row[i] * target
            for j in range(k):
                xtx[i][j] += row[i] * row[j]
    inv = invert_matrix(xtx)
    beta = [sum(inv[i][j] * xty[j] for j in range(k)) for i in range(k)]
    residuals = [target - sum(beta[i] * row[i] for i in range(k)) for row, target in zip(x, y, strict=True)]
    sse = sum(r * r for r in residuals)
    denom = max(n - k, 1)
    sigma2 = sse / denom
    std_errors = [math.sqrt(max(inv[i][i] * sigma2, 0.0)) for i in range(k)]
    return beta, std_errors, sse


def invert_matrix(matrix: list[list[float]]) -> list[list[float]]:
    n = len(matrix)
    aug = [row[:] + [1.0 if i == j else 0.0 for j in range(n)] for i, row in enumerate(matrix)]
    for col in range(n):
        pivot = max(range(col, n), key=lambda r: abs(aug[r][col]))
        if abs(aug[pivot][col]) < 1e-12:
            raise ValidationError("matrix is singular")
        aug[col], aug[pivot] = aug[pivot], aug[col]
        factor = aug[col][col]
        aug[col] = [value / factor for value in aug[col]]
        for row in range(n):
            if row == col:
                continue
            scale = aug[row][col]
            aug[row] = [v - scale * p for v, p in zip(aug[row], aug[col], strict=True)]
    return [row[n:] for row in aug]


def difference(values: list[float], order: int) -> list[float]:
    out = values[:]
    for _ in range(order):
        out = [b - a for a, b in zip(out[:-1], out[1:], strict=True)]
    return out


def integrate_forecast(history: list[float], diff_forecast: list[float], order: int) -> list[float]:
    if order <= 0:
        return diff_forecast
    if order == 1:
        last = history[-1]
        out = []
        for delta in diff_forecast:
            last += delta
            out.append(last)
        return out
    # Conservative fallback for higher differencing: repeatedly integrate
    # from the most recent lower-order difference.
    lower_history = difference(history, order - 1)
    lower_forecast = integrate_forecast(lower_history, diff_forecast, 1)
    return integrate_forecast(history, lower_forecast, order - 1)

"""Missing-data scanning and handling."""

from __future__ import annotations

from .schemas import (
    MissingDataDecision,
    MissingDataReport,
    MissingDataResult,
    SeriesCollection,
    SeriesQuality,
)
from .utils import median


SUPPORTED_STRATEGIES = {
    "auto",
    "none",
    "drop_series",
    "forward_fill",
    "backward_fill",
    "linear_interpolation",
    "time_interpolation",
    "seasonal_median",
    "rolling_median",
    "kalman",
    "iterative",
}


def handle_missing_data(
    collection: SeriesCollection,
    *,
    strategy: str = "auto",
    max_interpolation_gap: int = 3,
    seasonal_period: int | None = None,
    preserve_flags: bool = True,
) -> MissingDataResult:
    selected = strategy.lower()
    if selected not in SUPPORTED_STRATEGIES:
        from time_series_analysis_mcp.errors import ValidationError

        raise ValidationError(
            "unsupported missing-data strategy",
            strategy=strategy,
            supported=sorted(SUPPORTED_STRATEGIES),
        )

    out = collection.model_copy(deep=True)
    decisions: list[MissingDataDecision] = []
    kept_series = []

    for item in out.series:
        values = item.values[:]
        missing_before = _missing_count(values)
        warnings: list[str] = []
        filled_positions: list[int] = []
        if missing_before == 0:
            chosen = "none"
            reason = "series has no missing values"
        else:
            chosen, reason = _choose_strategy(
                values,
                selected,
                max_interpolation_gap=max_interpolation_gap,
                seasonal_period=seasonal_period,
            )

        if chosen == "drop_series":
            missing_after = missing_before
            decisions.append(
                MissingDataDecision(
                    series_id=item.series_id,
                    strategy=chosen,
                    reason=reason,
                    missing_before=missing_before,
                    missing_after=missing_after,
                    warnings=["series dropped from output"],
                )
            )
            continue
        if chosen in {"none"}:
            new_values = values
        elif chosen in {"linear_interpolation", "time_interpolation"}:
            new_values, filled_positions = _linear_fill(values, max_interpolation_gap)
        elif chosen == "forward_fill":
            new_values, filled_positions = _forward_fill(values)
        elif chosen == "backward_fill":
            new_values, filled_positions = _backward_fill(values)
        elif chosen == "seasonal_median":
            if not seasonal_period:
                warnings.append("seasonal_period was not supplied; falling back to linear interpolation")
                new_values, filled_positions = _linear_fill(values, max_interpolation_gap)
                chosen = "linear_interpolation"
            else:
                new_values, filled_positions = _seasonal_median_fill(values, seasonal_period)
        elif chosen == "rolling_median":
            new_values, filled_positions = _rolling_median_fill(values)
        elif chosen in {"kalman", "iterative"}:
            warnings.append(f"{chosen} fallback used pure-Python linear interpolation")
            new_values, filled_positions = _linear_fill(values, max_interpolation_gap)
        else:
            new_values = values

        missing_after = _missing_count(new_values)
        item.values = new_values
        if preserve_flags:
            flags = [idx in set(filled_positions) for idx in range(len(new_values))]
            item.metadata["imputation_flags"] = flags
            item.metadata["imputation_strategy"] = chosen
        item.quality = SeriesQuality(
            missing_count=missing_after,
            missing_rate=missing_after / len(new_values) if new_values else 0.0,
            calendar_gap_count=missing_after,
            outlier_count=item.quality.outlier_count,
        )
        kept_series.append(item)
        decisions.append(
            MissingDataDecision(
                series_id=item.series_id,
                strategy=chosen,
                reason=reason,
                missing_before=missing_before,
                missing_after=missing_after,
                filled_positions=filled_positions,
                warnings=warnings,
            )
        )

    out.series = kept_series
    out.quality.n_series = len(kept_series)
    total_before = sum(decision.missing_before for decision in decisions)
    total_after = sum(decision.missing_after for decision in decisions)
    return MissingDataResult(
        dataset=out,
        missing_data_report=MissingDataReport(
            total_missing_before=total_before,
            total_missing_after=total_after,
            series_decisions=decisions,
        ),
    )


def _choose_strategy(
    values: list[float | None],
    requested: str,
    *,
    max_interpolation_gap: int,
    seasonal_period: int | None,
) -> tuple[str, str]:
    if requested != "auto":
        return requested, "strategy selected by caller"
    missing = _missing_count(values)
    present = len(values) - missing
    if missing == 0:
        return "none", "series has no missing values"
    if present < 2:
        return "drop_series", "fewer than two observed values"
    if missing / len(values) > 0.5:
        return "none", "missing rate above 50%; leaving values unchanged"
    max_gap = max((stop - start + 1 for start, stop in _missing_runs(values)), default=0)
    if seasonal_period and len(values) >= seasonal_period * 2:
        return "seasonal_median", "seasonal period supplied and at least two seasons are present"
    if max_gap <= max_interpolation_gap:
        return "time_interpolation", f"maximum internal gap {max_gap} within threshold"
    return "linear_interpolation", "fallback interpolation for moderate missingness"


def _missing_count(values: list[float | None]) -> int:
    return sum(1 for value in values if value is None)


def _missing_runs(values: list[float | None]) -> list[tuple[int, int]]:
    runs = []
    start = None
    for idx, value in enumerate(values):
        if value is None and start is None:
            start = idx
        if value is not None and start is not None:
            runs.append((start, idx - 1))
            start = None
    if start is not None:
        runs.append((start, len(values) - 1))
    return runs


def _linear_fill(
    values: list[float | None],
    max_gap: int,
) -> tuple[list[float | None], list[int]]:
    out = values[:]
    filled: list[int] = []
    n = len(out)
    for start, stop in _missing_runs(out):
        gap = stop - start + 1
        left = start - 1
        right = stop + 1
        if left >= 0 and right < n and out[left] is not None and out[right] is not None:
            if gap <= max_gap:
                left_val = float(out[left])
                right_val = float(out[right])
                step = (right_val - left_val) / (gap + 1)
                for offset, idx in enumerate(range(start, stop + 1), start=1):
                    out[idx] = left_val + step * offset
                    filled.append(idx)
        elif left >= 0 and out[left] is not None:
            for idx in range(start, stop + 1):
                out[idx] = float(out[left])
                filled.append(idx)
        elif right < n and out[right] is not None:
            for idx in range(start, stop + 1):
                out[idx] = float(out[right])
                filled.append(idx)
    return out, filled


def _forward_fill(values: list[float | None]) -> tuple[list[float | None], list[int]]:
    out = values[:]
    filled = []
    last = None
    for idx, value in enumerate(out):
        if value is None and last is not None:
            out[idx] = last
            filled.append(idx)
        elif value is not None:
            last = value
    return out, filled


def _backward_fill(values: list[float | None]) -> tuple[list[float | None], list[int]]:
    reversed_values, filled = _forward_fill(list(reversed(values)))
    n = len(values)
    return list(reversed(reversed_values)), [n - 1 - idx for idx in filled]


def _seasonal_median_fill(
    values: list[float | None],
    seasonal_period: int,
) -> tuple[list[float | None], list[int]]:
    out = values[:]
    filled = []
    for idx, value in enumerate(out):
        if value is not None:
            continue
        same_position = [
            v for j, v in enumerate(out) if j % seasonal_period == idx % seasonal_period and v is not None
        ]
        if same_position:
            out[idx] = median([float(v) for v in same_position])
            filled.append(idx)
    if _missing_count(out):
        out, extra = _linear_fill(out, seasonal_period)
        filled.extend(extra)
    return out, sorted(set(filled))


def _rolling_median_fill(values: list[float | None], window: int = 2) -> tuple[list[float | None], list[int]]:
    out = values[:]
    filled = []
    for idx, value in enumerate(out):
        if value is not None:
            continue
        start = max(0, idx - window)
        stop = min(len(out), idx + window + 1)
        observed = [float(v) for v in out[start:stop] if v is not None]
        if observed:
            out[idx] = median(observed)
            filled.append(idx)
    if _missing_count(out):
        out, extra = _linear_fill(out, window * 2)
        filled.extend(extra)
    return out, sorted(set(filled))

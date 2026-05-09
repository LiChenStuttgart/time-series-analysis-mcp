"""Outlier detection and handling."""

from __future__ import annotations

from time_series_analysis_mcp.errors import ValidationError

from .missing_data import _linear_fill
from .schemas import OutlierDecision, OutlierReport, OutlierResult, SeriesCollection, SeriesQuality
from .utils import clean_values, median


SUPPORTED_DETECTION = {"auto", "mad", "iqr", "rolling_hampel", "stl_residual", "isolation_forest"}
SUPPORTED_HANDLING = {"flag", "winsorize", "interpolate", "seasonal_median", "null"}


def handle_outliers(
    collection: SeriesCollection,
    *,
    detection_method: str = "auto",
    handling_method: str = "flag",
    z_threshold: float = 3.5,
    iqr_multiplier: float = 1.5,
    seasonal_period: int | None = None,
    preserve_flags: bool = True,
) -> OutlierResult:
    method = detection_method.lower()
    handling = handling_method.lower()
    if method not in SUPPORTED_DETECTION:
        raise ValidationError("unsupported outlier detection method", method=method)
    if handling not in SUPPORTED_HANDLING:
        raise ValidationError("unsupported outlier handling method", method=handling)

    out = collection.model_copy(deep=True)
    decisions: list[OutlierDecision] = []

    for item in out.series:
        chosen = _choose_method(method, item.values, seasonal_period)
        positions = _detect(item.values, chosen, z_threshold, iqr_multiplier, seasonal_period)
        possible_breaks = _consecutive_runs(positions, min_run=3)
        warnings = []
        if possible_breaks:
            warnings.append("consecutive outliers may indicate a structural break")
        new_values = item.values[:]
        if positions and handling != "flag":
            new_values = _apply_handling(new_values, positions, handling, seasonal_period)
        item.values = new_values
        item.quality = SeriesQuality(
            missing_count=sum(1 for value in new_values if value is None),
            missing_rate=sum(1 for value in new_values if value is None) / len(new_values)
            if new_values
            else 0.0,
            calendar_gap_count=item.quality.calendar_gap_count,
            outlier_count=len(positions),
        )
        if preserve_flags:
            flag_set = set(positions)
            item.metadata["outlier_flags"] = [idx in flag_set for idx in range(len(new_values))]
            item.metadata["outlier_detection_method"] = chosen
            item.metadata["outlier_handling_method"] = handling
        decisions.append(
            OutlierDecision(
                series_id=item.series_id,
                detection_method=chosen,
                handling_method=handling,
                outlier_positions=positions,
                outlier_timestamps=[out.time_index[idx] for idx in positions],
                possible_break_positions=possible_breaks,
                warnings=warnings,
            )
        )

    return OutlierResult(
        dataset=out,
        outlier_report=OutlierReport(
            total_outliers=sum(len(decision.outlier_positions) for decision in decisions),
            series_decisions=decisions,
        ),
    )


def _choose_method(method: str, values: list[float | None], seasonal_period: int | None) -> str:
    if method != "auto":
        if method in {"stl_residual", "isolation_forest"}:
            return "rolling_hampel"
        return method
    observed = clean_values(values)
    if seasonal_period and len(observed) >= seasonal_period * 2:
        return "rolling_hampel"
    if len(observed) >= 12:
        return "rolling_hampel"
    return "mad"


def _detect(
    values: list[float | None],
    method: str,
    z_threshold: float,
    iqr_multiplier: float,
    seasonal_period: int | None,
) -> list[int]:
    if method == "mad":
        return _detect_mad(values, z_threshold)
    if method == "iqr":
        return _detect_iqr(values, iqr_multiplier)
    if method == "rolling_hampel":
        return _detect_hampel(values, z_threshold, seasonal_period or 3)
    return _detect_mad(values, z_threshold)


def _detect_mad(values: list[float | None], threshold: float) -> list[int]:
    indexed = [(idx, float(value)) for idx, value in enumerate(values) if value is not None]
    if len(indexed) < 3:
        return []
    center = median([value for _idx, value in indexed])
    deviations = [abs(value - center) for _idx, value in indexed]
    mad = median(deviations)
    if mad <= 1e-12:
        return []
    return [
        idx
        for idx, value in indexed
        if abs(0.6745 * (value - center) / mad) > threshold
    ]


def _detect_iqr(values: list[float | None], multiplier: float) -> list[int]:
    indexed = [(idx, float(value)) for idx, value in enumerate(values) if value is not None]
    observed = sorted(value for _idx, value in indexed)
    if len(observed) < 4:
        return []
    q1 = observed[len(observed) // 4]
    q3 = observed[(3 * len(observed)) // 4]
    iqr = q3 - q1
    lower = q1 - multiplier * iqr
    upper = q3 + multiplier * iqr
    return [idx for idx, value in indexed if value < lower or value > upper]


def _detect_hampel(values: list[float | None], threshold: float, window: int) -> list[int]:
    positions = []
    n = len(values)
    for idx, value in enumerate(values):
        if value is None:
            continue
        start = max(0, idx - window)
        stop = min(n, idx + window + 1)
        neighbors = [float(v) for j, v in enumerate(values[start:stop], start=start) if v is not None and j != idx]
        if len(neighbors) < 3:
            continue
        center = median(neighbors)
        mad = median([abs(v - center) for v in neighbors])
        if mad <= 1e-12:
            continue
        if abs(0.6745 * (float(value) - center) / mad) > threshold:
            positions.append(idx)
    # Global MAD catches isolated spikes near boundaries.
    return sorted(set(positions + _detect_mad(values, threshold)))


def _apply_handling(
    values: list[float | None],
    positions: list[int],
    handling: str,
    seasonal_period: int | None,
) -> list[float | None]:
    out = values[:]
    if handling == "null":
        for idx in positions:
            out[idx] = None
        return out
    if handling == "winsorize":
        observed = clean_values(values)
        low = median(observed) - 3 * median([abs(v - median(observed)) for v in observed])
        high = median(observed) + 3 * median([abs(v - median(observed)) for v in observed])
        for idx in positions:
            if out[idx] is not None:
                out[idx] = min(max(float(out[idx]), low), high)
        return out
    if handling == "seasonal_median" and seasonal_period:
        for idx in positions:
            same = [
                float(v)
                for j, v in enumerate(values)
                if j != idx and j % seasonal_period == idx % seasonal_period and v is not None
            ]
            out[idx] = median(same) if same else None
        out, _filled = _linear_fill(out, seasonal_period)
        return out
    if handling in {"interpolate", "seasonal_median"}:
        for idx in positions:
            out[idx] = None
        out, _filled = _linear_fill(out, max(3, seasonal_period or 3))
    return out


def _consecutive_runs(positions: list[int], min_run: int) -> list[int]:
    if not positions:
        return []
    out = []
    run = [positions[0]]
    for pos in positions[1:]:
        if pos == run[-1] + 1:
            run.append(pos)
        else:
            if len(run) >= min_run:
                out.append(run[0])
            run = [pos]
    if len(run) >= min_run:
        out.append(run[0])
    return out

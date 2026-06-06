"""Shared covariate preparation for forecasting tools."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import date
from typing import Any

from time_series_analysis_mcp.errors import ValidationError

from .schemas import SeriesCollection, TimeSeriesItem
from .utils import next_index_values, parse_float


@dataclass(frozen=True)
class CovariateBundle:
    """Aligned covariate values for one forecast target."""

    target_series_id: str
    known_future_names: list[str]
    past_only_names: list[str]
    history: dict[str, list[float]] = field(default_factory=dict)
    future_known: dict[str, list[float]] = field(default_factory=dict)
    sources: dict[str, str] = field(default_factory=dict)

    @property
    def all_names(self) -> list[str]:
        return [*self.known_future_names, *self.past_only_names]

    def history_rows(self, names: list[str] | None = None, *, end: int | None = None) -> list[list[float]]:
        selected = names if names is not None else self.all_names
        if not selected:
            return []
        history_end = self._history_end(end)
        return [
            [self.history[name][idx] for name in selected]
            for idx in range(history_end)
        ]

    def future_rows(self, *, history_end: int, prediction_length: int) -> list[list[float]]:
        if not self.known_future_names:
            return []
        values = self.future_values(history_end=history_end, prediction_length=prediction_length)
        return [
            [values[name][idx] for name in self.known_future_names]
            for idx in range(prediction_length)
        ]

    def future_values(self, *, history_end: int, prediction_length: int) -> dict[str, list[float]]:
        out: dict[str, list[float]] = {}
        for name in self.known_future_names:
            history = self.history[name]
            if history_end < len(history):
                values = history[history_end : history_end + prediction_length]
                if len(values) != prediction_length:
                    raise ValidationError(
                        "known-future covariates are missing holdout values",
                        target_series_id=self.target_series_id,
                        covariate=name,
                        expected=prediction_length,
                        actual=len(values),
                    )
            else:
                values = self.future_known.get(name, [])
                if len(values) != prediction_length:
                    raise ValidationError(
                        "known-future covariates must cover the forecast horizon",
                        target_series_id=self.target_series_id,
                        covariate=name,
                        expected=prediction_length,
                        actual=len(values),
                    )
            out[name] = values
        return out

    def metadata(self) -> dict[str, Any]:
        return {
            "known_future": list(self.known_future_names),
            "past_only": list(self.past_only_names),
            "sources": dict(self.sources),
        }

    def _history_end(self, end: int | None) -> int:
        if not self.history:
            return 0
        full_length = len(next(iter(self.history.values())))
        if end is None:
            return full_length
        if end < 0:
            return full_length + end
        return min(end, full_length)


def build_covariate_bundles(
    collection: SeriesCollection,
    targets: list[TimeSeriesItem],
    *,
    forecast_steps: int,
    known_future_covariate_metrics: list[str] | None = None,
    past_covariate_metrics: list[str] | None = None,
    future_covariates: dict[str, Any] | None = None,
    calendar_covariates: list[str] | None = None,
    allow_past_only: bool = True,
) -> dict[str, CovariateBundle]:
    known_metrics = _coerce_names(known_future_covariate_metrics, "known_future_covariate_metrics")
    past_metrics = _coerce_names(past_covariate_metrics, "past_covariate_metrics")
    calendar = _coerce_names(calendar_covariates, "calendar_covariates")
    duplicate_roles = sorted(set(known_metrics) & set(past_metrics))
    if duplicate_roles:
        raise ValidationError(
            "covariate metrics cannot be both known-future and past-only",
            covariates=duplicate_roles,
        )
    if past_metrics and not allow_past_only:
        raise ValidationError(
            "this forecasting method requires future values for every covariate; use known_future_covariate_metrics instead",
            past_covariate_metrics=past_metrics,
        )

    if not known_metrics and not past_metrics and not calendar:
        return {target.series_id: CovariateBundle(target.series_id, [], []) for target in targets}

    future_index = next_index_values(
        collection.time_index,
        collection.schema_.frequency,
        forecast_steps,
    )
    return {
        target.series_id: _build_target_bundle(
            collection,
            target,
            known_metrics=known_metrics,
            past_metrics=past_metrics,
            future_covariates=future_covariates,
            calendar=calendar,
            future_index=future_index,
            forecast_steps=forecast_steps,
        )
        for target in targets
    }


def covariate_metadata(bundles: dict[str, CovariateBundle]) -> dict[str, Any]:
    if not bundles:
        return {"known_future": [], "past_only": [], "sources": {}}
    known: list[str] = []
    past: list[str] = []
    sources: dict[str, str] = {}
    for bundle in bundles.values():
        for name in bundle.known_future_names:
            if name not in known:
                known.append(name)
        for name in bundle.past_only_names:
            if name not in past:
                past.append(name)
        sources.update(bundle.sources)
    return {"known_future": known, "past_only": past, "sources": sources}


def _build_target_bundle(
    collection: SeriesCollection,
    target: TimeSeriesItem,
    *,
    known_metrics: list[str],
    past_metrics: list[str],
    future_covariates: dict[str, Any] | None,
    calendar: list[str],
    future_index: list[str],
    forecast_steps: int,
) -> CovariateBundle:
    known_names: list[str] = []
    past_names: list[str] = []
    history: dict[str, list[float]] = {}
    future_known: dict[str, list[float]] = {}
    sources: dict[str, str] = {}

    for metric in known_metrics:
        item = _match_covariate_series(collection, target, metric)
        name = _available_name(metric, history)
        history[name] = _numeric_values(item.values, target_series_id=target.series_id, covariate=name)
        future_known[name] = _future_metric_values(
            future_covariates,
            target=target,
            covariate=item,
            feature_name=name,
            forecast_steps=forecast_steps,
        )
        known_names.append(name)
        sources[name] = f"metric:{item.series_id}"

    for feature in calendar:
        for name, hist_values, fut_values in _calendar_values(
            feature,
            collection.time_index,
            future_index,
        ):
            feature_name = _available_name(name, history)
            history[feature_name] = hist_values
            future_known[feature_name] = fut_values
            known_names.append(feature_name)
            sources[feature_name] = f"calendar:{feature}"

    for metric in past_metrics:
        item = _match_covariate_series(collection, target, metric)
        name = _available_name(metric, history)
        history[name] = _numeric_values(item.values, target_series_id=target.series_id, covariate=name)
        past_names.append(name)
        sources[name] = f"metric:{item.series_id}"

    expected_length = len(collection.time_index)
    for name, values in history.items():
        if len(values) != expected_length:
            raise ValidationError(
                "covariate history length must match the target time index",
                target_series_id=target.series_id,
                covariate=name,
                expected=expected_length,
                actual=len(values),
            )

    return CovariateBundle(
        target_series_id=target.series_id,
        known_future_names=known_names,
        past_only_names=past_names,
        history=history,
        future_known=future_known,
        sources=sources,
    )


def _coerce_names(values: list[str] | None, name: str) -> list[str]:
    if not values:
        return []
    out: list[str] = []
    for raw in values:
        value = str(raw).strip()
        if not value:
            raise ValidationError(f"{name} cannot contain empty values")
        if value not in out:
            out.append(value)
    return out


def _match_covariate_series(
    collection: SeriesCollection,
    target: TimeSeriesItem,
    metric: str,
) -> TimeSeriesItem:
    matches = [item for item in collection.series if item.metric == metric]
    if not matches:
        raise ValidationError(
            "covariate metric was not found in the series collection",
            target_series_id=target.series_id,
            covariate_metric=metric,
        )
    exact = [item for item in matches if item.dimensions == target.dimensions]
    if len(exact) == 1:
        return exact[0]
    global_matches = [item for item in matches if not item.dimensions]
    if len(global_matches) == 1:
        return global_matches[0]
    if len(matches) == 1:
        return matches[0]
    raise ValidationError(
        "covariate metric is ambiguous for the target dimensions",
        target_series_id=target.series_id,
        covariate_metric=metric,
        target_dimensions=target.dimensions,
        candidates=[item.series_id for item in matches],
    )


def _numeric_values(values: list[float | None], *, target_series_id: str, covariate: str) -> list[float]:
    out: list[float] = []
    for idx, value in enumerate(values):
        parsed = parse_float(value)
        if parsed is None or not math.isfinite(parsed):
            raise ValidationError(
                "covariate history must be complete and finite",
                target_series_id=target_series_id,
                covariate=covariate,
                position=idx,
            )
        out.append(float(parsed))
    return out


def _future_metric_values(
    payload: dict[str, Any] | None,
    *,
    target: TimeSeriesItem,
    covariate: TimeSeriesItem,
    feature_name: str,
    forecast_steps: int,
) -> list[float]:
    raw = _lookup_future_payload(payload, target=target, covariate=covariate, feature_name=feature_name)
    if raw is None:
        raise ValidationError(
            "future_covariates must provide values for every known-future covariate metric",
            target_series_id=target.series_id,
            covariate_metric=covariate.metric,
            covariate_series_id=covariate.series_id,
            expected=forecast_steps,
        )
    if isinstance(raw, dict) or isinstance(raw, str):
        raise ValidationError(
            "future covariate values must be a list of numeric values",
            target_series_id=target.series_id,
            covariate=feature_name,
        )
    try:
        values = list(raw)
    except TypeError as exc:
        raise ValidationError(
            "future covariate values must be a list of numeric values",
            target_series_id=target.series_id,
            covariate=feature_name,
        ) from exc
    if len(values) != forecast_steps:
        raise ValidationError(
            "future covariate values must cover exactly forecast_steps",
            target_series_id=target.series_id,
            covariate=feature_name,
            expected=forecast_steps,
            actual=len(values),
        )
    return _numeric_values(values, target_series_id=target.series_id, covariate=feature_name)


def _lookup_future_payload(
    payload: dict[str, Any] | None,
    *,
    target: TimeSeriesItem,
    covariate: TimeSeriesItem,
    feature_name: str,
) -> Any:
    if payload is None:
        return None
    target_payload = payload.get(target.series_id)
    if isinstance(target_payload, dict):
        nested = _lookup_keys(target_payload, covariate.series_id, covariate.metric, feature_name)
        if nested is not None:
            return nested
    direct = _lookup_keys(payload, covariate.series_id, covariate.metric, feature_name)
    if isinstance(direct, dict):
        return _lookup_keys(direct, target.series_id, covariate.series_id, "default", "__default__")
    return direct


def _lookup_keys(payload: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in payload:
            return payload[key]
    return None


def _calendar_values(
    feature: str,
    historical_index: list[str],
    future_index: list[str],
) -> list[tuple[str, list[float], list[float]]]:
    key = feature.strip().lower()
    historical_dates = [_date_from_stamp(stamp) for stamp in historical_index]
    future_dates = [_date_from_stamp(stamp) for stamp in future_index]
    if key in {"weekday", "day_of_week", "dow"}:
        return [
            (
                f"weekday_{weekday}",
                [1.0 if stamp.weekday() == weekday else 0.0 for stamp in historical_dates],
                [1.0 if stamp.weekday() == weekday else 0.0 for stamp in future_dates],
            )
            for weekday in range(7)
        ]
    if key in {"weekday_index", "day_of_week_index", "dow_index"}:
        return [
            (
                "weekday_index",
                [float(stamp.weekday()) for stamp in historical_dates],
                [float(stamp.weekday()) for stamp in future_dates],
            )
        ]
    if key in {"weekend", "is_weekend"}:
        return [
            (
                "is_weekend",
                [1.0 if stamp.weekday() >= 5 else 0.0 for stamp in historical_dates],
                [1.0 if stamp.weekday() >= 5 else 0.0 for stamp in future_dates],
            )
        ]
    if key == "month":
        return [
            (
                f"month_{month}",
                [1.0 if stamp.month == month else 0.0 for stamp in historical_dates],
                [1.0 if stamp.month == month else 0.0 for stamp in future_dates],
            )
            for month in range(1, 13)
        ]
    if key == "month_index":
        return [
            (
                "month_index",
                [float(stamp.month) for stamp in historical_dates],
                [float(stamp.month) for stamp in future_dates],
            )
        ]
    if key == "quarter":
        return [
            (
                f"quarter_{quarter}",
                [1.0 if _quarter(stamp) == quarter else 0.0 for stamp in historical_dates],
                [1.0 if _quarter(stamp) == quarter else 0.0 for stamp in future_dates],
            )
            for quarter in range(1, 5)
        ]
    raise ValidationError(
        "unsupported calendar covariate",
        calendar_covariate=feature,
        supported=[
            "weekday",
            "weekday_index",
            "weekend",
            "month",
            "month_index",
            "quarter",
        ],
    )


def _date_from_stamp(stamp: str) -> date:
    try:
        return date.fromisoformat(stamp[:10])
    except ValueError as exc:
        raise ValidationError("could not parse time index value for calendar covariates", timestamp=stamp) from exc


def _quarter(value: date) -> int:
    return (value.month - 1) // 3 + 1


def _available_name(preferred: str, existing: dict[str, list[float]]) -> str:
    if preferred not in existing:
        return preferred
    suffix = 2
    while f"{preferred}_{suffix}" in existing:
        suffix += 1
    return f"{preferred}_{suffix}"

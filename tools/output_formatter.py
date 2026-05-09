"""Simple JSON/markdown/text rendering for MCP tool outputs."""

from __future__ import annotations

import json
from typing import Any

from pydantic import BaseModel


def to_plain(value: Any) -> Any:
    if isinstance(value, BaseModel):
        return value.model_dump(by_alias=True)
    if isinstance(value, list):
        return [to_plain(item) for item in value]
    if isinstance(value, dict):
        return {key: to_plain(item) for key, item in value.items()}
    return value


def render(value: Any, output_format: str = "json") -> str:
    fmt = output_format.lower()
    plain = to_plain(value)
    if fmt == "json":
        return json.dumps(plain, indent=2, ensure_ascii=False)
    if fmt == "markdown":
        return to_markdown(plain)
    if fmt == "text":
        return str(plain)
    raise ValueError(f"unsupported output_format: {output_format!r}")


def to_markdown(payload: Any) -> str:
    if not isinstance(payload, dict):
        return f"```json\n{json.dumps(payload, indent=2, ensure_ascii=False)}\n```"
    lines = ["# Time Series Analysis Result", ""]
    if payload.get("ok") is True:
        lines.append("Status: ok")
        lines.append("")
    if "dataset" in payload:
        dataset = payload["dataset"]
        quality = dataset.get("quality", {})
        schema = dataset.get("schema", {})
        lines.extend(
            [
                "## Dataset",
                "",
                f"- Series: {quality.get('n_series')}",
                f"- Timestamps: {quality.get('n_timestamps')}",
                f"- Frequency: {schema.get('frequency')}",
                f"- Aggregation: {schema.get('aggregation')}",
                "",
            ]
        )
    for key, value in payload.items():
        if key in {"ok", "dataset"}:
            continue
        lines.extend([f"## {key.replace('_', ' ').title()}", ""])
        lines.append("```json")
        lines.append(json.dumps(value, indent=2, ensure_ascii=False))
        lines.append("```")
        lines.append("")
    return "\n".join(lines)

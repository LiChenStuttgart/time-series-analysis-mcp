"""Structured errors for MCP tools."""

from __future__ import annotations

import json
import traceback
from typing import Any


class ToolError(Exception):
    """Base class for structured tool errors."""

    code = "tool_error"

    def __init__(self, message: str, **details: Any) -> None:
        super().__init__(message)
        self.message = message
        self.details = details

    def to_payload(self) -> dict[str, Any]:
        return {
            "ok": False,
            "error": {
                "code": self.code,
                "message": self.message,
                "details": self.details,
            },
        }


class ValidationError(ToolError):
    """Input data did not meet the tool contract."""

    code = "validation_error"


class DependencyError(ToolError):
    """A method requires an optional dependency that is not installed."""

    code = "dependency_error"


class EstimationError(ToolError):
    """A model or diagnostic could not be estimated."""

    code = "estimation_error"


def format_exception(exc: BaseException, *, include_traceback: bool = False) -> str:
    if isinstance(exc, ToolError):
        payload = exc.to_payload()
    else:
        payload = {
            "ok": False,
            "error": {
                "code": type(exc).__name__,
                "message": str(exc),
                "details": {},
            },
        }
    if include_traceback:
        payload["error"]["traceback"] = traceback.format_exc()
    return json.dumps(payload, indent=2, ensure_ascii=False)

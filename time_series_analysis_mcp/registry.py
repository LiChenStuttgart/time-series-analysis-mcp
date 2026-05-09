"""Small tool registry used by the FastMCP server."""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from functools import wraps
from typing import Any

from .config import LIMITS
from .errors import format_exception

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    group: str
    handler: Callable[..., Any]


@dataclass
class Registry:
    tools: dict[str, ToolSpec] = field(default_factory=dict)

    def register(
        self,
        name: str,
        handler: Callable[..., Any],
        description: str,
        *,
        group: str,
    ) -> Callable[..., Any]:
        if name in self.tools:
            raise ValueError(f"duplicate tool name: {name!r}")
        wrapped = _wrap_errors(handler)
        self.tools[name] = ToolSpec(
            name=name,
            description=description,
            group=group,
            handler=wrapped,
        )
        return wrapped

    def groups(self) -> dict[str, list[ToolSpec]]:
        groups: dict[str, list[ToolSpec]] = {}
        for spec in self.tools.values():
            groups.setdefault(spec.group, []).append(spec)
        return groups

    def __iter__(self) -> Iterable[ToolSpec]:
        return iter(self.tools.values())

    def __len__(self) -> int:
        return len(self.tools)


REGISTRY = Registry()


def register(
    name: str,
    handler: Callable[..., Any],
    description: str,
    *,
    group: str,
) -> Callable[..., Any]:
    return REGISTRY.register(name, handler, description, group=group)


def _wrap_errors(func: Callable[..., Any]) -> Callable[..., Any]:
    @wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        try:
            return func(*args, **kwargs)
        except Exception as exc:  # noqa: BLE001
            log.exception("tool %s failed", getattr(func, "__name__", "?"))
            return format_exception(exc, include_traceback=LIMITS.debug)

    return wrapper

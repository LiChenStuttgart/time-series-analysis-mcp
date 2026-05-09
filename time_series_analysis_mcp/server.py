"""FastMCP server for time-series-analysis-mcp."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

from mcp.server.fastmcp import FastMCP

from . import __version__
from ._registrations import load_all
from .registry import REGISTRY

log = logging.getLogger(__name__)


def build_server() -> FastMCP:
    _configure_stdio_utf8()
    load_all()
    mcp = FastMCP("time-series-analysis-mcp")
    for spec in REGISTRY:
        mcp.tool(name=spec.name, description=spec.description)(spec.handler)

    @mcp.resource("guide://time-series-analysis")
    def time_series_guide() -> str:
        return _render_guide()

    @mcp.resource("guide://data-contracts")
    def data_contracts() -> str:
        return _load_static_resource("DATA_CONTRACTS.md")

    groups = REGISTRY.groups()
    log.info(
        "registered %d tools across %d groups: %s",
        len(REGISTRY),
        len(groups),
        ", ".join(f"{group}={len(tools)}" for group, tools in sorted(groups.items())),
    )
    return mcp


def _render_guide() -> str:
    lines = [
        "# time-series-analysis-mcp Tool Catalogue",
        "",
        f"Version {__version__} - {len(REGISTRY)} tools across {len(REGISTRY.groups())} groups.",
        "",
        "## Input conventions",
        "- Use `time_series_loader` to create a `SeriesCollection` for enterprise data.",
        "- Downstream tools accept `series_collection`, direct `data`, or file-based loader arguments.",
        "",
        "## Error payload",
        "Tool failures return a structured JSON object:",
        "```json",
        '{"ok": false, "error": {"code": "...", "message": "...", "details": {}}}',
        "```",
        "Set `TIME_SERIES_ANALYSIS_MCP_DEBUG=1` to include tracebacks.",
        "",
        "## Tools by group",
        "",
    ]
    for group, specs in sorted(REGISTRY.groups().items()):
        pretty = group.replace("_", " ").title()
        lines.append(f"### {pretty} ({len(specs)})")
        lines.append("")
        for spec in sorted(specs, key=lambda item: item.name):
            lines.append(f"- **`{spec.name}`** - {spec.description}")
        lines.append("")
    lines.append("See `guide://data-contracts` for the canonical `SeriesCollection` schema.")
    return "\n".join(lines)


def _load_static_resource(name: str) -> str:
    path = Path(__file__).resolve().parent.parent / "resources" / name
    if path.exists():
        return path.read_text(encoding="utf-8")
    return f"Resource {name!r} is not available."


def _configure_stdio_utf8() -> None:
    if sys.platform != "win32":
        return
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            try:
                reconfigure(encoding="utf-8")
            except Exception:  # noqa: BLE001
                pass


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        stream=sys.stderr,
    )
    log.info("time-series-analysis-mcp %s starting", __version__)
    build_server().run(transport="stdio")


if __name__ == "__main__":
    main()

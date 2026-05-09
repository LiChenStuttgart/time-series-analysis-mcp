# Architecture

The server follows three layers:

```text
time_series_analysis_mcp/  MCP server, CLI, registry, manifest, errors
tools/                     MCP adapter layer and output rendering
time_series_analysis/      Pure loading, data-quality, diagnostic, and forecasting code
```

Dependencies flow downward only:

- `time_series_analysis_mcp` may import `tools`.
- `tools` may import `time_series_analysis`.
- `time_series_analysis` must not import `tools` or server code.

Every public tool is registered in `time_series_analysis_mcp/_registrations.py`.
Adapters return JSON strings by default and raise normal Python exceptions;
the registry converts failures into structured JSON error payloads.

# Enterprise Operations

The server writes MCP protocol data only to stdout. Runtime logs go to stderr.

Supported environment limits:

- `TS_MCP_MAX_FILE_MB`
- `TS_MCP_MAX_ROWS`
- `TS_MCP_MAX_SERIES`
- `TS_MCP_MAX_TIMESTAMPS`
- `TIME_SERIES_ANALYSIS_MCP_DEBUG`
- `TS_MCP_DEBUG`

Set `TIME_SERIES_ANALYSIS_MCP_DEBUG=1` to include Python tracebacks in tool
error payloads during development.

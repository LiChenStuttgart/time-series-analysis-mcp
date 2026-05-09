# Testing

Run:

```bash
python -m pytest
```

The suite covers:

- registry and manifest shape;
- loader contract behavior;
- smoke calls for every registered tool;
- synthetic correctness checks for loader, missing data, outliers, diagnostics, and forecasts;
- MCP stdio protocol initialization and a real tool call.

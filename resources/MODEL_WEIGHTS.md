# Local Model Weights

The direct Chronos-2 and Toto 2.0 tools use ignored checkpoint files under the
repository-root `modelweights/` directory. The files are not committed, placed
in Python distributions, or downloaded during runtime inference.

## Pinned Models

| Model ID | Local directory | Upstream revision |
| --- | --- | --- |
| `amazon/chronos-2` | `modelweights/chronos-2` | `0f8a440441931157957e2be1a9bce66627d99c76` |
| `Datadog/Toto-2.0-22m` | `modelweights/toto-2.0-22m` | `685e4ae3e2be8d8998025e53dd98e7fdcb296a89` |

The tracked `MODEL_WEIGHTS_MANIFEST.json` is the source of truth for revision,
file size, and SHA-256 checksum information.

## Initial Setup

Install optional inference dependencies and download the pinned snapshots:

```bash
python -m pip install -e '.[chronos2,toto2]'
python scripts/manage_modelweights.py download
python scripts/manage_modelweights.py verify
```

Use another local root when needed:

```bash
export TS_MCP_MODEL_WEIGHTS_DIR=/path/to/modelweights
python scripts/manage_modelweights.py --weights-dir "$TS_MCP_MODEL_WEIGHTS_DIR" download
```

## Operations

```bash
python scripts/manage_modelweights.py status
python scripts/manage_modelweights.py verify
python scripts/manage_modelweights.py check-latest
python scripts/manage_modelweights.py download --force
```

`check-latest` only reports upstream revision differences. It never changes the
tracked manifest or installed weights.

To update a pinned model:

1. Review the upstream model revision, model card, license, and compatibility.
2. Update the revision, sizes, and SHA-256 values in the tracked manifest.
3. Pin compatible inference dependency versions in `pyproject.toml`.
4. Run `download --force`, `verify`, the full test suite, and real offline
   inference checks.
5. Commit only code, documentation, tests, and manifest changes. Never commit
   `modelweights/`.

## Runtime Rules

- Canonical manifest models must exist and verify locally.
- Canonical models never fall back to remote downloads.
- An explicit local model directory may be supplied through `model_id`.
- Non-vendored remote IDs require `TS_MCP_ALLOW_REMOTE_MODELS=1`.
- Forecast metadata records the canonical ID, revision, weight SHA-256, and
  model source.

Chronos-2 and Toto 2.0 weights are distributed upstream under Apache-2.0:

- https://huggingface.co/amazon/chronos-2
- https://huggingface.co/Datadog/Toto-2.0-22m

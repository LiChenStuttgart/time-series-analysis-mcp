"""Resolve and verify local model weights used by foundation-model tools."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from functools import cache
from pathlib import Path
from typing import Any

from .errors import DependencyError

_MANIFEST_NAME = "MODEL_WEIGHTS_MANIFEST.json"
_REQUIRED_LOCAL_FILES = ("config.json", "model.safetensors")


@dataclass(frozen=True)
class ResolvedModelAsset:
    """A model reference prepared for an upstream ``from_pretrained`` call."""

    model_id: str
    load_path: str
    source: str
    local_files_only: bool
    revision: str | None = None
    weights_sha256: str | None = None

    def inference_metadata(self) -> dict[str, Any]:
        metadata: dict[str, Any] = {
            "model_id": self.model_id,
            "model_source": self.source,
        }
        if self.revision:
            metadata["model_revision"] = self.revision
        if self.weights_sha256:
            metadata["weights_sha256"] = self.weights_sha256
        return metadata


def repository_root() -> Path:
    return Path(__file__).resolve().parent.parent


def default_model_weights_dir(configured: str | None = None) -> Path:
    if configured:
        return Path(configured).expanduser().resolve()
    return repository_root() / "modelweights"


def default_manifest_path() -> Path:
    repository_manifest = repository_root() / "resources" / _MANIFEST_NAME
    if repository_manifest.is_file():
        return repository_manifest
    return Path(__file__).resolve().with_name(_MANIFEST_NAME)


def load_model_weights_manifest(manifest_path: str | Path | None = None) -> dict[str, Any]:
    path = Path(manifest_path).resolve() if manifest_path else default_manifest_path()
    return _load_manifest(str(path))


@cache
def _load_manifest(path_text: str) -> dict[str, Any]:
    path = Path(path_text)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise DependencyError(
            "model weights manifest is not available",
            manifest_path=str(path),
        ) from exc
    except (OSError, json.JSONDecodeError) as exc:
        raise DependencyError(
            "model weights manifest could not be read",
            manifest_path=str(path),
            reason=str(exc),
        ) from exc
    if payload.get("schema_version") != 1 or not isinstance(payload.get("models"), dict):
        raise DependencyError(
            "model weights manifest has an unsupported structure",
            manifest_path=str(path),
        )
    return payload


def resolve_model_asset(
    model_id: str,
    *,
    weights_dir: str | None = None,
    allow_remote: bool = False,
    manifest_path: str | Path | None = None,
) -> ResolvedModelAsset:
    """Resolve canonical models to verified local weights.

    Canonical models declared in the manifest never fall back to the network.
    Non-vendored model IDs require ``allow_remote=True``.
    """

    manifest = load_model_weights_manifest(manifest_path)
    entry = manifest["models"].get(model_id)
    if entry is not None:
        root = default_model_weights_dir(weights_dir)
        model_path = _child_path(root, str(entry["directory"]))
        _verify_manifest_model(model_id, model_path, entry)
        weight_details = entry["files"]["model.safetensors"]
        return ResolvedModelAsset(
            model_id=model_id,
            load_path=str(model_path),
            source="local_modelweights",
            local_files_only=True,
            revision=str(entry["revision"]),
            weights_sha256=str(weight_details["sha256"]),
        )

    candidate = Path(model_id).expanduser()
    if candidate.is_dir():
        model_path = candidate.resolve()
        _verify_unmanaged_local_model(model_id, model_path)
        return ResolvedModelAsset(
            model_id=model_id,
            load_path=str(model_path),
            source="local_path",
            local_files_only=True,
        )

    if allow_remote:
        return ResolvedModelAsset(
            model_id=model_id,
            load_path=model_id,
            source="remote",
            local_files_only=False,
        )

    raise DependencyError(
        "model is not available in local modelweights and remote models are disabled",
        model_id=model_id,
        model_weights_dir=str(default_model_weights_dir(weights_dir)),
        allow_remote_models=False,
        enable_remote_models="set TS_MCP_ALLOW_REMOTE_MODELS=1",
    )


def injected_model_asset(model_id: str) -> ResolvedModelAsset:
    """Describe a caller-provided model object used by tests or embeddings."""

    return ResolvedModelAsset(
        model_id=model_id,
        load_path=model_id,
        source="injected",
        local_files_only=True,
    )


def _child_path(root: Path, relative: str) -> Path:
    resolved_root = root.expanduser().resolve()
    candidate = (resolved_root / relative).resolve()
    if not candidate.is_relative_to(resolved_root):
        raise DependencyError(
            "model weights manifest directory escapes the configured modelweights root",
            model_weights_dir=str(resolved_root),
            directory=relative,
        )
    return candidate


def _verify_manifest_model(model_id: str, model_path: Path, entry: dict[str, Any]) -> None:
    if not model_path.is_dir():
        raise DependencyError(
            "local model weights are not installed",
            model_id=model_id,
            model_path=str(model_path),
            install_command="python scripts/manage_modelweights.py download",
        )
    for name, details in entry["files"].items():
        _verify_file(model_id, model_path / name, details)


def _verify_unmanaged_local_model(model_id: str, model_path: Path) -> None:
    missing = [name for name in _REQUIRED_LOCAL_FILES if not (model_path / name).is_file()]
    if missing:
        raise DependencyError(
            "local model directory is missing required files",
            model_id=model_id,
            model_path=str(model_path),
            missing_files=missing,
        )


def _verify_file(model_id: str, path: Path, details: dict[str, Any]) -> None:
    if not path.is_file():
        raise DependencyError(
            "local model weights are incomplete",
            model_id=model_id,
            missing_file=str(path),
            install_command="python scripts/manage_modelweights.py download",
        )
    expected_size = int(details["size"])
    actual_size = path.stat().st_size
    if actual_size != expected_size:
        raise DependencyError(
            "local model weight file has an unexpected size",
            model_id=model_id,
            file=str(path),
            expected_size=expected_size,
            actual_size=actual_size,
            repair_command="python scripts/manage_modelweights.py download --force",
        )
    expected_sha256 = str(details["sha256"])
    actual_sha256 = _verified_sha256(
        str(path),
        actual_size,
        path.stat().st_mtime_ns,
        expected_sha256,
    )
    if actual_sha256 != expected_sha256:
        raise DependencyError(
            "local model weight file failed checksum verification",
            model_id=model_id,
            file=str(path),
            expected_sha256=expected_sha256,
            actual_sha256=actual_sha256,
            repair_command="python scripts/manage_modelweights.py download --force",
        )


@cache
def _verified_sha256(
    path_text: str,
    _size: int,
    _mtime_ns: int,
    _expected_sha256: str,
) -> str:
    digest = hashlib.sha256()
    with Path(path_text).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

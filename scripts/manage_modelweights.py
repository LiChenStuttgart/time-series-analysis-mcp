#!/usr/bin/env python3
"""Download and verify the repository-local foundation-model weights."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = REPOSITORY_ROOT / "resources" / "MODEL_WEIGHTS_MANIFEST.json"
DEFAULT_WEIGHTS_DIR = REPOSITORY_ROOT / "modelweights"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--weights-dir",
        default=os.environ.get("TS_MCP_MODEL_WEIGHTS_DIR"),
        help="modelweights destination; defaults to repository-root modelweights/",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("verify", "status", "check-latest"):
        child = subparsers.add_parser(command)
        child.add_argument("models", nargs="*", help="canonical model IDs or local directory names")
    download = subparsers.add_parser("download")
    download.add_argument("models", nargs="*", help="canonical model IDs or local directory names")
    download.add_argument("--force", action="store_true", help="replace files that already verify")
    return parser


def _manifest() -> dict[str, Any]:
    return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))


def _weights_dir(value: str | None) -> Path:
    return Path(value).expanduser().resolve() if value else DEFAULT_WEIGHTS_DIR


def _selected_models(manifest: dict[str, Any], requested: list[str]) -> list[tuple[str, dict[str, Any]]]:
    models = manifest["models"]
    if not requested:
        return list(models.items())
    aliases = {entry["directory"]: model_id for model_id, entry in models.items()}
    selected = []
    for name in requested:
        model_id = name if name in models else aliases.get(name)
        if model_id is None:
            raise ValueError(f"unknown model {name!r}; choose from {', '.join(models)}")
        selected.append((model_id, models[model_id]))
    return selected


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verify_file(path: Path, details: dict[str, Any]) -> str | None:
    if not path.is_file():
        return "missing"
    if path.stat().st_size != int(details["size"]):
        return f"wrong size ({path.stat().st_size}, expected {details['size']})"
    actual = _sha256(path)
    if actual != details["sha256"]:
        return f"wrong sha256 ({actual})"
    return None


def _verify_model(model_id: str, entry: dict[str, Any], root: Path) -> list[str]:
    model_path = root / entry["directory"]
    errors = []
    for name, details in entry["files"].items():
        problem = _verify_file(model_path / name, details)
        if problem:
            errors.append(f"{model_id}: {name}: {problem}")
    return errors


def _download_file(url: str, destination: Path, details: dict[str, Any]) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".part")
    digest = hashlib.sha256()
    written = 0
    try:
        with urllib.request.urlopen(url) as response, temporary.open("wb") as handle:
            while chunk := response.read(1024 * 1024):
                handle.write(chunk)
                digest.update(chunk)
                written += len(chunk)
        if written != int(details["size"]) or digest.hexdigest() != details["sha256"]:
            raise RuntimeError(
                f"downloaded file verification failed: size={written}, sha256={digest.hexdigest()}"
            )
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)


def _download_model(model_id: str, entry: dict[str, Any], root: Path, *, force: bool) -> None:
    destination = root / entry["directory"]
    for name, details in entry["files"].items():
        path = destination / name
        if not force and _verify_file(path, details) is None:
            print(f"{model_id}: {name}: already verified")
            continue
        url = f"{entry['source_url']}/resolve/{entry['revision']}/{name}?download=true"
        print(f"{model_id}: downloading {name} ({details['size']} bytes)")
        _download_file(url, path, details)
        print(f"{model_id}: {name}: verified")


def _latest_revision(model_id: str) -> str:
    url = f"https://huggingface.co/api/models/{model_id}"
    with urllib.request.urlopen(url) as response:
        return str(json.load(response)["sha"])


def main() -> int:
    args = _parser().parse_args()
    manifest = _manifest()
    try:
        selected = _selected_models(manifest, args.models)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    root = _weights_dir(args.weights_dir)
    try:
        if args.command == "download":
            for model_id, entry in selected:
                _download_model(model_id, entry, root, force=args.force)
            return 0
        if args.command in {"verify", "status"}:
            failures = []
            for model_id, entry in selected:
                errors = _verify_model(model_id, entry, root)
                failures.extend(errors)
                print(f"{model_id}: {'invalid' if errors else 'verified'}")
                for error in errors:
                    print(f"  {error}")
            return 1 if failures else 0
        if args.command == "check-latest":
            changed = False
            for model_id, entry in selected:
                latest = _latest_revision(model_id)
                current = entry["revision"]
                state = "current" if latest == current else "update available"
                changed = changed or latest != current
                print(f"{model_id}: {state}; manifest={current}; upstream={latest}")
            return 1 if changed else 0
    except (OSError, RuntimeError, urllib.error.URLError) as exc:
        print(f"{args.command} failed: {exc}", file=sys.stderr)
        return 1
    return 2


if __name__ == "__main__":
    raise SystemExit(main())

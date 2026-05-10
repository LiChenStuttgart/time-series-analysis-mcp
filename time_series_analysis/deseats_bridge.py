"""Bridge to the external R ``deseats`` package."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from time_series_analysis_mcp.config import LIMITS
from time_series_analysis_mcp.errors import DependencyError, EstimationError


class DeseatsBridge:
    """Run DeSeaTS work in a short-lived R subprocess."""

    def __init__(
        self,
        *,
        rscript_path: str | None = None,
        runner_path: Path | None = None,
        timeout_seconds: int | None = None,
    ) -> None:
        self.rscript_path = rscript_path or LIMITS.rscript_path
        self.runner_path = runner_path or Path(__file__).resolve().parent / "r" / "deseats_runner.R"
        self.timeout_seconds = timeout_seconds or LIMITS.deseats_timeout_seconds

    def run(self, payload: dict[str, Any]) -> dict[str, Any]:
        self._ensure_ready()
        with tempfile.NamedTemporaryFile("w", suffix=".json", encoding="utf-8", delete=False) as handle:
            json.dump(payload, handle, ensure_ascii=False)
            input_path = handle.name
        try:
            completed = subprocess.run(
                [self._resolved_rscript(), str(self.runner_path), input_path],
                check=False,
                capture_output=True,
                text=True,
                timeout=self.timeout_seconds,
                env=self._env(),
            )
        except subprocess.TimeoutExpired as exc:
            raise EstimationError(
                "DeSeaTS R subprocess timed out",
                timeout_seconds=self.timeout_seconds,
            ) from exc
        finally:
            try:
                os.unlink(input_path)
            except FileNotFoundError:
                pass

        stderr = completed.stderr.strip()
        if completed.returncode != 0:
            raise EstimationError(
                "DeSeaTS R subprocess failed",
                returncode=completed.returncode,
                stderr=stderr,
            )
        try:
            result = json.loads(completed.stdout)
        except json.JSONDecodeError as exc:
            raise EstimationError(
                "DeSeaTS R subprocess returned invalid JSON",
                stdout=completed.stdout[-2000:],
                stderr=stderr,
            ) from exc
        if stderr:
            result.setdefault("warnings", []).append(stderr)
        return result

    def dependency_status(self) -> dict[str, Any]:
        rscript = self._resolved_rscript(required=False)
        if rscript is None:
            return {
                "ok": False,
                "missing": ["Rscript"],
                "rscript_path": self.rscript_path,
            }
        if not self.runner_path.exists():
            return {
                "ok": False,
                "missing": ["deseats_runner.R"],
                "runner_path": str(self.runner_path),
            }
        probe = (
            'pkgs <- c("deseats", "jsonlite"); '
            'missing <- pkgs[!vapply(pkgs, requireNamespace, logical(1), quietly = TRUE)]; '
            'cat(paste(missing, collapse = ","))'
        )
        try:
            completed = subprocess.run(
                [rscript, "-e", probe],
                check=False,
                capture_output=True,
                text=True,
                timeout=min(self.timeout_seconds, 30),
                env=self._env(),
            )
        except subprocess.TimeoutExpired:
            return {
                "ok": False,
                "missing": ["dependency probe timeout"],
                "rscript_path": rscript,
            }
        if completed.returncode != 0:
            return {
                "ok": False,
                "missing": ["R dependency probe failed"],
                "rscript_path": rscript,
                "stderr": completed.stderr.strip(),
            }
        missing = [item for item in completed.stdout.strip().split(",") if item]
        return {
            "ok": not missing,
            "missing": missing,
            "rscript_path": rscript,
            "runner_path": str(self.runner_path),
        }

    def _ensure_ready(self) -> None:
        status = self.dependency_status()
        if status["ok"]:
            return
        missing = status.get("missing", [])
        raise DependencyError(
            "DeSeaTS requires Rscript plus the R packages 'deseats' and 'jsonlite'",
            missing=missing,
            rscript_path=status.get("rscript_path", self.rscript_path),
            runner_path=status.get("runner_path", str(self.runner_path)),
            install_commands=[
                'Rscript -e \'install.packages("jsonlite", repos="https://cloud.r-project.org")\'',
                "R CMD INSTALL /Users/lchen/github_phd/deseats",
            ],
            stderr=status.get("stderr"),
        )

    def _resolved_rscript(self, *, required: bool = True) -> str | None:
        if os.path.isabs(self.rscript_path):
            exists = Path(self.rscript_path).exists()
            if exists:
                return self.rscript_path
            if required:
                raise DependencyError("Rscript executable not found", rscript_path=self.rscript_path)
            return None
        resolved = shutil.which(self.rscript_path)
        if resolved:
            return resolved
        if required:
            raise DependencyError("Rscript executable not found", rscript_path=self.rscript_path)
        return None

    def _env(self) -> dict[str, str]:
        env = os.environ.copy()
        if LIMITS.r_libs_user:
            env["R_LIBS_USER"] = LIMITS.r_libs_user
        env["TS_MCP_DESEATS_BOOT_CORES"] = str(max(LIMITS.deseats_boot_cores, 1))
        return env

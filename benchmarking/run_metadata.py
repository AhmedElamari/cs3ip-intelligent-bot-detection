"""Helpers for benchmark run metadata."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from importlib import metadata
import os
import platform
from pathlib import Path
import shlex
import subprocess
import sys
import time
from datetime import datetime, timezone
from typing import Any


RUN_METADATA_FILENAME = "run_metadata.json"
RUN_METADATA_SCHEMA_VERSION = "RunMetadataV1"
ENVIRONMENT_FREEZE_FILENAME = "environment_freeze.txt"
DATASET_FILES = (
    ("train", "train.json"),
    ("val", "dev.json"),
    ("test", "test.json"),
)
PACKAGE_DISTRIBUTIONS = {
    "pandas": "pandas",
    "numpy": "numpy",
    "scikit-learn": "scikit-learn",
    "imbalanced-learn": "imbalanced-learn",
    "xgboost": "xgboost",
    "matplotlib": "matplotlib",
    "seaborn": "seaborn",
    "shap": "shap",
    "lime": "lime",
    "pyyaml": "PyYAML",
    "torch": "torch",
    "pytorch-tabnet": "pytorch-tabnet",
    "optuna": "optuna",
}


@dataclass
class BenchmarkRunContext:
    """Context needed to write benchmark run metadata."""

    argv: list[str]
    args: dict[str, Any]
    config_path: str | None
    repo_root: Path
    data_dir: Path
    output_dir: Path
    explainability: dict[str, Any]
    script_path: Path | None = None
    cwd: Path | None = None
    runtime: dict[str, Any] | None = None
    model_runtime_metadata: dict[str, Any] | None = None
    multi_seed_summary: dict[str, Any] | None = None
    run_start_perf: float | None = None


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_environment_freeze(output_dir: Path) -> dict[str, Any]:
    """Run ``pip freeze --all`` into ``environment_freeze.txt``; return metadata."""
    path = output_dir / ENVIRONMENT_FREEZE_FILENAME
    meta: dict[str, Any] = {
        "file": ENVIRONMENT_FREEZE_FILENAME,
        "sha256": None,
        "error": None,
    }
    try:
        result = subprocess.run(
            [sys.executable, "-m", "pip", "freeze", "--all"],
            capture_output=True,
            text=True,
            check=False,
            timeout=300,
        )
        if result.returncode != 0:
            msg = (result.stderr or result.stdout or "pip freeze failed").strip()
            meta["error"] = msg[:2000]
            return meta
        path.write_text(result.stdout, encoding="utf-8")
        meta["sha256"] = _sha256_file(path)
    except subprocess.TimeoutExpired:
        meta["error"] = "pip freeze timed out"
    except OSError as exc:
        meta["error"] = str(exc)[:2000]
    return meta


def _command_tokens(argv: list[str], script_path: Path) -> list[str]:
    return [sys.executable, str(script_path.resolve()), *argv]


def _command_line(tokens: list[str]) -> str:
    if sys.platform == "win32":
        return subprocess.list2cmdline(tokens)
    return shlex.join(tokens)


def _resolve_script_path(repo_root: Path, script_path: Path | None) -> Path:
    if script_path is not None:
        return Path(script_path)
    return Path(repo_root) / "run_benchmark.py"


def _hardware_metadata() -> dict[str, Any]:
    warnings: list[str] = []
    out: dict[str, Any] = {
        "processor": platform.processor() or None,
        "cpu_count_logical": os.cpu_count(),
        "memory_total_bytes": None,
        "torch_version": None,
        "cuda_available": None,
        "cuda_version": None,
        "cudnn_version": None,
        "gpus": [],
        "warnings": warnings,
    }
    try:
        import psutil  # type: ignore[import-untyped]

        out["memory_total_bytes"] = int(psutil.virtual_memory().total)
    except ImportError:
        pass
    except Exception as exc:
        warnings.append(f"psutil: {exc}")

    try:
        import torch

        out["torch_version"] = torch.__version__
        out["cuda_available"] = bool(torch.cuda.is_available())
        out["cuda_version"] = getattr(torch.version, "cuda", None)
        if torch.backends.cudnn.is_available():
            try:
                out["cudnn_version"] = str(torch.backends.cudnn.version())
            except Exception as exc:
                warnings.append(f"cudnn_version: {exc}")
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                out["gpus"].append(
                    {
                        "index": i,
                        "name": props.name,
                        "total_memory_bytes": int(props.total_memory),
                        "compute_capability": f"{props.major}.{props.minor}",
                    }
                )
    except ImportError:
        out["cuda_available"] = None
    except Exception as exc:
        warnings.append(f"torch: {exc}")

    return out


def write_run_metadata(context: BenchmarkRunContext) -> Path:
    """Write the benchmark run metadata after all other artifacts exist."""
    metadata_path = context.output_dir / RUN_METADATA_FILENAME
    freeze_meta = write_environment_freeze(context.output_dir)
    cwd = context.cwd or Path.cwd()
    script = _resolve_script_path(context.repo_root, context.script_path)
    tokens = _command_tokens(context.argv, script)

    invocation = {
        "cwd": str(cwd.resolve()),
        "repo_root": str(Path(context.repo_root).resolve()),
        "script": str(script.resolve()),
        "argv": list(context.argv),
        "command_tokens": tokens,
        "command": _command_line(tokens),
        "args": dict(context.args),
        "config_path": context.config_path,
        **context.explainability,
    }

    payload_warnings: list[str] = []
    if freeze_meta.get("error"):
        payload_warnings.append(f"environment_freeze: {freeze_meta['error']}")

    runtime_out: dict[str, Any] | None = None
    if context.runtime is not None:
        runtime_out = dict(context.runtime)
        runtime_out["ended_at_utc"] = (
            datetime.now(timezone.utc).replace(microsecond=0).isoformat()
        )
        if context.run_start_perf is not None:
            runtime_out["total_seconds"] = round(
                time.perf_counter() - context.run_start_perf, 6
            )

    payload = {
        "schema_version": RUN_METADATA_SCHEMA_VERSION,
        "python": _python_metadata(),
        "platform": _platform_metadata(),
        "hardware": _hardware_metadata(),
        "git": _git_metadata(context.repo_root),
        "invocation": invocation,
        "runtime": runtime_out,
        "dataset": _dataset_metadata(context.data_dir),
        "packages": _package_versions(),
        "environment": freeze_meta,
        "artifacts": {
            "output_dir": str(context.output_dir.resolve()),
            "files": sorted(
                {
                    path.name
                    for path in context.output_dir.iterdir()
                    if path.is_file()
                }
                | {RUN_METADATA_FILENAME}
            ),
        },
    }
    if context.model_runtime_metadata:
        payload["models_runtime"] = dict(context.model_runtime_metadata)
    if context.multi_seed_summary:
        payload["multi_seed"] = dict(context.multi_seed_summary)
    if payload_warnings:
        payload["warnings"] = payload_warnings

    metadata_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return metadata_path


def _python_metadata() -> dict[str, str]:
    return {
        "version": platform.python_version(),
        "full_version": sys.version,
        "implementation": platform.python_implementation(),
        "executable": sys.executable,
    }


def _platform_metadata() -> dict[str, str]:
    return {
        "platform": platform.platform(),
        "system": platform.system(),
        "release": platform.release(),
        "machine": platform.machine(),
    }


def _git_metadata(repo_root: Path) -> dict[str, Any]:
    commit = _run_git(repo_root, "rev-parse", "HEAD")
    branch = _run_git(repo_root, "rev-parse", "--abbrev-ref", "HEAD")
    dirty_state = _run_git(repo_root, "status", "--short")
    if commit is None and branch is None and dirty_state is None:
        return {
            "commit": None,
            "branch": None,
            "dirty": None,
        }

    return {
        "commit": commit or "unknown",
        "branch": branch or "unknown",
        "dirty": bool(dirty_state),
    }


def _run_git(repo_root: Path, *args: str) -> str | None:
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=repo_root,
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError:
        return None

    if result.returncode != 0:
        return None
    return result.stdout.strip()


def _dataset_metadata(data_dir: Path) -> dict[str, Any]:
    root = Path(data_dir).resolve()
    combined_hash = hashlib.sha256()
    file_metadata: dict[str, dict[str, Any]] = {}
    missing_files = False

    for split_name, filename in DATASET_FILES:
        metadata_row = _hash_dataset_file(root / filename, combined_hash)
        file_metadata[split_name] = {
            "file": filename,
            **metadata_row,
        }
        missing_files = missing_files or not metadata_row["exists"]

    return {
        "root": str(root),
        "combined_sha256": None if missing_files else combined_hash.hexdigest(),
        "files": file_metadata,
    }


def _hash_dataset_file(path: Path, combined_hash: Any) -> dict[str, Any]:
    if not path.exists():
        return {
            "exists": False,
            "sha256": None,
            "bytes": None,
        }

    file_hash = hashlib.sha256()
    total_bytes = 0
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            total_bytes += len(chunk)
            file_hash.update(chunk)
            combined_hash.update(chunk)

    return {
        "exists": True,
        "sha256": file_hash.hexdigest(),
        "bytes": total_bytes,
    }


def _package_versions() -> dict[str, str]:
    versions = {}
    for package_name, distribution_name in PACKAGE_DISTRIBUTIONS.items():
        if version := _distribution_version(distribution_name):
            versions[package_name] = version
    return versions


def _distribution_version(distribution_name: str) -> str | None:
    try:
        return metadata.version(distribution_name)
    except metadata.PackageNotFoundError:
        return None

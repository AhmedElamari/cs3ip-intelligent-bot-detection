"""Helpers for benchmark run metadata."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from importlib import metadata
import platform
from pathlib import Path
import subprocess
import sys
from typing import Any


RUN_METADATA_FILENAME = "run_metadata.json"
RUN_METADATA_SCHEMA_VERSION = "RunMetadataV1"
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


def write_run_metadata(context: BenchmarkRunContext) -> Path:
    """Write the benchmark run metadata after all other artifacts exist."""
    metadata_path = context.output_dir / RUN_METADATA_FILENAME
    artifact_files = sorted(
        {
            path.name
            for path in context.output_dir.iterdir()
            if path.is_file()
        }
        | {RUN_METADATA_FILENAME}
    )
    payload = {
        "schema_version": RUN_METADATA_SCHEMA_VERSION,
        "python": _python_metadata(),
        "platform": _platform_metadata(),
        "git": _git_metadata(context.repo_root),
        "invocation": {
            "argv": list(context.argv),
            "args": dict(context.args),
            "config_path": context.config_path,
            **context.explainability,
        },
        "dataset": _dataset_metadata(context.data_dir),
        "packages": _package_versions(),
        "artifacts": {
            "output_dir": str(context.output_dir.resolve()),
            "files": artifact_files,
        },
    }
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

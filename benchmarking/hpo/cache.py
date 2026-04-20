"""HPO cache signature (SHA-256) and atomic JSON writes."""
from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any

from config import Config
from benchmarking.run_metadata import _dataset_metadata


def _effective_preprocessing_subset(cfg_dict: dict[str, Any]) -> dict[str, Any]:
    prep = cfg_dict.get("preprocessing") or {}
    return {
        "preprocessing": prep,
        "time_split": cfg_dict.get("time_split"),
        "test_size": cfg_dict.get("test_size"),
        "val_size": cfg_dict.get("val_size"),
    }


def _feature_selection_fingerprint(prep: dict[str, Any]) -> dict[str, Any]:
    fs = prep.get("feature_selection")
    if isinstance(fs, dict):
        return {
            "enabled": fs.get("enabled", True),
            "n_features": fs.get("n_features", prep.get("n_features")),
        }
    return {
        "enabled": bool(fs),
        "n_features": prep.get("n_features"),
    }


def compute_signature(
    model_name: str,
    config: Config,
    feature_names_ordered: list[str],
    data_dir: Path,
    search_space_version: str,
    metric: str = "val_f1",
) -> str:
    cfg_dict = config.to_dict()
    prep = cfg_dict.get("preprocessing") or {}
    ds = _dataset_metadata(Path(data_dir))
    combined = ds.get("combined_sha256") or ""
    payload: dict[str, Any] = {
        "dataset_fingerprint": combined,
        "feature_names_ordered": list(feature_names_ordered),
        "feature_selection": _feature_selection_fingerprint(prep),
        "metric": metric,
        "model_name": model_name,
        "model_params": config.get_model_params(model_name),
        "random_state": cfg_dict.get("random_state"),
        "search_space_version": search_space_version,
        "effective_preprocessing": _effective_preprocessing_subset(cfg_dict),
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def cache_path_for_signature(
    cache_dir: Path,
    model_name: str,
    signature_hex: str,
) -> Path:
    return Path(cache_dir) / model_name / f"{signature_hex}.json"


def atomic_write_json(path: Path, data: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(
        suffix=".json",
        dir=str(path.parent),
        text=True,
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(data, handle, indent=2)
        os.replace(tmp, path)
    except BaseException:
        if os.path.isfile(tmp):
            os.unlink(tmp)
        raise


def read_hpo_json(path: Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)

"""TabNet HPO entry point; delegates to ``benchmarking.hpo.service.optimize_model``."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np

from config import Config


def _compute_virtual_batch_size(batch_size: int, ratio: int) -> int:
    return max(8, batch_size // ratio)


def optimize_tabnet(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    n_trials: int = 50,
    seed: int = 2112,
    output_path: Optional[Path] = None,
    device: str = "auto",
) -> dict:
    """Run Optuna HPO for TabNet; return HPOResultV1 dict."""
    from benchmarking.hpo.service import optimize_model

    result = optimize_model(
        "tabnet",
        X_train,
        y_train,
        X_val,
        y_val,
        config=Config(),
        n_trials=n_trials,
        seed=seed,
        enable_scaling=False,
        class_weights=None,
        feature_names=None,
        output_path=output_path,
        device=device,
    )
    print(
        f"TabNet HPO complete — best val_f1={result['best_score']:.4f} "
        f"({result['trial_count']} trials, device={result.get('device', 'cpu')})"
    )
    if output_path is not None:
        print(f"HPO result saved to: {output_path}")
    return result


def load_hpo_result(path: Path) -> dict:
    """Load and validate an HPOResultV1 artifact from disk."""
    with open(path, encoding="utf-8") as handle:
        data = json.load(handle)
    if data.get("schema_version") != "HPOResultV1":
        raise ValueError(
            f"Expected HPOResultV1 schema, got: {data.get('schema_version')!r}"
        )
    return data


def best_params_for_tabnetmodel(hpo_result: dict) -> dict:
    """Extract TabNetModel constructor kwargs from an HPOResultV1 result."""
    params = dict(hpo_result.get("best_params", {}))
    ratio = params.pop("virtual_batch_size_ratio", None)
    if "virtual_batch_size" not in params and ratio is not None:
        params["virtual_batch_size"] = _compute_virtual_batch_size(
            params.get("batch_size", 1024), ratio
        )
    return params

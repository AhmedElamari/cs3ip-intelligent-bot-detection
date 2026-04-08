"""
TabNet Hyperparameter Optimisation with Optuna
===============================================
Searches for the best TabNet configuration using Optuna (TPE sampler +
MedianPruner).  Persists the winner as an HPOResultV1 JSON artifact.

Usage:
    from benchmarking.tabnet_optuna import optimize_tabnet
    result = optimize_tabnet(X_train, y_train, X_val, y_val)

Install:  pip install -r requirements-dl.txt
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Optional

import numpy as np

# ---------------------------------------------------------------------------
# Dependency guards
# ---------------------------------------------------------------------------

def _require_optuna():
    try:
        import optuna  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "optuna is not installed. Install with: pip install -r requirements-dl.txt"
        ) from exc


def _require_tabnet():
    try:
        from pytorch_tabnet.tab_model import TabNetClassifier  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "pytorch-tabnet is not installed. Install with: pip install -r requirements-dl.txt"
        ) from exc


# ---------------------------------------------------------------------------
# Search space
# ---------------------------------------------------------------------------

def _compute_virtual_batch_size(batch_size: int, ratio: int) -> int:
    """Derive virtual_batch_size from batch_size and ratio (single source of truth)."""
    return max(8, batch_size // ratio)


def _suggest_params(trial) -> dict:
    """Define the TabNet hyperparameter search space."""
    n_d = trial.suggest_int("n_d", 8, 64, step=8)
    # n_a typically equals n_d; keep them tied to reduce search space
    n_a = n_d

    batch_size = trial.suggest_categorical("batch_size", [256, 512, 1024, 2048])
    ratio = trial.suggest_categorical("virtual_batch_size_ratio", [2, 4, 8, 16])
    vbs = _compute_virtual_batch_size(batch_size, ratio)

    return {
        "n_d": n_d,
        "n_a": n_a,
        "n_steps": trial.suggest_int("n_steps", 1, 6),
        "gamma": trial.suggest_float("gamma", 1.0, 2.0),
        "lambda_sparse": trial.suggest_float("lambda_sparse", 1e-5, 1e-2, log=True),
        "learning_rate": trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True),
        "batch_size": batch_size,
        "virtual_batch_size": vbs,
        "momentum": trial.suggest_float("momentum", 0.01, 0.4, log=True),
        "mask_type": trial.suggest_categorical("mask_type", ["sparsemax", "entmax"]),
    }


# ---------------------------------------------------------------------------
# Objective
# ---------------------------------------------------------------------------

def _build_objective(X_train, y_train, X_val, y_val, seed: int, device: str):
    """Return an Optuna objective function closing over the data splits."""
    import optuna
    from pytorch_tabnet.tab_model import TabNetClassifier
    from sklearn.metrics import f1_score

    def objective(trial) -> float:
        params = _suggest_params(trial)
        n_samples = len(X_train)
        batch_size = min(params["batch_size"], n_samples)
        vbs = min(params["virtual_batch_size"], batch_size)
        while batch_size % vbs != 0 and vbs > 1:
            vbs -= 1
        vbs = max(1, vbs)

        clf = TabNetClassifier(
            n_d=params["n_d"],
            n_a=params["n_a"],
            n_steps=params["n_steps"],
            gamma=params["gamma"],
            lambda_sparse=params["lambda_sparse"],
            optimizer_params={"lr": params["learning_rate"]},
            momentum=params["momentum"],
            mask_type=params["mask_type"],
            device_name=device,
            seed=seed,
            verbose=0,
        )
        clf.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            eval_name=["val"],
            max_epochs=200,
            patience=20,
            batch_size=batch_size,
            virtual_batch_size=vbs,
        )
        y_pred = clf.predict(X_val)
        score = f1_score(y_val, y_pred, average="binary", zero_division=0)

        trial.report(score, step=0)
        if trial.should_prune():
            raise optuna.TrialPruned()
        return score

    return objective


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

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
    """Run Optuna HPO for TabNet; return HPOResultV1-compliant dict.

    Args:
        X_train: float32 training features (after TabNetPrep).
        y_train: int binary training labels.
        X_val: float32 validation features.
        y_val: int binary validation labels.
        n_trials: Number of Optuna trials (balanced default: 50).
        seed: Random seed (should be 2112 to satisfy project reproducibility).
        output_path: If given, save JSON artifact to this path.
        device: 'auto' | 'cpu' | 'cuda'.

    Returns:
        HPOResultV1 dict.
    """
    _require_optuna()
    _require_tabnet()

    import optuna
    import torch

    # Resolve device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # Silence verbose optuna output
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    sampler = optuna.samplers.TPESampler(seed=seed)
    pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=0)
    study = optuna.create_study(
        direction="maximize",
        sampler=sampler,
        pruner=pruner,
    )

    X_tr = np.asarray(X_train, dtype=np.float32)
    y_tr = np.asarray(y_train, dtype=int)
    X_v = np.asarray(X_val, dtype=np.float32)
    y_v = np.asarray(y_val, dtype=int)

    warnings_list: list[str] = []
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        study.optimize(
            _build_objective(X_tr, y_tr, X_v, y_v, seed, device),
            n_trials=n_trials,
            show_progress_bar=False,
        )
        warnings_list = [str(w.message) for w in caught]

    best = study.best_trial
    best_params = dict(best.params)
    ratio = best_params.get("virtual_batch_size_ratio")
    if ratio is not None:
        best_params["virtual_batch_size"] = _compute_virtual_batch_size(
            best_params.get("batch_size", 1024), ratio
        )
    result: dict = {
        "schema_version": "HPOResultV1",
        "status": "ok",
        "best_params": best_params,
        "best_score": best.value,
        "trial_count": len(study.trials),
        "metric": "val_f1",
        "seed": seed,
        "device": device,
        "warnings": warnings_list[:10],  # cap to avoid bloating JSON
    }

    if output_path is not None:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(result, f, indent=2)
        print(f"HPO result saved to: {path}")

    print(
        f"TabNet HPO complete — best val_f1={best.value:.4f} "
        f"({len(study.trials)} trials, device={device})"
    )
    return result


def load_hpo_result(path: Path) -> dict:
    """Load and validate an HPOResultV1 artifact from disk."""
    with open(path) as f:
        data = json.load(f)
    if data.get("schema_version") != "HPOResultV1":
        raise ValueError(
            f"Expected HPOResultV1 schema, got: {data.get('schema_version')!r}"
        )
    return data


def best_params_for_tabnetmodel(hpo_result: dict) -> dict:
    """Extract TabNetModel constructor kwargs from an HPOResultV1 result.

    Drops ``virtual_batch_size_ratio`` (search-space helper only) and keeps
    ``learning_rate`` so callers can reproduce the HPO winner configuration.
    Recomputes ``virtual_batch_size`` from ratio when missing (backward compat
    for artifacts saved before the HPO fix). Callers can safely do
    ``TabNetModel(**best_params_for_tabnetmodel(result))``.

    Args:
        hpo_result: HPOResultV1 dict from optimize_tabnet or load_hpo_result.

    Returns:
        Cleaned kwargs dict ready for TabNetModel(**kwargs).
    """
    params = dict(hpo_result.get("best_params", {}))
    # Keep learning_rate for TabNetModel.optimizer_params
    ratio = params.pop("virtual_batch_size_ratio", None)
    if "virtual_batch_size" not in params and ratio is not None:
        params["virtual_batch_size"] = _compute_virtual_batch_size(
            params.get("batch_size", 1024), ratio
        )
    return params

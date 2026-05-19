"""Default HPO registry entries."""
from __future__ import annotations

from typing import Any, Callable, Optional

import numpy as np

from benchmarking.hpo.contracts import HPOEntry
from benchmarking.hpo.registry import register
from benchmarking.metrics import MetricsCalculator

SEARCH_SPACE_VERSION = "v1"


def _val_f1_score(model: Any, X_val: np.ndarray, y_val: np.ndarray) -> float:
    # HPO never optimizes test — val F1 is the selection objective.
    y_pred = model.predict(X_val)
    y_proba = (
        model.predict_proba(X_val) if hasattr(model, "predict_proba") else None
    )
    m = MetricsCalculator().compute_all_metrics(
        np.asarray(y_val), np.asarray(y_pred), y_proba
    )
    return float(m["f1"])


def _suggest_lr(trial: Any) -> dict[str, Any]:
    return {
        "C": trial.suggest_float("C", 1e-4, 10.0, log=True),
        "solver": trial.suggest_categorical(
            "solver", ["lbfgs", "liblinear", "saga"]
        ),
    }


def _suggest_svm(trial: Any) -> dict[str, Any]:
    kernel = trial.suggest_categorical("kernel", ["rbf", "linear", "poly"])
    out: dict[str, Any] = {
        "C": trial.suggest_float("C", 1e-3, 100.0, log=True),
        "kernel": kernel,
    }
    if kernel == "rbf":
        mode = trial.suggest_categorical("gamma_mode", ["scale", "auto", "value"])
        if mode == "value":
            out["gamma"] = trial.suggest_float("gamma", 1e-5, 1.0, log=True)
        else:
            out["gamma"] = mode
    elif kernel == "poly":
        out["gamma"] = trial.suggest_float("gamma_poly", 1e-4, 1.0, log=True)
    return out


def _suggest_dt(trial: Any) -> dict[str, Any]:
    return {
        "max_depth": trial.suggest_int("max_depth", 2, 30),
        "min_samples_split": trial.suggest_int("min_samples_split", 2, 50),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 30),
        "criterion": trial.suggest_categorical("criterion", ["gini", "entropy"]),
    }


def _suggest_rf(trial: Any) -> dict[str, Any]:
    kind = trial.suggest_categorical(
        "max_features_kind", ["sqrt", "log2", "none", "float"]
    )
    if kind == "none":
        mf: Any = None
    elif kind == "float":
        mf = trial.suggest_float("max_features_frac", 0.3, 0.9)
    else:
        mf = kind
    return {
        "n_estimators": trial.suggest_int("n_estimators", 50, 300),
        "max_depth": trial.suggest_int("max_depth", 3, 30),
        "min_samples_split": trial.suggest_int("min_samples_split", 2, 30),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 20),
        "max_features": mf,
    }


def _suggest_xgb(trial: Any) -> dict[str, Any]:
    return {
        "n_estimators": trial.suggest_int("n_estimators", 50, 400),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "max_depth": trial.suggest_int("max_depth", 2, 12),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
    }


def _compute_virtual_batch_size(batch_size: int, ratio: int) -> int:
    return max(8, batch_size // ratio)


def _suggest_tabnet(trial: Any) -> dict[str, Any]:
    n_d = trial.suggest_int("n_d", 8, 64, step=8)
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
        "virtual_batch_size_ratio": ratio,
        "virtual_batch_size": vbs,
        "momentum": trial.suggest_float("momentum", 0.01, 0.4, log=True),
        "mask_type": trial.suggest_categorical("mask_type", ["sparsemax", "entmax"]),
    }


def _median_pruner_factory() -> Any:
    import optuna

    return optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=0)


def register_default_hpo_entries() -> None:
    models: list[tuple[str, Callable[[Any], dict[str, Any]], bool, Optional[Callable[[], Any]]]] = [
        ("logistic_regression", _suggest_lr, False, None),
        ("svm", _suggest_svm, False, None),
        ("decision_tree", _suggest_dt, False, None),
        ("random_forest", _suggest_rf, False, None),
        ("xgboost", _suggest_xgb, False, None),
        ("tabnet", _suggest_tabnet, True, _median_pruner_factory),
    ]
    for name, suggest, requires_dl, pruner in models:
        register(
            HPOEntry(
                name=name,
                search_space_version=SEARCH_SPACE_VERSION,
                suggest_fn=suggest,
                score_fn=_val_f1_score,
                pruner_factory=pruner,
                requires_dl=requires_dl,
            )
        )

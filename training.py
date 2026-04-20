"""Training helpers for ``main.py`` (``train_and_evaluate``)."""

import numpy as np
from typing import Any, Dict, List, Optional

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from config import Config
from benchmarking.hpo.factory import build_model
from benchmarking.hpo.input_prep import build_model_inputs


def _validate_binary_labels(*label_arrays: np.ndarray) -> None:
    labels = np.concatenate([np.asarray(arr) for arr in label_arrays if arr is not None])
    if labels.size == 0:
        return
    unique = set(np.unique(labels))
    if not unique.issubset({0, 1}):
        raise ValueError(
            f"train_and_evaluate expects binary labels {{0, 1}}; found {sorted(unique)}"
        )


def train_and_evaluate(
    X_train: np.ndarray,
    X_val: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_val: np.ndarray,
    y_test: np.ndarray,
    model_type: str = "random_forest",
    class_weights: Optional[Dict[int, float]] = None,
    feature_names: Optional[List[str]] = None,
    *,
    enable_scaling: bool = False,
    model_params: Optional[Dict[str, Any]] = None,
    config: Optional[Config] = None,
) -> Dict[str, Any]:
    """Fit ``model_type`` (binary labels) and return metrics plus fitted wrapper."""
    _validate_binary_labels(y_train, y_val, y_test)

    if model_type not in ("random_forest", "logistic_regression", "svm", "tabnet"):
        raise ValueError(f"Unknown model type: {model_type}")

    cfg = config or Config()
    params = cfg.get_model_params(model_type)
    if model_params:
        params = {**params, **model_params}

    prep = build_model_inputs(
        model_type,
        X_train,
        X_val,
        X_test,
        enable_scaling=enable_scaling,
    )

    if model_type == "tabnet":
        effective_cw = class_weights if class_weights else "balanced"
    else:
        effective_cw = class_weights

    model = build_model(
        model_type,
        params,
        class_weights=effective_cw,
        tabnet_meta=prep.tabnet_meta,
    )

    if hasattr(model, "prepare_eval_set"):
        model.prepare_eval_set(prep.X_val, y_val)

    fit_names = feature_names
    if model_type == "tabnet" and prep.tabnet_meta is not None:
        fit_names = prep.tabnet_meta.feature_names or feature_names

    print(f"\nTraining {model_type}...")
    model.fit(prep.X_train, y_train, feature_names=fit_names)

    y_val_pred = model.predict(prep.X_val)
    val_metrics = {
        "accuracy": accuracy_score(y_val, y_val_pred),
        "precision": precision_score(y_val, y_val_pred, average="binary", zero_division=0),
        "recall": recall_score(y_val, y_val_pred, average="binary", zero_division=0),
        "f1": f1_score(y_val, y_val_pred, average="binary", zero_division=0),
    }

    print("\nValidation Results:")
    print(f"  Accuracy:  {val_metrics['accuracy']:.4f}")
    print(f"  Precision: {val_metrics['precision']:.4f}")
    print(f"  Recall:    {val_metrics['recall']:.4f}")
    print(f"  F1 Score:  {val_metrics['f1']:.4f}")

    y_test_pred = model.predict(prep.X_test)
    test_metrics = {
        "accuracy": accuracy_score(y_test, y_test_pred),
        "precision": precision_score(y_test, y_test_pred, average="binary", zero_division=0),
        "recall": recall_score(y_test, y_test_pred, average="binary", zero_division=0),
        "f1": f1_score(y_test, y_test_pred, average="binary", zero_division=0),
    }

    print("\nTest Results:")
    print(f"  Accuracy:  {test_metrics['accuracy']:.4f}")
    print(f"  Precision: {test_metrics['precision']:.4f}")
    print(f"  Recall:    {test_metrics['recall']:.4f}")
    print(f"  F1 Score:  {test_metrics['f1']:.4f}")

    print("\nClassification Report (Test):")
    print(
        classification_report(
            y_test,
            y_test_pred,
            labels=[0, 1],
            target_names=["Human", "Bot"],
            zero_division=0,
        )
    )

    print("\nConfusion Matrix (Test):")
    print(confusion_matrix(y_test, y_test_pred, labels=[0, 1]))

    return {
        "model": model,
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
    }

"""
Training helpers for the single-model pipeline.

This module supports the main CLI pipeline by encapsulating model selection,
training, and evaluation in one place. The single-model pipeline refers to the
`main.py` workflow that trains one chosen estimator and reports validation/test
metrics. The primary entry point is `train_and_evaluate`, which expects binary
labels (0/1), fits the requested model, and prints standard evaluation output.
"""

import numpy as np
from typing import Any, Dict, Optional
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix
)


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
    model_type: str = 'random_forest',
    class_weights: Optional[Dict[int, float]] = None
) -> Dict[str, Any]:
    """
    Train a model and evaluate on validation and test sets.

    Args:
        X_train: Training feature matrix.
        X_val: Validation feature matrix.
        X_test: Test feature matrix.
        y_train: Training labels.
        y_val: Validation labels.
        y_test: Test labels.
        model_type: Model identifier ('random_forest', 'logistic_regression', 'svm').
        class_weights: Optional class weight mapping for imbalance handling.

    Returns:
        Dictionary with the fitted model plus validation and test metrics.
    """
    _validate_binary_labels(y_train, y_val, y_test)
    if model_type == 'random_forest':
        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            class_weight=class_weights,
            random_state=2112,
            n_jobs=-1
        )
    elif model_type == 'logistic_regression':
        model = LogisticRegression(
            class_weight=class_weights,
            random_state=2112,
            max_iter=1000
        )
    elif model_type == 'svm':
        model = SVC(
            class_weight=class_weights,
            random_state=2112,
            kernel='rbf'
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    print(f"\nTraining {model_type}...")
    model.fit(X_train, y_train)

    y_val_pred = model.predict(X_val)
    val_metrics = {
        'accuracy': accuracy_score(y_val, y_val_pred),
        'precision': precision_score(y_val, y_val_pred, average='binary', zero_division=0),
        'recall': recall_score(y_val, y_val_pred, average='binary', zero_division=0),
        'f1': f1_score(y_val, y_val_pred, average='binary', zero_division=0)
    }

    print("\nValidation Results:")
    print(f"  Accuracy:  {val_metrics['accuracy']:.4f}")
    print(f"  Precision: {val_metrics['precision']:.4f}")
    print(f"  Recall:    {val_metrics['recall']:.4f}")
    print(f"  F1 Score:  {val_metrics['f1']:.4f}")

    y_test_pred = model.predict(X_test)
    test_metrics = {
        'accuracy': accuracy_score(y_test, y_test_pred),
        'precision': precision_score(y_test, y_test_pred, average='binary', zero_division=0),
        'recall': recall_score(y_test, y_test_pred, average='binary', zero_division=0),
        'f1': f1_score(y_test, y_test_pred, average='binary', zero_division=0)
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
            zero_division=0
        )
    )

    print("\nConfusion Matrix (Test):")
    print(confusion_matrix(y_test, y_test_pred, labels=[0, 1]))

    return {
        'model': model,
        'val_metrics': val_metrics,
        'test_metrics': test_metrics
    }

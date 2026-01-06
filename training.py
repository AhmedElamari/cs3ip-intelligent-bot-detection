"""
Training helpers for the single-model pipeline.
"""

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix
)


def train_and_evaluate(
    X_train, X_val, X_test,
    y_train, y_val, y_test,
    model_type: str = 'random_forest',
    class_weights: dict = None
) -> dict:
    """Train model and evaluate on validation and test sets."""
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

    print(f"\nValidation Results:")
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

    print(f"\nTest Results:")
    print(f"  Accuracy:  {test_metrics['accuracy']:.4f}")
    print(f"  Precision: {test_metrics['precision']:.4f}")
    print(f"  Recall:    {test_metrics['recall']:.4f}")
    print(f"  F1 Score:  {test_metrics['f1']:.4f}")

    print(f"\nClassification Report (Test):")
    test_labels = np.unique(np.concatenate([y_test, y_test_pred]))
    target_names = [
        "Human" if lbl == 0 else "Bot" if lbl == 1 else str(lbl)
        for lbl in test_labels
    ]
    print(
        classification_report(
            y_test,
            y_test_pred,
            labels=test_labels,
            target_names=target_names,
            zero_division=0
        )
    )

    print(f"\nConfusion Matrix (Test):")
    print(confusion_matrix(y_test, y_test_pred))

    return {
        'model': model,
        'val_metrics': val_metrics,
        'test_metrics': test_metrics
    }

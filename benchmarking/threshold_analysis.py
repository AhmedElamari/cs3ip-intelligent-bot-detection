"""Thresholds chosen on val only; test rows report, never tune (anti-leakage)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd

from benchmarking.metrics import MetricsCalculator


@dataclass(frozen=True)
class ThresholdSelection:
    threshold: float
    validation_metrics: Dict[str, float]


DISPLAY_METRICS = (
    ("Precision", "precision"),
    ("Recall", "recall"),
    ("F1", "f1"),
    ("F1-Macro", "f1_macro"),
    ("Balanced Accuracy", "balanced_accuracy"),
    ("MCC", "mcc"),
    ("False Positives", "false_positives"),
    ("False Negatives", "false_negatives"),
)


def _positive_class_proba(y_proba: np.ndarray) -> np.ndarray:
    proba = np.asarray(y_proba)
    return proba[:, 1] if proba.ndim > 1 else proba


def _candidate_thresholds(y_proba: np.ndarray) -> np.ndarray:
    scores = _positive_class_proba(y_proba)
    candidates = np.unique(np.concatenate([scores, np.array([0.5])]))
    return candidates[(0.0 <= candidates) & (candidates <= 1.0)]


def metrics_at_threshold(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    threshold: float,
) -> Dict[str, float]:
    """Compute classification metrics after applying a positive-class threshold."""
    scores = _positive_class_proba(y_proba)
    y_pred = (scores >= float(threshold)).astype(int)
    return MetricsCalculator().compute_all_metrics(np.asarray(y_true), y_pred, y_proba)


def _selection_rows(y_true: np.ndarray, y_proba: np.ndarray) -> list[ThresholdSelection]:
    return [
        ThresholdSelection(float(threshold), metrics_at_threshold(y_true, y_proba, threshold))
        for threshold in _candidate_thresholds(y_proba)
    ]


def _best_by(
    rows: list[ThresholdSelection],
    *metric_names: str,
) -> ThresholdSelection:
    return max(
        rows,
        key=lambda row: tuple(float(row.validation_metrics.get(metric, 0.0)) for metric in metric_names)
        + (-row.threshold,),
    )


def select_thresholds(
    y_val: np.ndarray,
    val_proba: np.ndarray,
    *,
    precision_floor: float = 0.80,
) -> Dict[str, ThresholdSelection]:
    """Select thresholds on validation data only."""
    rows = _selection_rows(y_val, val_proba)
    floor_label = f"precision_floor_{precision_floor:.2f}"
    floor_rows = [
        row for row in rows
        if float(row.validation_metrics.get("precision", 0.0)) >= precision_floor
    ]
    return {
        "threshold_0.50": ThresholdSelection(
            0.5,
            metrics_at_threshold(y_val, val_proba, 0.5),
        ),
        "best_macro_f1": _best_by(rows, "f1_macro", "mcc", "precision", "recall"),
        floor_label: _best_by(
            floor_rows or rows,
            "recall" if floor_rows else "precision",
            "f1_macro",
            "precision",
        ),
        "best_balanced_accuracy": _best_by(rows, "balanced_accuracy", "f1_macro", "mcc"),
    }


def _metric_columns(prefix: str, metrics: Dict[str, float]) -> Dict[str, float]:
    return {
        f"{prefix} {display}": metrics.get(key, np.nan)
        for display, key in DISPLAY_METRICS
    }


def _current_predict_row(model_name: str, result: Dict[str, Any]) -> Dict[str, Any]:
    row = {
        "Model": model_name,
        "Policy": "current_predict",
        "Threshold": np.nan,
        "Selection Source": "model.predict",
    }
    row.update(_metric_columns("Validation", result.get("val_metrics", {})))
    row.update(_metric_columns("Test", result.get("test_metrics", {})))
    return row


def build_threshold_analysis(
    benchmark: Any,
    *,
    precision_floor: float = 0.80,
) -> pd.DataFrame:
    """Build validation-selected threshold metrics for probability-capable models."""
    y_val = getattr(benchmark, "y_val", None)
    y_test = getattr(benchmark, "y_test", None)
    val_probas = getattr(benchmark, "validation_probabilities", {})
    test_probas = getattr(benchmark, "probabilities", {})
    results = getattr(benchmark, "results", {}) or {}
    if y_val is None or y_test is None:
        return pd.DataFrame()

    rows = []
    for model_name in sorted(set(val_probas) & set(test_probas) & set(results)):
        rows.append(_current_predict_row(model_name, results[model_name]))
        selections = select_thresholds(
            np.asarray(y_val),
            val_probas[model_name],
            precision_floor=precision_floor,
        )
        for policy, selection in selections.items():
            test_metrics = metrics_at_threshold(
                np.asarray(y_test),
                test_probas[model_name],
                selection.threshold,
            )
            row = {
                "Model": model_name,
                "Policy": policy,
                "Threshold": selection.threshold,
                "Selection Source": "validation",
            }
            row.update(_metric_columns("Validation", selection.validation_metrics))
            row.update(_metric_columns("Test", test_metrics))
            rows.append(row)

    return pd.DataFrame(rows)


def _markdown_table(frame: pd.DataFrame) -> str:
    header = "| " + " | ".join(frame.columns) + " |"
    separator = "| " + " | ".join("---" for _ in frame.columns) + " |"
    rows = [
        "| " + " | ".join(str(value) for value in row) + " |"
        for row in frame.itertuples(index=False, name=None)
    ]
    return "\n".join([header, separator, *rows])


def save_threshold_analysis_outputs(
    benchmark: Any,
    output_dir: Path,
    *,
    precision_floor: float = 0.80,
) -> pd.DataFrame:
    """Write threshold-analysis CSV/Markdown artifacts and return the frame."""
    frame = build_threshold_analysis(benchmark, precision_floor=precision_floor)
    if frame.empty:
        return frame

    output_dir = Path(output_dir)
    export = frame.copy()
    for column in export.select_dtypes(include=[np.number]).columns:
        export[column] = export[column].round(4)
    export.to_csv(output_dir / "threshold_analysis.csv", index=False)
    display = export.where(pd.notna(export), "")

    summary = (
        "Precision-recall threshold audit. Threshold policies marked with "
        "`Selection Source = validation` are validation-selected thresholds "
        "applied once to the held-out test split; test labels are not used for "
        "threshold selection. These rows diagnose operating-point sensitivity "
        "and do not replace the benchmark ranking."
    )
    (output_dir / "threshold_analysis.md").write_text(
        summary + "\n\n" + _markdown_table(display) + "\n",
        encoding="utf-8",
    )
    return frame

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from benchmarking.threshold_analysis import (
    build_threshold_analysis,
    metrics_at_threshold,
    select_thresholds,
)


class ThresholdAnalysisTest(unittest.TestCase):
    def test_metrics_at_threshold_uses_positive_class_probability(self):
        y_true = np.array([0, 0, 1, 1])
        y_proba = np.array(
            [
                [0.9, 0.1],
                [0.4, 0.6],
                [0.3, 0.7],
                [0.1, 0.9],
            ]
        )

        metrics = metrics_at_threshold(y_true, y_proba, threshold=0.65)

        self.assertEqual(metrics["true_negatives"], 2)
        self.assertEqual(metrics["false_positives"], 0)
        self.assertEqual(metrics["false_negatives"], 0)
        self.assertEqual(metrics["true_positives"], 2)
        self.assertAlmostEqual(metrics["precision"], 1.0)
        self.assertAlmostEqual(metrics["recall"], 1.0)
        self.assertAlmostEqual(metrics["f1_macro"], 1.0)

    def test_select_thresholds_uses_validation_precision_floor(self):
        y_val = np.array([0, 0, 1, 1])
        val_proba = np.array([0.2, 0.8, 0.7, 0.9])

        selections = select_thresholds(y_val, val_proba, precision_floor=0.8)

        precision_row = selections["precision_floor_0.80"]
        self.assertAlmostEqual(precision_row.threshold, 0.9)
        self.assertAlmostEqual(precision_row.validation_metrics["precision"], 1.0)
        self.assertAlmostEqual(precision_row.validation_metrics["recall"], 0.5)

    def test_build_threshold_analysis_applies_validation_threshold_to_test(self):
        class _Benchmark:
            y_val = np.array([0, 0, 1, 1])
            y_test = np.array([0, 1])
            predictions = {"model_a": np.array([1, 0])}
            probabilities = {
                "model_a": np.array(
                    [
                        [0.15, 0.85],
                        [0.9, 0.1],
                    ]
                )
            }
            validation_probabilities = {"model_a": np.array([0.2, 0.8, 0.7, 0.9])}
            results = {
                "model_a": {
                    "test_metrics": {"precision": 0.0, "recall": 0.0, "f1_macro": 0.0},
                    "val_metrics": {"precision": 0.0, "recall": 0.0, "f1_macro": 0.0},
                }
            }

        frame = build_threshold_analysis(_Benchmark(), precision_floor=0.8)
        precision_row = frame[
            (frame["Model"] == "model_a")
            & (frame["Policy"] == "precision_floor_0.80")
        ].iloc[0]

        self.assertAlmostEqual(float(precision_row["Threshold"]), 0.9)
        self.assertEqual(int(precision_row["Test False Positives"]), 0)
        self.assertEqual(int(precision_row["Test False Negatives"]), 1)
        self.assertAlmostEqual(float(precision_row["Test Precision"]), 0.0)


if __name__ == "__main__":
    unittest.main()

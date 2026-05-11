"""Tests for benchmarking.multi_seed helpers."""

import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from benchmarking import ModelBenchmark
from benchmarking.multi_seed import (
    METRIC_KEYS,
    build_summary,
    extract_per_seed_rows,
    validate_seeds,
    write_multi_seed_outputs,
)


class MultiSeedValidationTest(unittest.TestCase):
    def test_requires_three_seeds(self):
        with self.assertRaises(ValueError):
            validate_seeds([1, 2])
        with self.assertRaises(ValueError):
            validate_seeds([])

    def test_rejects_duplicates(self):
        with self.assertRaises(ValueError):
            validate_seeds([1, 2, 2])

    def test_accepts_three_unique(self):
        self.assertEqual([3, 1, 2], validate_seeds([3, 1, 2]))


class MultiSeedAggregationTest(unittest.TestCase):
    def test_sample_std_ddof_one(self):
        rows = [
            {"seed": 1, "model": "a", "f1_macro": 0.0, "pr_auc": 0.0, "mcc": 0.0, "balanced_accuracy": 0.0},
            {"seed": 2, "model": "a", "f1_macro": 2.0, "pr_auc": 0.0, "mcc": 0.0, "balanced_accuracy": 0.0},
            {"seed": 3, "model": "a", "f1_macro": 4.0, "pr_auc": 0.0, "mcc": 0.0, "balanced_accuracy": 0.0},
        ]
        summary = build_summary(rows)
        self.assertEqual(len(summary), 1)
        self.assertAlmostEqual(summary.iloc[0]["f1_macro_mean"], 2.0)
        self.assertAlmostEqual(summary.iloc[0]["f1_macro_std"], 2.0)

    def test_sort_by_f1_macro_then_pr_auc(self):
        rows = []
        for seed in (1, 2, 3):
            rows.append(
                {
                    "seed": seed,
                    "model": "low",
                    "f1_macro": 0.1,
                    "pr_auc": 0.9,
                    "mcc": 0.0,
                    "balanced_accuracy": 0.0,
                }
            )
            rows.append(
                {
                    "seed": seed,
                    "model": "high",
                    "f1_macro": 0.8,
                    "pr_auc": 0.1,
                    "mcc": 0.0,
                    "balanced_accuracy": 0.0,
                }
            )
        summary = build_summary(rows)
        self.assertEqual(summary.iloc[0]["model"], "high")

    def test_extract_per_seed_rows(self):
        b = ModelBenchmark(models={}, experiment_name="t")
        b.results = {
            "m1": {
                "test_metrics": {
                    "f1_macro": 0.5,
                    "pr_auc": 0.6,
                    "mcc": 0.07,
                    "balanced_accuracy": 0.55,
                }
            }
        }
        rows = extract_per_seed_rows(b, seed=42)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["seed"], 42)
        self.assertEqual(rows[0]["model"], "m1")
        for k in METRIC_KEYS:
            self.assertIn(k, rows[0])


class MultiSeedExportTest(unittest.TestCase):
    def test_writes_csv_and_markdown(self):
        rows = [
            {"seed": 1, "model": "x", "f1_macro": 0.5, "pr_auc": 0.5, "mcc": 0.0, "balanced_accuracy": 0.5},
            {"seed": 2, "model": "x", "f1_macro": 0.7, "pr_auc": 0.6, "mcc": 0.1, "balanced_accuracy": 0.6},
            {"seed": 3, "model": "x", "f1_macro": 0.6, "pr_auc": 0.55, "mcc": 0.05, "balanced_accuracy": 0.55},
        ]
        with TemporaryDirectory(dir=ROOT) as tmp:
            out = Path(tmp)
            write_multi_seed_outputs(rows, out)
            self.assertTrue((out / "multi_seed_results.csv").is_file())
            self.assertTrue((out / "multi_seed_summary.csv").is_file())
            md = (out / "multi_seed_summary.md").read_text(encoding="utf-8")
            self.assertIn("retraining", md.lower())
            for label in ("F1-Macro", "PR-AUC", "MCC", "Balanced Accuracy"):
                self.assertIn(label, md)


if __name__ == "__main__":
    unittest.main()

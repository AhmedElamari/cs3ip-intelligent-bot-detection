import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


class LimeMisclassifiedExportTest(unittest.TestCase):
    def test_export_selects_false_negative_bots_and_columns(self):
        from benchmarking import ModelBenchmark
        from benchmarking.xai_reporting import export_misclassified_bot_lime

        benchmark = ModelBenchmark(models={}, experiment_name="lime_fn")
        benchmark.y_test = np.array([1, 1, 1, 0])
        benchmark.predictions = {"xgboost": np.array([0, 1, 0, 0])}
        benchmark.probabilities = {
            "xgboost": np.array([
                [0.90, 0.10],
                [0.20, 0.80],
                [0.97, 0.03],
                [0.85, 0.15],
            ])
        }
        benchmark.test_metadata = pd.DataFrame({
            "user_id": ["bot_a", "bot_b", "bot_c", "human_a"],
            "row_index": [10, 11, 12, 13],
            "label": [1, 1, 1, 0],
        })
        X_train = pd.DataFrame({"f0": [0, 1, 2], "f1": [2, 1, 0]})
        X_test = pd.DataFrame({"f0": [0, 1, 2, 3], "f1": [3, 2, 1, 0]})
        benchmark.results = {
            "xgboost": {
                "model": object(),
                "training_time": 1.0,
                "val_metrics": {},
                "test_metrics": {"f1": 0.9},
                "feature_importance": None,
                "is_interpretable": False,
                "X_train": X_train,
                "X_val": X_train,
                "X_test": X_test,
                "feature_names": ["f0", "f1"],
                "scaler": None,
            }
        }

        explanation = {
            "feature_contributions": {
                "f0 <= 1.00": 0.4,
                "f1 > 0.50": -0.2,
                "bias": 0.1,
            }
        }
        with TemporaryDirectory(dir=ROOT) as tmp, mock.patch(
            "benchmarking.xai_reporting.LIMEExplainer.fit",
            return_value=None,
        ), mock.patch(
            "benchmarking.xai_reporting.LIMEExplainer.explain_instance",
            return_value=explanation,
        ):
            output_dir = Path(tmp)
            export_misclassified_bot_lime(benchmark, ["f0", "f1"], output_dir)

            csv_path = output_dir / "lime_misclassified_bots.csv"
            md_path = output_dir / "lime_misclassified_bots.md"
            self.assertTrue(csv_path.exists())
            self.assertTrue(md_path.exists())
            df = pd.read_csv(csv_path)

        self.assertEqual(df["user_id"].tolist(), ["bot_c", "bot_a"])
        self.assertEqual(df["test_row_index"].tolist(), [12, 10])
        self.assertListEqual(
            list(df.columns),
            [
                "user_id",
                "test_row_index",
                "true_label",
                "predicted_label",
                "predicted_bot_probability",
                "predicted_human_probability",
                "top_1_feature",
                "top_1_contribution",
                "top_1_direction",
                "top_2_feature",
                "top_2_contribution",
                "top_2_direction",
                "top_3_feature",
                "top_3_contribution",
                "top_3_direction",
            ],
        )
        self.assertNotIn("screen_name", df.columns)
        self.assertNotIn("description", df.columns)
        self.assertEqual(set(df["true_label"]), {1})
        self.assertEqual(set(df["predicted_label"]), {0})


if __name__ == "__main__":
    unittest.main()

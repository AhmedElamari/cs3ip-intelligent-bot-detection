import json
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from benchmarking import ModelBenchmark
from benchmarking.data_prep import prepare_data
from benchmarking.model_factory import create_models
from config import Config
from tests.test_benchmark_smoke import _make_synthetic_splits


class ResultsRepresentationContractTest(unittest.TestCase):
    def _build_benchmark(self, *, compute_statistics=False):
        splits = _make_synthetic_splits(n_samples=60)
        config = Config()
        for name in config.get("models", {}).keys():
            config.set(
                f"models.{name}.enabled",
                name in ("logistic_regression", "random_forest"),
            )
        config.set("preprocessing.scale_features", True)

        X_train, X_val, X_test, y_train, y_val, y_test, feature_names = prepare_data(
            splits,
            config,
        )
        benchmark = ModelBenchmark(models=create_models(config), experiment_name="results_repr")
        benchmark.run_benchmark(
            X_train,
            y_train,
            X_val,
            y_val,
            X_test,
            y_test,
            feature_names=feature_names,
            verbose=False,
            compute_statistics=compute_statistics,
            statistics_bootstrap_samples=50,
            statistics_metrics=["f1"],
            enable_scaling=True,
        )
        return benchmark

    def test_model_comparison_csv_matches_benchmark_results(self):
        benchmark = self._build_benchmark()

        with TemporaryDirectory(dir=ROOT) as tmp_dir:
            output_dir = Path(tmp_dir)
            benchmark.save_results(output_dir)
            comparison_df = pd.read_csv(output_dir / "model_comparison.csv")

        expected = benchmark.get_comparison_table().set_index("Model")
        actual = comparison_df.set_index("Model")
        self.assertListEqual(expected.index.tolist(), actual.index.tolist())
        for model_name in expected.index:
            for column in ("F1", "ACCURACY", "ROC_AUC", "MCC"):
                self.assertAlmostEqual(expected.loc[model_name, column], actual.loc[model_name, column])

    def test_results_json_matches_ranked_models_and_metrics(self):
        benchmark = self._build_benchmark(compute_statistics=True)

        with TemporaryDirectory(dir=ROOT) as tmp_dir:
            output_dir = Path(tmp_dir)
            benchmark.save_results(output_dir)
            results_json = json.loads((output_dir / "results.json").read_text(encoding="utf-8"))

        comparison_models = benchmark.get_comparison_table()["Model"].tolist()
        self.assertEqual("results_repr", results_json["experiment_name"])
        self.assertEqual("f1", results_json["sort_metric"])
        self.assertListEqual(comparison_models, results_json["ranked_models"])
        for rank, model_name in enumerate(comparison_models, start=1):
            entry = results_json["models"][model_name]
            self.assertEqual(rank, entry["rank"])
            self.assertAlmostEqual(
                benchmark.results[model_name]["test_metrics"]["f1"],
                entry["test_metrics"]["f1"],
            )
            self.assertIn("confidence_intervals", entry)

    def test_generate_report_uses_ranked_order_and_best_model(self):
        benchmark = self._build_benchmark(compute_statistics=True)
        comparison_models = benchmark.get_comparison_table()["Model"].tolist()
        best_name = comparison_models[0]

        report = benchmark.generate_report()

        self.assertIn(f"Best model by test F1: `{best_name}`", report)
        self.assertIn("## Summary", report)
        self.assertIn("```text", report)
        positions = [report.index(f"### {model_name}") for model_name in comparison_models]
        self.assertEqual(sorted(positions), positions)

    def test_feature_importance_exports_split_raw_and_normalized_contracts(self):
        benchmark = self._build_benchmark()

        with TemporaryDirectory(dir=ROOT) as tmp_dir:
            output_dir = Path(tmp_dir)
            benchmark.save_results(output_dir)
            raw_df = pd.read_csv(output_dir / "feature_importance.csv")
            comparison_df = pd.read_csv(output_dir / "feature_importance_comparison.csv")

        self.assertNotIn("Average", raw_df.columns)
        self.assertIn("Average_Normalized", comparison_df.columns)
        normalized_columns = [
            column
            for column in comparison_df.columns
            if column not in ("Feature", "Average_Normalized")
        ]
        for column in normalized_columns + ["Average_Normalized"]:
            self.assertTrue(((comparison_df[column] >= 0) & (comparison_df[column] <= 1)).all())

    def test_save_results_writes_canonical_comparison_file_only(self):
        benchmark = self._build_benchmark()

        with TemporaryDirectory(dir=ROOT) as tmp_dir:
            output_dir = Path(tmp_dir)
            benchmark.save_results(output_dir)
            self.assertTrue((output_dir / "model_comparison.csv").exists())
            self.assertFalse((output_dir / "comparison.csv").exists())


if __name__ == "__main__":
    unittest.main()

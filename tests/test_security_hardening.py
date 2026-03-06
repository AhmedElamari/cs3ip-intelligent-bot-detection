import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from explainability.shap_explainer import SHAPExplainer
from benchmarking import ModelBenchmark
from models.base import BaseModel


class DummyModel(BaseModel):
    def __init__(self):
        super().__init__(name="dummy")

    def _create_model(self, **kwargs):
        from sklearn.dummy import DummyClassifier

        return DummyClassifier(strategy="most_frequent", random_state=self.random_state)

    @property
    def is_interpretable(self) -> bool:
        return True

    @property
    def supports_feature_importance(self) -> bool:
        return False


class SecurityHardeningTest(unittest.TestCase):
    def test_model_load_requires_trusted_source(self):
        model = DummyModel()
        X = np.array([[0.0], [1.0], [2.0], [3.0]])
        y = np.array([0, 0, 1, 1])
        model.fit(X, y, feature_names=["feature_0"])

        with tempfile.TemporaryDirectory(dir=ROOT) as tmp_dir:
            model_path = Path(tmp_dir) / "dummy_model.pkl"
            model.save(str(model_path))

            with self.assertRaises(ValueError):
                DummyModel.load(str(model_path), trusted_source=False)

            loaded = DummyModel.load(str(model_path), trusted_source=True)
            self.assertTrue(loaded.is_fitted)
            self.assertEqual(loaded.feature_names, ["feature_0"])

    def test_model_save_rejects_path_outside_workspace(self):
        model = DummyModel()
        X = np.array([[0.0], [1.0]])
        y = np.array([0, 1])
        model.fit(X, y)

        outside_path = ROOT.parent / "outside_model.pkl"
        with self.assertRaises(ValueError):
            model.save(str(outside_path))

    def test_shap_save_explanations_writes_npz(self):
        explainer = SHAPExplainer(model=object(), feature_names=["f1", "f2"])
        explainer.shap_values = np.array([[0.2, -0.1], [0.1, 0.3]])

        with tempfile.TemporaryDirectory(dir=ROOT) as tmp_dir:
            path = Path(tmp_dir) / "explanations.npz"
            explainer.save_explanations(str(path))
            self.assertTrue(path.exists())

            with np.load(path, allow_pickle=False) as archive:
                self.assertIn("shap_values", archive.files)
                self.assertIn("feature_names", archive.files)
                shap_arr = archive["shap_values"]
                names_arr = archive["feature_names"]
                np.testing.assert_array_equal(shap_arr, [[0.2, -0.1], [0.1, 0.3]])
                self.assertEqual(list(names_arr), ["f1", "f2"])

    def test_shap_save_rejects_path_outside_workspace(self):
        explainer = SHAPExplainer(model=object(), feature_names=["f1"])
        explainer.shap_values = np.array([[0.2]])

        outside_path = ROOT.parent / "outside_explanations.npz"
        with self.assertRaises(ValueError):
            explainer.save_explanations(str(outside_path))

    def test_benchmark_save_results_rejects_path_outside_workspace(self):
        benchmark = ModelBenchmark()
        outside_dir = ROOT.parent / "outside_results"
        with self.assertRaises(ValueError):
            benchmark.save_results(str(outside_dir))

    def test_benchmark_save_results_accepts_workspace_subdir(self):
        benchmark = ModelBenchmark()
        benchmark.results = {
            "dummy": {
                "training_time": 0.1,
                "val_metrics": {"f1": 0.5},
                "test_metrics": {"f1": 0.6},
                "is_interpretable": True,
                "feature_importance": None,
            }
        }
        with tempfile.TemporaryDirectory(dir=ROOT) as tmp_dir:
            benchmark.save_results(tmp_dir)
            self.assertTrue((Path(tmp_dir) / "comparison.csv").exists())
            self.assertTrue((Path(tmp_dir) / "results.json").exists())


if __name__ == "__main__":
    unittest.main()

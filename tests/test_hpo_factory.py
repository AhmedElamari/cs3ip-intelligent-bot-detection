"""Tests for benchmarking.hpo.factory.build_model."""
import unittest

from config import Config
from benchmarking.hpo.factory import build_model


class TestBuildModel(unittest.TestCase):
    def test_build_each_registered_model(self):
        cfg = Config()
        for name in (
            "logistic_regression",
            "svm",
            "decision_tree",
            "random_forest",
            "xgboost",
        ):
            params = cfg.get_model_params(name)
            m = build_model(name, params)
            self.assertIsNotNone(m)
            self.assertEqual(m.__class__.__name__.endswith("Model"), True)
        try:
            params = cfg.get_model_params("tabnet")
            m = build_model("tabnet", params)
            self.assertIsNotNone(m)
        except ImportError:
            self.skipTest("pytorch-tabnet not installed")

    def test_random_forest_max_features(self):
        cfg = Config()
        params = cfg.get_model_params("random_forest")
        params["max_features"] = "log2"
        m = build_model("random_forest", params)
        self.assertEqual(m.model.max_features, "log2")


if __name__ == "__main__":
    unittest.main()

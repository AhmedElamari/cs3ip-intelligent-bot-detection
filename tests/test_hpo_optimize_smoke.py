"""Smoke: optimize_model runs for LR with 1 trial (requires optuna)."""
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from config import Config
from benchmarking.hpo.service import optimize_model


class TestOptimizeModelSmoke(unittest.TestCase):
    def test_one_trial_logistic_regression(self):
        cfg = Config()
        X_train = np.array([[0.0, 1.0], [1.0, 0.0], [1.0, 1.0], [0.0, 0.0]], dtype=np.float64)
        y_train = np.array([0, 0, 1, 1])
        X_val = np.array([[0.5, 0.5], [0.9, 0.1]], dtype=np.float64)
        y_val = np.array([0, 1])
        with TemporaryDirectory() as tmp:
            out = Path(tmp) / "hpo.json"
            res = optimize_model(
                "logistic_regression",
                X_train,
                y_train,
                X_val,
                y_val,
                config=cfg,
                n_trials=1,
                seed=2112,
                enable_scaling=True,
                class_weights=None,
                feature_names=["a", "b"],
                output_path=out,
            )
            self.assertEqual(res.get("schema_version"), "HPOResultV1")
            self.assertEqual(res.get("status"), "ok")
            self.assertIn("C", res.get("best_params", {}))
            self.assertTrue(out.is_file())


if __name__ == "__main__":
    unittest.main()

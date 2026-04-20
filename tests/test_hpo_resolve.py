"""resolve_hpo behaviour (no-tune skip, unknown model fail-fast)."""
import unittest
from pathlib import Path

import numpy as np

from config import Config
from benchmarking.hpo.service import HPOCliOverrides, resolve_hpo


class TestResolveHPO(unittest.TestCase):
    def test_no_tune_skips(self):
        cfg = Config()
        cfg.set("hpo.enabled", True)
        X = np.zeros((4, 2))
        y = np.array([0, 1, 0, 1])
        res, audit = resolve_hpo(
            "logistic_regression",
            cfg,
            X_train=X,
            y_train=y,
            X_val=X[:2],
            y_val=y[:2],
            feature_names_ordered=["a", "b"],
            data_dir=Path(__file__).resolve().parents[1] / "data",
            enable_scaling=False,
            class_weights={0: 1.0, 1: 1.0},
            cli=HPOCliOverrides(no_tune=True),
        )
        self.assertEqual(res.get("status"), "skipped")
        self.assertTrue(audit.get("skipped"))

    def test_unknown_model_raises_when_fail_fast(self):
        cfg = Config()
        cfg.set("hpo.fail_fast", True)
        X = np.zeros((4, 2))
        y = np.array([0, 1, 0, 1])
        with self.assertRaises(ValueError) as ctx:
            resolve_hpo(
                "not_a_real_model",
                cfg,
                X_train=X,
                y_train=y,
                X_val=X[:2],
                y_val=y[:2],
                feature_names_ordered=["a", "b"],
                data_dir=Path(__file__).resolve().parents[1] / "data",
                enable_scaling=False,
                class_weights=None,
            )
        self.assertIn("not_a_real_model", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()

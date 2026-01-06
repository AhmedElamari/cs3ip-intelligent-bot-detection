import importlib.util
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

SKLEARN_AVAILABLE = importlib.util.find_spec("sklearn") is not None
NUMPY_AVAILABLE = importlib.util.find_spec("numpy") is not None


class GradientBoostingWeightsTest(unittest.TestCase):
    def setUp(self):
        if not (SKLEARN_AVAILABLE and NUMPY_AVAILABLE):
            self.skipTest("Required dependencies not installed")
        import numpy as np
        from models.gradient_boosting import GradientBoostingModel
        self.np = np
        self.model_cls = GradientBoostingModel

    def test_balanced_weights(self):
        y = self.np.array([0, 0, 1, 1])
        weights = self.model_cls._compute_sample_weight(y, "balanced")
        self.assertTrue((weights == 1.0).all())

    def test_dict_weights(self):
        y = self.np.array([0, 1, 1])
        weights = self.model_cls._compute_sample_weight(y, {0: 1.0, 1: 2.0})
        self.assertListEqual(weights.tolist(), [1.0, 2.0, 2.0])

    def test_unknown_labels_raise(self):
        y = self.np.array([0, 2])
        with self.assertRaises(ValueError):
            self.model_cls._compute_sample_weight(y, {0: 1.0, 1: 1.0})

    def test_fit_sets_fitted_flag(self):
        X = self.np.array([
            [0.0, 0.1],
            [0.2, 0.0],
            [1.0, 1.1],
            [1.2, 1.0],
        ])
        y = self.np.array([0, 0, 1, 1])
        model = self.model_cls(class_weight="balanced")
        model.fit(X, y)
        self.assertTrue(model.is_fitted)


if __name__ == "__main__":
    unittest.main()

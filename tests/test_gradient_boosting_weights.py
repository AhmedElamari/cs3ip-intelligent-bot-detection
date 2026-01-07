import importlib.util
import re
import sys
import unittest
from importlib import metadata
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

MIN_SKLEARN_VERSION = "1.5.0"
MIN_NUMPY_VERSION = "1.23.5"

#Required in the event of a mismatch
def _version_at_least(version_str, minimum):
    try:
        from packaging.version import Version
    except Exception:
        if not re.fullmatch(r"\d+(\.\d+)*", version_str):
            return False
        current = tuple(int(part) for part in version_str.split("."))
        required = tuple(int(part) for part in minimum.split("."))
        max_len = max(len(current), len(required))
        current += (0,) * (max_len - len(current))
        required += (0,) * (max_len - len(required))
        return current >= required
    return Version(version_str) >= Version(minimum)


def _has_min_version(module_name, dist_name, minimum):
    if importlib.util.find_spec(module_name) is None:
        return False
    try:
        version_str = metadata.version(dist_name)
    except metadata.PackageNotFoundError:
        return False
    return _version_at_least(version_str, minimum)


SKLEARN_AVAILABLE = _has_min_version("sklearn", "scikit-learn", MIN_SKLEARN_VERSION)
NUMPY_AVAILABLE = _has_min_version("numpy", "numpy", MIN_NUMPY_VERSION)


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

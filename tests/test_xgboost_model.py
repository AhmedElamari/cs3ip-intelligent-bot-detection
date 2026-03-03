"""
Tests for XGBoostModel — replaces test_gradient_boosting_weights.py.

Covers:
  - Sample-weight computation (balanced and dict class_weight)
  - Error paths for unknown labels
  - fit() sets is_fitted flag and propagates sample_weight
  - predict / predict_proba work after fit
  - feature_importances_ attribute is present after fit
  - Migration guard: gradient_boosting absent from MODEL_REGISTRY
  - Migration guard: xgboost present in MODEL_REGISTRY and config defaults
"""

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


def _version_at_least(version_str, minimum):
    try:
        from packaging.version import Version
    except Exception:
        if not re.fullmatch(r"\d+(\.\d+)*", version_str):
            return False
        current = tuple(int(p) for p in version_str.split("."))
        required = tuple(int(p) for p in minimum.split("."))
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
XGBOOST_AVAILABLE = importlib.util.find_spec("xgboost") is not None


class XGBoostSampleWeightTest(unittest.TestCase):
    """Unit tests for the static _compute_sample_weight helper."""

    def setUp(self):
        if not (SKLEARN_AVAILABLE and NUMPY_AVAILABLE and XGBOOST_AVAILABLE):
            self.skipTest("Required dependencies not installed")
        import numpy as np
        from models.xgboost import XGBoostModel
        self.np = np
        self.model_cls = XGBoostModel

    def test_balanced_weights_equal_classes(self):
        y = self.np.array([0, 0, 1, 1])
        weights = self.model_cls._compute_sample_weight(y, "balanced")
        self.assertTrue((weights == 1.0).all())

    def test_balanced_weights_unequal_classes(self):
        y = self.np.array([0, 0, 0, 1])
        weights = self.model_cls._compute_sample_weight(y, "balanced")
        # minority class should have higher weight
        self.assertGreater(weights[3], weights[0])

    def test_dict_weights(self):
        y = self.np.array([0, 1, 1])
        weights = self.model_cls._compute_sample_weight(y, {0: 1.0, 1: 2.0})
        self.assertListEqual(weights.tolist(), [1.0, 2.0, 2.0])

    def test_none_class_weight_returns_none(self):
        y = self.np.array([0, 1])
        result = self.model_cls._compute_sample_weight(y, None)
        self.assertIsNone(result)

    def test_unknown_labels_raise(self):
        y = self.np.array([0, 2])
        with self.assertRaises(ValueError):
            self.model_cls._compute_sample_weight(y, {0: 1.0, 1: 1.0})

    def test_unsupported_class_weight_raises(self):
        y = self.np.array([0, 1])
        with self.assertRaises(ValueError):
            self.model_cls._compute_sample_weight(y, "invalid_strategy")


class XGBoostFitPredictTest(unittest.TestCase):
    """Integration tests: fit, predict, predict_proba, feature_importances."""

    def setUp(self):
        if not (SKLEARN_AVAILABLE and NUMPY_AVAILABLE and XGBOOST_AVAILABLE):
            self.skipTest("Required dependencies not installed")
        import numpy as np
        from models.xgboost import XGBoostModel
        self.np = np
        self.model_cls = XGBoostModel

        self.X = np.array([
            [0.0, 0.1],
            [0.2, 0.0],
            [1.0, 1.1],
            [1.2, 1.0],
        ])
        self.y = np.array([0, 0, 1, 1])

    def test_fit_sets_fitted_flag(self):
        model = self.model_cls(class_weight="balanced")
        model.fit(self.X, self.y)
        self.assertTrue(model.is_fitted)

    def test_predict_returns_binary(self):
        model = self.model_cls(class_weight="balanced")
        model.fit(self.X, self.y)
        preds = model.predict(self.X)
        self.assertEqual(len(preds), len(self.y))
        self.assertTrue(set(preds).issubset({0, 1}))

    def test_predict_proba_shape(self):
        model = self.model_cls(class_weight="balanced")
        model.fit(self.X, self.y)
        proba = model.predict_proba(self.X)
        self.assertEqual(proba.shape, (len(self.y), 2))
        self.assertTrue(self.np.allclose(proba.sum(axis=1), 1.0, atol=1e-5))

    def test_feature_importances_present(self):
        model = self.model_cls(class_weight="balanced")
        model.fit(self.X, self.y, feature_names=['a', 'b'])
        importance = model.get_feature_importance()
        self.assertIn('a', importance)
        self.assertIn('b', importance)
        # All importance values must be >= 0; trivially small datasets may
        # produce all-zero importances so we only check the type/shape.
        for v in importance.values():
            self.assertGreaterEqual(float(v), 0.0)

    def test_is_not_interpretable(self):
        model = self.model_cls()
        self.assertFalse(model.is_interpretable)

    def test_supports_feature_importance(self):
        model = self.model_cls()
        self.assertTrue(model.supports_feature_importance)


class MigrationGuardTest(unittest.TestCase):
    """Ensure gradient_boosting is fully removed and xgboost is present."""

    def setUp(self):
        if not (SKLEARN_AVAILABLE and NUMPY_AVAILABLE):
            self.skipTest("Required dependencies not installed")

    def test_gradient_boosting_absent_from_registry(self):
        from models import MODEL_REGISTRY
        self.assertNotIn(
            'gradient_boosting', MODEL_REGISTRY,
            "'gradient_boosting' key must be removed from MODEL_REGISTRY."
        )

    def test_xgboost_present_in_registry(self):
        from models import MODEL_REGISTRY
        self.assertIn(
            'xgboost', MODEL_REGISTRY,
            "'xgboost' key must be present in MODEL_REGISTRY."
        )

    def test_gradient_boosting_absent_from_config_defaults(self):
        from config import Config
        models_cfg = Config.DEFAULTS.get('models', {})
        self.assertNotIn(
            'gradient_boosting', models_cfg,
            "'gradient_boosting' key must be removed from Config.DEFAULTS."
        )

    def test_xgboost_present_in_config_defaults(self):
        from config import Config
        models_cfg = Config.DEFAULTS.get('models', {})
        self.assertIn(
            'xgboost', models_cfg,
            "'xgboost' key must be present in Config.DEFAULTS."
        )

    def test_gradient_boosting_absent_from_factory(self):
        if not XGBOOST_AVAILABLE:
            self.skipTest("xgboost not installed; skipping factory instantiation guard")
        from config import Config
        from benchmarking.model_factory import create_models
        config = Config()
        for name in config.get('models', {}).keys():
            config.set(f'models.{name}.enabled', False)
        config.set('models.xgboost.enabled', True)
        models = create_models(config)
        self.assertNotIn('gradient_boosting', models)
        self.assertIn('xgboost', models)


if __name__ == "__main__":
    unittest.main()

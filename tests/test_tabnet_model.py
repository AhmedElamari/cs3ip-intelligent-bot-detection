"""
Tests for TabNetModel and TabNetPrep.

Covers:
  - Dependency guard raises ImportError with install instructions
  - fit() sets is_fitted flag
  - predict() returns binary labels with correct shape
  - predict_proba() returns Nx2 floats summing to 1
  - get_feature_importance() returns dict with correct keys
  - is_interpretable is True; supports_feature_importance is True
  - prepare_eval_set() registers eval data without error
  - _safe_batch_sizes() caps correctly
  - TabNetPrep fit_transform produces float32 array and metadata
  - TabNetPrep transform raises if not fitted
  - sample_weights helper handles balanced and dict specs
"""

import importlib.util
import sys
import unittest
from pathlib import Path
from types import ModuleType
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

TABNET_AVAILABLE = (
    importlib.util.find_spec("pytorch_tabnet") is not None
    and importlib.util.find_spec("torch") is not None
)
NUMPY_AVAILABLE = importlib.util.find_spec("numpy") is not None


class TabNetPrepTest(unittest.TestCase):
    """Unit tests for TabNetPrep (no pytorch required)."""

    def setUp(self):
        if not NUMPY_AVAILABLE:
            self.skipTest("numpy not installed")
        import numpy as np
        import pandas as pd
        self.np = np
        self.pd = pd
        from benchmarking.tabnet_prep import TabNetPrep, TabNetMeta
        self.PrepCls = TabNetPrep
        self.MetaCls = TabNetMeta

    def _make_df(self, n=20):
        rng = self.np.random.default_rng(42)
        return self.pd.DataFrame({
            "a": rng.standard_normal(n),
            "b": rng.standard_normal(n),
        })

    def test_fit_transform_returns_float32(self):
        prep = self.PrepCls()
        X, meta = prep.fit_transform(self._make_df())
        self.assertEqual(X.dtype, self.np.float32)

    def test_fit_transform_returns_meta(self):
        prep = self.PrepCls()
        _, meta = prep.fit_transform(self._make_df())
        self.assertIsInstance(meta, self.MetaCls)
        self.assertEqual(meta.feature_names, ["a", "b"])
        self.assertEqual(meta.cat_idxs, [])
        self.assertEqual(meta.cat_dims, [])

    def test_transform_produces_consistent_shape(self):
        prep = self.PrepCls()
        X_tr, _ = prep.fit_transform(self._make_df(20))
        X_val = prep.transform(self._make_df(5))
        self.assertEqual(X_val.shape[1], X_tr.shape[1])

    def test_transform_before_fit_raises(self):
        with self.assertRaises(RuntimeError):
            self.PrepCls().transform(self._make_df())

    def test_nan_imputation_uses_train_median(self):
        import pandas as pd
        import numpy as np
        # Insert NaN in val; should be filled with training median
        prep = self.PrepCls()
        X_tr = pd.DataFrame({"a": [1.0, 2.0, 3.0, 4.0]})
        prep.fit_transform(X_tr)
        X_v = pd.DataFrame({"a": [float("nan")]})
        X_out = prep.transform(X_v)
        expected_median = 2.5  # median of [1, 2, 3, 4]
        self.assertAlmostEqual(float(X_out[0, 0]), expected_median)

    def test_numpy_input_accepted(self):
        import numpy as np
        prep = self.PrepCls()
        X_np = np.random.default_rng(1).standard_normal((10, 3)).astype(np.float32)
        X_out, meta = prep.fit_transform(X_np)
        self.assertEqual(X_out.shape, (10, 3))


class TabNetSafeBatchSizesTest(unittest.TestCase):
    """Unit tests for _safe_batch_sizes (no pytorch required)."""

    def setUp(self):
        if not NUMPY_AVAILABLE:
            self.skipTest("numpy not installed")

    def test_caps_batch_size_to_n_samples(self):
        from models.tabnet import TabNetModel
        bs, vbs = TabNetModel._safe_batch_sizes(50, 1024, 128)
        self.assertEqual(bs, 50)
        self.assertLessEqual(vbs, bs)

    def test_virtual_batch_size_divides_batch_size(self):
        from models.tabnet import TabNetModel
        bs, vbs = TabNetModel._safe_batch_sizes(512, 512, 128)
        self.assertEqual(bs % vbs, 0)

    def test_minimum_virtual_batch_size_is_one(self):
        from models.tabnet import TabNetModel
        _, vbs = TabNetModel._safe_batch_sizes(3, 3, 128)
        self.assertGreaterEqual(vbs, 1)

    def test_create_model_forwards_device_name(self):
        from models.tabnet import TabNetModel

        fake_module = ModuleType("pytorch_tabnet.tab_model")
        captured = {}

        class FakeTabNetClassifier:
            def __init__(self, **kwargs):
                captured.update(kwargs)

        fake_module.TabNetClassifier = FakeTabNetClassifier
        with mock.patch.dict(sys.modules, {"pytorch_tabnet.tab_model": fake_module}):
            model = TabNetModel(device_name="cuda")
            model._create_model(**model.get_params())

        self.assertEqual(captured.get("device_name"), "cuda")


class SampleWeightsTest(unittest.TestCase):
    """Unit tests for _sample_weights helper."""

    def setUp(self):
        if not NUMPY_AVAILABLE:
            self.skipTest("numpy not installed")
        import numpy as np
        self.np = np

    def test_none_returns_none(self):
        from models.tabnet import _sample_weights
        self.assertIsNone(_sample_weights(self.np.array([0, 1]), None))

    def test_balanced_minority_higher_weight(self):
        from models.tabnet import _sample_weights
        y = self.np.array([0, 0, 0, 1])
        w = _sample_weights(y, "balanced")
        self.assertGreater(w[3], w[0])

    def test_dict_weights_applied(self):
        from models.tabnet import _sample_weights
        y = self.np.array([0, 1, 1])
        w = _sample_weights(y, {0: 1.0, 1: 3.0})
        self.assertListEqual(w.tolist(), [1.0, 3.0, 3.0])

    def test_dict_weight_missing_label_raises(self):
        from models.tabnet import _sample_weights
        with self.assertRaises(ValueError) as ctx:
            _sample_weights(self.np.array([0, 1]), {1: 2.0})
        self.assertIn("class_weight dict missing label(s): [0]", str(ctx.exception))

    def test_unsupported_spec_raises(self):
        from models.tabnet import _sample_weights
        with self.assertRaises(ValueError):
            _sample_weights(self.np.array([0, 1]), "unknown")


@unittest.skipUnless(TABNET_AVAILABLE, "pytorch-tabnet not installed")
class TabNetModelTest(unittest.TestCase):
    """Integration tests requiring pytorch-tabnet."""

    def setUp(self):
        import numpy as np
        rng = np.random.default_rng(2112)
        self.X = rng.standard_normal((80, 5)).astype(np.float32)
        self.y = np.array([0] * 40 + [1] * 40, dtype=int)
        self.X_val = rng.standard_normal((20, 5)).astype(np.float32)
        self.y_val = np.array([0] * 10 + [1] * 10, dtype=int)
        self.np = np

    def _make_fitted(self):
        from models.tabnet import TabNetModel
        model = TabNetModel(
            random_state=2112,
            n_d=8, n_a=8, n_steps=1,
            batch_size=32, virtual_batch_size=8,
            max_epochs=2, patience=2,
        )
        model.fit(self.X, self.y, feature_names=[f"f{i}" for i in range(5)])
        return model

    def test_fit_sets_is_fitted(self):
        model = self._make_fitted()
        self.assertTrue(model.is_fitted)

    def test_runtime_metadata_after_fit(self):
        model = self._make_fitted()
        meta = model.get_runtime_metadata()
        self.assertIsInstance(meta, dict)
        self.assertIn("requested_device", meta)
        self.assertIn("actual_device", meta)
        self.assertIsInstance(meta.get("cuda_available"), bool)

    def test_predict_returns_binary(self):
        model = self._make_fitted()
        preds = model.predict(self.X)
        self.assertEqual(preds.shape[0], 80)
        self.assertTrue(set(preds.tolist()).issubset({0, 1}))

    def test_predict_proba_shape_and_sum(self):
        model = self._make_fitted()
        proba = model.predict_proba(self.X)
        self.assertEqual(proba.shape, (80, 2))
        self.assertTrue(self.np.allclose(proba.sum(axis=1), 1.0, atol=1e-4))

    def test_feature_importance_keys(self):
        model = self._make_fitted()
        fi = model.get_feature_importance()
        self.assertIsNotNone(fi)
        self.assertEqual(set(fi.keys()), {f"f{i}" for i in range(5)})

    def test_is_interpretable(self):
        from models.tabnet import TabNetModel
        self.assertTrue(TabNetModel(max_epochs=1).is_interpretable)

    def test_supports_feature_importance(self):
        from models.tabnet import TabNetModel
        self.assertTrue(TabNetModel(max_epochs=1).supports_feature_importance)

    def test_prepare_eval_set_registers_data(self):
        from models.tabnet import TabNetModel
        model = TabNetModel(max_epochs=1)
        model.prepare_eval_set(self.X_val, self.y_val)
        self.assertIsNotNone(model._eval_set)
        self.assertEqual(len(model._eval_set), 1)

    def test_predict_before_fit_raises(self):
        from models.tabnet import TabNetModel
        with self.assertRaises(RuntimeError):
            TabNetModel().predict(self.X)

    def test_save_metadata_excludes_model(self):
        import pickle
        import tempfile

        model = self._make_fitted()
        with tempfile.TemporaryDirectory(dir=ROOT) as tmp_dir:
            path = Path(tmp_dir) / "tabnet_model.pkl"
            model.save(str(path))

            with open(path, "rb") as f:
                data = pickle.load(f)
            self.assertNotIn("model", data)
            self.assertEqual(data["feature_names"], model.feature_names)
            self.assertIsNotNone(data.get("tabnet_path"))

    def test_load_restores_from_tabnet_path(self):
        import tempfile
        from models.tabnet import TabNetModel

        model = self._make_fitted()
        with tempfile.TemporaryDirectory(dir=ROOT) as tmp_dir:
            path = Path(tmp_dir) / "tabnet_model.pkl"
            model.save(str(path))

            loaded = TabNetModel.load(str(path), trusted_source=True)
            self.assertTrue(loaded.is_fitted)
            self.assertEqual(loaded.feature_names, model.feature_names)


SHAP_AVAILABLE = importlib.util.find_spec("shap") is not None


@unittest.skipUnless(TABNET_AVAILABLE and SHAP_AVAILABLE, "pytorch-tabnet or shap not installed")
class TabNetSHAPIntegrationTest(unittest.TestCase):
    """Verify SHAP KernelExplainer path works for a fitted TabNetModel."""

    def setUp(self):
        import numpy as np
        rng = np.random.default_rng(2112)
        self.X = rng.standard_normal((60, 4)).astype(np.float32)
        self.y = np.array([0] * 30 + [1] * 30, dtype=int)

    def test_shap_explainer_produces_values(self):
        import numpy as np
        from models.tabnet import TabNetModel
        from explainability.shap_explainer import SHAPExplainer

        model = TabNetModel(
            random_state=2112,
            n_d=8, n_a=8, n_steps=1,
            batch_size=32, virtual_batch_size=8,
            max_epochs=2, patience=2,
        )
        model.fit(self.X, self.y, feature_names=[f"f{i}" for i in range(4)])

        explainer = SHAPExplainer(model, [f"f{i}" for i in range(4)])
        explainer.fit(self.X, max_samples=20)
        explainer.explain(self.X[:5])

        self.assertIsNotNone(explainer.shap_values)
        importance = explainer.get_global_importance()
        self.assertEqual(set(importance.keys()), {f"f{i}" for i in range(4)})


@unittest.skipUnless(not TABNET_AVAILABLE, "pytorch-tabnet IS installed - skip dep-guard test")
class TabNetDependencyGuardTest(unittest.TestCase):
    """Verify ImportError is raised with install instructions when deps missing."""

    def test_import_error_with_instructions(self):
        from models.tabnet import _require_tabnet
        with self.assertRaises(ImportError) as ctx:
            _require_tabnet()
        self.assertIn("requirements-dl.txt", str(ctx.exception))


class TabNetLegacyLoadTest(unittest.TestCase):
    def test_load_falls_back_to_legacy_pickle(self):
        import pickle
        import tempfile
        from models.tabnet import TabNetModel

        payload = {
            "name": "TabNet",
            "feature_names": ["f0"],
            "params": {},
            "training_time": 0.0,
            "model": "legacy-model",
        }

        with tempfile.TemporaryDirectory(dir=ROOT) as tmp_dir:
            path = Path(tmp_dir) / "legacy_tabnet.pkl"
            with open(path, "wb") as f:
                pickle.dump(payload, f)

            loaded = TabNetModel.load(str(path), trusted_source=True)
            self.assertEqual(loaded.model, "legacy-model")

if __name__ == "__main__":
    unittest.main()

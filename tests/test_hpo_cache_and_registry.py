"""HPO cache signature stability and registry suggest smoke tests."""
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from config import Config
from benchmarking.hpo.cache import compute_signature
from benchmarking.hpo.registry import get
from benchmarking.hpo.factory import build_model


class TestHPOCacheSignature(unittest.TestCase):
    def test_signature_stable_identical_config(self):
        cfg = Config()
        names = ["f1", "f2"]
        data_dir = Path(__file__).resolve().parents[1] / "data"
        s1 = compute_signature(
            "logistic_regression",
            cfg,
            names,
            data_dir,
            "v1",
        )
        s2 = compute_signature(
            "logistic_regression",
            cfg,
            names,
            data_dir,
            "v1",
        )
        self.assertEqual(s1, s2)

    def test_signature_changes_with_search_space_version(self):
        cfg = Config()
        names = ["f1"]
        data_dir = Path(__file__).resolve().parents[1] / "data"
        s1 = compute_signature("random_forest", cfg, names, data_dir, "v1")
        s2 = compute_signature("random_forest", cfg, names, data_dir, "v2")
        self.assertNotEqual(s1, s2)

    def test_signature_changes_when_fixed_model_params_change(self):
        cfg_a = Config()
        cfg_b = Config()
        names = ["f1"]
        data_dir = Path(__file__).resolve().parents[1] / "data"

        cfg_a.set("models.tabnet.params.max_epochs", 200)
        cfg_b.set("models.tabnet.params.max_epochs", 500)

        s1 = compute_signature("tabnet", cfg_a, names, data_dir, "v1")
        s2 = compute_signature("tabnet", cfg_b, names, data_dir, "v1")

        self.assertNotEqual(s1, s2)


class TestHPORegistrySuggestConstructs(unittest.TestCase):
    """Each registry entry's suggested params must construct the wrapper."""

    def test_suggested_params_construct(self):
        cfg = Config()
        rng = np.random.default_rng(2112)
        for name in (
            "logistic_regression",
            "svm",
            "decision_tree",
            "random_forest",
            "xgboost",
        ):
            with self.subTest(model=name):
                entry = get(name)
                params = cfg.get_model_params(name)
                trial_params = entry.suggest_fn(_FakeTrial(rng))
                merged = {**params, **trial_params}
                m = build_model(name, merged)
                self.assertIsNotNone(m.model)


class _FakeTrial:
    """Minimal Optuna trial API surface for suggest_* calls."""

    def __init__(self, rng: np.random.Generator):
        self._rng = rng

    def suggest_float(self, name: str, low: float, high: float, log: bool = False):
        u = self._rng.random()
        if log:
            return float(np.exp(np.log(low) + u * (np.log(high) - np.log(low))))
        return float(low + u * (high - low))

    def suggest_int(self, name: str, low: int, high: int, step: int = 1):
        return int(self._rng.integers(low, high + 1))

    def suggest_categorical(self, name: str, choices: list):
        idx = int(self._rng.integers(0, len(choices)))
        return choices[idx]

    def report(self, *args, **kwargs):
        pass

    def should_prune(self):
        return False


if __name__ == "__main__":
    unittest.main()

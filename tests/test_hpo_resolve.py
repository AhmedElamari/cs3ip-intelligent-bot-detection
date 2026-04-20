"""resolve_hpo behaviour (no-tune skip, unknown model fail-fast)."""
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from config import Config
from benchmarking.hpo.service import HPOCliOverrides, optimize_model, resolve_hpo


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

    def test_disabled_hpo_skips_unknown_model_without_registry_entry(self):
        cfg = Config()
        cfg.set("hpo.enabled", False)
        X = np.zeros((4, 2))
        y = np.array([0, 1, 0, 1])
        res, audit = resolve_hpo(
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
        self.assertEqual(res.get("status"), "skipped")
        self.assertTrue(audit.get("skipped"))


class TestOptimizeModelPrepReuse(unittest.TestCase):
    def test_build_model_inputs_runs_once_per_hpo_run(self):
        cfg = Config()
        X_train = np.array([[0.0], [1.0]], dtype=np.float64)
        y_train = np.array([0, 1])
        X_val = np.array([[0.0]], dtype=np.float64)
        y_val = np.array([0])

        calls = {"count": 0}

        def fake_build_model_inputs(*args, **kwargs):
            calls["count"] += 1
            return mock.Mock(
                X_train=X_train,
                X_val=X_val,
                X_test=X_val,
                tabnet_meta=None,
            )

        class Trial:
            def __init__(self):
                self.params = {}
                self.value = 0.5

            def report(self, *args, **kwargs):
                return None

            def should_prune(self):
                return False

            def suggest_float(self, *args, **kwargs):
                return 1.0

            def suggest_categorical(self, name, choices):
                return choices[0]

            def suggest_int(self, *args, **kwargs):
                return 2

        class Study:
            def __init__(self):
                self.trials = []
                self.best_trial = Trial()

            def optimize(self, objective, n_trials, show_progress_bar=False):
                for _ in range(n_trials):
                    trial = Trial()
                    self.trials.append(trial)
                    objective(trial)

        class FakeOptuna:
            class logging:
                WARNING = 0

                @staticmethod
                def set_verbosity(level):
                    return None

            class samplers:
                class TPESampler:
                    def __init__(self, seed):
                        self.seed = seed

            @staticmethod
            def create_study(direction, sampler, pruner):
                return Study()

        with mock.patch(
            "benchmarking.hpo.service.require_optuna",
            return_value=FakeOptuna,
        ), mock.patch(
            "benchmarking.hpo.service.build_model_inputs",
            side_effect=fake_build_model_inputs,
        ), mock.patch(
            "benchmarking.hpo.service._fit_and_val_f1",
            return_value=0.5,
        ):
            optimize_model(
                "random_forest",
                X_train,
                y_train,
                X_val,
                y_val,
                config=cfg,
                n_trials=3,
                seed=2112,
                enable_scaling=False,
                class_weights=None,
                feature_names=["a"],
            )

        self.assertEqual(calls["count"], 1)


if __name__ == "__main__":
    unittest.main()

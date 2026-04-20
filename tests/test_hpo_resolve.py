"""resolve_hpo and optimize_model regression coverage."""
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from config import Config
from benchmarking.hpo.service import HPOCliOverrides, optimize_model, resolve_hpo

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
X_TRAIN = np.array([[0.0], [1.0]], dtype=np.float64)
Y_TRAIN = np.array([0, 1])
X_VAL = np.array([[0.0]], dtype=np.float64)
Y_VAL = np.array([0])


def _resolve_kwargs():
    return {
        "X_train": np.zeros((4, 2)),
        "y_train": np.array([0, 1, 0, 1]),
        "X_val": np.zeros((2, 2)),
        "y_val": np.array([0, 1]),
        "feature_names_ordered": ["a", "b"],
        "data_dir": DATA_DIR,
        "enable_scaling": False,
    }


def _prepared_inputs(*, tabnet_meta=None):
    return mock.Mock(
        X_train=X_TRAIN,
        X_val=X_VAL,
        X_test=X_VAL,
        tabnet_meta=tabnet_meta,
    )


class _Trial:
    def __init__(self, *, float_values=None):
        self._float_values = float_values or {}
        self.params = {}
        self.user_attrs = {}
        self.value = 0.0

    def report(self, *args, **kwargs):
        return None

    def should_prune(self):
        return False

    def set_user_attr(self, key, value):
        self.user_attrs[key] = value

    def suggest_float(self, name, *args, **kwargs):
        value = self._float_values.get(name, 1.0)
        self.params[name] = value
        return value

    def suggest_categorical(self, name, choices):
        value = choices[0]
        self.params[name] = value
        return value

    def suggest_int(self, name, *args, **kwargs):
        value = 2
        self.params[name] = value
        return value


class _Study:
    def __init__(self, trial_factory):
        self._trial_factory = trial_factory
        self.trials = []
        self.best_trial = trial_factory()

    def optimize(self, objective, n_trials, show_progress_bar=False):
        for _ in range(n_trials):
            trial = self._trial_factory()
            trial.value = objective(trial)
            self.trials.append(trial)
            self.best_trial = trial


def _fake_optuna(trial_factory):
    study = _Study(trial_factory)

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
            return study

    return FakeOptuna


class TestResolveHPO(unittest.TestCase):
    def _resolve(self, config, model_name, *, class_weights=None, cli=None):
        return resolve_hpo(
            model_name,
            config,
            class_weights=class_weights,
            cli=cli,
            **_resolve_kwargs(),
        )

    def test_no_tune_skips(self):
        cfg = Config()
        cfg.set("hpo.enabled", True)
        res, audit = self._resolve(
            cfg,
            "logistic_regression",
            class_weights={0: 1.0, 1: 1.0},
            cli=HPOCliOverrides(no_tune=True),
        )
        self.assertEqual(res.get("status"), "skipped")
        self.assertTrue(audit.get("skipped"))

    def test_unknown_model_raises_when_fail_fast(self):
        cfg = Config()
        cfg.set("hpo.fail_fast", True)
        with self.assertRaises(ValueError) as ctx:
            self._resolve(cfg, "not_a_real_model")
        self.assertIn("not_a_real_model", str(ctx.exception))

    def test_disabled_hpo_skips_unknown_model_without_registry_entry(self):
        cfg = Config()
        cfg.set("hpo.enabled", False)
        res, audit = self._resolve(cfg, "not_a_real_model")
        self.assertEqual(res.get("status"), "skipped")
        self.assertTrue(audit.get("skipped"))

    def test_missing_registry_entry_skips_when_fail_fast_disabled(self):
        cfg = Config()
        cfg.set("hpo.fail_fast", False)
        res, audit = self._resolve(cfg, "not_a_real_model")
        self.assertEqual(res.get("status"), "skipped")
        self.assertEqual(res.get("search_space_version"), "missing")
        self.assertTrue(audit.get("skipped"))


class TestOptimizeModel(unittest.TestCase):
    def test_build_model_inputs_runs_once_per_hpo_run(self):
        cfg = Config()
        calls = {"count": 0}

        def fake_build_model_inputs(*args, **kwargs):
            calls["count"] += 1
            return _prepared_inputs()

        with mock.patch(
            "benchmarking.hpo.service.require_optuna",
            return_value=_fake_optuna(_Trial),
        ), mock.patch(
            "benchmarking.hpo.service.build_model_inputs",
            side_effect=fake_build_model_inputs,
        ), mock.patch(
            "benchmarking.hpo.service._fit_and_val_f1",
            return_value=0.5,
        ):
            optimize_model(
                "random_forest",
                X_TRAIN,
                Y_TRAIN,
                X_VAL,
                Y_VAL,
                config=cfg,
                n_trials=3,
                seed=2112,
                enable_scaling=False,
                class_weights=None,
                feature_names=["a"],
            )

        self.assertEqual(calls["count"], 1)

    def test_returns_normalized_model_params_not_raw_optuna_names(self):
        cfg = Config()
        fake_entry = mock.Mock(
            requires_dl=False,
            search_space_version="v-test",
            pruner_factory=None,
        )
        fake_entry.suggest_fn = lambda trial: {
            "gamma": trial.suggest_float("gamma_poly", 1e-4, 1.0, log=True),
        }
        fake_entry.score_fn = mock.Mock(return_value=0.5)

        with mock.patch(
            "benchmarking.hpo.service.require_optuna",
            return_value=_fake_optuna(lambda: _Trial(float_values={"gamma_poly": 0.25})),
        ), mock.patch(
            "benchmarking.hpo.service.build_model_inputs",
            return_value=_prepared_inputs(),
        ), mock.patch(
            "benchmarking.hpo.service._fit_and_val_f1",
            return_value=0.5,
        ):
            result = optimize_model(
                "svm",
                X_TRAIN,
                Y_TRAIN,
                X_VAL,
                Y_VAL,
                config=cfg,
                n_trials=1,
                seed=2112,
                enable_scaling=False,
                class_weights=None,
                feature_names=["a"],
                entry=fake_entry,
            )

        self.assertEqual(result["best_params"], {"gamma": 0.25})

    def test_tabnet_device_is_forwarded_to_model_build(self):
        cfg = Config()
        fake_entry = mock.Mock(
            requires_dl=True,
            search_space_version="v-test",
            pruner_factory=None,
            suggest_fn=lambda trial: {"n_d": 8, "n_a": 8},
            score_fn=mock.Mock(return_value=0.5),
        )
        build_model = mock.Mock(
            return_value=mock.Mock(
                fit=mock.Mock(return_value=None),
                predict=mock.Mock(return_value=np.array([0])),
            )
        )

        with mock.patch(
            "benchmarking.hpo.service.require_optuna",
            return_value=_fake_optuna(_Trial),
        ), mock.patch(
            "benchmarking.hpo.service.require_tabnet_dl",
            return_value=None,
        ), mock.patch(
            "benchmarking.hpo.service.build_model_inputs",
            return_value=_prepared_inputs(
                tabnet_meta=mock.Mock(
                    feature_names=["feature_0"],
                    cat_idxs=[],
                    cat_dims=[],
                )
            ),
        ), mock.patch(
            "benchmarking.hpo.service.build_model",
            build_model,
        ):
            optimize_model(
                "tabnet",
                X_TRAIN,
                Y_TRAIN,
                X_VAL,
                Y_VAL,
                config=cfg,
                n_trials=1,
                seed=2112,
                enable_scaling=False,
                class_weights=None,
                feature_names=["real_feature"],
                device="cuda",
                entry=fake_entry,
            )

        self.assertEqual(build_model.call_args.args[1]["device_name"], "cuda")


if __name__ == "__main__":
    unittest.main()

"""Tests for top-scoreboard multi-seed retraining."""

import importlib.util
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

SKLEARN = importlib.util.find_spec("sklearn") is not None
NUMPY = importlib.util.find_spec("numpy") is not None
PANDAS = importlib.util.find_spec("pandas") is not None


def _make_synthetic_splits(n_samples: int = 60):
    import numpy as np
    import pandas as pd

    rng = np.random.default_rng(2112)
    df = pd.DataFrame(
        {
            "account_creation_date": pd.date_range("2020-01-01", periods=n_samples, freq="D"),
            "is_verified": rng.integers(0, 2, size=n_samples),
            "followers_count": rng.integers(0, 100, size=n_samples),
            "friends_count": rng.integers(0, 100, size=n_samples),
            "listed_count": rng.integers(0, 20, size=n_samples),
            "statuses_count": rng.integers(0, 200, size=n_samples),
            "favourites_count": rng.integers(0, 300, size=n_samples),
            "default_profile": rng.integers(0, 2, size=n_samples),
            "default_profile_image": rng.integers(0, 2, size=n_samples),
            "has_extended_profile": rng.integers(0, 2, size=n_samples),
            "geo_enabled": rng.integers(0, 2, size=n_samples),
            "protected": rng.integers(0, 2, size=n_samples),
            "description": ["test user"] * n_samples,
            "url": [None] * n_samples,
            "screen_name": [f"user_{i}" for i in range(n_samples)],
            "tweet_count": rng.integers(0, 50, size=n_samples),
            "following_sample_count": rng.integers(0, 50, size=n_samples),
            "follower_sample_count": rng.integers(0, 50, size=n_samples),
            "label": [0, 1] * (n_samples // 2),
        }
    )
    return {
        "train": df.iloc[: int(n_samples * 0.6)].copy(),
        "val": df.iloc[int(n_samples * 0.6) : int(n_samples * 0.8)].copy(),
        "test": df.iloc[int(n_samples * 0.8) :].copy(),
    }


@unittest.skipUnless(SKLEARN and NUMPY and PANDAS, "sklearn/numpy/pandas required")
class MultiSeedRetrainingTest(unittest.TestCase):
    def test_writes_artifacts_for_two_seeds_top2(self):
        from benchmarking import ModelBenchmark
        from benchmarking.data_prep import prepare_data
        from benchmarking.model_factory import create_models
        from benchmarking.multi_seed import run_multi_seed_retraining
        from config import Config

        splits = _make_synthetic_splits(80)
        config = Config()
        for name in config.get("models", {}):
            config.set(f"models.{name}.enabled", name in ("logistic_regression", "decision_tree"))

        X_train, X_val, X_test, y_train, y_val, y_test, feature_names = prepare_data(splits, config)
        models = create_models(config)
        benchmark = ModelBenchmark(models=models, experiment_name="base")
        benchmark.run_benchmark(
            X_train,
            y_train,
            X_val,
            y_val,
            X_test,
            y_test,
            feature_names=feature_names,
            verbose=False,
            compute_statistics=False,
        )

        with TemporaryDirectory(dir=str(ROOT)) as tmp:
            out = Path(tmp)
            payload = run_multi_seed_retraining(
                benchmark=benchmark,
                config=config,
                X_train=X_train,
                y_train=y_train,
                X_val=X_val,
                y_val=y_val,
                X_test=X_test,
                y_test=y_test,
                feature_names=feature_names,
                output_dir=out,
                seeds=[2112, 2113],
                top_k=2,
                enable_scaling=False,
            )

            self.assertEqual("MultiSeedRetrainingV1", payload["schema_version"])
            self.assertEqual("ok", payload["status"])
            self.assertEqual(2, len(payload["top_models"]))
            self.assertEqual(4, len(payload["rows"]))
            self.assertTrue((out / "multi_seed_retraining.csv").is_file())
            self.assertTrue((out / "multi_seed_summary.csv").is_file())
            self.assertTrue((out / "multi_seed_retraining.json").is_file())


import sys
import unittest
from pathlib import Path
from unittest import mock

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import main
from config import Config


class _DetectorStub:
    selected_features = []

    def get_class_weights(self, y):
        return {0: 1.0, 1: 1.0}


class MainHPOSkipContractTest(unittest.TestCase):
    def _build_splits(self):
        frame = pd.DataFrame(
            {
                "account_creation_date": pd.date_range("2020-01-01", periods=9, freq="D"),
                "feature_a": [0.0, 1.0, 0.0, 1.0, 0.2, 0.8, 0.1, 0.9, 0.5],
                "feature_b": [1.0, 0.0, 1.0, 0.0, 0.8, 0.2, 0.9, 0.1, 0.5],
                "label": [0, 1, 0, 1, 0, 1, 0, 1, 0],
            }
        )
        return {
            "train": frame.iloc[:5].copy(),
            "val": frame.iloc[5:7].copy(),
            "test": frame.iloc[7:].copy(),
        }

    def _run_pipeline(self, *, config: Config, no_tune: bool) -> None:
        with mock.patch.object(
            main,
            "load_and_prepare_data",
            return_value=self._build_splits(),
        ), mock.patch.object(
            main,
            "engineer_features",
            side_effect=lambda df, reference_date=None: df,
        ), mock.patch.object(
            main,
            "preprocess_split",
            side_effect=lambda detector, df: df,
        ), mock.patch.object(
            main,
            "BotDetector",
            return_value=_DetectorStub(),
        ), mock.patch.object(
            main,
            "resolve_hpo",
            return_value=(
                {
                    "schema_version": "HPOResultV1",
                    "status": "skipped",
                    "best_params": {},
                    "best_score": float("nan"),
                    "trial_count": 0,
                    "metric": "val_f1",
                    "seed": 2112,
                    "warnings": [],
                    "model_name": "random_forest",
                    "search_space_version": "none",
                },
                {
                    "cache_hit": False,
                    "artifact": None,
                    "search_space_version": None,
                    "trial_count": 0,
                    "best_score": None,
                    "skipped": True,
                },
            ),
        ), mock.patch.object(
            main,
            "merge_hpo_into_config_params",
        ), mock.patch.object(
            main,
            "train_and_evaluate",
            return_value={
                "model": object(),
                "val_metrics": {"f1": 1.0},
                "test_metrics": {"f1": 1.0},
            },
        ):
            main.run_pipeline(
                model_type="random_forest",
                use_smote=False,
                use_scaling=False,
                num_features=None,
                use_time_split=False,
                no_tune=no_tune,
                config=config,
            )

    def test_no_tune_does_not_require_hpo_registry_entry(self):
        cfg = Config()
        self.assertFalse(hasattr(main, "get_hpo_entry"))
        self._run_pipeline(config=cfg, no_tune=True)


if __name__ == "__main__":
    unittest.main()

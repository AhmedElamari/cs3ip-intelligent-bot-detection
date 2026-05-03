"""Concept-drift temporal splits and delta tables."""
import importlib.util
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

PANDAS_AVAILABLE = importlib.util.find_spec("pandas") is not None


class TimeStratifiedResultsTest(unittest.TestCase):
    def setUp(self):
        if not PANDAS_AVAILABLE:
            self.skipTest("pandas not installed")

    def test_build_temporal_split_dict_orders_train_before_test(self):
        import pandas as pd
        from benchmarking.time_stratified_results import build_temporal_split_dict

        df = pd.DataFrame({
            "account_creation_date": pd.date_range("2020-01-01", periods=30, freq="D"),
            "label": [i % 2 for i in range(30)],
        })
        splits = {"train": df.iloc[:10].copy(), "val": df.iloc[10:20].copy(), "test": df.iloc[20:].copy()}
        out = build_temporal_split_dict(
            splits,
            val_size=0.2,
            test_size=0.1,
            time_col="account_creation_date",
            random_state=2112,
        )
        tmax = pd.to_datetime(out["train"]["account_creation_date"]).max()
        tmin_test = pd.to_datetime(out["test"]["account_creation_date"]).min()
        self.assertLess(tmax, tmin_test)

    def test_build_concept_drift_delta(self):
        import pandas as pd
        from benchmarking.dissertation_scoreboard import DISPLAY_ORDER
        from benchmarking.time_stratified_results import build_concept_drift_delta

        main = pd.DataFrame([
            {"Rank": 1, "Model": "m", "Precision": 0.9, "Recall": 0.8, "F1-Macro": 0.85,
             "F1-Weighted": 0.84, "PR-AUC": 0.7, "ROC-AUC": 0.72, "MCC": 0.1,
             "Balanced Accuracy": 0.75, "Train Time (s)": 1.0},
        ], columns=DISPLAY_ORDER)
        drift = pd.DataFrame([
            {"Rank": 1, "Model": "m", "Precision": 0.7, "Recall": 0.6, "F1-Macro": 0.65,
             "F1-Weighted": 0.64, "PR-AUC": 0.5, "ROC-AUC": 0.52, "MCC": 0.05,
             "Balanced Accuracy": 0.55, "Train Time (s)": 1.0},
        ], columns=DISPLAY_ORDER)
        d = build_concept_drift_delta(main, drift)
        self.assertAlmostEqual(float(d.loc[0, "F1-Macro Δ (drift−baseline)"]), -0.2)

    def test_prepare_data_temporal_protocol_uses_combined_reference(self):
        from benchmarking.data_prep import prepare_data
        from config import Config

        import pandas as pd

        n = 30
        df_all = pd.DataFrame({
            "account_creation_date": pd.date_range("2020-01-01", periods=n, freq="D"),
            "label": [i % 2 for i in range(n)],
            "statuses_count": [100 + i for i in range(n)],
            "followers_count": [200 + i for i in range(n)],
            "favourites_count": [50 + i for i in range(n)],
        })
        splits = {
            "train": df_all.iloc[:15].copy(),
            "val": df_all.iloc[15:24].copy(),
            "test": df_all.iloc[24:].copy(),
        }
        config = Config({"time_split": False})
        X_train, X_val, X_test, *_ = prepare_data(
            splits, config, return_metadata=False, temporal_protocol=True
        )
        self.assertTrue((X_val["account_age_days"] > 1).any())
        self.assertTrue((X_test["account_age_days"] > 1).any())


if __name__ == "__main__":
    unittest.main()

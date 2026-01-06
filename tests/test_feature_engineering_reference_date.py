import importlib.util
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

PANDAS_AVAILABLE = importlib.util.find_spec("pandas") is not None


class ReferenceDateTest(unittest.TestCase):
    def setUp(self):
        if not PANDAS_AVAILABLE:
            self.skipTest("pandas not installed")
        import pandas as pd
        from FeatureEngineering import derive_reference_date
        self.pd = pd
        self.derive_reference_date = derive_reference_date

    def test_returns_max_for_valid_dates(self):
        df = self.pd.DataFrame({
            "account_creation_date": self.pd.to_datetime(
                ["2020-01-01", "2020-01-03", "2020-01-02"]
            )
        })
        ref = self.derive_reference_date(df)
        self.assertEqual(ref, self.pd.Timestamp("2020-01-03"))

    def test_returns_none_for_all_nat(self):
        df = self.pd.DataFrame({
            "account_creation_date": [self.pd.NaT, self.pd.NaT]
        })
        ref = self.derive_reference_date(df)
        self.assertIsNone(ref)

    def test_preserves_timezone_for_aware_dates(self):
        dates = self.pd.date_range("2021-01-01", periods=3, tz="UTC")
        df = self.pd.DataFrame({"account_creation_date": dates})
        ref = self.derive_reference_date(df)
        self.assertEqual(ref, dates.max())
        self.assertIsNotNone(ref.tzinfo)

    def test_returns_none_without_column(self):
        df = self.pd.DataFrame({"other": [1, 2, 3]})
        ref = self.derive_reference_date(df)
        self.assertIsNone(ref)


if __name__ == "__main__":
    unittest.main()

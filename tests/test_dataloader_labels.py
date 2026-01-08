import importlib.util
import json
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

PANDAS_AVAILABLE = importlib.util.find_spec("pandas") is not None


class DataLoaderLabelsTest(unittest.TestCase):
    def setUp(self):
        if not PANDAS_AVAILABLE:
            self.skipTest("pandas not installed")
        import pandas as pd
        from DataLoader import TwiBotDataLoader
        self.pd = pd
        self.loader_cls = TwiBotDataLoader

    def _write_json(self, path: Path, users: list) -> None:
        with open(path, 'w', encoding='utf-8') as handle:
            json.dump(users, handle)

    def test_embedded_labels_are_normalized(self):
        users = [
            {"ID": "1", "label": "bot", "profile": {}},
            {"ID": "2", "label": "human", "profile": {}},
        ]
        with TemporaryDirectory() as tmp:
            json_path = Path(tmp) / "data.json"
            self._write_json(json_path, users)
            loader = self.loader_cls(json_path=json_path)
            df = loader.load()
            self.assertIn("label", df.columns)
            self.assertTrue(df["label"].notna().all())
            self.assertEqual(set(df["label"].tolist()), {0, 1})

    def test_fractional_embedded_labels_are_ignored(self):
        users = [
            {"ID": "1", "label": 0.5, "profile": {}},
            {"ID": "2", "label": 1, "profile": {}},
        ]
        with TemporaryDirectory() as tmp:
            json_path = Path(tmp) / "data.json"
            self._write_json(json_path, users)
            loader = self.loader_cls(json_path=json_path)
            df = loader.load()
            labels_by_id = df.set_index("user_id")["label"]
            self.assertTrue(self.pd.isna(labels_by_id.loc["1"]))
            self.assertEqual(labels_by_id.loc["2"], 1)

    def test_string_zero_is_false_for_boolean_fields(self):
        users = [
            {
                "ID": "1",
                "profile": {
                    "protected": "0",
                    "geo_enabled": "1",
                    "default_profile": "false",
                    "default_profile_image": "true",
                },
            }
        ]
        with TemporaryDirectory() as tmp:
            json_path = Path(tmp) / "data.json"
            self._write_json(json_path, users)
            loader = self.loader_cls(json_path=json_path)
            df = loader.load()
            row = df.iloc[0]
            self.assertEqual(row["protected"], 0)
            self.assertEqual(row["geo_enabled"], 1)
            self.assertEqual(row["default_profile"], 0)
            self.assertEqual(row["default_profile_image"], 1)


if __name__ == "__main__":
    unittest.main()

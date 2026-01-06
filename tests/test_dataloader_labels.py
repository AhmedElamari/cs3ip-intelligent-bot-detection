 
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

    def test_external_labels_are_merged(self):
        users = [
            {"ID": "1", "profile": {}},
            {"ID": "2", "profile": {}},
        ]
        labels = self.pd.DataFrame({
            "ID": ["1", "2"],
            "label": ["bot", "human"],
        })
        with TemporaryDirectory() as tmp:
            json_path = Path(tmp) / "data.json"
            csv_path = Path(tmp) / "labels.csv"
            self._write_json(json_path, users)
            labels.to_csv(csv_path, index=False)
            loader = self.loader_cls(json_path=json_path, label_path=str(csv_path))
            df = loader.load()
            self.assertIn("label", df.columns)
            self.assertTrue(df["label"].notna().all())
            self.assertEqual(set(df["label"].tolist()), {0, 1})

    def test_missing_label_column_raises(self):
        users = [{"ID": "1", "profile": {}}]
        labels = self.pd.DataFrame({"ID": ["1"]})
        with TemporaryDirectory() as tmp:
            json_path = Path(tmp) / "data.json"
            csv_path = Path(tmp) / "labels.csv"
            self._write_json(json_path, users)
            labels.to_csv(csv_path, index=False)
            loader = self.loader_cls(json_path=json_path, label_path=str(csv_path))
            with self.assertRaises(ValueError) as ctx:
                loader.load()
            self.assertIn("label", str(ctx.exception).lower())

    def test_non_binary_labels_raise(self):
        users = [{"ID": "1", "profile": {}}]
        labels = self.pd.DataFrame({"ID": ["1"], "label": [2]})
        with TemporaryDirectory() as tmp:
            json_path = Path(tmp) / "data.json"
            csv_path = Path(tmp) / "labels.csv"
            self._write_json(json_path, users)
            labels.to_csv(csv_path, index=False)
            loader = self.loader_cls(json_path=json_path, label_path=str(csv_path))
            with self.assertRaises(ValueError) as ctx:
                loader.load()
            self.assertIn("binary", str(ctx.exception).lower())

    def test_label_merge_with_no_matches_raises(self):
        users = [{"ID": "1", "profile": {}}]
        labels = self.pd.DataFrame({"ID": ["999"], "label": [1]})
        with TemporaryDirectory() as tmp:
            json_path = Path(tmp) / "data.json"
            csv_path = Path(tmp) / "labels.csv"
            self._write_json(json_path, users)
            labels.to_csv(csv_path, index=False)
            loader = self.loader_cls(json_path=json_path, label_path=str(csv_path))
            with self.assertRaises(ValueError) as ctx:
                loader.load()
            self.assertIn("zero matches", str(ctx.exception).lower())


if __name__ == "__main__":
    unittest.main()

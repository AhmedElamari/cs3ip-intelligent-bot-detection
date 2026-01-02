import importlib.util
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

ROOT = Path(__file__).resolve().parents[1]

PANDAS_AVAILABLE = importlib.util.find_spec("pandas") is not None


class DataLoaderLabelsTest(unittest.TestCase):
    def setUp(self):
        if not PANDAS_AVAILABLE:
            self.skipTest("pandas not installed")
        self.temp_dir = TemporaryDirectory()
        self.temp_path = Path(self.temp_dir.name)

    def tearDown(self):
        if hasattr(self, "temp_dir"):
            self.temp_dir.cleanup()

    def _write_json(self, path: Path) -> None:
        data = [
            {
                "ID": 1,
                "profile": {
                    "id": 1,
                    "created_at": "Wed Oct 10 20:19:24 +0000 2018",
                    "verified": False,
                },
                "tweet": [],
                "neighbor": {},
                "domain": [],
            }
        ]
        path.write_text(json.dumps(data), encoding="utf-8")

    def test_missing_labels_file_raises(self):
        from DataLoader import TwiBotDataLoader

        json_path = self.temp_path / "sample.json"
        self._write_json(json_path)
        labels_path = self.temp_path / "missing_labels.csv"

        loader = TwiBotDataLoader(str(json_path), str(labels_path))
        with self.assertRaises(FileNotFoundError):
            loader.load()

    def test_missing_label_column_raises(self):
        import pandas as pd
        from DataLoader import TwiBotDataLoader

        json_path = self.temp_path / "sample.json"
        self._write_json(json_path)

        labels_path = self.temp_path / "labels.csv"
        pd.DataFrame({"ID": [1]}).to_csv(labels_path, index=False)

        loader = TwiBotDataLoader(str(json_path), str(labels_path))
        with self.assertRaises(ValueError):
            loader.load()

    def test_label_merge_with_user_id(self):
        import pandas as pd
        from DataLoader import TwiBotDataLoader

        json_path = self.temp_path / "sample.json"
        self._write_json(json_path)

        labels_path = self.temp_path / "labels.csv"
        pd.DataFrame({"user_id": ["1"], "label": [1]}).to_csv(labels_path, index=False)

        loader = TwiBotDataLoader(str(json_path), str(labels_path))
        df = loader.load()

        self.assertIn("label", df.columns)
        self.assertEqual(int(df.loc[0, "label"]), 1)


if __name__ == "__main__":
    unittest.main()

import importlib.util
import json
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

PANDAS_AVAILABLE = importlib.util.find_spec("pandas") is not None


class DataLoaderSplitsTest(unittest.TestCase):
    def setUp(self):
        if not PANDAS_AVAILABLE:
            self.skipTest("pandas not installed")
        from DataLoader import TwiBotDataLoader, load_twibot_splits_as_dict
        self.loader_cls = TwiBotDataLoader
        self.load_splits = load_twibot_splits_as_dict

    def _write_json(self, path: Path, users: list) -> None:
        with open(path, 'w', encoding='utf-8') as handle:
            json.dump(users, handle)

    def test_missing_split_files_raise(self):
        users = [{"ID": "1", "label": 0, "profile": {}}]
        with TemporaryDirectory() as tmp:
            data_dir = Path(tmp)
            self._write_json(data_dir / "train.json", users)
            self._write_json(data_dir / "dev.json", users)
            with self.assertRaises(FileNotFoundError) as ctx:
                self.load_splits(data_dir)
            self.assertIn("Missing split files", str(ctx.exception))

    def test_load_splits_returns_expected_keys(self):
        users_train = [{"ID": "1", "label": 0, "profile": {}}]
        users_val = [{"ID": "2", "label": 1, "profile": {}}]
        users_test = [{"ID": "3", "label": 0, "profile": {}}]
        with TemporaryDirectory() as tmp:
            data_dir = Path(tmp)
            self._write_json(data_dir / "train.json", users_train)
            self._write_json(data_dir / "dev.json", users_val)
            self._write_json(data_dir / "test.json", users_test)
            splits = self.load_splits(data_dir)
            self.assertEqual(set(splits.keys()), {"train", "val", "test"})
            self.assertEqual(len(splits["train"]), 1)
            self.assertEqual(len(splits["val"]), 1)
            self.assertEqual(len(splits["test"]), 1)
            self.assertIn("label", splits["train"].columns)

    def test_constructor_requires_exclusive_paths(self):
        with self.assertRaises(ValueError):
            self.loader_cls(json_path="a.json", json_paths=["b.json"])


if __name__ == "__main__":
    unittest.main()

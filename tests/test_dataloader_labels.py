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

    def test_string_label_bot_normalized_to_1(self):
        """Verify 'bot' string label is normalized to 1."""
        import pandas as pd
        from DataLoader import TwiBotDataLoader

        json_path = self.temp_path / "sample.json"
        self._write_json(json_path)

        labels_path = self.temp_path / "labels.csv"
        pd.DataFrame({"ID": ["1"], "label": ["bot"]}).to_csv(labels_path, index=False)

        loader = TwiBotDataLoader(str(json_path), str(labels_path))
        df = loader.load()

        self.assertIn("label", df.columns)
        self.assertEqual(int(df.loc[0, "label"]), 1)

    def test_string_label_human_normalized_to_0(self):
        """Verify 'human' string label is normalized to 0."""
        import pandas as pd
        from DataLoader import TwiBotDataLoader

        json_path = self.temp_path / "sample.json"
        self._write_json(json_path)

        labels_path = self.temp_path / "labels.csv"
        pd.DataFrame({"ID": ["1"], "label": ["human"]}).to_csv(labels_path, index=False)

        loader = TwiBotDataLoader(str(json_path), str(labels_path))
        df = loader.load()

        self.assertIn("label", df.columns)
        self.assertEqual(int(df.loc[0, "label"]), 0)

    def test_string_label_fake_normalized_to_1(self):
        """Verify 'fake' string label is normalized to 1."""
        import pandas as pd
        from DataLoader import TwiBotDataLoader

        json_path = self.temp_path / "sample.json"
        self._write_json(json_path)

        labels_path = self.temp_path / "labels.csv"
        pd.DataFrame({"ID": ["1"], "label": ["fake"]}).to_csv(labels_path, index=False)

        loader = TwiBotDataLoader(str(json_path), str(labels_path))
        df = loader.load()

        self.assertIn("label", df.columns)
        self.assertEqual(int(df.loc[0, "label"]), 1)

    def test_string_label_real_normalized_to_0(self):
        """Verify 'real' string label is normalized to 0."""
        import pandas as pd
        from DataLoader import TwiBotDataLoader

        json_path = self.temp_path / "sample.json"
        self._write_json(json_path)

        labels_path = self.temp_path / "labels.csv"
        pd.DataFrame({"ID": ["1"], "label": ["real"]}).to_csv(labels_path, index=False)

        loader = TwiBotDataLoader(str(json_path), str(labels_path))
        df = loader.load()

        self.assertIn("label", df.columns)
        self.assertEqual(int(df.loc[0, "label"]), 0)

    def test_string_labels_case_insensitive(self):
        """Verify string labels are normalized regardless of case."""
        import pandas as pd
        from DataLoader import TwiBotDataLoader

        json_path = self.temp_path / "sample.json"
        # Write multiple users
        data = [
            {
                "ID": i,
                "profile": {
                    "id": i,
                    "created_at": "Wed Oct 10 20:19:24 +0000 2018",
                    "verified": False,
                },
                "tweet": [],
                "neighbor": {},
                "domain": [],
            }
            for i in range(1, 5)
        ]
        json_path.write_text(json.dumps(data), encoding="utf-8")

        labels_path = self.temp_path / "labels.csv"
        pd.DataFrame({
            "ID": ["1", "2", "3", "4"],
            "label": ["BOT", "Human", "FAKE", "ReAl"]
        }).to_csv(labels_path, index=False)

        loader = TwiBotDataLoader(str(json_path), str(labels_path))
        df = loader.load()

        self.assertIn("label", df.columns)
        # BOT -> 1, Human -> 0, FAKE -> 1, ReAl -> 0
        expected = {1: 1, 2: 0, 3: 1, 4: 0}
        for user_id, expected_label in expected.items():
            row = df[df["user_id"] == str(user_id)]
            self.assertEqual(
                int(row["label"].values[0]),
                expected_label,
                f"User {user_id} label mismatch"
            )

    def test_string_labels_with_whitespace_trimmed(self):
        """Verify string labels with leading/trailing whitespace are normalized."""
        import pandas as pd
        from DataLoader import TwiBotDataLoader

        json_path = self.temp_path / "sample.json"
        data = [
            {
                "ID": i,
                "profile": {
                    "id": i,
                    "created_at": "Wed Oct 10 20:19:24 +0000 2018",
                    "verified": False,
                },
                "tweet": [],
                "neighbor": {},
                "domain": [],
            }
            for i in range(1, 3)
        ]
        json_path.write_text(json.dumps(data), encoding="utf-8")

        labels_path = self.temp_path / "labels.csv"
        pd.DataFrame({
            "ID": ["1", "2"],
            "label": ["  bot  ", "  human  "]
        }).to_csv(labels_path, index=False)

        loader = TwiBotDataLoader(str(json_path), str(labels_path))
        df = loader.load()

        self.assertIn("label", df.columns)
        row_bot = df[df["user_id"] == "1"]
        row_human = df[df["user_id"] == "2"]
        self.assertEqual(int(row_bot["label"].values[0]), 1)
        self.assertEqual(int(row_human["label"].values[0]), 0)

    def test_mixed_numeric_and_string_labels(self):
        """Verify mixed numeric and string labels are normalized correctly."""
        import pandas as pd
        from DataLoader import TwiBotDataLoader

        json_path = self.temp_path / "sample.json"
        data = [
            {
                "ID": i,
                "profile": {
                    "id": i,
                    "created_at": "Wed Oct 10 20:19:24 +0000 2018",
                    "verified": False,
                },
                "tweet": [],
                "neighbor": {},
                "domain": [],
            }
            for i in range(1, 5)
        ]
        json_path.write_text(json.dumps(data), encoding="utf-8")

        labels_path = self.temp_path / "labels.csv"
        # Mix of string and numeric-as-string labels
        pd.DataFrame({
            "ID": ["1", "2", "3", "4"],
            "label": ["bot", "0", "human", "1"]
        }).to_csv(labels_path, index=False)

        loader = TwiBotDataLoader(str(json_path), str(labels_path))
        df = loader.load()

        self.assertIn("label", df.columns)
        expected = {1: 1, 2: 0, 3: 0, 4: 1}
        for user_id, expected_label in expected.items():
            row = df[df["user_id"] == str(user_id)]
            self.assertEqual(
                int(row["label"].values[0]),
                expected_label,
                f"User {user_id} label mismatch"
            )

    def test_invalid_string_label_raises(self):
        """Verify invalid string labels raise ValueError."""
        import pandas as pd
        from DataLoader import TwiBotDataLoader

        json_path = self.temp_path / "sample.json"
        self._write_json(json_path)

        labels_path = self.temp_path / "labels.csv"
        pd.DataFrame({"ID": ["1"], "label": ["invalid_label"]}).to_csv(
            labels_path, index=False
        )

        loader = TwiBotDataLoader(str(json_path), str(labels_path))
        with self.assertRaises(ValueError) as ctx:
            loader.load()
        # Invalid labels result in NaN which triggers validation error
        error_msg = str(ctx.exception).lower()
        self.assertTrue(
            "no valid label" in error_msg or "binary" in error_msg,
            f"Unexpected error message: {ctx.exception}"
        )


if __name__ == "__main__":
    unittest.main()

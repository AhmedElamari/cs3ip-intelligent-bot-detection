import importlib.util
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

SKLEARN_AVAILABLE = importlib.util.find_spec("sklearn") is not None
PANDAS_AVAILABLE = importlib.util.find_spec("pandas") is not None
NUMPY_AVAILABLE = importlib.util.find_spec("numpy") is not None


@unittest.skipUnless(
    SKLEARN_AVAILABLE and PANDAS_AVAILABLE and NUMPY_AVAILABLE,
    "sklearn/pandas/numpy not installed",
)
class FeatureSpaceTSNETest(unittest.TestCase):
    def test_attack_feature_space_tsne_writes_artifacts(self):
        import pandas as pd

        from benchmarking import ModelBenchmark
        from benchmarking.feature_space_tsne import save_attack_feature_space_tsne

        feature_names = [
            "has_description",
            "description_length",
            "screen_name_has_digits",
            "default_profile_image",
            "default_profile",
            "has_extended_profile",
            "followers_count",
            "friends_count",
        ]
        train = pd.DataFrame(
            [
                [1, 30, 0, 0, 0, 1, 100, 90],
                [1, 25, 0, 0, 0, 1, 120, 80],
                [0, 0, 1, 1, 1, 0, 20, 200],
                [0, 0, 1, 1, 1, 0, 25, 210],
            ],
            columns=feature_names,
        )
        test = pd.DataFrame(
            [
                [1, 22, 0, 0, 0, 1, 130, 70],
                [1, 28, 0, 0, 0, 1, 140, 75],
                [0, 0, 1, 1, 1, 0, 15, 220],
                [0, 0, 1, 1, 1, 0, 18, 230],
            ],
            columns=feature_names,
        )
        benchmark = ModelBenchmark(models={}, experiment_name="tsne")
        benchmark.base_train_inputs = train
        benchmark.base_test_inputs = test
        benchmark.base_y_train = pd.Series([0, 0, 1, 1]).to_numpy()
        benchmark.y_test = pd.Series([0, 0, 1, 1]).to_numpy()

        with TemporaryDirectory(dir=ROOT) as tmp:
            output_dir = Path(tmp)
            save_attack_feature_space_tsne(
                benchmark,
                feature_names,
                output_dir,
                perplexity=2,
            )

            self.assertTrue((output_dir / "attack_feature_space_tsne.png").exists())
            csv_path = output_dir / "attack_feature_space_tsne.csv"
            caption_path = output_dir / "attack_feature_space_tsne_caption.md"
            self.assertTrue(csv_path.exists())
            self.assertTrue(caption_path.exists())
            df = pd.read_csv(csv_path)
            self.assertListEqual(
                list(df.columns),
                ["tsne_x", "tsne_y", "scenario", "true_label", "source_test_row_index"],
            )
            self.assertTrue({"human_clean", "bot_clean"}.issubset(set(df["scenario"])))
            caption = caption_path.read_text(encoding="utf-8").lower()
            self.assertIn("qualitative", caption)
            self.assertIn("random_state=2112", caption)


if __name__ == "__main__":
    unittest.main()

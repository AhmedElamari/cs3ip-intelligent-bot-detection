"""Regression tests for poster SHAP exports."""

import sys
import types
import unittest
from unittest import mock
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[1]


class PosterShapExportTest(unittest.TestCase):
    def test_export_poster_shap_caption_uses_model_and_top_n(self):
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            self.skipTest("matplotlib not installed")

        from explainability.poster_shap import export_poster_shap

        fake_shap = types.SimpleNamespace(
            summary_plot=lambda *_args, **_kwargs: plt.scatter([0, 1], [0, 1], s=10)
        )
        with TemporaryDirectory(dir=str(_REPO_ROOT)) as tmp, mock.patch.dict(
            sys.modules, {"shap": fake_shap}
        ):
            output_dir = Path(tmp)
            png = export_poster_shap(
                np.array([[0.2, -0.1], [0.1, 0.3]]),
                np.array([[1.0, 2.0], [3.0, 4.0]]),
                ["followers_count", "friends_count"],
                model_name="random_forest",
                output_dir=output_dir,
                top_n=7,
            )

            caption = (output_dir / "shap_summary_random_forest_poster_caption.md").read_text(
                encoding="utf-8"
            )
            self.assertTrue(png.exists())
            self.assertIn("Random Forest", caption)
            self.assertIn("top 7 features", caption)
            self.assertNotIn("XGBoost", caption)
            self.assertNotIn("top 10 features", caption)
            self.assertIn("Followers (count)", caption)

    def test_export_poster_shap_sets_takeaway_title_by_default(self):
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            self.skipTest("matplotlib not installed")

        from explainability.poster_shap import export_poster_shap

        fake_shap = types.SimpleNamespace(
            summary_plot=lambda *_args, **_kwargs: plt.scatter([0, 1], [0, 1], s=10)
        )
        with TemporaryDirectory(dir=str(_REPO_ROOT)) as tmp, mock.patch.dict(
            sys.modules, {"shap": fake_shap}
        ), mock.patch("matplotlib.pyplot.title") as mock_title:
            export_poster_shap(
                np.array([[0.2, -0.1], [0.1, 0.3]]),
                np.array([[1.0, 2.0], [3.0, 4.0]]),
                ["followers_count", "friends_count"],
                model_name="xgboost",
                output_dir=Path(tmp),
                top_n=7,
            )

        mock_title.assert_called_once()
        title = mock_title.call_args.args[0]
        self.assertIn("XGBoost", title)
        self.assertIn("interpretable", title.lower())


if __name__ == "__main__":
    unittest.main()

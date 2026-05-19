"""Tests for VIVA demo Tab 2 (Explainability) HTML helpers."""

from __future__ import annotations

import unittest

from streamlit_demo.data import FEATURE_IMPORTANCE_ROWS, RESILIENCE_ROWS, SHAP_SUMMARY_RF_PATH
from streamlit_demo.explainability import (
    SHAP_SUMMARY_FILENAME,
    _aside_html,
    build_explainability_html,
    build_shap_placeholder_html,
    format_resilience_rows,
    top_feature_names,
)


class ExplainabilityHelpersTest(unittest.TestCase):
    def test_feature_importance_order_and_top_three(self) -> None:
        names = [r[0] for r in FEATURE_IMPORTANCE_ROWS]
        self.assertEqual(
            names[:3],
            ["is_verified", "followers_to_friends_ratio", "followers_count"],
        )
        self.assertEqual(top_feature_names(FEATURE_IMPORTANCE_ROWS, n=3), names[:3])

    def test_build_explainability_html_structure(self) -> None:
        html = build_explainability_html(
            FEATURE_IMPORTANCE_ROWS,
            RESILIENCE_ROWS,
            shap_image_exists=False,
            shap_image_src="",
        )
        self.assertIn("demo-explain-grid", html)
        self.assertIn("demo-feature-bars", html)
        self.assertIn("demo-figure-head", html)
        self.assertIn("Section 02", html)
        self.assertIn("Why did it predict that?", html)
        self.assertIn("demo-appendix", html)
        self.assertIn("<details", html)
        self.assertIn("demo-resilience-table", html)
        self.assertIn("demo-aside-note", html)
        self.assertIn("top 10 of 24", html)
        self.assertIn("500 held-out", html)

    def test_format_resilience_flip_rates(self) -> None:
        rows = format_resilience_rows(RESILIENCE_ROWS)
        self.assertEqual(len(rows), 5)
        self.assertEqual(rows[0]["Feature"], "is_verified")
        self.assertEqual(rows[0]["Flip Rate"], "8%")
        self.assertEqual(rows[2]["Flip Rate"], "41%")

    def test_shap_placeholder_not_fake_chart(self) -> None:
        html = build_shap_placeholder_html(SHAP_SUMMARY_FILENAME)
        self.assertIn("shap_summary_random_forest.png", html)
        self.assertIn("run_benchmark.py", html)
        self.assertIn("demo_assets", html)
        lowered = html.lower()
        self.assertNotIn("plotly", lowered)
        self.assertNotIn("canvas", lowered)

    def test_aside_html_pads_short_top_three(self) -> None:
        html = _aside_html(["is_verified"])
        self.assertIn("is_verified", html)
        self.assertIn("demo-aside-note", html)

    def test_shap_image_branch_uses_img_tag(self) -> None:
        src = str(SHAP_SUMMARY_RF_PATH).replace("\\", "/")
        html = build_explainability_html(
            FEATURE_IMPORTANCE_ROWS,
            RESILIENCE_ROWS,
            shap_image_exists=True,
            shap_image_src=src,
        )
        self.assertIn("<img ", html)
        self.assertIn("demo-shap-img", html)


if __name__ == "__main__":
    unittest.main()

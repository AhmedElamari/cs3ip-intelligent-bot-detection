"""Tests for VIVA demo Tab 1 (Model Arena) helpers."""

from __future__ import annotations

import unittest

from streamlit_demo.data import DEMO_ARENA_BENCHMARK_REL, DEMO_DATASET_META, MODEL_ARENA_ROWS
from streamlit_demo.model_arena import (
    METRIC_KEYS,
    arena_provenance_footnote_html,
    bar_width_pct,
    build_model_arena_html,
    compute_best_per_metric,
)


class ModelArenaHelpersTest(unittest.TestCase):
    def test_compute_best_per_metric(self) -> None:
        best = compute_best_per_metric(MODEL_ARENA_ROWS)
        self.assertAlmostEqual(best["f1"], 0.8541)
        self.assertAlmostEqual(best["pr_auc"], 0.8607)
        self.assertAlmostEqual(best["roc_auc"], 0.8704)
        self.assertAlmostEqual(best["mcc"], 0.6644)
        self.assertAlmostEqual(best["train_seconds"], 0.06)

    def test_bar_width_matches_design_denominator(self) -> None:
        self.assertAlmostEqual(bar_width_pct(0.8541, "f1"), 85.41)
        self.assertAlmostEqual(bar_width_pct(0.05, "train_seconds"), (0.05 / 50.0) * 100.0)

    def test_html_contains_snapshot_and_highlights(self) -> None:
        html = build_model_arena_html(MODEL_ARENA_ROWS, DEMO_DATASET_META)
        self.assertIn("demo-winner-card", html)
        self.assertIn("demo-metric-table", html)
        self.assertIn("demo-f1-note", html)
        self.assertIn("Six models", html)
        self.assertNotIn("Three classifiers", html)
        self.assertIn("SVM", html)
        self.assertIn("Random Forest", html)
        self.assertIn("TabNet", html)
        self.assertIn("XGBoost", html)
        self.assertGreaterEqual(html.count("demo-metric-row"), 6)
        self.assertIn("0.854", html)
        self.assertIn("0.664", html)
        self.assertIn("Decision Tree", html)
        self.assertIn(DEMO_DATASET_META["account_count_label"], html)
        self.assertIn("Why F1 over accuracy?", html)
        self.assertIn("Best", html)
        self.assertIn("demo-winner-badge", html)

    def test_arena_provenance_footnote_lists_paths(self) -> None:
        foot = arena_provenance_footnote_html()
        self.assertIn(DEMO_ARENA_BENCHMARK_REL, foot)
        self.assertIn("results.json", foot)
        self.assertIn("<code>", foot)

    def test_metric_keys_order(self) -> None:
        self.assertEqual(
            METRIC_KEYS,
            ("f1", "pr_auc", "roc_auc", "mcc", "train_seconds"),
        )

    def test_empty_rows_returns_message_html(self) -> None:
        html = build_model_arena_html([], DEMO_DATASET_META)
        self.assertIn("demo-arena-empty", html)
        self.assertIn("No benchmark rows", html)


if __name__ == "__main__":
    unittest.main()

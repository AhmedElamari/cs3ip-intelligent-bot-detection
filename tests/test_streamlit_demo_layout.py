"""Tests for VIVA demo shell (custom header, query-param tab keys)."""

from __future__ import annotations

import unittest

from streamlit_demo.layout import build_header_html, normalize_tab


class LayoutHelpersTest(unittest.TestCase):
    def test_normalize_tab_defaults_and_aliases(self) -> None:
        self.assertEqual(normalize_tab(None), "arena")
        self.assertEqual(normalize_tab(""), "arena")
        self.assertEqual(normalize_tab("arena"), "arena")
        self.assertEqual(normalize_tab("explainability"), "explainability")
        self.assertEqual(normalize_tab("EXPLAINABILITY"), "explainability")
        self.assertEqual(normalize_tab("live"), "live")
        self.assertEqual(normalize_tab("prediction"), "live")
        self.assertEqual(normalize_tab("tab3"), "live")
        self.assertEqual(normalize_tab("3"), "live")

    def test_header_html_structure(self) -> None:
        h = build_header_html("arena")
        self.assertIn("demo-topnav", h)
        self.assertIn("BD", h)
        self.assertIn("Bot Detection", h)
        self.assertIn("TwiBot-20", h)
        self.assertIn("Model Arena", h)
        self.assertIn("Explainability", h)
        self.assertIn("Live Prediction", h)
        self.assertNotIn("demo-tab-disabled", h)
        self.assertIn("?tab=live", h)
        self.assertIn("demo-status-text", h)
        self.assertIn("demo-tab-active", h)
        self.assertIn('data-demo-active="arena"', h)
        self.assertIn("?tab=arena", h)
        self.assertIn("?tab=explainability", h)

    def test_header_status_text_escaped(self) -> None:
        h = build_header_html("arena", status_text="RF · Optuna HPO (live)")
        self.assertIn("RF · Optuna HPO (live)", h)

    def test_header_active_explainability(self) -> None:
        h = build_header_html("explainability")
        self.assertIn('data-demo-active="explainability"', h)
        self.assertIn("?tab=explainability", h)
        self.assertIn("demo-tab-active", h)

    def test_header_active_live(self) -> None:
        h = build_header_html("live")
        self.assertIn('data-demo-active="live"', h)
        self.assertIn("?tab=live", h)
        self.assertNotIn("demo-tab-disabled", h)


if __name__ == "__main__":
    unittest.main()

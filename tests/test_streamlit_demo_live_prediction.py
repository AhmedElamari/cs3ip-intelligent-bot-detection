"""Tests for Tab 3 live prediction helpers."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier

from FeatureEngineering import BotFeatureExtractor
from streamlit_demo.live_prediction import (
    LivePredictor,
    build_artifact_missing_html,
    build_fallback_predictor_for_demo,
    build_gauge_html,
    build_shap_waterfall_html,
    format_shap_feature_label,
    build_verdict_html,
    compute_derived_pair,
    format_derived_display,
    live_header_status,
    load_live_predictor,
)


class DerivedStatsTest(unittest.TestCase):
    def test_tweets_per_day_cap(self) -> None:
        tpd, _ = compute_derived_pair(50.0, 2000.0, 1.0, 5000.0)
        self.assertAlmostEqual(tpd, float(BotFeatureExtractor.TWEETS_PER_DAY_CAP))

    def test_ff_ratio_cap_and_guard(self) -> None:
        _, r = compute_derived_pair(50000.0, 10.0, 100.0, 0.0)
        self.assertAlmostEqual(r, float(BotFeatureExtractor.FOLLOWERS_FRIENDS_RATIO_CAP))

    def test_display_matches_screenshot_defaults(self) -> None:
        tpd, ff = compute_derived_pair(50, 2000, 3, 600)
        tpd_s, ff_s = format_derived_display(tpd, ff)
        self.assertEqual(tpd_s, "200.0")
        self.assertEqual(ff_s, "0.025")


class RowAssemblyTest(unittest.TestCase):
    def test_overrides_align_with_feature_order(self) -> None:
        fo = ["followers_count", "is_verified", "tweets_per_day"]
        med = np.array([10.0, 1.0, 3.0])
        rng = np.random.RandomState(0)
        X = rng.rand(20, 3)
        y = (X[:, 0] > 0.4).astype(int)
        model = RandomForestClassifier(n_estimators=5, random_state=2112)
        model.fit(X, y)
        lp = LivePredictor(model=model, feature_order=list(fo), medians=med)
        ui = {
            "followers_count": 99.0,
            "friends_count": 100.0,
            "account_age_days": 10.0,
            "statuses_count": 500.0,
            "description_length": 12.0,
            "is_verified": 0,
            "default_profile_image": 1,
            "screen_name_has_digits": 0,
        }
        row = lp.assemble_row(ui)
        self.assertAlmostEqual(row[0], 99.0)
        self.assertAlmostEqual(row[1], 0.0)
        tpd, _ = compute_derived_pair(99.0, 100.0, 10.0, 500.0)
        self.assertAlmostEqual(row[2], tpd)


class FallbackPredictorTest(unittest.TestCase):
    def test_synthetic_fallback_predicts(self) -> None:
        lp = build_fallback_predictor_for_demo()
        ui = {
            "followers_count": 50.0,
            "friends_count": 2000.0,
            "account_age_days": 3.0,
            "statuses_count": 600.0,
            "description_length": 0.0,
            "is_verified": 0,
            "default_profile_image": 1,
            "screen_name_has_digits": 1,
        }
        score, contrib = lp.predict(ui)
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)
        self.assertGreater(len(contrib), 0)


class PredictAndShapTest(unittest.TestCase):
    def test_joblib_roundtrip_and_shap_subset(self) -> None:
        fo = ["followers_count", "is_verified", "tweets_per_day"]
        rng = np.random.RandomState(2112)
        X = rng.rand(50, 3)
        y = (X.sum(axis=1) > 1.2).astype(int)
        model = RandomForestClassifier(n_estimators=12, random_state=2112)
        model.fit(X, y)
        medians = np.median(X, axis=0)
        payload = {
            "schema_version": "LivePredictorV1",
            "feature_order": fo,
            "medians": medians,
            "model": model,
        }
        with tempfile.NamedTemporaryFile(suffix=".joblib", delete=False) as tmp:
            path = Path(tmp.name)
        try:
            joblib.dump(payload, path)
            lp = load_live_predictor(path)
            assert lp is not None
            ui = {
                "followers_count": float(X[0, 0]),
                "friends_count": 100.0,
                "account_age_days": 50.0,
                "statuses_count": 1000.0,
                "description_length": 20.0,
                "is_verified": int(X[0, 1]),
                "default_profile_image": 1,
                "screen_name_has_digits": 0,
            }
            bot_score, contrib = lp.predict(ui)
            self.assertGreaterEqual(bot_score, 0.0)
            self.assertLessEqual(bot_score, 1.0)
            self.assertTrue(set(contrib.keys()).issubset(set(fo)))
            self.assertGreater(len(contrib), 0)
        finally:
            path.unlink(missing_ok=True)


class GaugeHtmlTest(unittest.TestCase):
    def test_structure_and_animation_hook(self) -> None:
        h = build_gauge_html(0.4, 60, 7)
        self.assertIn('viewBox="0 0 280 155"', h)
        self.assertIn("requestAnimationFrame", h)
        self.assertIn("demo-gauge-prev-angle", h)
        self.assertIn("0.40000000", h)


class WaterfallHtmlTest(unittest.TestCase):
    def test_sort_and_cap(self) -> None:
        h = build_shap_waterfall_html(
            [
                ("low", 0.05),
                ("high", 0.5),
                ("mid", -0.2),
            ]
        )
        hi = h.index("high")
        mi = h.index("mid")
        lo = h.index("low")
        self.assertLess(hi, mi)
        self.assertLess(mi, lo)
        self.assertIn("pushes toward bot", h)

    def test_empty_fallback(self) -> None:
        h = build_shap_waterfall_html([])
        self.assertIn("No strong signals", h)

    def test_ff_ratio_display_label(self) -> None:
        h = build_shap_waterfall_html([("followers_to_friends_ratio", 0.42)])
        self.assertIn("ff_ratio", h)
        self.assertNotIn("followers_to_friends_ra", h)

    def test_format_shap_feature_label(self) -> None:
        self.assertEqual(
            format_shap_feature_label("followers_to_friends_ratio"),
            "ff_ratio",
        )


class VerdictHtmlTest(unittest.TestCase):
    def test_thresholds(self) -> None:
        self.assertIn("Bot detected", build_verdict_html(0.6))
        self.assertIn("60%", build_verdict_html(0.6))
        self.assertIn("Human", build_verdict_html(0.4))
        self.assertIn("Bot detected", build_verdict_html(0.5))


class ArtifactMissingTest(unittest.TestCase):
    def test_missing_load_and_banner(self) -> None:
        self.assertIsNone(load_live_predictor(Path("/nonexistent/nope.joblib")))
        b = build_artifact_missing_html("demo_assets/live_predictor.joblib")
        self.assertIn("bake_live_artifact", b)
        self.assertIn("--train-split-dir data", b)
        self.assertIn("demo_assets/live_predictor.joblib", b)


class LiveHeaderStatusTest(unittest.TestCase):
    def test_loaded_predictor_shows_model_loaded(self) -> None:
        fb = build_fallback_predictor_for_demo()
        self.assertEqual(live_header_status(fb, "fallback"), "Model loaded")
        lp_plain = LivePredictor(
            model=fb.model,
            feature_order=list(fb.feature_order),
            medians=fb.medians.copy(),
            hpo_provenance=None,
        )
        self.assertEqual(live_header_status(lp_plain, "artifact"), "Model loaded")
        lp_hpo = LivePredictor(
            model=fb.model,
            feature_order=list(fb.feature_order),
            medians=fb.medians.copy(),
            hpo_provenance={"best_val_f1": 0.84},
        )
        self.assertEqual(live_header_status(lp_hpo, "artifact"), "Model loaded")
        self.assertEqual(live_header_status(None, "error"), "Predictor unavailable")

    def test_legacy_predictor_without_hpo_attr(self) -> None:
        from streamlit_demo.live_prediction import _hpo_provenance

        class LegacyArtifact:
            model = object()
            feature_order = ["a"]
            medians = np.array([1.0])

        leg = LegacyArtifact()
        self.assertIsNone(_hpo_provenance(leg))
        self.assertEqual(live_header_status(leg, "artifact"), "Model loaded")


class LiveIntroHtmlTest(unittest.TestCase):
    def test_compact_intro_copy(self) -> None:
        from streamlit_demo.live_prediction import build_live_intro_html

        html_out = build_live_intro_html()
        self.assertIn("Live prediction", html_out)
        self.assertIn("remaining 16", html_out)
        self.assertIn("24 features", html_out)
        self.assertIn("Adjust the 8", html_out)


if __name__ == "__main__":
    unittest.main()

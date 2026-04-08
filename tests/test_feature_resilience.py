import unittest

import numpy as np

from explainability.resilience import FeatureResilienceAnalyzer


class FeatureResilienceAnalyzerTest(unittest.TestCase):

    def test_rank_stability_is_normalized(self):
        before = np.array([2, 0, 1])
        after = np.array([1, 0, 2])
        stability = FeatureResilienceAnalyzer.feature_rank_stability(before, after, feature_index=0)
        self.assertGreaterEqual(stability, 0.0)
        self.assertLessEqual(stability, 1.0)
        self.assertAlmostEqual(stability, 0.5)

    def test_frs_returns_nan_when_no_baseline_detected_bots(self):
        score = FeatureResilienceAnalyzer.compute_feature_resilience(
            importance=0.8,
            stability=0.9,
            flips_to_human=0,
            baseline_detected_bots=0,
        )
        self.assertTrue(np.isnan(score))

    def test_top_k_pivot_metadata_tracks_entries_and_exits(self):
        features = ['bio', 'age', 'ratio']
        before = np.array([0.9, 0.3, 0.2])
        after = np.array([0.1, 0.8, 0.2])
        pivot = FeatureResilienceAnalyzer.top_k_pivot_metadata(
            feature_names=features,
            before_values=before,
            after_values=after,
            top_k=1,
        )
        self.assertEqual(pivot['top_feature_before'], 'bio')
        self.assertEqual(pivot['top_feature_after'], 'age')
        self.assertEqual(pivot['entered_top_k'], 'age')
        self.assertEqual(pivot['dropped_top_k'], 'bio')


if __name__ == '__main__':
    unittest.main()

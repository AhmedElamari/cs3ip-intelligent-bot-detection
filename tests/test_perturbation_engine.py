import unittest

import numpy as np
import pandas as pd

from adversarial.perturbation import RealisticPerturbationEngine


class PerturbationEngineTest(unittest.TestCase):

    def setUp(self):
        self.feature_names = [
            'has_description',
            'description_length',
            'screen_name_has_digits',
            'default_profile_image',
            'default_profile',
            'has_extended_profile',
            'followers_count',
            'friends_count',
            'followers_to_friends_ratio',
            'followers_per_day',
            'tweets_per_day',
            'favourites_per_day',
            'account_age_days',
        ]
        self.X_train = pd.DataFrame(
            [
                [1, 20, 0, 0, 0, 1, 100, 80, 100 / 81, 10, 2, 1, 10],
                [1, 30, 0, 0, 0, 1, 110, 90, 110 / 91, 11, 2, 1, 10],
                [0, 0, 1, 1, 1, 0, 10, 50, 10 / 51, 1, 4, 3, 10],
                [0, 0, 1, 1, 1, 0, 12, 55, 12 / 56, 1.2, 4.5, 3.2, 10],
            ],
            columns=self.feature_names,
        )
        self.y_train = np.array([0, 0, 1, 1])
        self.X_eval = pd.DataFrame(
            [[0, 0, 1, 1, 1, 0, 10, 50, 10 / 51, 1, 4, 3, 10]],
            columns=self.feature_names,
        )
        self.engine = RealisticPerturbationEngine(
            feature_names=self.feature_names,
            X_train=self.X_train,
            y_train=self.y_train,
            expensive_nudge_fraction=0.05,
        )

    def test_has_description_attack_adds_bio_and_length(self):
        result = self.engine.apply_single_feature_attack(self.X_eval, 'has_description')
        self.assertTrue(result.applied)
        self.assertEqual(result.data.loc[0, 'has_description'], 1)
        self.assertEqual(result.data.loc[0, 'description_length'], 25)

    def test_realistic_mixed_profile_bounds_expensive_nudges_and_recomputes(self):
        result = self.engine.apply_profile(self.X_eval, 'realistic_mixed')
        self.assertTrue(result.applied)
        self.assertEqual(result.data.loc[0, 'account_age_days'], 10)
        self.assertLessEqual(result.data.loc[0, 'followers_count'], 10.5)
        self.assertGreaterEqual(result.data.loc[0, 'friends_count'], 47.5)
        self.assertAlmostEqual(
            result.data.loc[0, 'followers_to_friends_ratio'],
            result.data.loc[0, 'followers_count'] / (result.data.loc[0, 'friends_count'] + 1),
        )
        self.assertAlmostEqual(
            result.data.loc[0, 'followers_per_day'],
            result.data.loc[0, 'followers_count'] / result.data.loc[0, 'account_age_days'],
        )

    def test_removed_feature_is_skipped_with_reason(self):
        reduced = self.X_eval.drop(columns=['default_profile_image'])
        engine = RealisticPerturbationEngine(
            feature_names=reduced.columns.tolist(),
            X_train=self.X_train[reduced.columns],
            y_train=self.y_train,
            expensive_nudge_fraction=0.05,
        )
        result = engine.apply_single_feature_attack(reduced, 'default_profile_image')
        self.assertFalse(result.applied)
        self.assertIn('not available', result.skip_reason)


if __name__ == '__main__':
    unittest.main()

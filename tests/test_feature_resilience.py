"""Tests for Feature Resilience Score (FRS) helpers and markdown layout."""

import unittest

import numpy as np
import pandas as pd

from adversarial.perturbation import RealisticPerturbationEngine
from benchmarking.robustness import (
    _minmax_importance,
    _spearman_rho_pair,
    build_feature_resilience_markdown,
    compute_frs,
)


class FeatureResilienceHelpersTest(unittest.TestCase):

    def test_compute_frs_matches_spec(self):
        self.assertAlmostEqual(compute_frs(0.8, 0.9, 0.25), 0.8 * 0.9 * 0.75)

    def test_minmax_importance_flat_vector(self):
        imp = {'a': 1.0, 'b': 1.0}
        n = _minmax_importance(['a', 'b'], imp)
        self.assertEqual(n['a'], 1.0)
        self.assertEqual(n['b'], 1.0)

    def test_spearman_identity_is_one(self):
        v = np.array([3.0, 1.0, 2.0, 4.0])
        self.assertAlmostEqual(_spearman_rho_pair(v, v), 1.0, places=5)

    def test_feature_resilience_markdown_layout(self):
        text = build_feature_resilience_markdown(
            [('account_age_days', 0.94), ('followers_count', 0.88)],
            [('has_description', 0.12), ('default_avatar', 0.15)],
        )
        self.assertIn('Top 5 Resilient Features (High FRS)', text)
        self.assertIn('Top 5 Vulnerable Features (Low FRS)', text)
        self.assertIn('| `account_age_days` | 0.94 |', text)
        self.assertIn('FRS quantifies', text)

    def test_register_dynamic_recipe_adds_generic_mask(self):
        feature_names = ['f_num', 'f_bin', 'followers_to_friends_ratio']
        X_train = pd.DataFrame({
            'f_num': [1.0, 2.0, 0.5, 0.25],
            'f_bin': [0, 1, 1, 0],
            'followers_to_friends_ratio': [0.1, 0.2, 0.15, 0.12],
        }, columns=feature_names)
        y_train = np.array([0, 0, 1, 1])
        engine = RealisticPerturbationEngine(feature_names, X_train, y_train)
        n_builtin = len(engine.available_single_feature_attacks(builtin_only=True))
        self.assertTrue(engine.register_dynamic_recipe('f_num'))
        self.assertTrue(engine.register_dynamic_recipe('f_bin'))
        extended = engine.available_single_feature_attacks(builtin_only=False)
        self.assertGreater(len(extended), n_builtin)
        attack = engine.apply_single_feature_attack(X_train.iloc[:1].copy(), 'f_num')
        self.assertTrue(attack.applied)


if __name__ == '__main__':
    unittest.main()

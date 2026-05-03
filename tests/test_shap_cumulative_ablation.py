"""Tests for cumulative ablation fill values and fidelity gating."""

import unittest

import numpy as np
import pandas as pd

from benchmarking.robustness import _training_fill_value


class TrainingFillValueTest(unittest.TestCase):

    def test_numeric_uses_median(self):
        df = pd.DataFrame({'x': [1.0, 2.0, 3.0, 100.0]})
        self.assertAlmostEqual(_training_fill_value(df, 'x'), 2.5)

    def test_binary_uses_majority(self):
        df = pd.DataFrame({'b': [1, 1, 0]})
        self.assertEqual(_training_fill_value(df, 'b'), 1)


class FidelityPassedLogicTest(unittest.TestCase):

    def test_fidelity_passed_requires_all_positive_drops(self):
        rows = [
            {'macro_f1_drop': 0.01, 'pr_auc_drop': 0.02},
            {'macro_f1_drop': 0.0, 'pr_auc_drop': 0.01},
        ]
        self.assertFalse(
            all(float(r['macro_f1_drop']) > 0.0 and float(r['pr_auc_drop']) > 0.0 for r in rows)
        )
        good = [
            {'macro_f1_drop': 0.01, 'pr_auc_drop': 0.001},
            {'macro_f1_drop': 0.02, 'pr_auc_drop': 0.003},
        ]
        self.assertTrue(
            all(float(r['macro_f1_drop']) > 0.0 and float(r['pr_auc_drop']) > 0.0 for r in good)
        )


if __name__ == '__main__':
    unittest.main()

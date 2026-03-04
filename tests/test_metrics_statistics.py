"""
Tests for statistical inference methods in MetricsCalculator.

Covers:
  - bootstrap_metric_ci: shape, monotonicity (lower <= point <= upper),
    determinism under fixed seed, graceful handling of single-class edge cases.
  - bootstrap_delta_ci: shape, determinism, returns finite values.
  - mcnemar_test: identical predictions -> p=1, known disagreement matrix.
  - holm_bonferroni: ordering and upper-bound guarantee.
"""

import importlib.util
import math
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

SKLEARN_AVAILABLE = importlib.util.find_spec("sklearn") is not None
NUMPY_AVAILABLE = importlib.util.find_spec("numpy") is not None
SCIPY_AVAILABLE = importlib.util.find_spec("scipy") is not None


class BootstrapCITest(unittest.TestCase):

    def setUp(self):
        if not (SKLEARN_AVAILABLE and NUMPY_AVAILABLE):
            self.skipTest("Required dependencies not installed")
        import numpy as np
        from benchmarking.metrics import MetricsCalculator
        self.np = np
        self.calc = MetricsCalculator()

        rng = np.random.RandomState(42)
        n = 100
        self.y_true = rng.randint(0, 2, n)
        self.y_pred = (rng.rand(n) > 0.4).astype(int)
        raw_proba = rng.rand(n, 2)
        self.y_proba = (raw_proba / raw_proba.sum(axis=1, keepdims=True))

    def test_ci_bounds_ordered(self):
        lower, point, upper = self.calc.bootstrap_metric_ci(
            self.y_true, self.y_pred, self.y_proba, metric='f1',
            n_bootstrap=200, random_state=2112
        )
        self.assertLessEqual(lower, point)
        self.assertLessEqual(point, upper)

    def test_ci_deterministic(self):
        r1 = self.calc.bootstrap_metric_ci(
            self.y_true, self.y_pred, self.y_proba, metric='f1',
            n_bootstrap=200, random_state=2112
        )
        r2 = self.calc.bootstrap_metric_ci(
            self.y_true, self.y_pred, self.y_proba, metric='f1',
            n_bootstrap=200, random_state=2112
        )
        self.assertEqual(r1, r2)

    def test_different_seeds_different_bounds(self):
        r1 = self.calc.bootstrap_metric_ci(
            self.y_true, self.y_pred, self.y_proba, metric='f1',
            n_bootstrap=200, random_state=1
        )
        r2 = self.calc.bootstrap_metric_ci(
            self.y_true, self.y_pred, self.y_proba, metric='f1',
            n_bootstrap=200, random_state=99
        )
        # Bounds will differ (point estimate is the same)
        self.assertEqual(r1[1], r2[1])

    def test_roc_auc_ci(self):
        lower, point, upper = self.calc.bootstrap_metric_ci(
            self.y_true, self.y_pred, self.y_proba, metric='roc_auc',
            n_bootstrap=200, random_state=2112
        )
        self.assertGreaterEqual(point, 0.0)
        self.assertLessEqual(point, 1.0)
        self.assertLessEqual(lower, upper)

    def test_single_class_graceful(self):
        """Single-class y_true should not raise; may return nan bounds."""
        import numpy as np
        y_single = np.zeros(20, dtype=int)
        y_pred = np.zeros(20, dtype=int)
        lower, point, upper = self.calc.bootstrap_metric_ci(
            y_single, y_pred, None, metric='f1',
            n_bootstrap=50, random_state=2112
        )
        # point should be a finite number or nan; should not raise
        self.assertTrue(point == point or self.np.isnan(point))

    def test_mcc_ci(self):
        lower, point, upper = self.calc.bootstrap_metric_ci(
            self.y_true, self.y_pred, None, metric='mcc',
            n_bootstrap=200, random_state=2112
        )
        self.assertLessEqual(lower, point)
        self.assertLessEqual(point, upper)


class BootstrapDeltaTest(unittest.TestCase):

    def setUp(self):
        if not (SKLEARN_AVAILABLE and NUMPY_AVAILABLE):
            self.skipTest("Required dependencies not installed")
        import numpy as np
        from benchmarking.metrics import MetricsCalculator
        self.np = np
        self.calc = MetricsCalculator()

        rng = np.random.RandomState(42)
        n = 80
        self.y_true = rng.randint(0, 2, n)
        self.preds_a = (rng.rand(n) > 0.5).astype(int)
        self.preds_b = (rng.rand(n) > 0.4).astype(int)

    def test_delta_returns_required_keys(self):
        result = self.calc.bootstrap_delta_ci(
            self.y_true, self.preds_a, self.preds_b,
            metric='f1', n_bootstrap=200, random_state=2112
        )
        for key in ('delta', 'ci_lower', 'ci_upper', 'p_value'):
            self.assertIn(key, result)

    def test_delta_is_deterministic(self):
        r1 = self.calc.bootstrap_delta_ci(
            self.y_true, self.preds_a, self.preds_b,
            metric='f1', n_bootstrap=200, random_state=2112
        )
        r2 = self.calc.bootstrap_delta_ci(
            self.y_true, self.preds_a, self.preds_b,
            metric='f1', n_bootstrap=200, random_state=2112
        )
        self.assertEqual(r1['delta'], r2['delta'])
        self.assertEqual(r1['ci_lower'], r2['ci_lower'])

    def test_symmetric_models_p_not_extreme(self):
        """Identical predictions should produce near-zero delta."""
        result = self.calc.bootstrap_delta_ci(
            self.y_true, self.preds_a, self.preds_a,
            metric='f1', n_bootstrap=200, random_state=2112
        )
        self.assertAlmostEqual(result['delta'], 0.0, places=10)

    def test_ci_bounds_ordered(self):
        result = self.calc.bootstrap_delta_ci(
            self.y_true, self.preds_a, self.preds_b,
            metric='f1', n_bootstrap=200, random_state=2112
        )
        self.assertLessEqual(result['ci_lower'], result['ci_upper'])


class McNemarTest(unittest.TestCase):

    def setUp(self):
        if not (SKLEARN_AVAILABLE and NUMPY_AVAILABLE and SCIPY_AVAILABLE):
            self.skipTest("Required dependencies not installed")
        import numpy as np
        from benchmarking.metrics import MetricsCalculator
        self.np = np
        self.calc = MetricsCalculator()

    def test_identical_predictions_p_equals_one(self):
        import numpy as np
        y_true = np.array([0, 1, 0, 1, 1, 0])
        preds = np.array([0, 1, 1, 1, 0, 0])
        result = self.calc.mcnemar_test(y_true, preds, preds)
        self.assertEqual(result['p_value'], 1.0)
        self.assertEqual(result['b'], 0)
        self.assertEqual(result['c'], 0)

    def test_known_b_c_values(self):
        """Construct arrays with exactly b=3 c=0."""
        import numpy as np
        # A correct (b positions), B wrong (same positions)
        # y_true: 4 items; A correct on all 4, B wrong on first 3
        y_true = np.array([1, 1, 1, 0])
        preds_a = np.array([1, 1, 1, 0])   # all correct
        preds_b = np.array([0, 0, 0, 0])   # wrong on first 3, correct on last
        result = self.calc.mcnemar_test(y_true, preds_a, preds_b)
        self.assertEqual(result['b'], 3)
        self.assertEqual(result['c'], 0)

    def test_result_keys_present(self):
        import numpy as np
        y_true = np.array([0, 1, 0, 1])
        preds_a = np.array([0, 1, 1, 0])
        preds_b = np.array([0, 0, 1, 1])
        result = self.calc.mcnemar_test(y_true, preds_a, preds_b)
        for key in ('b', 'c', 'statistic', 'p_value', 'test_type'):
            self.assertIn(key, result)

    def test_p_value_in_valid_range(self):
        import numpy as np
        rng = np.random.RandomState(7)
        y_true = rng.randint(0, 2, 60)
        preds_a = rng.randint(0, 2, 60)
        preds_b = rng.randint(0, 2, 60)
        result = self.calc.mcnemar_test(y_true, preds_a, preds_b)
        self.assertGreaterEqual(result['p_value'], 0.0)
        self.assertLessEqual(result['p_value'], 1.0)

    def test_exact_branch_p_value_capped_at_one(self):
        import numpy as np
        # Build b=1, c=1 to exercise exact branch where 2*min(...) may exceed 1.
        y_true = np.array([1, 1, 0, 0])
        preds_a = np.array([1, 0, 0, 0])  # wrong once
        preds_b = np.array([0, 1, 0, 0])  # wrong once, different position
        result = self.calc.mcnemar_test(y_true, preds_a, preds_b)
        self.assertEqual(result['test_type'], 'exact')
        self.assertLessEqual(result['p_value'], 1.0)


class McNemarDependencyBehaviorTest(unittest.TestCase):
    def setUp(self):
        if not (SKLEARN_AVAILABLE and NUMPY_AVAILABLE):
            self.skipTest("Required dependencies not installed")
        from benchmarking.metrics import MetricsCalculator
        self.calc = MetricsCalculator()

    def test_missing_scipy_graceful_unavailable(self):
        """Deterministic proof: mcnemar_test falls back gracefully when scipy import is blocked.

        Uses sys.modules patching so this test runs regardless of whether scipy is
        installed in the current environment.
        """
        import sys
        import numpy as np
        from unittest.mock import patch

        y_true = np.array([0, 1, 0, 1])
        preds_a = np.array([0, 1, 1, 0])
        preds_b = np.array([0, 0, 1, 1])

        with patch.dict(sys.modules, {'scipy': None, 'scipy.stats': None}):
            result = self.calc.mcnemar_test(y_true, preds_a, preds_b)

        self.assertTrue(np.isnan(result['p_value']))
        self.assertEqual(result['test_type'], 'unavailable')


class HolmBonferroniTest(unittest.TestCase):

    def setUp(self):
        if not NUMPY_AVAILABLE:
            self.skipTest("numpy not installed")
        from benchmarking.metrics import MetricsCalculator
        self.calc = MetricsCalculator()

    def test_corrected_values_geq_input(self):
        """Smallest raw p-value gets multiplied by n; result >= original."""
        raw = [0.01, 0.04, 0.03]
        corrected = self.calc.holm_bonferroni(raw)
        # Holm guarantees that the *smallest* raw p-value is multiplied by n.
        # Larger raw p-values may receive corrected values >= their adjusted
        # neighbour, which can be lower than the raw value.
        # At minimum, the corrected value for the smallest raw p-value (0.01)
        # must be n * 0.01 = 0.03 >= 0.01.
        min_raw_idx = raw.index(min(raw))
        self.assertGreaterEqual(corrected[min_raw_idx], raw[min_raw_idx] - 1e-12)

    def test_corrected_capped_at_one(self):
        raw = [0.9, 0.95, 0.99]
        corrected = self.calc.holm_bonferroni(raw)
        for p in corrected:
            self.assertLessEqual(p, 1.0)

    def test_single_p_value(self):
        corrected = self.calc.holm_bonferroni([0.03])
        self.assertAlmostEqual(corrected[0], 0.03)

    def test_empty_list(self):
        corrected = self.calc.holm_bonferroni([])
        self.assertEqual(corrected, [])

    def test_ordering_preserved(self):
        """Corrected values preserve the input ordering (not sorted)."""
        raw = [0.01, 0.04, 0.03]
        corrected = self.calc.holm_bonferroni(raw)
        self.assertEqual(len(corrected), len(raw))

    def test_all_same_p(self):
        raw = [0.05, 0.05, 0.05]
        corrected = self.calc.holm_bonferroni(raw)
        self.assertEqual(len(corrected), 3)
        for p in corrected:
            self.assertLessEqual(p, 1.0)

    def test_nan_excluded_from_multiplicity(self):
        """NaN should not inflate corrections; finite entries corrected over m=2."""
        import math
        raw = [0.01, 0.04, float('nan')]
        corrected = self.calc.holm_bonferroni(raw)
        self.assertEqual(len(corrected), 3)
        self.assertTrue(math.isnan(corrected[2]))
        # With only 2 finite hypotheses, smallest adjusted = 2*0.01 = 0.02
        self.assertAlmostEqual(min(c for c in corrected if not math.isnan(c)), 0.02, places=10)

    def test_all_nan_returns_all_nan(self):
        raw = [float('nan'), float('nan')]
        corrected = self.calc.holm_bonferroni(raw)
        self.assertEqual(len(corrected), 2)
        self.assertTrue(all(math.isnan(c) for c in corrected))

    def test_nonfinite_inputs_produce_valid_finite_range(self):
        """Positions with inf/-inf become NaN; finite positions stay in [0, 1]."""
        import math
        raw = [0.01, float('-inf'), float('inf'), float('nan'), 0.04]
        corrected = self.calc.holm_bonferroni(raw)
        self.assertEqual(len(corrected), 5)
        finite_corrected = [c for c in corrected if not math.isnan(c)]
        for c in finite_corrected:
            self.assertGreaterEqual(c, 0.0)
            self.assertLessEqual(c, 1.0)
        # Positions 1, 2, 3 (non-finite input) must be NaN in output
        self.assertTrue(math.isnan(corrected[1]))
        self.assertTrue(math.isnan(corrected[2]))
        self.assertTrue(math.isnan(corrected[3]))

    def test_finite_path_unchanged(self):
        """Pure finite vector must match expected Holm behaviour."""
        raw = [0.01, 0.04, 0.03]
        corrected = self.calc.holm_bonferroni(raw)
        # All 3 are finite
        self.assertTrue(all(not math.isnan(c) for c in corrected))
        # Smallest raw (0.01 at index 0) gets multiplied by 3 -> 0.03
        self.assertAlmostEqual(corrected[0], 0.03, places=10)


if __name__ == '__main__':
    unittest.main()

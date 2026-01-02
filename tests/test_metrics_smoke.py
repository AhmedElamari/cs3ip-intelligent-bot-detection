import importlib.util
import unittest

SKLEARN_AVAILABLE = importlib.util.find_spec("sklearn") is not None


class MetricsSmokeTest(unittest.TestCase):
    def setUp(self):
        if not SKLEARN_AVAILABLE:
            self.skipTest("scikit-learn not installed")

    def test_compute_all_metrics_binary(self):
        import numpy as np
        from benchmarking.metrics import MetricsCalculator

        y_true = np.array([0, 1, 0, 1])
        y_pred = np.array([0, 1, 1, 1])
        y_proba = np.array([
            [0.9, 0.1],
            [0.2, 0.8],
            [0.4, 0.6],
            [0.3, 0.7],
        ])

        metrics = MetricsCalculator().compute_all_metrics(y_true, y_pred, y_proba)

        self.assertIn('accuracy', metrics)
        self.assertIn('roc_auc', metrics)
        self.assertEqual(metrics['true_positives'] + metrics['false_negatives'], int((y_true == 1).sum()))

    def test_compute_all_metrics_single_class(self):
        import numpy as np
        from benchmarking.metrics import MetricsCalculator

        y_true = np.zeros(5, dtype=int)
        y_pred = np.zeros(5, dtype=int)

        metrics = MetricsCalculator().compute_all_metrics(y_true, y_pred)

        self.assertIn('specificity', metrics)
        self.assertEqual(metrics['false_positives'], 0)


if __name__ == '__main__':
    unittest.main()

import importlib.util
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

SKLEARN_AVAILABLE = importlib.util.find_spec("sklearn") is not None
NUMPY_AVAILABLE = importlib.util.find_spec("numpy") is not None


class StratifiedSplitTest(unittest.TestCase):
    def setUp(self):
        if not (SKLEARN_AVAILABLE and NUMPY_AVAILABLE):
            self.skipTest("Required dependencies not installed")
        import numpy as np
        import pipeline_utils
        self.np = np
        self.splitters = (
            pipeline_utils.safe_stratified_split,
        )

    def _assert_split(self, splitter, labels, split_name, test_size):
        indices = self.np.arange(len(labels))
        idx_train, idx_test, y_train, y_test = splitter(
            indices,
            labels,
            test_size=test_size,
            random_state=2112,
            split_name=split_name
        )
        self.assertEqual(len(idx_train) + len(idx_test), len(labels))
        self.assertEqual(len(y_train), len(idx_train))
        self.assertEqual(len(y_test), len(idx_test))

    def _assert_fallback_split(self, splitter, labels, split_name, test_size):
        indices = self.np.arange(len(labels))
        with self.assertLogs(splitter.__module__, level="WARNING") as logs:
            idx_train, idx_test, y_train, y_test = splitter(
                indices,
                labels,
                test_size=test_size,
                random_state=2112,
                split_name=split_name
            )
        self.assertEqual(len(idx_train) + len(idx_test), len(labels))
        self.assertEqual(len(y_train), len(idx_train))
        self.assertEqual(len(y_test), len(idx_test))
        self.assertTrue(any("Stratified" in message for message in logs.output))

    def test_single_class_fallback(self):
        labels = self.np.zeros(10, dtype=int)
        for splitter in self.splitters:
            with self.subTest(splitter=splitter.__module__):
                self._assert_split(
                    splitter,
                    labels,
                    split_name="test",
                    test_size=0.2
                )

    def test_too_few_samples_fallback(self):
        labels = self.np.array([0, 0, 0, 1])
        for splitter in self.splitters:
            with self.subTest(splitter=splitter.__module__):
                self._assert_fallback_split(
                    splitter,
                    labels,
                    split_name="validation",
                    test_size=0.5
                )


if __name__ == "__main__":
    unittest.main()

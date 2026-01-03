import importlib.util
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

SKLEARN_AVAILABLE = importlib.util.find_spec("sklearn") is not None
NUMPY_AVAILABLE = importlib.util.find_spec("numpy") is not None
PANDAS_AVAILABLE = importlib.util.find_spec("pandas") is not None


class PreprocessingTest(unittest.TestCase):
    def setUp(self):
        if not (SKLEARN_AVAILABLE and NUMPY_AVAILABLE and PANDAS_AVAILABLE):
            self.skipTest("Required dependencies not installed")
        import numpy as np
        from Preprocessing import BotDetector
        self.np = np
        self.detector = BotDetector()

    def test_get_class_weights_with_empty_labels_raises_value_error(self):
        """Verify ValueError is raised when y_train is empty."""
        y_train = self.np.array([])
        
        with self.assertRaises(ValueError) as ctx:
            self.detector.get_class_weights(y_train)
        
        # Verify the error message is helpful
        error_msg = str(ctx.exception)
        self.assertIn("empty", error_msg.lower())
        self.assertIn("training labels", error_msg.lower())

    def test_get_class_weights_with_empty_labels_has_helpful_message(self):
        """Verify error message is helpful when y_train is empty."""
        y_train = self.np.array([])
        
        with self.assertRaises(ValueError) as ctx:
            self.detector.get_class_weights(y_train)
        
        error_msg = str(ctx.exception)
        # Check that the message includes key information
        self.assertIn("Cannot compute class weights", error_msg)
        self.assertIn("training labels are empty", error_msg)

    def test_get_class_weights_with_missing_classes_raises_value_error(self):
        """Verify ValueError is raised when some classes have no samples."""
        # Create a training set with class 0 and 2, but missing class 1
        # This will cause bincount to return [count_0, 0, count_2]
        y_train = self.np.array([0, 0, 2, 2])
        
        with self.assertRaises(ValueError) as ctx:
            self.detector.get_class_weights(y_train)
        
        # Verify the error message mentions missing classes
        error_msg = str(ctx.exception)
        self.assertIn("no samples found for classes", error_msg)

    def test_get_class_weights_with_missing_classes_has_helpful_message(self):
        """Verify error message is helpful when classes are missing."""
        # Create a training set with class 0 and 3, but missing classes 1 and 2
        # This will cause bincount to return [count_0, 0, 0, count_3]
        y_train = self.np.array([0, 0, 3, 3, 3])
        
        with self.assertRaises(ValueError) as ctx:
            self.detector.get_class_weights(y_train)
        
        error_msg = str(ctx.exception)
        # Check that the message includes actionable guidance
        self.assertIn("Cannot compute class weights", error_msg)
        self.assertIn("no samples found for classes", error_msg)
        self.assertIn("train/validation split", error_msg)
        self.assertIn("resampling strategy", error_msg)

    def test_get_class_weights_with_balanced_classes(self):
        """Verify class weights are computed correctly for balanced classes."""
        y_train = self.np.array([0, 0, 1, 1])
        
        weights = self.detector.get_class_weights(y_train)
        
        # With balanced classes, weights should be equal
        self.assertAlmostEqual(weights[0], 1.0)
        self.assertAlmostEqual(weights[1], 1.0)

    def test_get_class_weights_with_imbalanced_classes(self):
        """Verify class weights are computed correctly for imbalanced classes."""
        # 3 samples of class 0, 1 sample of class 1
        y_train = self.np.array([0, 0, 0, 1])
        
        weights = self.detector.get_class_weights(y_train)
        
        # Class 1 (minority) should have higher weight
        self.assertGreater(weights[1], weights[0])
        # Verify the actual weight calculation
        # total = 4, num_classes = 2
        # weight_0 = 4 / (2 * 3) = 4/6 = 0.666...
        # weight_1 = 4 / (2 * 1) = 4/2 = 2.0
        self.assertAlmostEqual(weights[0], 4.0 / 6.0)
        self.assertAlmostEqual(weights[1], 2.0)


if __name__ == "__main__":
    unittest.main()

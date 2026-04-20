import importlib.util
import sys
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

SKLEARN_AVAILABLE = importlib.util.find_spec("sklearn") is not None
NUMPY_AVAILABLE = importlib.util.find_spec("numpy") is not None


class TrainingPipelineTest(unittest.TestCase):
    def setUp(self):
        if not (SKLEARN_AVAILABLE and NUMPY_AVAILABLE):
            self.skipTest("Required dependencies not installed")
        import numpy as np
        from training import train_and_evaluate
        self.np = np
        self.train_and_evaluate = train_and_evaluate

        X = self.np.array([
            [0.0, 0.1],
            [0.2, 0.0],
            [0.1, 0.2],
            [0.0, 0.3],
            [1.0, 1.1],
            [1.2, 1.0],
            [1.1, 1.2],
            [1.0, 1.3],
            [0.9, 1.0],
            [1.3, 0.9],
        ])
        y = self.np.array([0, 0, 0, 0, 1, 1, 1, 1, 1, 1])
        self.X_train, self.y_train = X[:6], y[:6]
        self.X_val, self.y_val = X[6:8], y[6:8]
        self.X_test, self.y_test = X[8:], y[8:]

    def test_metrics_present_and_in_range(self):
        results = self.train_and_evaluate(
            self.X_train,
            self.X_val,
            self.X_test,
            self.y_train,
            self.y_val,
            self.y_test,
            model_type="logistic_regression",
            class_weights={0: 1.0, 1: 1.0}
        )
        metrics = results["test_metrics"]
        for key in ("accuracy", "precision", "recall", "f1"):
            self.assertIn(key, metrics)
            self.assertGreaterEqual(metrics[key], 0.0)
            self.assertLessEqual(metrics[key], 1.0)

    def test_model_types_supported(self):
        model_types = {
            "random_forest": "RandomForestModel",
            "logistic_regression": "LogisticRegressionModel",
            "svm": "SVMModel",
        }
        for model_type, expected_class in model_types.items():
            results = self.train_and_evaluate(
                self.X_train,
                self.X_val,
                self.X_test,
                self.y_train,
                self.y_val,
                self.y_test,
                model_type=model_type,
                class_weights={0: 1.0, 1: 2.0}
            )
            self.assertEqual(results["model"].__class__.__name__, expected_class)

    def test_class_weight_applied(self):
        class_weights = {0: 1.0, 1: 2.0}
        results = self.train_and_evaluate(
            self.X_train,
            self.X_val,
            self.X_test,
            self.y_train,
            self.y_val,
            self.y_test,
            model_type="svm",
            class_weights=class_weights
        )
        self.assertEqual(results["model"]._params.get("class_weight"), class_weights)

    def test_invalid_model_type_raises(self):
        with self.assertRaises(ValueError):
            self.train_and_evaluate(
                self.X_train,
                self.X_val,
                self.X_test,
                self.y_train,
                self.y_val,
                self.y_test,
                model_type="not_a_model",
                class_weights=None
            )

    def test_non_binary_labels_raise(self):
        y_bad = self.np.array([0, 1, 2, 0, 1, 2])
        with self.assertRaises(ValueError):
            self.train_and_evaluate(
                self.X_train,
                self.X_val,
                self.X_test,
                y_bad,
                self.y_val,
                self.y_test,
                model_type="random_forest",
                class_weights=None
            )

    def test_tabnet_prefers_caller_feature_names_over_placeholder_meta(self):
        model = mock.Mock()
        model.predict.return_value = self.np.zeros(len(self.X_val), dtype=int)
        prep = mock.Mock(
            X_train=self.X_train,
            X_val=self.X_val,
            X_test=self.X_test,
            tabnet_meta=mock.Mock(feature_names=["feature_0", "feature_1"]),
        )

        with mock.patch("training.build_model_inputs", return_value=prep), mock.patch(
            "training.build_model",
            return_value=model,
        ), mock.patch(
            "training.classification_report",
            return_value="report",
        ), mock.patch(
            "training.confusion_matrix",
            return_value=self.np.array([[1, 0], [0, 1]]),
        ):
            self.train_and_evaluate(
                self.X_train,
                self.X_val,
                self.X_test,
                self.y_train,
                self.y_val,
                self.y_test,
                model_type="tabnet",
                class_weights={0: 1.0, 1: 1.0},
                feature_names=["selected_a", "selected_b"],
            )

        self.assertEqual(
            model.fit.call_args.kwargs["feature_names"],
            ["selected_a", "selected_b"],
        )


if __name__ == "__main__":
    unittest.main()

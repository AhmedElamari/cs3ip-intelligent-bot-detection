import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

SKLEARN_AVAILABLE = importlib.util.find_spec("sklearn") is not None
NUMPY_AVAILABLE = importlib.util.find_spec("numpy") is not None
PANDAS_AVAILABLE = importlib.util.find_spec("pandas") is not None


def _make_splits():
    df = pd.DataFrame({
        'account_creation_date': pd.date_range('2020-01-01', periods=12, freq='D'),
        'is_verified': [0] * 12,
        'followers_count': [100, 120, 110, 95, 10, 12, 9, 8, 11, 10, 9, 8],
        'friends_count': [80, 85, 82, 78, 50, 52, 55, 51, 50, 53, 55, 54],
        'listed_count': [1] * 12,
        'statuses_count': [20] * 12,
        'favourites_count': [10] * 12,
        'default_profile': [0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1],
        'default_profile_image': [0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1],
        'has_extended_profile': [1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
        'geo_enabled': [0] * 12,
        'protected': [0] * 12,
        'description': ['normal user'] * 4 + [''] * 8,
        'url': [None] * 12,
        'screen_name': ['alice', 'bob', 'charlie', 'david', 'bot001', 'bot002', 'bot003', 'bot004', 'bot005', 'bot006', 'bot007', 'bot008'],
        'label': [0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1],
    })
    return {
        'train': df.iloc[:6].copy(),
        'val': df.iloc[6:9].copy(),
        'test': df.iloc[9:].copy(),
    }


@unittest.skipUnless(
    SKLEARN_AVAILABLE and NUMPY_AVAILABLE and PANDAS_AVAILABLE,
    "sklearn/numpy/pandas not installed"
)
class RobustnessAnalysisTest(unittest.TestCase):

    def _build_benchmark(self):
        from benchmarking.data_prep import prepare_data
        from benchmarking.model_factory import create_models
        from benchmarking import ModelBenchmark
        from config import Config

        splits = _make_splits()
        config = Config()
        for name in config.get('models', {}).keys():
            config.set(f'models.{name}.enabled', name == 'logistic_regression')
        config.set('preprocessing.scale_features', True)
        X_train, X_val, X_test, y_train, y_val, y_test, feature_names = prepare_data(splits, config)
        benchmark = ModelBenchmark(models=create_models(config), experiment_name='robustness')
        benchmark.run_benchmark(
            X_train, y_train, X_val, y_val, X_test, y_test,
            feature_names=feature_names,
            verbose=False,
            compute_statistics=False,
            enable_scaling=True,
        )
        return benchmark, config, feature_names

    def test_prepare_eval_inputs_reuses_model_scaler(self):
        benchmark, _, _ = self._build_benchmark()
        expected_train, _, expected_test = benchmark.get_prepared_inputs('logistic_regression')
        replayed_test = benchmark.prepare_eval_inputs('logistic_regression', benchmark.base_test_inputs)
        np.testing.assert_allclose(replayed_test, expected_test)
        self.assertEqual(expected_train.shape[1], replayed_test.shape[1])

    def test_disabled_robustness_returns_no_results(self):
        from benchmarking.robustness import run_robustness_analysis

        benchmark, config, feature_names = self._build_benchmark()
        config.set('robustness.enabled', False)
        with tempfile.TemporaryDirectory(dir=ROOT) as tmp:
            results = run_robustness_analysis(benchmark, feature_names, config, Path(tmp))
        self.assertEqual(results, {})

    def test_enabled_robustness_outputs_expected_columns(self):
        from benchmarking.robustness import run_robustness_analysis

        benchmark, config, feature_names = self._build_benchmark()
        config.set('robustness.enabled', True)
        config.set('robustness.max_shap_samples', 2)
        with tempfile.TemporaryDirectory(dir=ROOT) as tmp:
            output_dir = Path(tmp)
            results = run_robustness_analysis(benchmark, feature_names, config, output_dir)
            self.assertTrue((output_dir / 'robustness_summary.csv').exists())
        self.assertIn('summary', results)
        self.assertIn('feature_attacks', results)
        summary = results['summary']
        feature_attacks = results['feature_attacks']
        for column in ('model', 'profile', 'attacked_true_bots', 'baseline_detected_bots', 'flip_rate'):
            self.assertIn(column, summary.columns)
        for column in ('model', 'feature', 'cost_tier', 'baseline_detected_bots', 'confidence_drop_mean'):
            self.assertIn(column, feature_attacks.columns)


if __name__ == '__main__':
    unittest.main()

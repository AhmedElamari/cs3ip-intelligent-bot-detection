import importlib.util
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

SKLEARN_AVAILABLE = importlib.util.find_spec("sklearn") is not None
NUMPY_AVAILABLE = importlib.util.find_spec("numpy") is not None
PANDAS_AVAILABLE = importlib.util.find_spec("pandas") is not None
XGBOOST_AVAILABLE = importlib.util.find_spec("xgboost") is not None


def _make_synthetic_splits(n_samples=40):
    """Build a minimal synthetic TwiBot-20-shaped DataFrame for smoke tests."""
    import numpy as np
    import pandas as pd
    rng = np.random.default_rng(2112)
    df = pd.DataFrame({
        'account_creation_date': pd.date_range('2020-01-01', periods=n_samples, freq='D'),
        'is_verified': rng.integers(0, 2, size=n_samples),
        'followers_count': rng.integers(0, 100, size=n_samples),
        'friends_count': rng.integers(0, 100, size=n_samples),
        'listed_count': rng.integers(0, 20, size=n_samples),
        'statuses_count': rng.integers(0, 200, size=n_samples),
        'favourites_count': rng.integers(0, 300, size=n_samples),
        'default_profile': rng.integers(0, 2, size=n_samples),
        'default_profile_image': rng.integers(0, 2, size=n_samples),
        'has_extended_profile': rng.integers(0, 2, size=n_samples),
        'geo_enabled': rng.integers(0, 2, size=n_samples),
        'protected': rng.integers(0, 2, size=n_samples),
        'description': ['test user'] * n_samples,
        'url': [None] * n_samples,
        'screen_name': [f'user_{i}' for i in range(n_samples)],
        'tweet_count': rng.integers(0, 50, size=n_samples),
        'following_sample_count': rng.integers(0, 50, size=n_samples),
        'follower_sample_count': rng.integers(0, 50, size=n_samples),
        'label': [0, 1] * (n_samples // 2),
    })
    return {
        'train': df.iloc[:int(n_samples * 0.6)].copy(),
        'val':   df.iloc[int(n_samples * 0.6):int(n_samples * 0.8)].copy(),
        'test':  df.iloc[int(n_samples * 0.8):].copy(),
    }


class BenchmarkSmokeTest(unittest.TestCase):
    """Existing LR-only smoke test — regression guard."""

    def test_benchmark_pipeline(self):
        if not (SKLEARN_AVAILABLE and NUMPY_AVAILABLE and PANDAS_AVAILABLE):
            self.skipTest("Required dependencies not installed")

        from benchmarking.data_prep import prepare_data
        from benchmarking.model_factory import create_models
        from benchmarking import ModelBenchmark
        from config import Config

        splits = _make_synthetic_splits()
        config = Config()
        for model_name in config.get('models', {}).keys():
            config.set(f'models.{model_name}.enabled', model_name == 'logistic_regression')

        X_train, X_val, X_test, y_train, y_val, y_test, feature_names = prepare_data(splits, config)
        models = create_models(config)

        benchmark = ModelBenchmark(models=models, experiment_name='smoke')
        results = benchmark.run_benchmark(
            X_train, y_train, X_val, y_val, X_test, y_test,
            feature_names=feature_names, verbose=False
        )

        self.assertIn('logistic_regression', results)
        self.assertIsNotNone(benchmark.y_test)
        self.assertEqual(len(benchmark.y_test), len(y_test))

        try:
            import matplotlib.pyplot as plt
        except ImportError:
            self.skipTest("matplotlib not installed")

        fig = benchmark.plot_roc_curves()
        self.assertIsNotNone(fig)
        plt.close(fig)


class BenchmarkStatisticsIntegrationTest(unittest.TestCase):
    """Verify that CIs and pairwise significance are generated after run_benchmark."""

    def setUp(self):
        if not (SKLEARN_AVAILABLE and NUMPY_AVAILABLE and PANDAS_AVAILABLE):
            self.skipTest("Required dependencies not installed")

    def _run_lr_rf_benchmark(self):
        from benchmarking.data_prep import prepare_data
        from benchmarking.model_factory import create_models
        from benchmarking import ModelBenchmark
        from config import Config

        splits = _make_synthetic_splits(n_samples=60)
        config = Config()
        for name in config.get('models', {}).keys():
            config.set(f'models.{name}.enabled', name in ('logistic_regression', 'random_forest'))

        X_train, X_val, X_test, y_train, y_val, y_test, feature_names = prepare_data(splits, config)
        models = create_models(config)

        benchmark = ModelBenchmark(models=models, experiment_name='stats_smoke')
        benchmark.run_benchmark(
            X_train, y_train, X_val, y_val, X_test, y_test,
            feature_names=feature_names, verbose=False
        )
        return benchmark

    def test_confidence_intervals_present(self):
        benchmark = self._run_lr_rf_benchmark()
        ci_df = benchmark.get_confidence_intervals()
        self.assertFalse(ci_df.empty, "Confidence intervals DataFrame must not be empty.")
        self.assertIn('model', ci_df.columns)
        self.assertIn('metric', ci_df.columns)
        self.assertIn('lower', ci_df.columns)
        self.assertIn('upper', ci_df.columns)
        self.assertIn('point', ci_df.columns)

    def test_ci_bounds_ordered(self):
        import numpy as np
        benchmark = self._run_lr_rf_benchmark()
        ci_df = benchmark.get_confidence_intervals()
        for _, row in ci_df.iterrows():
            if np.isnan(row['lower']) or np.isnan(row['upper']):
                continue
            self.assertLessEqual(row['lower'], row['point'])
            self.assertLessEqual(row['point'], row['upper'])

    def test_pairwise_significance_present(self):
        benchmark = self._run_lr_rf_benchmark()
        sig_df = benchmark.get_pairwise_significance()
        self.assertFalse(sig_df.empty, "Pairwise significance DataFrame must not be empty.")
        for col in ('model_a', 'model_b', 'metric', 'delta', 'bootstrap_p', 'mcnemar_p'):
            self.assertIn(col, sig_df.columns)

    def test_holm_corrected_column_present(self):
        benchmark = self._run_lr_rf_benchmark()
        sig_df = benchmark.get_pairwise_significance()
        self.assertIn('bootstrap_p_corrected', sig_df.columns)

    def test_comparison_table_unchanged(self):
        """Existing get_comparison_table() output is not broken by stats layer."""
        benchmark = self._run_lr_rf_benchmark()
        df = benchmark.get_comparison_table()
        self.assertIn('Model', df.columns)
        self.assertIn('F1', df.columns)

    def test_generate_report_includes_ci_section(self):
        benchmark = self._run_lr_rf_benchmark()
        report = benchmark.generate_report()
        self.assertIn('CONFIDENCE INTERVALS', report)
        self.assertIn('PAIRWISE MODEL SIGNIFICANCE', report)


class XGBoostBenchmarkSmokeTest(unittest.TestCase):
    """Smoke test for the xgboost model path in the benchmark pipeline."""

    def setUp(self):
        if not (SKLEARN_AVAILABLE and NUMPY_AVAILABLE and PANDAS_AVAILABLE and XGBOOST_AVAILABLE):
            self.skipTest("xgboost or other required dependencies not installed")

    def test_xgboost_runs_in_benchmark(self):
        from benchmarking.data_prep import prepare_data
        from benchmarking.model_factory import create_models
        from benchmarking import ModelBenchmark
        from config import Config

        splits = _make_synthetic_splits(n_samples=60)
        config = Config()
        for name in config.get('models', {}).keys():
            config.set(f'models.{name}.enabled', name == 'xgboost')

        X_train, X_val, X_test, y_train, y_val, y_test, feature_names = prepare_data(splits, config)
        models = create_models(config)

        self.assertIn('xgboost', models)

        benchmark = ModelBenchmark(models=models, experiment_name='xgb_smoke')
        results = benchmark.run_benchmark(
            X_train, y_train, X_val, y_val, X_test, y_test,
            feature_names=feature_names, verbose=False
        )

        self.assertIn('xgboost', results)
        self.assertNotIn('gradient_boosting', results)
        test_m = results['xgboost']['test_metrics']
        self.assertIn('f1', test_m)
        self.assertIn('roc_auc', test_m)

    def test_xgboost_produces_feature_importance(self):
        from benchmarking.data_prep import prepare_data
        from benchmarking.model_factory import create_models
        from benchmarking import ModelBenchmark
        from config import Config

        splits = _make_synthetic_splits(n_samples=60)
        config = Config()
        for name in config.get('models', {}).keys():
            config.set(f'models.{name}.enabled', name == 'xgboost')

        X_train, X_val, X_test, y_train, y_val, y_test, feature_names = prepare_data(splits, config)
        models = create_models(config)
        benchmark = ModelBenchmark(models=models, experiment_name='xgb_fi')
        benchmark.run_benchmark(
            X_train, y_train, X_val, y_val, X_test, y_test,
            feature_names=feature_names, verbose=False
        )

        fi = benchmark.results['xgboost']['feature_importance']
        self.assertIsNotNone(fi)
        self.assertGreater(len(fi), 0)


if __name__ == '__main__':
    unittest.main()

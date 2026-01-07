import importlib.util
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

SKLEARN_AVAILABLE = importlib.util.find_spec("sklearn") is not None
NUMPY_AVAILABLE = importlib.util.find_spec("numpy") is not None
PANDAS_AVAILABLE = importlib.util.find_spec("pandas") is not None


class BenchmarkSmokeTest(unittest.TestCase):
    def test_benchmark_pipeline(self):
        if not (SKLEARN_AVAILABLE and NUMPY_AVAILABLE and PANDAS_AVAILABLE):
            self.skipTest("Required dependencies not installed")

        import numpy as np
        import pandas as pd

        from benchmarking.data_prep import prepare_data
        from benchmarking.model_factory import create_models
        from benchmarking import ModelBenchmark
        from config import Config

        rng = np.random.default_rng(2112)
        n_samples = 30
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
            'label': np.array([0, 1] * (n_samples // 2)),
        })

        config = Config()
        for model_name in config.get('models', {}).keys():
            config.set(f'models.{model_name}.enabled', model_name == 'logistic_regression')

        X_train, X_val, X_test, y_train, y_val, y_test, feature_names = prepare_data(df, config)
        models = create_models(config)

        benchmark = ModelBenchmark(models=models, experiment_name='smoke')
        results = benchmark.run_benchmark(
            X_train, y_train,
            X_val, y_val,
            X_test, y_test,
            feature_names=feature_names,
            verbose=False
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


if __name__ == '__main__':
    unittest.main()

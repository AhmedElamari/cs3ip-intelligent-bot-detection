import importlib.util
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

SKLEARN_AVAILABLE = importlib.util.find_spec("sklearn") is not None
NUMPY_AVAILABLE = importlib.util.find_spec("numpy") is not None
PANDAS_AVAILABLE = importlib.util.find_spec("pandas") is not None
XGBOOST_AVAILABLE = importlib.util.find_spec("xgboost") is not None
SCIPY_AVAILABLE = importlib.util.find_spec("scipy") is not None


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
        config.set('preprocessing.scale_features', True)

        X_train, X_val, X_test, y_train, y_val, y_test, feature_names = prepare_data(splits, config)
        models = create_models(config)

        benchmark = ModelBenchmark(models=models, experiment_name='smoke')
        results = benchmark.run_benchmark(
            X_train, y_train, X_val, y_val, X_test, y_test,
            feature_names=feature_names, verbose=False,
            compute_statistics=False,
            enable_scaling=True,
        )

        self.assertIn('logistic_regression', results)
        self.assertIsNotNone(benchmark.y_test)
        self.assertEqual(len(benchmark.y_test), len(y_test))

        try:
            import matplotlib.pyplot as plt
        except ImportError:
            self.skipTest("matplotlib not installed")

        fig_pr = benchmark.plot_pr_curves_top(top_n=1)
        self.assertIsNotNone(fig_pr)
        plt.close(fig_pr)

        fig_cm_norm = benchmark.plot_best_confusion_matrix(normalize="true")
        self.assertIsNotNone(fig_cm_norm)
        plt.close(fig_cm_norm)

        fig_cm_raw = benchmark.plot_best_confusion_matrix(normalize=None)
        self.assertIsNotNone(fig_cm_raw)
        plt.close(fig_cm_raw)


class BenchmarkStatisticsIntegrationTest(unittest.TestCase):
    """Verify that CIs and pairwise significance are generated after run_benchmark."""

    _shared_benchmark = None  # class-level cache

    @classmethod
    def setUpClass(cls):
        if not (SKLEARN_AVAILABLE and NUMPY_AVAILABLE and PANDAS_AVAILABLE):
            return
        cls._shared_benchmark = cls._build_lr_rf_benchmark()

    def setUp(self):
        if not (SKLEARN_AVAILABLE and NUMPY_AVAILABLE and PANDAS_AVAILABLE):
            self.skipTest("Required dependencies not installed")

    @classmethod
    def _build_lr_rf_benchmark(cls):
        from benchmarking.data_prep import prepare_data
        from benchmarking.model_factory import create_models
        from benchmarking import ModelBenchmark
        from config import Config

        splits = _make_synthetic_splits(n_samples=60)
        config = Config()
        for name in config.get('models', {}).keys():
            config.set(f'models.{name}.enabled', name in ('logistic_regression', 'random_forest'))
        config.set('preprocessing.scale_features', True)

        X_train, X_val, X_test, y_train, y_val, y_test, feature_names = prepare_data(splits, config)
        models = create_models(config)

        benchmark = ModelBenchmark(models=models, experiment_name='stats_smoke')
        benchmark.run_benchmark(
            X_train, y_train, X_val, y_val, X_test, y_test,
            feature_names=feature_names, verbose=False,
            compute_statistics=True,
            statistics_bootstrap_samples=100,
            statistics_metrics=['f1'],
            enable_scaling=True,
        )
        return benchmark

    def _run_lr_rf_benchmark(self):
        return self._shared_benchmark

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
        if not SCIPY_AVAILABLE:
            self.assertTrue((sig_df['mcnemar_type'] == 'unavailable').all())

    def test_default_statistics_include_f1_macro(self):
        from benchmarking.data_prep import prepare_data
        from benchmarking.model_factory import create_models
        from benchmarking import ModelBenchmark
        from config import Config

        splits = _make_synthetic_splits(n_samples=60)
        config = Config()
        for name in config.get('models', {}).keys():
            config.set(f'models.{name}.enabled', name in ('logistic_regression', 'random_forest'))
        config.set('preprocessing.scale_features', True)

        X_train, X_val, X_test, y_train, y_val, y_test, feature_names = prepare_data(splits, config)
        benchmark = ModelBenchmark(models=create_models(config), experiment_name='stats_defaults')
        benchmark.run_benchmark(
            X_train, y_train, X_val, y_val, X_test, y_test,
            feature_names=feature_names, verbose=False,
            compute_statistics=True,
            statistics_bootstrap_samples=20,
            enable_scaling=True,
        )
        self.assertIn('f1_macro', set(benchmark.get_confidence_intervals()['metric']))
        self.assertIn('f1_macro', set(benchmark.get_pairwise_significance()['metric']))

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
        self.assertIn('Confidence Intervals', report)
        self.assertIn('Pairwise Model Significance', report)

    def test_statistics_mcnemar_unavailable_when_scipy_absent(self):
        """Benchmark pairwise rows carry mcnemar_type='unavailable' when scipy is blocked.

        Uses a freshly trained benchmark (not the shared cache) so that the
        re-compute call cannot contaminate other tests.
        """
        import sys
        from unittest.mock import patch

        # Fresh instance: no shared-state mutation risk
        benchmark = self._build_lr_rf_benchmark()

        with patch.dict(sys.modules, {'scipy': None, 'scipy.stats': None}):
            benchmark._compute_statistics(
                metrics=['f1'], n_bootstrap=20, include_mcnemar=True, verbose=False
            )

        sig_df = benchmark.get_pairwise_significance()
        self.assertFalse(sig_df.empty)
        self.assertTrue((sig_df['mcnemar_type'] == 'unavailable').all())

    def test_statistics_can_be_skipped(self):
        from benchmarking.data_prep import prepare_data
        from benchmarking.model_factory import create_models
        from benchmarking import ModelBenchmark
        from config import Config

        splits = _make_synthetic_splits(n_samples=60)
        config = Config()
        for name in config.get('models', {}).keys():
            config.set(f'models.{name}.enabled', name in ('logistic_regression', 'random_forest'))
        config.set('preprocessing.scale_features', True)

        X_train, X_val, X_test, y_train, y_val, y_test, feature_names = prepare_data(splits, config)
        models = create_models(config)

        benchmark = ModelBenchmark(models=models, experiment_name='stats_skip_smoke')
        benchmark.run_benchmark(
            X_train, y_train, X_val, y_val, X_test, y_test,
            feature_names=feature_names, verbose=False, compute_statistics=False,
            enable_scaling=True,
        )
        self.assertTrue(benchmark.get_confidence_intervals().empty)
        self.assertTrue(benchmark.get_pairwise_significance().empty)

    def test_prepare_data_return_metadata_alignment(self):
        from benchmarking.data_prep import prepare_data
        from config import Config

        splits = _make_synthetic_splits(n_samples=60)
        X_train, X_val, X_test, y_train, y_val, y_test, feature_names, metadata = prepare_data(
            splits,
            Config(),
            return_metadata=True,
        )
        self.assertEqual(len(metadata), len(X_test))
        self.assertEqual(metadata['label'].tolist(), y_test.astype(int).tolist())
        self.assertIn('user_id', metadata.columns)
        self.assertIn('row_index', metadata.columns)


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
            feature_names=feature_names, verbose=False,
            compute_statistics=False,
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
            feature_names=feature_names, verbose=False,
            compute_statistics=False,
        )

        fi = benchmark.results['xgboost']['feature_importance']
        self.assertIsNotNone(fi)
        self.assertGreater(len(fi), 0)


TABNET_AVAILABLE = (
    importlib.util.find_spec("pytorch_tabnet") is not None
    and importlib.util.find_spec("torch") is not None
)


@unittest.skipUnless(
    SKLEARN_AVAILABLE and NUMPY_AVAILABLE and PANDAS_AVAILABLE and TABNET_AVAILABLE,
    "pytorch-tabnet or core deps not installed"
)
class TabNetBenchmarkSmokeTest(unittest.TestCase):
    """Smoke test for TabNet through the full benchmark pipeline."""

    def test_tabnet_benchmark_produces_results(self):
        from benchmarking.data_prep import prepare_data
        from benchmarking.model_factory import create_models
        from benchmarking import ModelBenchmark
        from config import Config

        splits = _make_synthetic_splits(n_samples=80)
        config = Config()
        for name in config.get('models', {}).keys():
            config.set(f'models.{name}.enabled', False)
        config.set('models.tabnet.enabled', True)
        # Small params for speed in CI
        config.set('models.tabnet.params.n_d', 8)
        config.set('models.tabnet.params.n_a', 8)
        config.set('models.tabnet.params.n_steps', 1)
        config.set('models.tabnet.params.max_epochs', 3)
        config.set('models.tabnet.params.patience', 2)
        config.set('models.tabnet.params.batch_size', 32)
        config.set('models.tabnet.params.virtual_batch_size', 8)

        X_train, X_val, X_test, y_train, y_val, y_test, feature_names = prepare_data(splits, config)
        models = create_models(config)

        benchmark = ModelBenchmark(models=models, experiment_name='tabnet_smoke')
        results = benchmark.run_benchmark(
            X_train, y_train, X_val, y_val, X_test, y_test,
            feature_names=feature_names, verbose=False,
            compute_statistics=False,
        )

        self.assertIn('tabnet', results)
        result = results['tabnet']
        self.assertIn('f1', result['val_metrics'])
        self.assertIn('f1', result['test_metrics'])
        # predict_proba path → roc_auc should be present
        self.assertIn('roc_auc', result['test_metrics'])

    def test_tabnet_feature_importance_present(self):
        from benchmarking.data_prep import prepare_data
        from benchmarking.model_factory import create_models
        from benchmarking import ModelBenchmark
        from config import Config

        splits = _make_synthetic_splits(n_samples=80)
        config = Config()
        for name in config.get('models', {}).keys():
            config.set(f'models.{name}.enabled', False)
        config.set('models.tabnet.enabled', True)
        config.set('models.tabnet.params.n_d', 8)
        config.set('models.tabnet.params.n_a', 8)
        config.set('models.tabnet.params.n_steps', 1)
        config.set('models.tabnet.params.max_epochs', 3)
        config.set('models.tabnet.params.patience', 2)
        config.set('models.tabnet.params.batch_size', 32)
        config.set('models.tabnet.params.virtual_batch_size', 8)

        X_train, X_val, X_test, y_train, y_val, y_test, feature_names = prepare_data(splits, config)
        models = create_models(config)

        benchmark = ModelBenchmark(models=models, experiment_name='tabnet_fi')
        benchmark.run_benchmark(
            X_train, y_train, X_val, y_val, X_test, y_test,
            feature_names=feature_names, verbose=False,
            compute_statistics=False,
        )

        fi = benchmark.results['tabnet']['feature_importance']
        self.assertIsNotNone(fi)
        self.assertGreater(len(fi), 0)


@unittest.skipUnless(
    SKLEARN_AVAILABLE and NUMPY_AVAILABLE and PANDAS_AVAILABLE and TABNET_AVAILABLE,
    "pytorch-tabnet or core deps not installed"
)
class TabNetStatisticsSmokeTest(unittest.TestCase):
    """Verify CIs and pairwise significance work when TabNet is in the comparison set."""

    def test_tabnet_lr_statistics_populated(self):
        from benchmarking.data_prep import prepare_data
        from benchmarking.model_factory import create_models
        from benchmarking import ModelBenchmark
        from config import Config

        splits = _make_synthetic_splits(n_samples=100)
        config = Config()
        for name in config.get('models', {}).keys():
            config.set(f'models.{name}.enabled', name in ('logistic_regression', 'tabnet'))
        config.set('preprocessing.scale_features', True)
        config.set('models.tabnet.params.n_d', 8)
        config.set('models.tabnet.params.n_a', 8)
        config.set('models.tabnet.params.n_steps', 1)
        config.set('models.tabnet.params.max_epochs', 3)
        config.set('models.tabnet.params.patience', 2)
        config.set('models.tabnet.params.batch_size', 32)
        config.set('models.tabnet.params.virtual_batch_size', 8)

        X_train, X_val, X_test, y_train, y_val, y_test, feature_names = prepare_data(splits, config)
        models = create_models(config)

        benchmark = ModelBenchmark(models=models, experiment_name='tabnet_stats_smoke')
        benchmark.run_benchmark(
            X_train, y_train, X_val, y_val, X_test, y_test,
            feature_names=feature_names, verbose=False,
            compute_statistics=True,
            statistics_bootstrap_samples=50,
            statistics_metrics=['f1'],
            enable_scaling=True,
        )

        ci_df = benchmark.get_confidence_intervals()
        self.assertFalse(ci_df.empty, "CIs must not be empty when TabNet included.")
        self.assertIn('tabnet', ci_df['model'].values)

        sig_df = benchmark.get_pairwise_significance()
        self.assertFalse(sig_df.empty, "Pairwise significance must not be empty.")


@unittest.skipUnless(
    SKLEARN_AVAILABLE and NUMPY_AVAILABLE and PANDAS_AVAILABLE,
    "sklearn/numpy/pandas not installed"
)
class ScalingBackwardCompatTest(unittest.TestCase):
    """LR/SVM receive scaled inputs when enable_scaling=True; backward compat restored by run_benchmark."""

    def test_lr_receives_scaled_inputs_when_enable_scaling_true(self):
        from benchmarking.data_prep import prepare_data
        from benchmarking.model_factory import create_models
        from benchmarking import ModelBenchmark
        from config import Config
        import numpy as np

        splits = _make_synthetic_splits(n_samples=60)
        config = Config()
        for name in config.get('models', {}).keys():
            config.set(f'models.{name}.enabled', name == 'logistic_regression')
        config.set('preprocessing.scale_features', True)

        X_train, X_val, X_test, y_train, y_val, y_test, feature_names = prepare_data(splits, config)
        models = create_models(config)
        benchmark = ModelBenchmark(models=models, experiment_name='scaling_lr')
        benchmark.run_benchmark(
            X_train, y_train, X_val, y_val, X_test, y_test,
            feature_names=feature_names, verbose=False,
            compute_statistics=False, enable_scaling=True,
        )
        X_tr, _, _ = benchmark.get_prepared_inputs('logistic_regression')
        # Scaled data has ~0 mean; non-constant columns have ~1 std (constant cols → std 0)
        col_means = np.mean(X_tr, axis=0)
        col_stds = np.std(X_tr, axis=0)
        np.testing.assert_allclose(col_means, 0, atol=1e-5)
        nonconst = col_stds > 0.1
        np.testing.assert_allclose(col_stds[nonconst], 1, atol=1e-5)

    def test_rf_receives_raw_inputs_when_enable_scaling_false(self):
        from benchmarking.data_prep import prepare_data
        from benchmarking.model_factory import create_models
        from benchmarking import ModelBenchmark
        from config import Config
        import numpy as np

        splits = _make_synthetic_splits(n_samples=60)
        config = Config()
        for name in config.get('models', {}).keys():
            config.set(f'models.{name}.enabled', name == 'random_forest')
        config.set('preprocessing.scale_features', False)

        X_train, X_val, X_test, y_train, y_val, y_test, feature_names = prepare_data(splits, config)
        models = create_models(config)
        benchmark = ModelBenchmark(models=models, experiment_name='scaling_rf')
        benchmark.run_benchmark(
            X_train, y_train, X_val, y_val, X_test, y_test,
            feature_names=feature_names, verbose=False,
            compute_statistics=False, enable_scaling=False,
        )
        X_tr, _, _ = benchmark.get_prepared_inputs('random_forest')
        np.testing.assert_array_almost_equal(X_tr, X_train)


@unittest.skipUnless(
    SKLEARN_AVAILABLE and NUMPY_AVAILABLE and PANDAS_AVAILABLE,
    "sklearn/numpy/pandas not installed"
)
class XAIPreparedInputsContractTest(unittest.TestCase):
    """XAI uses model-specific prepared inputs from benchmark.get_prepared_inputs()."""

    def test_get_prepared_inputs_returns_stored_arrays(self):
        from benchmarking.data_prep import prepare_data
        from benchmarking.model_factory import create_models
        from benchmarking import ModelBenchmark
        from config import Config
        import numpy as np

        splits = _make_synthetic_splits(n_samples=60)
        config = Config()
        for name in config.get('models', {}).keys():
            config.set(f'models.{name}.enabled', name in ('logistic_regression', 'random_forest'))
        config.set('preprocessing.scale_features', True)

        X_train, X_val, X_test, y_train, y_val, y_test, feature_names = prepare_data(splits, config)
        models = create_models(config)
        benchmark = ModelBenchmark(models=models, experiment_name='xai_contract')
        benchmark.run_benchmark(
            X_train, y_train, X_val, y_val, X_test, y_test,
            feature_names=feature_names, verbose=False,
            compute_statistics=False, enable_scaling=True,
        )
        X_tr_lr, X_val_lr, X_te_lr = benchmark.get_prepared_inputs('logistic_regression')
        X_tr_rf, X_val_rf, X_te_rf = benchmark.get_prepared_inputs('random_forest')
        # LR gets scaled; RF gets raw
        np.testing.assert_allclose(np.mean(X_tr_lr, axis=0), 0, atol=1e-5)
        np.testing.assert_array_almost_equal(X_tr_rf, X_train)

    def test_model_results_do_not_store_redundant_base_inputs(self):
        from benchmarking.data_prep import prepare_data
        from benchmarking.model_factory import create_models
        from benchmarking import ModelBenchmark
        from config import Config

        splits = _make_synthetic_splits(n_samples=60)
        config = Config()
        for name in config.get('models', {}).keys():
            config.set(f'models.{name}.enabled', name == 'logistic_regression')
        config.set('preprocessing.scale_features', True)

        X_train, X_val, X_test, y_train, y_val, y_test, feature_names = prepare_data(splits, config)
        models = create_models(config)
        benchmark = ModelBenchmark(models=models, experiment_name='dedupe_inputs')
        results = benchmark.run_benchmark(
            X_train, y_train, X_val, y_val, X_test, y_test,
            feature_names=feature_names, verbose=False,
            compute_statistics=False, enable_scaling=True,
        )

        result = results['logistic_regression']
        self.assertNotIn('base_X_train', result)
        self.assertNotIn('base_X_val', result)
        self.assertNotIn('base_X_test', result)
        self.assertIsNotNone(benchmark.base_train_inputs)
        self.assertIsNotNone(benchmark.base_val_inputs)
        self.assertIsNotNone(benchmark.base_test_inputs)

    def test_run_explainability_analysis_uses_prepared_inputs(self):
        from benchmarking.data_prep import prepare_data
        from benchmarking.model_factory import create_models
        from benchmarking import ModelBenchmark
        from benchmarking.xai_reporting import run_explainability_analysis
        from config import Config
        from pathlib import Path
        import tempfile

        splits = _make_synthetic_splits(n_samples=60)
        config = Config()
        for name in config.get('models', {}).keys():
            config.set(f'models.{name}.enabled', name == 'logistic_regression')
        config.set('preprocessing.scale_features', True)
        config.set('explainability.shap.enabled', False)
        config.set('explainability.feature_importance.enabled', False)

        X_train, X_val, X_test, y_train, y_val, y_test, feature_names = prepare_data(splits, config)
        models = create_models(config)
        benchmark = ModelBenchmark(models=models, experiment_name='xai_prepared')
        benchmark.run_benchmark(
            X_train, y_train, X_val, y_val, X_test, y_test,
            feature_names=feature_names, verbose=False,
            compute_statistics=False, enable_scaling=True,
        )
        with tempfile.TemporaryDirectory(dir=ROOT) as tmp:
            out = Path(tmp)
            results = run_explainability_analysis(benchmark, feature_names, config, out)
        self.assertIn('lime_explanations', results)

    def test_optional_robustness_analysis_writes_artifacts(self):
        from benchmarking.data_prep import prepare_data
        from benchmarking.model_factory import create_models
        from benchmarking import ModelBenchmark
        from benchmarking.output_utils import (
            save_feature_vulnerability_outputs,
            save_robustness_degradation_figure,
        )
        from benchmarking.robustness import run_robustness_analysis
        from config import Config

        splits = _make_synthetic_splits(n_samples=60)
        config = Config()
        for name in config.get('models', {}).keys():
            config.set(f'models.{name}.enabled', name == 'logistic_regression')
        config.set('preprocessing.scale_features', True)
        config.set('robustness.enabled', True)

        X_train, X_val, X_test, y_train, y_val, y_test, feature_names = prepare_data(splits, config)
        models = create_models(config)
        benchmark = ModelBenchmark(models=models, experiment_name='robustness_smoke')
        benchmark.run_benchmark(
            X_train, y_train, X_val, y_val, X_test, y_test,
            feature_names=feature_names, verbose=False,
            compute_statistics=False, enable_scaling=True,
        )

        with TemporaryDirectory(dir=ROOT) as tmp:
            out = Path(tmp)
            results = run_robustness_analysis(benchmark, feature_names, config, out)
            self.assertIn('summary', results)
            self.assertTrue((out / 'robustness_summary.csv').exists())
            self.assertTrue((out / 'feature_attack_results.csv').exists())
            self.assertTrue((out / 'robustness_degradation.csv').exists())
            self.assertTrue((out / 'feature_resilience.csv').exists())
            self.assertTrue((out / 'feature_resilience.md').exists())
            self.assertTrue((out / 'shap_rank_stability.csv').exists())
            self.assertTrue((out / 'shap_cumulative_ablation.csv').exists())
            try:
                import matplotlib  # noqa: F401
            except ImportError:
                pass
            else:
                save_robustness_degradation_figure(benchmark, out)
                save_feature_vulnerability_outputs(benchmark, out)
                self.assertTrue((out / 'robustness_profile_degradation.png').exists())
                self.assertTrue((out / 'top_feature_vulnerabilities.csv').exists())
                self.assertTrue((out / 'feature_attack_flip_rates_best_model.png').exists())


if __name__ == '__main__':
    unittest.main()

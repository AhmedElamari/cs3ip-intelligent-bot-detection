import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

try:
    import numpy as np
except ImportError:  # pragma: no cover - exercised by dependency-aware skip
    np = None

try:
    import pandas as pd
except ImportError:  # pragma: no cover - exercised by dependency-aware skip
    pd = None

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

SKLEARN_AVAILABLE = importlib.util.find_spec("sklearn") is not None
NUMPY_AVAILABLE = np is not None
PANDAS_AVAILABLE = pd is not None


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

    def _build_benchmark(
        self,
        *,
        feature_selection=False,
        n_features=5,
        enabled_models=("logistic_regression",),
    ):
        from benchmarking.data_prep import prepare_data
        from benchmarking.model_factory import create_models
        from benchmarking import ModelBenchmark
        from config import Config

        splits = _make_splits()
        config = Config()
        for name in config.get('models', {}).keys():
            config.set(f'models.{name}.enabled', name in enabled_models)
        config.set('preprocessing.scale_features', True)
        config.set('preprocessing.feature_selection', feature_selection)
        config.set('preprocessing.n_features', n_features)
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

    def test_prepare_eval_inputs_aligns_wider_dataframe_to_model_features(self):
        benchmark, _, feature_names = self._build_benchmark(feature_selection=True, n_features=5)
        expected_test = benchmark.get_prepared_inputs('logistic_regression')[2]
        wider_eval = pd.DataFrame(
            benchmark.base_test_inputs,
            columns=feature_names,
        )
        wider_eval = wider_eval.assign(extra_noise=1.0)
        wider_eval = wider_eval[['extra_noise', *feature_names]]

        replayed_test = benchmark.prepare_eval_inputs('logistic_regression', wider_eval)

        np.testing.assert_allclose(replayed_test, expected_test)
        self.assertEqual(expected_test.shape[1], replayed_test.shape[1])

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
        with tempfile.TemporaryDirectory(dir=ROOT) as tmp:
            output_dir = Path(tmp)
            results = run_robustness_analysis(benchmark, feature_names, config, output_dir)
            self.assertTrue((output_dir / 'robustness_summary.csv').exists())
            self.assertTrue((output_dir / 'robustness_degradation.csv').exists())
        self.assertIn('summary', results)
        self.assertIn('feature_attacks', results)
        self.assertIn('degradation', results)
        summary = results['summary']
        feature_attacks = results['feature_attacks']
        degradation = results['degradation']
        for column in ('model', 'profile', 'attacked_true_bots', 'baseline_detected_bots', 'flip_rate'):
            self.assertIn(column, summary.columns)
        for column in (
            'model', 'feature', 'cost_tier', 'baseline_detected_bots',
            'confidence_drop_mean',
        ):
            self.assertIn(column, feature_attacks.columns)
        for column in ('model', 'scenario', 'macro_f1', 'pr_auc'):
            self.assertIn(column, degradation.columns)

    def test_robustness_degradation_csv_schema(self):
        from benchmarking.robustness import RobustnessAnalyzer

        benchmark, config, feature_names = self._build_benchmark()
        config.set('robustness.enabled', True)
        analyzer = RobustnessAnalyzer(benchmark, feature_names, config)
        results = analyzer.run()
        deg = results['degradation']
        profiles = list(config.get('robustness.profiles', []))
        for model in benchmark.results:
            sub = deg[deg['model'] == model].sort_values('scenario')
            scenarios = set(sub['scenario'].tolist())
            self.assertIn('baseline', scenarios)
            for p in profiles:
                self.assertIn(p, scenarios)
            for _, row in sub.iterrows():
                self.assertTrue(0.0 <= row['macro_f1'] <= 1.0)
                self.assertTrue(0.0 <= row['pr_auc'] <= 1.0)
        with tempfile.TemporaryDirectory(dir=ROOT) as tmp:
            analyzer.save_outputs(Path(tmp), results)
            self.assertTrue((Path(tmp) / 'robustness_degradation.csv').exists())

    def test_save_outputs_writes_compact_json_manifest(self):
        from benchmarking.robustness import RobustnessAnalyzer

        benchmark, config, feature_names = self._build_benchmark()
        config.set('robustness.enabled', True)
        analyzer = RobustnessAnalyzer(benchmark, feature_names, config)

        with tempfile.TemporaryDirectory(dir=ROOT) as tmp:
            output_dir = Path(tmp)
            results = analyzer.run()
            analyzer.save_outputs(output_dir, results)
            report = json.loads((output_dir / 'robustness_report.json').read_text(encoding='utf-8'))

        self.assertIn('artifacts', report)
        self.assertIn('overview', report)
        self.assertNotIn('summary_rows', report)
        self.assertNotIn('shap_rank_stability_rows', report)

        summary_manifest = report['artifacts']['summary']
        self.assertEqual('robustness_summary.csv', summary_manifest['file'])
        self.assertEqual(len(results['summary']), summary_manifest['rows'])
        self.assertListEqual(list(results['summary'].columns), summary_manifest['columns'])
        self.assertEqual(['logistic_regression'], report['overview']['models'])
        self.assertIn('degradation', report['artifacts'])

    def test_save_outputs_sorts_frames_and_pretty_prints_json(self):
        from benchmarking.robustness import RobustnessAnalyzer

        benchmark, config, feature_names = self._build_benchmark(
            enabled_models=("logistic_regression", "random_forest")
        )
        config.set('robustness.enabled', True)
        analyzer = RobustnessAnalyzer(benchmark, feature_names, config)

        with tempfile.TemporaryDirectory(dir=ROOT) as tmp:
            output_dir = Path(tmp)
            results = analyzer.run()
            analyzer.save_outputs(output_dir, results)
            summary_df = pd.read_csv(output_dir / 'robustness_summary.csv')
            feature_attacks_df = pd.read_csv(output_dir / 'feature_attack_results.csv')
            json_text = (output_dir / 'robustness_report.json').read_text(encoding='utf-8')
            report = json.loads(json_text)

        from benchmarking.output_formatting import format_frame_for_export

        expected_summary = format_frame_for_export(
            results['summary'].sort_values(['model', 'profile']).reset_index(drop=True)
        )
        expected_feature_attacks = format_frame_for_export(
            results['feature_attacks'].sort_values(['model', 'feature']).reset_index(drop=True)
        )
        summary_df = summary_df.fillna('')
        expected_summary = expected_summary.fillna('')
        feature_attacks_df = feature_attacks_df.fillna('')
        expected_feature_attacks = expected_feature_attacks.fillna('')

        pd.testing.assert_frame_equal(summary_df, expected_summary, check_dtype=False)
        pd.testing.assert_frame_equal(
            feature_attacks_df,
            expected_feature_attacks,
            check_dtype=False,
        )
        self.assertIn('\n  "artifacts"', json_text)
        self.assertEqual(sorted(summary_df['model'].dropna().unique().tolist()), report['overview']['models'])

    def test_stable_reducers_ignore_input_order(self):
        from benchmarking.robustness import RobustnessAnalyzer

        values = np.array([0.1, 0.3, 0.2, 0.4, 0.5], dtype=np.float64)
        reversed_values = values[::-1]

        self.assertEqual(
            RobustnessAnalyzer._stable_mean(values).hex(),
            RobustnessAnalyzer._stable_mean(reversed_values).hex(),
        )
        self.assertEqual(
            RobustnessAnalyzer._stable_std(values).hex(),
            RobustnessAnalyzer._stable_std(reversed_values).hex(),
        )
        self.assertEqual(
            RobustnessAnalyzer._stable_median(values).hex(),
            RobustnessAnalyzer._stable_median(reversed_values).hex(),
        )

    def test_precision_policy_rounds_metrics_and_preserves_counts(self):
        from benchmarking.robustness import RobustnessAnalyzer

        benchmark, config, feature_names = self._build_benchmark()
        config.set('robustness.enabled', True)
        analyzer = RobustnessAnalyzer(benchmark, feature_names, config)

        with tempfile.TemporaryDirectory(dir=ROOT) as tmp:
            output_dir = Path(tmp)
            results = analyzer.run()
            analyzer.save_outputs(output_dir, results)
            summary_text = (output_dir / 'robustness_summary.csv').read_text(encoding='utf-8')
            report = json.loads((output_dir / 'robustness_report.json').read_text(encoding='utf-8'))
            summary_df = pd.read_csv(output_dir / 'robustness_summary.csv')

        self.assertRegex(summary_text, r"\d+\.\d{4}")
        self.assertTrue(pd.api.types.is_integer_dtype(summary_df['attacked_true_bots']))
        self.assertTrue(pd.api.types.is_integer_dtype(summary_df['baseline_detected_bots']))
        self.assertTrue(
            all(
                len(str(value).split('.')[-1]) <= 4
                for value in report['overview'].values()
                if isinstance(value, float)
            )
        )

    def test_save_outputs_are_byte_stable(self):
        from benchmarking.robustness import RobustnessAnalyzer

        benchmark, config, feature_names = self._build_benchmark(
            enabled_models=("logistic_regression", "random_forest")
        )
        config.set('robustness.enabled', True)
        analyzer = RobustnessAnalyzer(benchmark, feature_names, config)

        with tempfile.TemporaryDirectory(dir=ROOT) as tmp_a, tempfile.TemporaryDirectory(dir=ROOT) as tmp_b:
            results_a = analyzer.run()
            analyzer.save_outputs(Path(tmp_a), results_a)
            results_b = analyzer.run()
            analyzer.save_outputs(Path(tmp_b), results_b)

            for filename in (
                'robustness_summary.csv',
                'feature_attack_results.csv',
                'robustness_degradation.csv',
                'robustness_report.json',
            ):
                self.assertEqual(
                    (Path(tmp_a) / filename).read_text(encoding='utf-8'),
                    (Path(tmp_b) / filename).read_text(encoding='utf-8'),
                )

    def test_robustness_metrics_ignore_equivalent_input_order(self):
        from benchmarking.robustness import RobustnessAnalyzer

        benchmark, config, feature_names = self._build_benchmark(
            enabled_models=("logistic_regression", "random_forest")
        )
        config.set('robustness.enabled', True)

        analyzer_a = RobustnessAnalyzer(benchmark, feature_names, config)
        analyzer_b = RobustnessAnalyzer(benchmark, feature_names, config)
        analyzer_b.base_test = analyzer_b.base_test.sample(frac=1, random_state=2112).reset_index(drop=True)
        analyzer_b.y_test = analyzer_b.y_test[::-1]

        results_a = analyzer_a.run()
        results_b = analyzer_b.run()

        pd.testing.assert_frame_equal(
            results_a['summary'].sort_values(['model', 'profile']).reset_index(drop=True).fillna(''),
            results_b['summary'].sort_values(['model', 'profile']).reset_index(drop=True).fillna(''),
            check_dtype=False,
        )
        pd.testing.assert_frame_equal(
            results_a['feature_attacks'].sort_values(['model', 'feature']).reset_index(drop=True).fillna(''),
            results_b['feature_attacks'].sort_values(['model', 'feature']).reset_index(drop=True).fillna(''),
            check_dtype=False,
        )


if __name__ == '__main__':
    unittest.main()

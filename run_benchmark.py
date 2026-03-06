"""
Bot Detection Benchmark Pipeline
================================
CLI entry point for multi-model benchmarking with XAI analysis.

Usage:
    python run_benchmark.py
    python run_benchmark.py --config config/config.yaml
    python run_benchmark.py --explain --save-plots
"""

import argparse
from pathlib import Path
from datetime import datetime

from config import Config, load_config
from benchmarking import ModelBenchmark
from benchmarking.data_prep import load_data, prepare_data
from benchmarking.model_factory import create_models
from benchmarking.output_utils import save_comparison_outputs, save_final_outputs
from benchmarking.xai_reporting import run_explainability_analysis

REPO_ROOT = Path(__file__).resolve().parent
TWIBOT20_DATA_DIR = REPO_ROOT / "data"


def main():
    parser = argparse.ArgumentParser(
        description='Bot Detection Model Benchmarking Pipeline (TwiBot-20 only)',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument(
        '--config', '-c',
        type=str,
        default=None,
        help='Path to configuration file (YAML or JSON)'
    )
    parser.add_argument(
        '--output', '-o',
        type=str,
        default='results',
        help='Output directory for results'
    )
    parser.add_argument(
        '--explain',
        action='store_true',
        help='Run explainability analysis'
    )
    parser.add_argument(
        '--save-plots',
        action='store_true',
        help='Save visualization plots'
    )
    parser.add_argument(
        '--models',
        type=str,
        nargs='+',
        default=None,
        help='Specific models to benchmark (e.g., random_forest svm)'
    )
    parser.add_argument(
        '--smote',
        action='store_true',
        help='Use SMOTE for class balancing'
    )
    parser.add_argument(
        '--scale',
        action='store_true',
        help='Scale features'
    )
    parser.add_argument(
        '--skip-statistics',
        action='store_true',
        help='Skip bootstrap confidence intervals and pairwise significance tests'
    )
    parser.add_argument(
        '--statistics-bootstrap-samples',
        type=int,
        default=1000,
        help='Bootstrap resamples for inferential statistics (default: 1000)'
    )
    parser.add_argument(
        '--skip-mcnemar',
        action='store_true',
        help='Skip McNemar paired test in pairwise significance output'
    )
    args = parser.parse_args()

    if args.config:
        config = load_config(args.config)
    else:
        config = Config()

    if args.smote:
        config.set('preprocessing.handle_imbalance', True)
        config.set('preprocessing.imbalance_method', 'smote')
    if args.scale:
        config.set('preprocessing.scale_features', True)
    if args.save_plots or args.explain:
        config.set('output.save_plots', True)
    if args.models:
        known_models = set(config.get('models', {}).keys())
        unknown = [m for m in args.models if m not in known_models]
        if unknown:
            parser.error(
                f"Unknown model(s): {unknown}. Available: {sorted(known_models)}\n"
                "Note: 'gradient_boosting' has been replaced by 'xgboost'."
            )
        for model_name in known_models:
            config.set(f'models.{model_name}.enabled', model_name in args.models)

    enabled_models = config.get_enabled_models()
    scale_from_config = config.get('preprocessing.scale_features')
    scaled_models = {'logistic_regression', 'svm'}
    needs_scaling = any(m in enabled_models for m in scaled_models)
    if not scale_from_config and needs_scaling:
        config.set('preprocessing.scale_features', True)
        print(
            "\n[Compatibility] Scaling disabled by config but logistic_regression/svm enabled; "
            "auto-restoring scaling for those models."
        )

    output_dir = Path(args.output)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = output_dir / f'benchmark_{timestamp}'
    output_dir.mkdir(parents=True, exist_ok=True)

    data_source_str = str(TWIBOT20_DATA_DIR)

    print("="*60)
    print("BOT DETECTION MODEL BENCHMARK")
    print("="*60)
    print(f"Data:   {data_source_str}")
    print(f"Output: {output_dir}")
    print(f"Models: {config.get_enabled_models()}")

    data_splits = load_data(TWIBOT20_DATA_DIR)

    X_train, X_val, X_test, y_train, y_val, y_test, feature_names = prepare_data(data_splits, config)

    models = create_models(config)

    benchmark = ModelBenchmark(
        models=models,
        experiment_name=f'benchmark_{timestamp}'
    )

    benchmark.run_benchmark(
        X_train, y_train,
        X_val, y_val,
        X_test, y_test,
        feature_names=feature_names,
        verbose=True,
        compute_statistics=not args.skip_statistics,
        statistics_bootstrap_samples=args.statistics_bootstrap_samples,
        include_mcnemar=not args.skip_mcnemar,
        enable_scaling=config.get('preprocessing.scale_features'),
    )

    benchmark.print_summary()

    save_comparison_outputs(benchmark, output_dir, config)

    if args.explain or config.get('explainability.enabled'):
        xai_results = run_explainability_analysis(
            benchmark,
            feature_names,
            config,
            output_dir
        )

        if 'feature_importance' in xai_results:
            xai_results['feature_importance'].to_csv(
                output_dir / 'feature_importance_comparison.csv'
            )

    save_final_outputs(benchmark, output_dir, config)

    print("\n" + "="*60)
    print("BENCHMARK COMPLETE")
    print("="*60)
    print(f"Results saved to: {output_dir}")

    best_name, best_model, best_metrics = benchmark.get_best_model('f1')
    print(f"\n[BEST] Model: {best_name}")
    print(f"   F1 Score:  {best_metrics['f1']:.4f}")
    print(f"   ROC-AUC:   {best_metrics.get('roc_auc', 'N/A')}")

    return benchmark


if __name__ == '__main__':
    main()

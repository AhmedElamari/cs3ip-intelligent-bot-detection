"""
Bot Detection Benchmark Pipeline
================================
Main script to run comprehensive model benchmarking with XAI analysis.

Usage:
    python benchmark.py
    python benchmark.py --config config/config.yaml
    python benchmark.py --explain --save-plots
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
    args = parser.parse_args()
    
    # Load configuration
    if args.config:
        config = load_config(args.config)
    else:
        config = Config()
    
    # Override config with command line arguments
    if args.smote:
        config.set('preprocessing.handle_imbalance', True)
        config.set('preprocessing.imbalance_method', 'smote')
    if args.scale:
        config.set('preprocessing.scale_features', True)
    if args.save_plots or args.explain:
        config.set('output.save_plots', True)
    if args.models:
        # Disable models not specified
        for model_name in config.get('models', {}).keys():
            config.set(f'models.{model_name}.enabled', model_name in args.models)

    enabled_models = config.get_enabled_models()
    if (
        not config.get('preprocessing.scale_features')
        and any(name in enabled_models for name in ('logistic_regression', 'svm'))
    ):
        print("\nEnabling feature scaling for logistic_regression/svm...")
        config.set('preprocessing.scale_features', True)
    
    # Set up output directory
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
    
    # Load data
    data_splits = load_data(TWIBOT20_DATA_DIR)
    
    # Prepare data
    X_train, X_val, X_test, y_train, y_val, y_test, feature_names = prepare_data(data_splits, config)
    
    # Create models
    models = create_models(config)
    
    # Run benchmark
    benchmark = ModelBenchmark(
        models=models,
        experiment_name=f'benchmark_{timestamp}'
    )
    
    benchmark.run_benchmark(
        X_train, y_train,
        X_val, y_val,
        X_test, y_test,
        feature_names=feature_names,
        verbose=True
    )
    
    # Print summary
    benchmark.print_summary()
    
    save_comparison_outputs(benchmark, output_dir, config)
    
    # Run explainability analysis
    if args.explain or config.get('explainability.enabled'):
        xai_results = run_explainability_analysis(
            benchmark,
            X_train, X_test, y_test,
            feature_names,
            config,
            output_dir
        )
        
        # Save feature importance comparison
        if 'feature_importance' in xai_results:
            xai_results['feature_importance'].to_csv(
                output_dir / 'feature_importance_comparison.csv'
            )
    
    save_final_outputs(benchmark, output_dir, config)
    
    print("\n" + "="*60)
    print("BENCHMARK COMPLETE")
    print("="*60)
    print(f"Results saved to: {output_dir}")
    
    # Return best model info
    best_name, best_model, best_metrics = benchmark.get_best_model('f1')
    print(f"\n[BEST] Model: {best_name}")
    print(f"   F1 Score:  {best_metrics['f1']:.4f}")
    print(f"   ROC-AUC:   {best_metrics.get('roc_auc', 'N/A')}")
    
    return benchmark


if __name__ == '__main__':
    main()

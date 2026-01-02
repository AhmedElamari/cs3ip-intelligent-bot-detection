"""
Bot Detection Benchmark Pipeline
================================
Main script to run comprehensive model benchmarking with XAI analysis.

Usage:
    python benchmark.py --data TwiBot-20_sample.json --labels labels.csv
    python benchmark.py --data data.csv --config config/config.yaml
    python benchmark.py --data data.csv --explain --save-plots
"""

import argparse
import sys
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# Project imports
from DataLoader import load_twibot_json
from FeatureEngineering import BotFeatureExtractor
from Preprocessing import BotDetector
from config import Config, load_config
from models import (
    get_model, get_all_models,
    LogisticRegressionModel, SVMModel, DecisionTreeModel,
    RandomForestModel, GradientBoostingModel
)
from benchmarking import ModelBenchmark, MetricsCalculator
from explainability import SHAPExplainer, LIMEExplainer, FeatureImportanceAnalyzer


def load_data(data_path: str, label_path: str = None) -> pd.DataFrame:
    """Load data from JSON or CSV file."""
    path = Path(data_path)
    
    if path.suffix.lower() == '.json':
        print(f"Loading JSON data from: {data_path}")
        df = load_twibot_json(data_path, label_path)
    else:
        print(f"Loading CSV data from: {data_path}")
        df = pd.read_csv(data_path)
        
        if label_path:
            labels = pd.read_csv(label_path)
            # Try to merge on common ID column
            id_cols = ['user_id', 'ID', 'id']
            for col in id_cols:
                if col in df.columns and col in labels.columns:
                    df = df.merge(labels[[col, 'label']], on=col, how='left')
                    break
    
    print(f"Loaded {len(df)} samples with {df.shape[1]} columns")
    return df


def prepare_data(df: pd.DataFrame, config: Config) -> tuple:
    """Prepare data: feature engineering, preprocessing, splitting."""
    print("\n" + "="*60)
    print("DATA PREPARATION")
    print("="*60)
    
    # Check for labels
    if 'label' not in df.columns:
        print("\n[WARNING] No 'label' column found.")
        print("Creating synthetic labels for demonstration...")
        np.random.seed(config.get('random_state'))
        df['label'] = np.random.randint(0, 2, size=len(df))
    
    df = df.dropna(subset=['label'])
    
    random_state = config.get('random_state')
    test_size = config.get('test_size', 0.1)
    val_size = config.get('val_size', 0.2)
    
    # Split indices first to avoid leakage when deriving reference dates
    indices = df.index.to_numpy()
    labels = df['label'].to_numpy()
    idx_temp, idx_test, labels_temp, _ = train_test_split(
        indices, labels, test_size=test_size, random_state=random_state
    )
    val_ratio = val_size / (1 - test_size)
    idx_train, idx_val, _, _ = train_test_split(
        idx_temp, labels_temp, test_size=val_ratio, random_state=random_state
    )
    
    # Feature engineering
    print("\nExtracting features...")
    reference_date = None
    if 'account_creation_date' in df.columns:
        account_creation = pd.to_datetime(
            df.loc[idx_train, 'account_creation_date'], errors='coerce'
        )
        if account_creation.notna().any():
            reference_date = account_creation.max()
        else:
            full_account_creation = pd.to_datetime(
                df['account_creation_date'], errors='coerce'
            )
            reference_date = pd.Timestamp.utcnow()
            if full_account_creation.dt.tz is not None:
                reference_date = reference_date.tz_localize(full_account_creation.dt.tz)
    extractor = BotFeatureExtractor(reference_date=reference_date)
    df = extractor.extract_all_features(df)
    print(f"Extracted {len(extractor.get_feature_names())} features")
    
    # Preprocessing
    print("\nPreprocessing...")
    detector = BotDetector()
    detector.data = df
    df = detector.preprocess()
    
    # Split data
    print(f"\nSplitting data (train/val/test: {(1-test_size-val_size)*100:.0f}%/{val_size*100:.0f}%/{test_size*100:.0f}%)...")
    feature_names = (
        df.drop(columns=['label'])
        .select_dtypes(include=[np.number])
        .columns
        .tolist()
    )
    X = df[feature_names]
    y = df['label']
    X_train, X_val, X_test = X.loc[idx_train], X.loc[idx_val], X.loc[idx_test]
    y_train, y_val, y_test = y.loc[idx_train], y.loc[idx_val], y.loc[idx_test]
    
    # Handle imbalance if configured
    if config.get('preprocessing.handle_imbalance'):
        method = config.get('preprocessing.imbalance_method', 'smote')
        print(f"\nApplying {method.upper()} for class balancing...")
        X_train, y_train = detector.handle_imbalance(X_train, y_train, method=method)
    
    # Scale features if configured
    if config.get('preprocessing.scale_features'):
        print("\nScaling features...")
        X_train, X_val, X_test = detector.scale_features(X_train, X_val, X_test)
    
    # Feature selection if configured
    if config.get('preprocessing.feature_selection'):
        n_features = config.get('preprocessing.n_features', 20)
        print(f"\nSelecting top {n_features} features...")
        X_train = detector.select_features(X_train, y_train, k=n_features)
        X_val = detector.apply_feature_selection(X_val)
        X_test = detector.apply_feature_selection(X_test)
        # Update feature names
        selected_indices = detector.selected_features
        feature_names = [feature_names[i] for i in selected_indices]
    
    print(f"\nFinal data shapes:")
    print(f"  Training:   {X_train.shape}")
    print(f"  Validation: {X_val.shape}")
    print(f"  Test:       {X_test.shape}")
    
    return X_train, X_val, X_test, y_train, y_val, y_test, feature_names


def create_models(config: Config) -> dict:
    """Create model instances based on configuration."""
    models = {}
    random_state = config.get('random_state')
    
    enabled_models = config.get_enabled_models()
    
    model_classes = {
        'logistic_regression': LogisticRegressionModel,
        'svm': SVMModel,
        'decision_tree': DecisionTreeModel,
        'random_forest': RandomForestModel,
        'gradient_boosting': GradientBoostingModel,
    }
    
    for model_name in enabled_models:
        if model_name in model_classes:
            params = config.get_model_params(model_name)
            models[model_name] = model_classes[model_name](**params)
    
    return models


def run_explainability_analysis(
    benchmark: ModelBenchmark,
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    feature_names: list,
    config: Config,
    output_dir: Path
) -> dict:
    """Run XAI analysis on trained models."""
    print("\n" + "="*60)
    print("EXPLAINABILITY ANALYSIS (XAI)")
    print("="*60)
    
    xai_results = {}
    
    # Feature importance analysis
    if config.get('explainability.feature_importance.enabled', True):
        print("\n--- Feature Importance Analysis ---")
        
        analyzer = FeatureImportanceAnalyzer(feature_names)
        importance_comparison = {}
        
        for model_name, result in benchmark.results.items():
            model = result['model']
            
            if model.supports_feature_importance:
                print(f"\n{model_name}:")
                
                # Built-in importance
                importance = analyzer.analyze_model_importance(model)
                importance_comparison[model_name] = importance
                
                # Print top features
                top_features = analyzer.get_top_features(importance, n=5)
                for feat, imp in top_features:
                    print(f"  {feat}: {imp:.4f}")
        
        if importance_comparison:
            # Compare across models
            comparison_df = analyzer.compare_importances(importance_comparison)
            xai_results['feature_importance'] = comparison_df
            
            # Save plot
            if config.get('output.save_plots'):
                try:
                    fig = analyzer.plot_importance_comparison(comparison_df)
                    fig.savefig(output_dir / 'feature_importance_comparison.png', dpi=150, bbox_inches='tight')
                    print(f"\nSaved feature importance plot")
                except Exception as e:
                    print(f"Could not save plot: {e}")
    
    # SHAP analysis for complex models
    if config.get('explainability.shap.enabled', True):
        print("\n--- SHAP Analysis ---")
        
        # Focus on less interpretable models
        target_models = ['random_forest', 'gradient_boosting', 'svm']
        
        for model_name in target_models:
            if model_name not in benchmark.results:
                continue
            
            model = benchmark.results[model_name]['model']
            print(f"\nAnalyzing {model_name} with SHAP...")
            
            try:
                shap_explainer = SHAPExplainer(model, feature_names)
                max_samples = config.get('explainability.shap.max_samples', 100)
                shap_explainer.fit(X_train, max_samples=max_samples)
                
                # Explain test set
                shap_values = shap_explainer.explain(X_test[:min(50, len(X_test))])
                
                # Get global importance from SHAP
                shap_importance = shap_explainer.get_global_importance()
                print(f"Top SHAP features for {model_name}:")
                sorted_shap = sorted(shap_importance.items(), key=lambda x: x[1], reverse=True)[:5]
                for feat, imp in sorted_shap:
                    print(f"  {feat}: {imp:.4f}")
                
                xai_results[f'shap_{model_name}'] = shap_importance
                
                # Save SHAP summary plot
                if config.get('output.save_plots'):
                    try:
                        fig = shap_explainer.plot_summary(
                            X_test[:min(50, len(X_test))],
                            max_display=10
                        )
                        fig.savefig(
                            output_dir / f'shap_summary_{model_name}.png',
                            dpi=150, bbox_inches='tight'
                        )
                    except Exception as e:
                        print(f"Could not save SHAP plot: {e}")
                        
            except Exception as e:
                print(f"SHAP analysis failed for {model_name}: {e}")
    
    # LIME analysis for individual predictions
    if config.get('explainability.lime.enabled', True):
        print("\n--- LIME Analysis (Sample Explanations) ---")
        
        # Pick best model for LIME
        best_name, best_model, _ = benchmark.get_best_model('f1')
        print(f"\nExplaining predictions from best model: {best_name}")
        
        try:
            lime_explainer = LIMEExplainer(best_model, feature_names)
            lime_explainer.fit(X_train)
            
            # Explain a few test instances
            n_explain = min(3, len(X_test))
            for i in range(n_explain):
                explanation = lime_explainer.explain_instance(
                    X_test[i],
                    num_features=config.get('explainability.lime.num_features', 10)
                )
                
                print(f"\nInstance {i+1} - Predicted: {explanation['predicted_class']}")
                print(f"  Probabilities: {explanation['prediction_proba']}")
                print("  Top contributing features:")
                for feat, contrib in list(explanation['feature_contributions'].items())[:5]:
                    direction = "+" if contrib > 0 else ""
                    print(f"    {feat}: {direction}{contrib:.4f}")
            
            xai_results['lime_explanations'] = True
            
        except Exception as e:
            print(f"LIME analysis failed: {e}")
    
    return xai_results


def main():
    parser = argparse.ArgumentParser(
        description='Bot Detection Model Benchmarking Pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        '--data', '-d',
        type=str,
        default='TwiBot-20_sample.json',
        help='Path to data file (JSON or CSV)'
    )
    parser.add_argument(
        '--labels', '-l',
        type=str,
        default=None,
        help='Path to labels CSV file'
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
    
    # Set up output directory
    output_dir = Path(args.output)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = output_dir / f'benchmark_{timestamp}'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*60)
    print("BOT DETECTION MODEL BENCHMARK")
    print("="*60)
    print(f"Data:   {args.data}")
    print(f"Output: {output_dir}")
    print(f"Models: {config.get_enabled_models()}")
    
    # Load data
    df = load_data(args.data, args.labels)
    
    # Prepare data
    X_train, X_val, X_test, y_train, y_val, y_test, feature_names = prepare_data(df, config)
    
    # Create models
    models = create_models(config)
    
    # Run benchmark
    benchmark = ModelBenchmark(
        models=models,
        experiment_name=f'benchmark_{timestamp}'
    )
    
    results = benchmark.run_benchmark(
        X_train, y_train,
        X_val, y_val,
        X_test, y_test,
        feature_names=feature_names,
        verbose=True
    )
    
    # Print summary
    benchmark.print_summary()
    
    # Save comparison table
    comparison_df = benchmark.get_comparison_table()
    comparison_df.to_csv(output_dir / 'model_comparison.csv', index=False)
    print(f"\nSaved comparison table to {output_dir / 'model_comparison.csv'}")
    
    # Generate and save plots
    if config.get('output.save_plots'):
        try:
            # Performance comparison
            fig = benchmark.plot_comparison()
            fig.savefig(output_dir / 'performance_comparison.png', dpi=150, bbox_inches='tight')
            
            # Training times
            fig = benchmark.plot_training_times()
            fig.savefig(output_dir / 'training_times.png', dpi=150, bbox_inches='tight')
            
            print(f"Saved performance plots to {output_dir}")
        except Exception as e:
            print(f"Warning: Could not save plots: {e}")
    
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
    
    # Save benchmark results
    benchmark.save_results(output_dir)
    
    # Generate report
    report = benchmark.generate_report()
    with open(output_dir / 'benchmark_report.txt', 'w') as f:
        f.write(report)
    
    # Save configuration
    config.to_json(output_dir / 'config.json')
    
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

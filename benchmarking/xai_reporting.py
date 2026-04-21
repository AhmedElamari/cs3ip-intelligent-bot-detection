"""
XAI reporting helpers for the benchmark pipeline.
"""

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from config import Config
from explainability import SHAPExplainer, LIMEExplainer, FeatureImportanceAnalyzer
from explainability.poster_shap import export_poster_shap


def run_explainability_analysis(
    benchmark: Any,
    feature_names: list,
    config: Config,
    output_dir: Path
) -> dict:
    """Run XAI analysis (feature importance, SHAP, LIME) using model-specific prepared inputs."""
    print("\n" + "=" * 60)
    print("EXPLAINABILITY ANALYSIS (XAI)")
    print("=" * 60)

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

            if config.get('output.save_plots'):
                try:
                    fig = analyzer.plot_importance_comparison(comparison_df)
                    fig.savefig(output_dir / 'feature_importance_comparison.png', dpi=150, bbox_inches='tight')
                    plt.close(fig)
                    print("\nSaved feature importance plot")
                except Exception as e:
                    print(f"Could not save plot: {e}")

    # SHAP analysis for complex models
    if config.get('explainability.shap.enabled', True):
        print("\n--- SHAP Analysis ---")
        print(
            "Note: Positive SHAP contributions push toward Bot (class 1), "
            "negative toward Human (class 0)."
        )

        # SHAP for tree-based models + TabNet (uses model-agnostic KernelExplainer path).
        target_models = ['random_forest', 'xgboost', 'tabnet']

        for model_name in target_models:
            if model_name not in benchmark.results:
                continue

            X_train_m, _, X_test_m = benchmark.get_prepared_inputs(model_name)
            model = benchmark.results[model_name]['model']
            print(f"\nAnalyzing {model_name} with SHAP...")

            try:
                shap_explainer = SHAPExplainer(model, feature_names)
                max_samples = config.get('explainability.shap.max_samples', 100)
                shap_explainer.fit(X_train_m, max_samples=max_samples)

                if len(X_test_m) == 0:
                    print("No test samples available for SHAP explanations.")
                    continue

                # Explain test set
                shap_explainer.explain(X_test_m[:min(50, len(X_test_m))])
                global_shap_values = shap_explainer.shap_values

                # Get global importance from SHAP
                shap_importance = shap_explainer.get_global_importance()
                print(f"Top SHAP features for {model_name}:")
                sorted_shap = sorted(
                    shap_importance.items(),
                    key=lambda x: x[1],
                    reverse=True
                )[:5]
                for feat, imp in sorted_shap:
                    print(f"  {feat}: {imp:.4f}")

                if len(X_test_m) > 0:
                    print(f"\nExample SHAP explanations for {model_name}:")
                    n_explain = min(2, len(X_test_m))
                    for i in range(n_explain):
                        if isinstance(X_test_m, pd.DataFrame):
                            instance = X_test_m.iloc[i:i + 1]
                        else:
                            instance = X_test_m[i:i + 1]
                        pred = model.predict(instance)[0]
                        pred_label = "Bot" if pred == 1 else "Human"
                        try:
                            proba = model.predict_proba(instance)[0]
                            confidence = proba[1] if pred == 1 else proba[0]
                            confidence_str = f"{confidence:.1%}"
                        except Exception:  # predict_proba unavailable or malformed
                            confidence_str = "N/A"
                        print(
                            f"\n  Instance {i+1} - Predicted: {pred_label} "
                            f"({confidence_str} confidence)"
                        )
                        print("  Top contributing features:")
                        instance_expl = shap_explainer.explain_instance(
                            instance,
                            instance_idx=0
                        )
                        sorted_contrib = sorted(
                            instance_expl.items(),
                            key=lambda x: abs(x[1]),
                            reverse=True
                        )[:5]
                        for feat, contrib in sorted_contrib:
                            direction = "->Bot" if contrib > 0 else "->Human"
                            print(f"    {feat}: {contrib:+.4f} {direction}")
                    shap_explainer.shap_values = global_shap_values

                xai_results[f'shap_{model_name}'] = shap_importance

                if config.get('output.save_plots'):
                    try:
                        n_shap = min(50, len(X_test_m))
                        X_shap = X_test_m[:n_shap]
                        fig = shap_explainer.plot_summary(X_shap, max_display=10)
                        fig.savefig(output_dir / f'shap_summary_{model_name}.png', dpi=150, bbox_inches='tight')
                        plt.close(fig)
                        if (
                            config.get('explainability.poster.enabled', False)
                            and model_name == config.get('explainability.poster.model', 'xgboost')
                        ):
                            export_poster_shap(
                                shap_explainer.shap_values,
                                X_shap,
                                list(feature_names),
                                model_name=model_name,
                                output_dir=output_dir / 'poster',
                                top_n=int(config.get('explainability.poster.top_n', 10)),
                            )
                    except Exception as e:
                        print(f"Could not save SHAP plot: {e}")

            except Exception as e:
                print(f"SHAP analysis failed for {model_name}: {e}")

    # LIME analysis for individual predictions
    if config.get('explainability.lime.enabled', True):
        print("\n--- LIME Analysis (Sample Explanations) ---")
        print(
            "Note: Positive LIME contributions push toward the predicted class; "
            "negative toward the opposite class."
        )

        # Pick best model for LIME
        best_name, best_model, _ = benchmark.get_best_model('f1')
        print(f"\nExplaining predictions from best model: {best_name}")

        try:
            X_train_m, _, X_test_m = benchmark.get_prepared_inputs(best_name)
            lime_explainer = LIMEExplainer(best_model, feature_names)
            lime_explainer.fit(X_train_m)

            # Explain a few test instances
            n_explain = min(3, len(X_test_m))
            if n_explain == 0:
                print("No test samples available for LIME explanations.")
            for i in range(n_explain):
                if isinstance(X_test_m, pd.DataFrame):
                    instance = X_test_m.iloc[i].to_numpy()
                else:
                    instance = X_test_m[i]
                explanation = lime_explainer.explain_instance(
                    instance,
                    num_features=config.get('explainability.lime.num_features', 10)
                )

                print(f"\nInstance {i+1} - Predicted: {explanation['predicted_class']}")
                print(f"  Probabilities: {explanation['prediction_proba']}")
                print("  Top contributing features:")
                predicted_class = explanation['predicted_class']
                opposite_class = "Human" if predicted_class == "Bot" else "Bot"
                for feat, contrib in list(explanation['feature_contributions'].items())[:5]:
                    direction = predicted_class if contrib >= 0 else opposite_class
                    print(f"    {feat}: {contrib:+.4f} -> {direction}")

            xai_results['lime_explanations'] = True

        except Exception as e:
            print(f"LIME analysis failed: {e}")

    return xai_results

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


def export_poster_shap(*args, **kwargs):
    """Lazily import poster SHAP export support only when it is actually used."""
    from explainability.poster_shap import export_poster_shap as _export_poster_shap

    return _export_poster_shap(*args, **kwargs)
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

        if getattr(benchmark, "test_metadata", None) is not None:
            try:
                export_misclassified_bot_lime(
                    benchmark,
                    feature_names,
                    output_dir,
                    n_examples=3,
                    num_lime_features=config.get('explainability.lime.num_features', 10),
                )
            except Exception as e:
                print(f"Misclassified-bot LIME export failed: {e}")

    return xai_results


def export_misclassified_bot_lime(
    benchmark: Any,
    feature_names: list,
    output_dir: Path,
    n_examples: int = 3,
    num_lime_features: int = 10,
) -> None:
    """Export LIME explanations for false-negative bot accounts from the best model."""
    best_name, best_model, _ = benchmark.get_best_model('f1')
    y_true = np.asarray(benchmark.y_test)
    y_pred = np.asarray(benchmark.predictions[best_name])
    proba = np.asarray(benchmark.probabilities[best_name])
    if proba.ndim == 1:
        proba = np.column_stack([1.0 - proba, proba])

    false_negative_idx = np.where((y_true == 1) & (y_pred == 0))[0]
    if false_negative_idx.size == 0:
        (output_dir / 'lime_misclassified_bots.md').write_text(
            "No false-negative bot accounts were available for LIME export on this test split.\n",
            encoding='utf-8',
        )
        pd.DataFrame(columns=_lime_export_columns()).to_csv(
            output_dir / 'lime_misclassified_bots.csv',
            index=False,
        )
        return

    order = np.argsort(proba[false_negative_idx, 1], kind='mergesort')
    selected_idx = false_negative_idx[order[:n_examples]]

    X_train_m, _, X_test_m = benchmark.get_prepared_inputs(best_name)
    lime_explainer = LIMEExplainer(best_model, feature_names)
    lime_explainer.fit(X_train_m)

    metadata = benchmark.test_metadata.reset_index(drop=True)
    rows = []
    for test_idx in selected_idx:
        instance = X_test_m.iloc[test_idx].to_numpy() if isinstance(X_test_m, pd.DataFrame) else X_test_m[test_idx]
        explanation = lime_explainer.explain_instance(instance, num_features=num_lime_features)
        rows.append(_lime_export_row(metadata, test_idx, y_true, y_pred, proba, explanation))

    frame = pd.DataFrame(rows, columns=_lime_export_columns())
    frame.to_csv(output_dir / 'lime_misclassified_bots.csv', index=False)
    (output_dir / 'lime_misclassified_bots.md').write_text(
        _lime_markdown(frame, best_name),
        encoding='utf-8',
    )


def _lime_export_row(metadata, test_idx, y_true, y_pred, proba, explanation):
    meta = metadata.iloc[int(test_idx)] if int(test_idx) < len(metadata) else {}
    row = {
        'user_id': str(meta.get('user_id', 'n/a')) if hasattr(meta, 'get') else 'n/a',
        'test_row_index': int(meta.get('row_index', test_idx)) if hasattr(meta, 'get') else int(test_idx),
        'true_label': int(y_true[test_idx]),
        'predicted_label': int(y_pred[test_idx]),
        'predicted_bot_probability': float(proba[test_idx, 1]),
        'predicted_human_probability': float(proba[test_idx, 0]),
    }
    contributions = sorted(
        explanation['feature_contributions'].items(),
        key=lambda item: abs(item[1]),
        reverse=True,
    )[:3]
    for index in range(3):
        feature, contribution = contributions[index] if index < len(contributions) else ('', np.nan)
        row[f'top_{index + 1}_feature'] = feature
        row[f'top_{index + 1}_contribution'] = contribution
        row[f'top_{index + 1}_direction'] = _lime_direction(contribution)
    return row


def _lime_direction(contribution) -> str:
    if pd.isna(contribution):
        return ''
    return 'toward_human' if contribution >= 0 else 'toward_bot'


def _lime_export_columns() -> list:
    base = [
        'user_id',
        'test_row_index',
        'true_label',
        'predicted_label',
        'predicted_bot_probability',
        'predicted_human_probability',
    ]
    contributor_columns = []
    for index in range(1, 4):
        contributor_columns.extend([
            f'top_{index}_feature',
            f'top_{index}_contribution',
            f'top_{index}_direction',
        ])
    return base + contributor_columns


def _lime_markdown(frame: pd.DataFrame, model_name: str) -> str:
    lines = [
        "# Misclassified Bot LIME Examples",
        "",
        f"Best model: `{model_name}`. Examples are false-negative bots sorted by lowest predicted bot probability.",
    ]
    for _, row in frame.iterrows():
        lines.extend([
            "",
            f"## Test row {int(row['test_row_index'])}",
            f"- user_id: `{row['user_id']}`",
            f"- true label: `{int(row['true_label'])}`; predicted label: `{int(row['predicted_label'])}`",
            f"- predicted bot probability: {float(row['predicted_bot_probability']):.4f}",
            f"- predicted human probability: {float(row['predicted_human_probability']):.4f}",
            "- top LIME contributors:",
        ])
        for index in range(1, 4):
            feature = row[f'top_{index}_feature']
            if not feature:
                continue
            lines.append(
                f"  - {feature}: {float(row[f'top_{index}_contribution']):+.4f} "
                f"({row[f'top_{index}_direction']})"
            )
    return "\n".join(lines) + "\n"

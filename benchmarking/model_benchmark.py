"""Benchmarking system for comparing multiple bot detection models."""

from typing import Dict, List, Optional, Any, Union, Set, Tuple
import numpy as np
import pandas as pd
from pathlib import Path
import time
import json
from datetime import datetime

from .metrics import MetricsCalculator


class ModelBenchmark:
    """Benchmark multiple bot detection models; compare metrics, generate reports and plots."""

    def __init__(
        self,
        models: Dict[str, Any] = None,
        metrics_calculator: MetricsCalculator = None,
        experiment_name: str = None
    ):
        self.models = models or {}
        self.metrics_calculator = metrics_calculator or MetricsCalculator()
        self.experiment_name = experiment_name or f"experiment_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        self.results: Dict[str, Dict[str, Any]] = {}
        self.training_times: Dict[str, float] = {}
        self.predictions: Dict[str, np.ndarray] = {}
        self.probabilities: Dict[str, np.ndarray] = {}
        self.y_val: Optional[np.ndarray] = None
        self.y_test: Optional[np.ndarray] = None
        self.confidence_intervals: Dict[str, Dict[str, Any]] = {}
        self.pairwise_significance: List[Dict[str, Any]] = []

    def add_model(self, name: str, model: Any) -> 'ModelBenchmark':
        self.models[name] = model
        return self

    def run_benchmark(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        feature_names: List[str] = None,
        verbose: bool = True,
        compute_statistics: bool = False,
        statistics_metrics: Optional[List[str]] = None,
        statistics_bootstrap_samples: int = 1000,
        statistics_alpha: float = 0.05,
        statistics_random_state: int = 2112,
        include_mcnemar: bool = True,
        enable_scaling: bool = True,
    ) -> Dict[str, Dict[str, Any]]:
        if verbose:
            print(f"\n{'='*60}")
            print(f"BENCHMARKING {len(self.models)} MODELS")
            print(f"{'='*60}")
            print(f"Training samples: {len(X_train)}")
            print(f"Validation samples: {len(X_val)}")
            print(f"Test samples: {len(X_test)}")
            print(f"Features: {X_train.shape[1]}")

        self.y_val = y_val
        self.y_test = y_test

        scaled_models = {'logistic_regression', 'svm'}

        for name, model in self.models.items():
            if verbose:
                print(f"\n{'-'*40}")
                print(f"Training: {name}")
                print(f"{'-'*40}")

            X_train_model, X_val_model, X_test_model = self._prepare_model_inputs(
                model_name=name,
                X_train=X_train,
                X_val=X_val,
                X_test=X_test,
                enable_scaling=enable_scaling,
                scaled_models=scaled_models,
                verbose=verbose,
            )

            if hasattr(model, 'prepare_eval_set'):
                model.prepare_eval_set(X_val_model, y_val)

            start_time = time.time()
            model.fit(X_train_model, y_train, feature_names=feature_names)
            training_time = time.time() - start_time
            self.training_times[name] = training_time

            if verbose:
                print(f"Training time: {training_time:.2f}s")

            y_val_pred = model.predict(X_val_model)
            y_test_pred = model.predict(X_test_model)
            self.predictions[name] = y_test_pred

            try:
                y_val_proba = model.predict_proba(X_val_model)
                y_test_proba = model.predict_proba(X_test_model)
                self.probabilities[name] = y_test_proba
            except (NotImplementedError, AttributeError):
                y_val_proba = None
                y_test_proba = None

            val_metrics = self.metrics_calculator.compute_all_metrics(
                y_val, y_val_pred, y_val_proba
            )
            test_metrics = self.metrics_calculator.compute_all_metrics(
                y_test, y_test_pred, y_test_proba
            )

            feature_importance = None
            if (
                hasattr(model, 'get_feature_importance')
                and hasattr(model, 'supports_feature_importance')
                and model.supports_feature_importance
            ):
                feature_importance = model.get_feature_importance()

            self.results[name] = {
                'model': model,
                'training_time': training_time,
                'val_metrics': val_metrics,
                'test_metrics': test_metrics,
                'feature_importance': feature_importance,
                'is_interpretable': model.is_interpretable if hasattr(model, 'is_interpretable') else False,
                'X_train': X_train_model,
                'X_val': X_val_model,
                'X_test': X_test_model,
            }

            if verbose:
                print(f"Validation F1: {val_metrics['f1']:.4f}")
                print(f"Test F1: {test_metrics['f1']:.4f}")
                if 'roc_auc' in test_metrics:
                    print(f"Test ROC-AUC: {test_metrics['roc_auc']:.4f}")

        if verbose:
            print(f"\n{'='*60}")
            print("BENCHMARK COMPLETE")
            print(f"{'='*60}")

        if compute_statistics:
            self._compute_statistics(
                metrics=statistics_metrics,
                n_bootstrap=statistics_bootstrap_samples,
                alpha=statistics_alpha,
                random_state=statistics_random_state,
                include_mcnemar=include_mcnemar,
                verbose=verbose,
            )
        else:
            self.confidence_intervals = {}
            self.pairwise_significance = []

        return self.results

    def _compute_statistics(
        self,
        metrics: Optional[List[str]] = None,
        n_bootstrap: int = 1000,
        alpha: float = 0.05,
        random_state: int = 2112,
        include_mcnemar: bool = True,
        verbose: bool = False,
    ) -> None:
        if self.y_test is None:
            return
        if metrics is None:
            metrics = ['f1', 'roc_auc', 'pr_auc', 'mcc', 'balanced_accuracy']

        if verbose:
            print("\nComputing bootstrap confidence intervals…")

        calc = self.metrics_calculator
        model_names = list(self.predictions.keys())

        for name in model_names:
            y_pred = self.predictions[name]
            y_proba = self.probabilities.get(name)
            cis: Dict[str, Any] = {}
            for m in metrics:
                lower, point, upper = calc.bootstrap_metric_ci(
                    self.y_test, y_pred, y_proba,
                    metric=m,
                    n_bootstrap=n_bootstrap,
                    alpha=alpha,
                    random_state=random_state,
                )
                cis[m] = {'lower': lower, 'point': point, 'upper': upper}
            self.confidence_intervals[name] = cis

        if verbose:
            print("Computing pairwise significance tests…")

        sig_rows = []
        for i, name_a in enumerate(model_names):
            for name_b in model_names[i + 1:]:
                preds_a = self.predictions[name_a]
                preds_b = self.predictions[name_b]
                probas_a = self.probabilities.get(name_a)
                probas_b = self.probabilities.get(name_b)

                mcnemar_result = {
                    'b': float('nan'),
                    'c': float('nan'),
                    'p_value': float('nan'),
                    'test_type': 'disabled',
                }
                if include_mcnemar:
                    mcnemar_result = calc.mcnemar_test(self.y_test, preds_a, preds_b)

                for m in metrics:
                    delta_result = calc.bootstrap_delta_ci(
                        self.y_test, preds_a, preds_b,
                        probas_a=probas_a, probas_b=probas_b,
                        metric=m,
                        n_bootstrap=n_bootstrap,
                        alpha=alpha,
                        random_state=random_state,
                    )
                    sig_rows.append({
                        'model_a': name_a,
                        'model_b': name_b,
                        'metric': m,
                        'delta': delta_result['delta'],
                        'ci_lower': delta_result['ci_lower'],
                        'ci_upper': delta_result['ci_upper'],
                        'bootstrap_p': delta_result['p_value'],
                        'mcnemar_b': mcnemar_result['b'],
                        'mcnemar_c': mcnemar_result['c'],
                        'mcnemar_p': mcnemar_result['p_value'],
                        'mcnemar_type': mcnemar_result['test_type'],
                    })

        metrics_seen = list(dict.fromkeys(r['metric'] for r in sig_rows))
        for m in metrics_seen:
            subset_idx = [i for i, r in enumerate(sig_rows) if r['metric'] == m]
            raw_ps = [sig_rows[i]['bootstrap_p'] for i in subset_idx]
            corrected = calc.holm_bonferroni(raw_ps)
            for i, idx in enumerate(subset_idx):
                sig_rows[idx]['bootstrap_p_corrected'] = corrected[i]

        self.pairwise_significance = sig_rows

    def get_confidence_intervals(self, format: str = 'dataframe'):
        if format == 'dict':
            return self.confidence_intervals

        rows = []
        for model_name, cis in self.confidence_intervals.items():
            for metric, bounds in cis.items():
                rows.append({
                    'model': model_name,
                    'metric': metric,
                    'lower': bounds['lower'],
                    'point': bounds['point'],
                    'upper': bounds['upper'],
                })
        return pd.DataFrame(rows)

    def get_pairwise_significance(self) -> pd.DataFrame:
        return pd.DataFrame(self.pairwise_significance)

    def get_comparison_table(
        self,
        metrics: List[str] = None,
        sort_by: str = 'f1',
        dataset: str = 'test'
    ) -> pd.DataFrame:
        if metrics is None:
            metrics = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc', 'mcc']

        metric_key = f'{dataset}_metrics'

        data = []
        for name, result in self.results.items():
            row = {'Model': name, 'Training Time (s)': result['training_time']}
            row['Interpretable'] = result['is_interpretable']

            for metric in metrics:
                if metric in result[metric_key]:
                    row[metric.upper()] = result[metric_key][metric]

            data.append(row)

        df = pd.DataFrame(data)

        sort_col = sort_by.upper() if sort_by.upper() in df.columns else 'F1'
        df = df.sort_values(sort_col, ascending=False)

        return df

    def get_best_model(self, metric: str = 'f1', dataset: str = 'test') -> tuple:
        if not self.results:
            raise ValueError("No benchmark results available. Run run_benchmark() first.")

        metric_key = f'{dataset}_metrics'

        best_name = None
        best_score = -float('inf')

        for name, result in self.results.items():
            score = result[metric_key].get(metric, 0)
            if score > best_score:
                best_score = score
                best_name = name

        return (
            best_name,
            self.results[best_name]['model'],
            self.results[best_name][metric_key]
        )

    def get_interpretable_models(self) -> List[str]:
        return [
            name for name, result in self.results.items()
            if result['is_interpretable']
        ]

    def get_prepared_inputs(self, model_name: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return (X_train, X_val, X_test) used for the given model. Raises if model not found."""
        if model_name not in self.results:
            raise KeyError(f"Model {model_name!r} not in benchmark results.")
        r = self.results[model_name]
        for k in ('X_train', 'X_val', 'X_test'):
            if k not in r:
                raise ValueError(
                    f"Prepared inputs not stored for {model_name}. "
                    "Ensure run_benchmark() completed successfully."
                )
        return r['X_train'], r['X_val'], r['X_test']

    def get_feature_importance_comparison(self) -> pd.DataFrame:
        importance_data = {}

        for name, result in self.results.items():
            if result['feature_importance'] is not None:
                importance_data[name] = result['feature_importance']

        if not importance_data:
            return pd.DataFrame()

        df = pd.DataFrame(importance_data)
        df.index.name = 'Feature'

        df['Average'] = df.mean(axis=1)
        df = df.sort_values('Average', ascending=False)

        return df

    def print_summary(self) -> None:
        print(f"\n{'='*70}")
        print(f"BENCHMARK SUMMARY: {self.experiment_name}")
        print(f"{'='*70}")

        for metric in ['f1', 'roc_auc', 'accuracy']:
            try:
                best_name, _, best_metrics = self.get_best_model(metric)
                print(f"\nBest by {metric.upper()}: {best_name} ({best_metrics[metric]:.4f})")
            except (KeyError, TypeError, ValueError):
                continue

        print(f"\n{'-'*70}")
        print("TEST SET PERFORMANCE")
        print(f"{'-'*70}")
        df = self.get_comparison_table()
        print(df.to_string(index=False))

        interpretable = self.get_interpretable_models()
        if interpretable:
            print(f"\nInterpretable models: {', '.join(interpretable)}")

    def plot_comparison(
        self,
        metrics: List[str] = None,
        figsize: tuple = (12, 6)
    ):
        import matplotlib.pyplot as plt

        if metrics is None:
            metrics = ['accuracy', 'precision', 'recall', 'f1']

        df = self.get_comparison_table(metrics=metrics)

        fig, ax = plt.subplots(figsize=figsize)

        x = np.arange(len(df))
        width = 0.8 / len(metrics)

        for i, metric in enumerate(metrics):
            col = metric.upper()
            if col in df.columns:
                offset = (i - len(metrics)/2 + 0.5) * width
                ax.bar(x + offset, df[col], width, label=metric.upper(), alpha=0.8)

        ax.set_ylabel('Score')
        ax.set_title('Model Performance Comparison')
        ax.set_xticks(x)
        ax.set_xticklabels(df['Model'], rotation=45, ha='right')
        ax.legend()
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()
        return fig

    def plot_training_times(self, figsize: tuple = (10, 5)):
        import matplotlib.pyplot as plt

        models = list(self.training_times.keys())
        times = list(self.training_times.values())

        fig, ax = plt.subplots(figsize=figsize)
        ax.barh(models, times, color='steelblue', alpha=0.7)
        ax.set_xlabel('Training Time (seconds)')
        ax.set_title('Model Training Times')

        for i, time in enumerate(times):
            ax.text(time + 0.01, i, f'{time:.2f}s', va='center')

        plt.tight_layout()
        return fig

    def plot_roc_curves(self, figsize: tuple = (10, 8)):
        import matplotlib.pyplot as plt
        from sklearn.metrics import roc_curve, roc_auc_score

        fig, ax = plt.subplots(figsize=figsize)

        colors = plt.cm.Set1(np.linspace(0, 1, len(self.probabilities)))

        for (name, proba), color in zip(self.probabilities.items(), colors):
            if proba is not None:
                y_proba = proba[:, 1] if len(proba.shape) > 1 else proba

                if self.y_test is None:
                    raise RuntimeError("y_test not available. Run run_benchmark() first.")

                auc = self.results[name]['test_metrics'].get('roc_auc', 0)
                fpr, tpr, _ = roc_curve(self.y_test, y_proba)
                ax.plot(fpr, tpr, color=color, label=f'{name} (AUC={auc:.3f})')

        ax.plot([0, 1], [0, 1], 'k--', label='Random')
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title('ROC Curves Comparison')
        ax.legend(loc='lower right')
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        return fig

    def save_results(self, path: str) -> None:
        path = self._validate_output_path(path)
        path.mkdir(parents=True, exist_ok=True)

        df = self.get_comparison_table()
        df.to_csv(path / 'comparison.csv', index=False)

        fi_df = self.get_feature_importance_comparison()
        if not fi_df.empty:
            fi_df.to_csv(path / 'feature_importance.csv')

        ci_df = self.get_confidence_intervals()
        if not ci_df.empty:
            ci_df.to_csv(path / 'metric_confidence_intervals.csv', index=False)

        sig_df = self.get_pairwise_significance()
        if not sig_df.empty:
            sig_df.to_csv(path / 'pairwise_significance.csv', index=False)

        results_json = {}
        for name, result in self.results.items():
            results_json[name] = {
                'training_time': result['training_time'],
                'val_metrics': result['val_metrics'],
                'test_metrics': result['test_metrics'],
                'is_interpretable': result['is_interpretable'],
                'confidence_intervals': self.confidence_intervals.get(name, {}),
            }

        with open(path / 'results.json', 'w') as f:
            json.dump(results_json, f, indent=2, default=str)

        print(f"Results saved to {path}")

    @staticmethod
    def _validate_output_path(path: str) -> Path:
        resolved = Path(path).expanduser().resolve()
        workspace = Path.cwd().resolve()
        if resolved != workspace and workspace not in resolved.parents:
            raise ValueError(
                f"Path must stay within workspace: {workspace}"
            )
        return resolved

    @staticmethod
    def _scale_features(X_train, X_val, X_test):
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        X_test_scaled = scaler.transform(X_test)
        return X_train_scaled, X_val_scaled, X_test_scaled

    @classmethod
    def _prepare_model_inputs(
        cls,
        model_name: str,
        X_train: np.ndarray,
        X_val: np.ndarray,
        X_test: np.ndarray,
        enable_scaling: bool,
        scaled_models: Set[str],
        verbose: bool,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        if enable_scaling and model_name in scaled_models:
            if verbose:
                print("  (Applying feature scaling)")
            return cls._scale_features(X_train, X_val, X_test)
        return X_train, X_val, X_test

    def generate_report(self) -> str:
        lines = []
        lines.append(f"{'='*70}")
        lines.append(f"BOT DETECTION MODEL BENCHMARK REPORT")
        lines.append(f"Experiment: {self.experiment_name}")
        lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append(f"{'='*70}")

        lines.append("\n## SUMMARY")
        lines.append(f"Models evaluated: {len(self.models)}")

        best_name, _, best_metrics = self.get_best_model('f1')
        lines.append(f"Best model (F1): {best_name} ({best_metrics['f1']:.4f})")

        lines.append("\n## DETAILED RESULTS")
        for name, result in self.results.items():
            lines.append(f"\n### {name}")
            lines.append(f"Interpretable: {result['is_interpretable']}")
            lines.append(f"Training time: {result['training_time']:.2f}s")
            lines.append("\nTest Metrics:")
            for metric, value in result['test_metrics'].items():
                if isinstance(value, float):
                    lines.append(f"  {metric}: {value:.4f}")

        fi_df = self.get_feature_importance_comparison()
        if not fi_df.empty:
            lines.append("\n## TOP FEATURES (by average importance)")
            top_features = fi_df.head(10)
            lines.append(top_features.to_string())

        ci_df = self.get_confidence_intervals()
        if not ci_df.empty:
            lines.append("\n## METRIC CONFIDENCE INTERVALS (bootstrap)")
            lines.append("  Format: point [lower, upper]")
            for model_name in ci_df['model'].unique():
                lines.append(f"\n  {model_name}:")
                subset = ci_df[ci_df['model'] == model_name]
                for _, row in subset.iterrows():
                    lines.append(
                        f"    {row['metric']:20s}: "
                        f"{row['point']:.4f} [{row['lower']:.4f}, {row['upper']:.4f}]"
                    )

        sig_df = self.get_pairwise_significance()
        if not sig_df.empty:
            lines.append("\n## PAIRWISE MODEL SIGNIFICANCE (test set)")
            lines.append(
                "  delta = metric_B - metric_A; "
                "bootstrap_p_corrected: Holm-Bonferroni adjusted p-value; "
                "mcnemar_p: McNemar exact test on prediction disagreements."
            )
            for m in sig_df['metric'].unique():
                mdf = sig_df[sig_df['metric'] == m][
                    ['model_a', 'model_b', 'delta', 'ci_lower', 'ci_upper',
                     'bootstrap_p_corrected', 'mcnemar_p']
                ]
                lines.append(f"\n  Metric: {m}")
                lines.append(mdf.to_string(index=False))

        return '\n'.join(lines)

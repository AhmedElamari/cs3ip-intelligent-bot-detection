"""Benchmarking system for comparing multiple bot detection models."""

from datetime import datetime
import json
from pathlib import Path
import time
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import numpy as np
import pandas as pd

from .metrics import MetricsCalculator
from .output_formatting import format_frame_for_export, format_payload_for_export


DEFAULT_COMPARISON_METRICS = [
    "accuracy",
    "precision",
    "recall",
    "f1",
    "roc_auc",
    "mcc",
]
DEFAULT_STATISTICS_METRICS = [
    "f1",
    "roc_auc",
    "pr_auc",
    "mcc",
    "balanced_accuracy",
]


class ModelBenchmark:
    """Benchmark multiple bot detection models; compare metrics, generate reports and plots."""

    def __init__(
        self,
        models: Dict[str, Any] = None,
        metrics_calculator: MetricsCalculator = None,
        experiment_name: str = None,
    ):
        self.models = models or {}
        self.metrics_calculator = metrics_calculator or MetricsCalculator()
        self.experiment_name = (
            experiment_name
            or f"experiment_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )

        self.results: Dict[str, Dict[str, Any]] = {}
        self.training_times: Dict[str, float] = {}
        self.predictions: Dict[str, np.ndarray] = {}
        self.probabilities: Dict[str, np.ndarray] = {}
        self.y_val: Optional[np.ndarray] = None
        self.y_test: Optional[np.ndarray] = None
        self.confidence_intervals: Dict[str, Dict[str, Any]] = {}
        self.pairwise_significance: List[Dict[str, Any]] = []
        self.base_train_inputs: Optional[Union[np.ndarray, pd.DataFrame]] = None
        self.base_val_inputs: Optional[Union[np.ndarray, pd.DataFrame]] = None
        self.base_test_inputs: Optional[Union[np.ndarray, pd.DataFrame]] = None
        self.base_y_train: Optional[np.ndarray] = None
        self.base_feature_names: List[str] = []
        self.robustness_summary: Optional[pd.DataFrame] = None

    def add_model(self, name: str, model: Any) -> "ModelBenchmark":
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
        enable_scaling: bool = False,
    ) -> Dict[str, Dict[str, Any]]:
        if verbose:
            print(f"\n{'=' * 60}")
            print(f"BENCHMARKING {len(self.models)} MODELS")
            print(f"{'=' * 60}")
            print(f"Training samples: {len(X_train)}")
            print(f"Validation samples: {len(X_val)}")
            print(f"Test samples: {len(X_test)}")
            print(f"Features: {X_train.shape[1]}")

        self.y_val = y_val
        self.y_test = y_test
        self.base_train_inputs = self._copy_input(X_train)
        self.base_val_inputs = self._copy_input(X_val)
        self.base_test_inputs = self._copy_input(X_test)
        self.base_y_train = np.asarray(y_train)
        self.base_feature_names = list(feature_names or [])

        scaled_models = {"logistic_regression", "svm"}

        for name, model in self.models.items():
            if verbose:
                print(f"\n{'-' * 40}")
                print(f"Training: {name}")
                print(f"{'-' * 40}")

            X_train_model, X_val_model, X_test_model, scaler = self._prepare_model_inputs(
                model_name=name,
                X_train=X_train,
                X_val=X_val,
                X_test=X_test,
                enable_scaling=enable_scaling,
                scaled_models=scaled_models,
                verbose=verbose,
            )

            if hasattr(model, "prepare_eval_set"):
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
                hasattr(model, "get_feature_importance")
                and hasattr(model, "supports_feature_importance")
                and model.supports_feature_importance
            ):
                feature_importance = model.get_feature_importance()

            self.results[name] = {
                "model": model,
                "training_time": training_time,
                "val_metrics": val_metrics,
                "test_metrics": test_metrics,
                "feature_importance": feature_importance,
                "is_interpretable": (
                    model.is_interpretable if hasattr(model, "is_interpretable") else False
                ),
                "X_train": X_train_model,
                "X_val": X_val_model,
                "X_test": X_test_model,
                "feature_names": list(feature_names or []),
                "scaler": scaler,
            }

            if verbose:
                print(f"Validation F1: {val_metrics['f1']:.4f}")
                print(f"Test F1: {test_metrics['f1']:.4f}")
                if "roc_auc" in test_metrics:
                    print(f"Test ROC-AUC: {test_metrics['roc_auc']:.4f}")

        if verbose:
            print(f"\n{'=' * 60}")
            print("BENCHMARK COMPLETE")
            print(f"{'=' * 60}")

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
            metrics = list(DEFAULT_STATISTICS_METRICS)

        if verbose:
            print("\nComputing bootstrap confidence intervals...")

        calc = self.metrics_calculator
        model_names = list(self.predictions.keys())

        for name in model_names:
            y_pred = self.predictions[name]
            y_proba = self.probabilities.get(name)
            cis: Dict[str, Any] = {}
            for metric in metrics:
                lower, point, upper = calc.bootstrap_metric_ci(
                    self.y_test,
                    y_pred,
                    y_proba,
                    metric=metric,
                    n_bootstrap=n_bootstrap,
                    alpha=alpha,
                    random_state=random_state,
                )
                cis[metric] = {"lower": lower, "point": point, "upper": upper}
            self.confidence_intervals[name] = cis

        if verbose:
            print("Computing pairwise significance tests...")

        sig_rows = []
        for index, name_a in enumerate(model_names):
            for name_b in model_names[index + 1 :]:
                preds_a = self.predictions[name_a]
                preds_b = self.predictions[name_b]
                probas_a = self.probabilities.get(name_a)
                probas_b = self.probabilities.get(name_b)

                mcnemar_result = {
                    "b": float("nan"),
                    "c": float("nan"),
                    "p_value": float("nan"),
                    "test_type": "disabled",
                }
                if include_mcnemar:
                    mcnemar_result = calc.mcnemar_test(self.y_test, preds_a, preds_b)

                for metric in metrics:
                    delta_result = calc.bootstrap_delta_ci(
                        self.y_test,
                        preds_a,
                        preds_b,
                        probas_a=probas_a,
                        probas_b=probas_b,
                        metric=metric,
                        n_bootstrap=n_bootstrap,
                        alpha=alpha,
                        random_state=random_state,
                    )
                    sig_rows.append(
                        {
                            "model_a": name_a,
                            "model_b": name_b,
                            "metric": metric,
                            "delta": delta_result["delta"],
                            "ci_lower": delta_result["ci_lower"],
                            "ci_upper": delta_result["ci_upper"],
                            "bootstrap_p": delta_result["p_value"],
                            "mcnemar_b": mcnemar_result["b"],
                            "mcnemar_c": mcnemar_result["c"],
                            "mcnemar_p": mcnemar_result["p_value"],
                            "mcnemar_type": mcnemar_result["test_type"],
                        }
                    )

        metrics_seen = list(dict.fromkeys(row["metric"] for row in sig_rows))
        for metric in metrics_seen:
            subset_idx = [idx for idx, row in enumerate(sig_rows) if row["metric"] == metric]
            raw_ps = [sig_rows[idx]["bootstrap_p"] for idx in subset_idx]
            corrected = calc.holm_bonferroni(raw_ps)
            for corrected_idx, row_idx in enumerate(subset_idx):
                sig_rows[row_idx]["bootstrap_p_corrected"] = corrected[corrected_idx]

        self.pairwise_significance = sig_rows

    def get_confidence_intervals(self, format: str = "dataframe"):
        if format == "dict":
            return self.confidence_intervals

        rows = []
        for model_name, cis in self.confidence_intervals.items():
            for metric, bounds in cis.items():
                rows.append(
                    {
                        "model": model_name,
                        "metric": metric,
                        "lower": bounds["lower"],
                        "point": bounds["point"],
                        "upper": bounds["upper"],
                    }
                )
        return pd.DataFrame(rows)

    def get_pairwise_significance(self) -> pd.DataFrame:
        return pd.DataFrame(self.pairwise_significance)

    def get_comparison_table(
        self,
        metrics: List[str] = None,
        sort_by: str = "f1",
        dataset: str = "test",
    ) -> pd.DataFrame:
        if metrics is None:
            metrics = list(DEFAULT_COMPARISON_METRICS)

        ranked_results = self._ranked_result_items(sort_by=sort_by, dataset=dataset)
        rows = []
        metric_key = self._metric_key(dataset)
        for rank, (name, result) in enumerate(ranked_results, start=1):
            row = {
                "Rank": rank,
                "Model": name,
                "Training Time (s)": result["training_time"],
                "Interpretable": result["is_interpretable"],
            }
            for metric in metrics:
                if metric in result[metric_key]:
                    row[metric.upper()] = result[metric_key][metric]
            rows.append(row)
        return pd.DataFrame(rows)

    def get_best_model(self, metric: str = "f1", dataset: str = "test") -> tuple:
        ranked_results = self._ranked_result_items(sort_by=metric, dataset=dataset)
        if not ranked_results:
            raise ValueError("No benchmark results available. Run run_benchmark() first.")

        best_name, best_result = ranked_results[0]
        return best_name, best_result["model"], best_result[self._metric_key(dataset)]

    def get_interpretable_models(self) -> List[str]:
        return [
            name for name, result in self.results.items() if result["is_interpretable"]
        ]

    def get_prepared_inputs(
        self, model_name: str
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return (X_train, X_val, X_test) used for the given model."""
        if model_name not in self.results:
            raise KeyError(f"Model {model_name!r} not in benchmark results.")
        result = self.results[model_name]
        for key in ("X_train", "X_val", "X_test"):
            if key not in result:
                raise ValueError(
                    f"Prepared inputs not stored for {model_name}. "
                    "Ensure run_benchmark() completed successfully."
                )
        return result["X_train"], result["X_val"], result["X_test"]

    def prepare_eval_inputs(
        self,
        model_name: str,
        X_eval: Union[np.ndarray, pd.DataFrame],
    ) -> Union[np.ndarray, pd.DataFrame]:
        if model_name not in self.results:
            raise KeyError(f"Model {model_name!r} not in benchmark results.")
        result = self.results[model_name]
        X_eval = self._align_eval_input(X_eval, result.get("feature_names") or [])
        scaler = result.get("scaler")
        if scaler is None:
            return self._copy_input(X_eval)
        if isinstance(X_eval, pd.DataFrame) and not hasattr(scaler, "feature_names_in_"):
            X_eval = X_eval.to_numpy()
        return scaler.transform(X_eval)

    def get_feature_importance_raw(self) -> pd.DataFrame:
        importance_data = self._feature_importance_data()
        if not importance_data:
            return pd.DataFrame()

        comparison_df = self.get_feature_importance_comparison()
        ordered_features = comparison_df.index.tolist() if not comparison_df.empty else sorted(
            {feature for values in importance_data.values() for feature in values}
        )
        raw_df = pd.DataFrame(importance_data).fillna(0.0)
        raw_df = raw_df.reindex(ordered_features)
        raw_df.index.name = "Feature"
        return raw_df

    def get_feature_importance_comparison(self) -> pd.DataFrame:
        importance_data = self._feature_importance_data()
        if not importance_data:
            return pd.DataFrame()

        all_features = sorted(
            {feature for values in importance_data.values() for feature in values}
        )
        normalized = {}
        for model_name, values in importance_data.items():
            column = np.array([float(values.get(feature, 0.0)) for feature in all_features])
            max_value = float(np.max(column)) if len(column) else 0.0
            if max_value > 0:
                column = column / max_value
            normalized[model_name] = column

        df = pd.DataFrame(normalized, index=all_features)
        df.index.name = "Feature"
        df["Average_Normalized"] = df.mean(axis=1)
        df = df.sort_values("Average_Normalized", ascending=False)
        return df

    def print_summary(self) -> None:
        print(f"\n{'=' * 70}")
        print(f"BENCHMARK SUMMARY: {self.experiment_name}")
        print(f"{'=' * 70}")

        for metric in ("f1", "roc_auc", "accuracy"):
            try:
                best_name, _, best_metrics = self.get_best_model(metric)
                print(
                    f"\nBest by {metric.upper()}: {best_name} "
                    f"({best_metrics[metric]:.4f})"
                )
            except (KeyError, TypeError, ValueError):
                continue

        print(f"\n{'-' * 70}")
        print("TEST SET PERFORMANCE")
        print(f"{'-' * 70}")
        comparison_df = self._display_dataframe(self.get_comparison_table())
        print(comparison_df.to_string(index=False))

        interpretable = self.get_interpretable_models()
        if interpretable:
            print(f"\nInterpretable models: {', '.join(interpretable)}")

    def plot_comparison(
        self,
        metrics: List[str] = None,
        figsize: tuple = (12, 6),
    ):
        import matplotlib.pyplot as plt

        if metrics is None:
            metrics = ["accuracy", "precision", "recall", "f1"]

        df = self.get_comparison_table(metrics=metrics)
        fig, ax = plt.subplots(figsize=figsize)
        x_positions = np.arange(len(df))
        width = 0.8 / max(len(metrics), 1)

        for index, metric in enumerate(metrics):
            column = metric.upper()
            if column not in df.columns:
                continue
            offset = (index - len(metrics) / 2 + 0.5) * width
            values = df[column].to_numpy()
            bars = ax.bar(
                x_positions + offset,
                values,
                width,
                label=column,
                alpha=0.8,
            )
            for bar, value in zip(bars, values):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    value + 0.01,
                    f"{value:.3f}",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                )

        ax.set_ylabel("Score")
        ax.set_title("Model Performance Comparison")
        ax.set_xticks(x_positions)
        ax.set_xticklabels(df["Model"], rotation=45, ha="right")
        ax.legend()
        ax.set_ylim(0, 1.1)
        ax.grid(True, alpha=0.3, axis="y")

        plt.tight_layout()
        return fig

    def plot_training_times(self, figsize: tuple = (10, 5)):
        import matplotlib.pyplot as plt

        comparison_df = self.get_comparison_table()
        plot_df = comparison_df.sort_values("Training Time (s)", ascending=False)
        fig, ax = plt.subplots(figsize=figsize)
        bars = ax.barh(
            plot_df["Model"],
            plot_df["Training Time (s)"],
            color="steelblue",
            alpha=0.7,
        )
        ax.set_xlabel("Training Time (seconds)")
        ax.set_title("Model Training Times")

        for bar, value in zip(bars, plot_df["Training Time (s)"]):
            ax.text(value + 0.01, bar.get_y() + bar.get_height() / 2, f"{value:.2f}s", va="center")

        plt.tight_layout()
        return fig

    def plot_roc_curves(self, figsize: tuple = (10, 8)):
        import matplotlib.pyplot as plt
        from sklearn.metrics import roc_curve

        if self.y_test is None:
            raise RuntimeError("y_test not available. Run run_benchmark() first.")

        ranked_names = [
            name
            for name, _ in self._ranked_result_items(sort_by="roc_auc", dataset="test")
            if name in self.probabilities
        ]
        fig, ax = plt.subplots(figsize=figsize)
        colors = plt.cm.Set1(np.linspace(0, 1, max(len(ranked_names), 1)))

        for color, name in zip(colors, ranked_names):
            proba = self.probabilities.get(name)
            if proba is None:
                continue
            y_proba = proba[:, 1] if len(proba.shape) > 1 else proba
            auc = self.results[name]["test_metrics"].get("roc_auc", 0.0)
            fpr, tpr, _ = roc_curve(self.y_test, y_proba)
            ax.plot(fpr, tpr, color=color, label=f"{name} (AUC={auc:.3f})")

        ax.plot([0, 1], [0, 1], "k--", label="Random")
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title("ROC Curves Comparison")
        ax.legend(loc="lower right")
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        return fig

    def save_results(self, path: str) -> None:
        path = self._validate_output_path(path)
        path.mkdir(parents=True, exist_ok=True)

        comparison_df = self._export_dataframe(self.get_comparison_table())
        comparison_df.to_csv(path / "model_comparison.csv", index=False)

        raw_fi_df = self._export_dataframe(self.get_feature_importance_raw())
        if not raw_fi_df.empty:
            raw_fi_df.to_csv(path / "feature_importance.csv")

        normalized_fi_df = self._export_dataframe(self.get_feature_importance_comparison())
        if not normalized_fi_df.empty:
            normalized_fi_df.to_csv(path / "feature_importance_comparison.csv")

        ci_df = self._export_dataframe(self.get_confidence_intervals())
        if not ci_df.empty:
            ci_df.sort_values(["model", "metric"]).to_csv(
                path / "metric_confidence_intervals.csv",
                index=False,
            )

        sig_df = self._export_dataframe(self.get_pairwise_significance())
        if not sig_df.empty:
            sort_columns = [
                column
                for column in ("metric", "bootstrap_p_corrected", "model_a", "model_b")
                if column in sig_df.columns
            ]
            sig_df.sort_values(sort_columns).to_csv(
                path / "pairwise_significance.csv",
                index=False,
            )

        with open(path / "results.json", "w", encoding="utf-8") as handle:
            json.dump(
                format_payload_for_export(self._results_payload()),
                handle,
                indent=2,
                default=str,
                sort_keys=True,
            )

        print(f"Results saved to {path}")

    def generate_report(self) -> str:
        comparison_df = self._display_dataframe(self.get_comparison_table())
        best_name, _, best_metrics = self.get_best_model("f1")
        lines = [
            "# Bot Detection Model Benchmark Report",
            "",
            f"- Experiment: `{self.experiment_name}`",
            f"- Generated: `{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}`",
            "",
            "## Dataset Overview",
            f"- Models evaluated: {len(self.models)}",
            f"- Training samples: {self._safe_length(self.base_train_inputs)}",
            f"- Validation samples: {self._safe_length(self.base_val_inputs)}",
            f"- Test samples: {self._safe_length(self.base_test_inputs)}",
            f"- Feature count: {len(self.base_feature_names)}",
            "",
            "## Summary",
            f"- Best model by test F1: `{best_name}` ({best_metrics['f1']:.4f})",
            "",
            "```text",
            comparison_df.to_string(index=False),
            "```",
        ]

        lines.extend(self._metric_description_lines())

        lines.extend(["", "## Detailed Results"])
        for name, result in self._ranked_result_items(sort_by="f1", dataset="test"):
            lines.extend(
                [
                    "",
                    f"### {name}",
                    f"- Interpretable: {result['is_interpretable']}",
                    f"- Training time: {result['training_time']:.2f}s",
                    "- Test metrics:",
                ]
            )
            for metric, value in result["test_metrics"].items():
                lines.append(f"  - {metric}: {self._format_metric_value(value)}")

        normalized_fi_df = self._display_dataframe(
            self.get_feature_importance_comparison().head(10)
        )
        if not normalized_fi_df.empty:
            lines.extend(
                [
                    "",
                    "## Feature Importance",
                    "- Cross-model ranking uses normalized per-model importances.",
                    "- Raw per-model values are saved in `feature_importance.csv`.",
                    "",
                    "```text",
                    normalized_fi_df.to_string(),
                    "```",
                ]
            )

        ci_df = self._display_dataframe(self.get_confidence_intervals())
        if not ci_df.empty:
            lines.extend(
                [
                    "",
                    "## CONFIDENCE INTERVALS",
                    "```text",
                    ci_df.sort_values(["model", "metric"]).to_string(index=False),
                    "```",
                ]
            )

        sig_df = self._display_dataframe(self.get_pairwise_significance())
        if not sig_df.empty:
            lines.extend(
                [
                    "",
                    "## PAIRWISE MODEL SIGNIFICANCE",
                    "```text",
                    sig_df.sort_values(
                        [
                            column
                            for column in (
                                "metric",
                                "bootstrap_p_corrected",
                                "model_a",
                                "model_b",
                            )
                            if column in sig_df.columns
                        ]
                    ).to_string(index=False),
                    "```",
                ]
            )

        robustness_df = self._display_dataframe(self.robustness_summary)
        if robustness_df is not None and not robustness_df.empty:
            lines.extend(
                [
                    "",
                    "## Adversarial Robustness Audit",
                    "```text",
                    robustness_df.to_string(index=False),
                    "```",
                ]
            )

        return "\n".join(lines)

    @staticmethod
    def _validate_output_path(path: str) -> Path:
        resolved = Path(path).expanduser().resolve()
        workspace = Path.cwd().resolve()
        if resolved != workspace and workspace not in resolved.parents:
            raise ValueError(f"Path must stay within workspace: {workspace}")
        return resolved

    @staticmethod
    def _scale_features(X_train, X_val, X_test):
        from sklearn.preprocessing import StandardScaler

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        X_test_scaled = scaler.transform(X_test)
        return X_train_scaled, X_val_scaled, X_test_scaled, scaler

    @staticmethod
    def _copy_input(data: Union[np.ndarray, pd.DataFrame]) -> Union[np.ndarray, pd.DataFrame]:
        return data.copy() if hasattr(data, "copy") else np.array(data, copy=True)

    @staticmethod
    def _align_eval_input(
        X_eval: Union[np.ndarray, pd.DataFrame],
        feature_names: List[str],
    ) -> Union[np.ndarray, pd.DataFrame]:
        if not isinstance(X_eval, pd.DataFrame) or not feature_names:
            return X_eval

        missing = [name for name in feature_names if name not in X_eval.columns]
        if missing:
            raise ValueError(
                "Evaluation input is missing required model features: "
                f"{missing}"
            )
        return X_eval.loc[:, feature_names]

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
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Any]:
        if enable_scaling and model_name in scaled_models:
            if verbose:
                print("  (Applying feature scaling)")
            return cls._scale_features(X_train, X_val, X_test)
        return X_train, X_val, X_test, None

    @staticmethod
    def _safe_length(data: Optional[Union[np.ndarray, pd.DataFrame]]) -> int:
        return len(data) if data is not None else 0

    @staticmethod
    def _format_metric_value(value: Any) -> str:
        if isinstance(value, (float, np.floating)):
            return f"{float(value):.4f}"
        return str(value)

    @staticmethod
    def _display_dataframe(frame: Optional[pd.DataFrame]) -> Optional[pd.DataFrame]:
        return ModelBenchmark._export_dataframe(frame)

    @staticmethod
    def _export_dataframe(frame: Optional[pd.DataFrame]) -> Optional[pd.DataFrame]:
        return format_frame_for_export(frame)

    @staticmethod
    def _metric_key(dataset: str) -> str:
        return f"{dataset}_metrics"

    @staticmethod
    def _metric_sort_value(value: Any) -> float:
        if value is None:
            return float("-inf")
        try:
            value = float(value)
        except (TypeError, ValueError):
            return float("-inf")
        if np.isnan(value):
            return float("-inf")
        return value

    def _ranked_result_items(
        self,
        sort_by: str = "f1",
        dataset: str = "test",
    ) -> List[Tuple[str, Dict[str, Any]]]:
        metric_key = self._metric_key(dataset)
        ranked = []
        for name, result in self.results.items():
            metric_value = self._metric_sort_value(result.get(metric_key, {}).get(sort_by))
            ranked.append((name, result, metric_value))
        ranked.sort(key=lambda item: (-item[2], item[0]))
        return [(name, result) for name, result, _ in ranked]

    def _feature_importance_data(self) -> Dict[str, Dict[str, float]]:
        return {
            name: result["feature_importance"]
            for name, result in self.results.items()
            if result.get("feature_importance")
        }

    def _metric_description_lines(self) -> List[str]:
        description_keys = ("f1", "roc_auc", "pr_auc", "mcc", "balanced_accuracy")
        lines = ["", "## Metric Notes"]
        for key in description_keys:
            description = self.metrics_calculator.METRIC_DESCRIPTIONS.get(key)
            if description:
                lines.append(f"- `{key}`: {description}")
        return lines

    def _results_payload(self) -> Dict[str, Any]:
        ranked_results = self._ranked_result_items(sort_by="f1", dataset="test")
        payload = {
            "experiment_name": self.experiment_name,
            "sort_metric": "f1",
            "ranked_models": [name for name, _ in ranked_results],
            "models": {},
        }
        for rank, (name, result) in enumerate(ranked_results, start=1):
            payload["models"][name] = {
                "rank": rank,
                "training_time": result["training_time"],
                "val_metrics": result["val_metrics"],
                "test_metrics": result["test_metrics"],
                "is_interpretable": result["is_interpretable"],
                "confidence_intervals": self.confidence_intervals.get(name, {}),
                "feature_importance_available": result["feature_importance"] is not None,
            }
        return payload

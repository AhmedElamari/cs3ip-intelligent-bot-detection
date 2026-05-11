"""Benchmarking system for comparing multiple bot detection models."""

from dataclasses import dataclass, field
from datetime import datetime
import json
from pathlib import Path
import time
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from .dissertation_scoreboard import SCOREBOARD_METRIC_KEYS as SCOREBOARD_METRICS
from .dissertation_scoreboard import build_scoreboard
from .metrics import MetricsCalculator
from .output_formatting import format_frame_for_export, format_payload_for_export
from benchmarking.hpo.input_prep import build_model_inputs


DEFAULT_METRICS = {
    "comparison": ["accuracy", "precision", "recall", "f1", "roc_auc", "mcc"],
    "statistics": ["f1", "f1_macro", "pr_auc", "mcc", "balanced_accuracy", "roc_auc"],
}

MODEL_DISPLAY_NAMES = {
    "xgboost": "XGBoost",
    "random_forest": "Random Forest",
    "logistic_regression": "Logistic Regression",
    "decision_tree": "Decision Tree",
    "naive_bayes": "Naive Bayes",
    "svm": "SVM",
    "tabnet": "TabNet",
}


def _display_model_name(name: str) -> str:
    return MODEL_DISPLAY_NAMES.get(str(name), str(name).replace("_", " ").title())

@dataclass
class ModelBenchmarkConfig:
    metrics: Dict[str, List[str]] = field(default_factory=lambda: DEFAULT_METRICS)

@dataclass
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
        self.validation_probabilities: Dict[str, np.ndarray] = {}
        self.y_val: Optional[np.ndarray] = None
        self.y_test: Optional[np.ndarray] = None
        self.test_metadata: Optional[pd.DataFrame] = None
        self.confidence_intervals: Dict[str, Dict[str, Any]] = {}
        self.pairwise_significance: List[Dict[str, Any]] = []
        self.base_train_inputs: Optional[Union[np.ndarray, pd.DataFrame]] = None
        self.base_val_inputs: Optional[Union[np.ndarray, pd.DataFrame]] = None
        self.base_test_inputs: Optional[Union[np.ndarray, pd.DataFrame]] = None
        self.base_y_train: Optional[np.ndarray] = None
        self.base_feature_names: List[str] = []
        self.robustness_summary: Optional[pd.DataFrame] = None
        self.robustness_degradation: Optional[pd.DataFrame] = None
        self.feature_attack_results: Optional[pd.DataFrame] = None
        self.robustness_fidelity: Optional[Dict[str, Any]] = None
        self.hpo_audit_by_model: Dict[str, Dict[str, Any]] = {}

    def add_model(self, name: str, model: Any) -> "ModelBenchmark":
        self.models[name] = model
        return self

    def set_test_metadata(self, metadata: Optional[pd.DataFrame]) -> None:
        self.test_metadata = metadata.copy() if metadata is not None else None

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

        for name, model in self.models.items():
            if verbose:
                print(f"\n{'-' * 40}")
                print(f"Training: {name}")
                print(f"{'-' * 40}")

            if enable_scaling and name in ("logistic_regression", "svm") and verbose:
                print("  (Applying feature scaling)")

            prep = build_model_inputs(
                name,
                X_train,
                X_val,
                X_test,
                enable_scaling=enable_scaling,
            )
            X_train_model = prep.X_train
            X_val_model = prep.X_val
            X_test_model = prep.X_test
            scaler = prep.scaler

            fit_feature_names = feature_names
            if name == "tabnet" and prep.tabnet_meta is not None:
                fit_feature_names = prep.tabnet_meta.feature_names or feature_names

            if hasattr(model, "prepare_eval_set"):
                model.prepare_eval_set(X_val_model, y_val)

            start_time = time.time()
            model.fit(X_train_model, y_train, feature_names=fit_feature_names)
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
                self.validation_probabilities[name] = y_val_proba
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

            runtime_metadata = None
            if hasattr(model, "get_runtime_metadata") and callable(
                getattr(model, "get_runtime_metadata")
            ):
                runtime_metadata = model.get_runtime_metadata()

            self.results[name] = {
                "model": model,
                "training_time": training_time,
                "val_metrics": val_metrics,
                "test_metrics": test_metrics,
                "feature_importance": feature_importance,
                "runtime_metadata": runtime_metadata,
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
            metrics = DEFAULT_METRICS["statistics"]

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
            metrics = DEFAULT_METRICS["comparison"]

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

    def get_scoreboard_table(self, dataset: str = "test") -> pd.DataFrame:
        from benchmarking.dissertation_scoreboard import build_scoreboard

        return build_scoreboard(self, dataset=dataset)

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
            if isinstance(X_eval, pd.DataFrame) and not isinstance(result.get("X_train"), pd.DataFrame):
                return X_eval.to_numpy()
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

    def plot_pr_curves_top(self, top_n: int = 3, figsize: tuple = (8, 6)):
        """Test-set PR curves for top ``top_n`` models by dissertation scoreboard order (F1-Macro, ROC-AUC)."""
        import matplotlib.pyplot as plt

        if self.y_test is None:
            raise RuntimeError("y_test not available. Run run_benchmark() first.")

        scoreboard_df = build_scoreboard(self, dataset="test")
        ranked_with_proba: List[str] = []
        for _, row in scoreboard_df.iterrows():
            name = str(row["Model"])
            if name not in self.probabilities:
                continue
            ranked_with_proba.append(name)
            if len(ranked_with_proba) >= top_n:
                break

        if len(ranked_with_proba) < top_n:
            print(
                f"Warning: Skipping PR curve comparison: need {top_n} models with "
                f"test-set probabilities, found {len(ranked_with_proba)}."
            )
            return None

        fig, ax = plt.subplots(figsize=figsize)
        colors = plt.cm.Set1(np.linspace(0, 1, max(top_n, 1)))

        for color, name in zip(colors, ranked_with_proba):
            pr_data = self.metrics_calculator.get_precision_recall_curve(
                self.y_test, self.probabilities[name]
            )
            pr_auc = float(self.results[name]["test_metrics"].get("pr_auc", 0.0))
            ax.plot(
                pr_data["recall"],
                pr_data["precision"],
                color=color,
                label=f"{name} (PR-AUC = {pr_auc:.3f})",
            )

        prevalence = float(np.mean(np.asarray(self.y_test, dtype=float)))
        ax.axhline(
            y=prevalence,
            linestyle="--",
            color="grey",
            label="Class prevalence",
        )
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.set_title("Precision-Recall Curves Comparison (test set)")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.legend(loc="lower left")
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        return fig

    def plot_best_confusion_matrix(
        self,
        normalize: Optional[str] = "true",
        figsize: tuple = (5.5, 4.5),
    ):
        """Confusion matrix for best model by scoreboard (F1-Macro); test predictions only."""
        import matplotlib.pyplot as plt
        import seaborn as sns

        if self.y_test is None:
            raise RuntimeError("y_test not available. Run run_benchmark() first.")

        scoreboard_df = build_scoreboard(self, dataset="test")
        if scoreboard_df.empty:
            raise ValueError("No benchmark results for confusion matrix.")

        best_name = str(scoreboard_df.iloc[0]["Model"])
        if best_name not in self.predictions:
            raise KeyError(f"No test predictions stored for best model {best_name!r}.")

        y_pred = self.predictions[best_name]
        cm = self.metrics_calculator.get_confusion_matrix(
            self.y_test, y_pred, normalize=normalize
        )
        display_name = _display_model_name(best_name)
        fmt = ".2f" if normalize else "d"
        if normalize == "true":
            human_recall = float(cm[0, 0])
            bot_recall = float(cm[1, 1])
            if bot_recall >= 0.95 and human_recall < 0.75:
                title = f"{display_name} catches nearly all bots but over-flags some humans"
            elif min(human_recall, bot_recall) >= 0.8:
                title = f"{display_name} separates humans and bots cleanly on the test split"
            else:
                title = f"{display_name} shows uneven error trade-offs across the two classes"
        else:
            title = f"Test-set prediction counts for {display_name}"

        fig, ax = plt.subplots(figsize=figsize)
        sns.heatmap(
            cm,
            annot=True,
            fmt=fmt,
            cmap="Blues",
            ax=ax,
            cbar=True,
            xticklabels=["Human", "Bot"],
            yticklabels=["Human", "Bot"],
        )
        ax.set_xlabel("Predicted label")
        ax.set_ylabel("True label")
        ax.set_title(title)
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
        ]

        try:
            best_name, _, best_metrics = self.get_best_model("f1")
            lines.append(f"- Best model by test F1: `{best_name}` ({best_metrics['f1']:.4f})")
        except (ValueError, KeyError):
            lines.append("- No model results available.")

        if comparison_df is not None and not comparison_df.empty:
            lines.extend(
                [
                    "",
                    "```text",
                    comparison_df.to_string(index=False),
                    "```",
                ]
            )

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
                    "## Confidence Intervals",
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
                    "## Pairwise Model Significance",
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
            entry = {
                "rank": rank,
                "training_time": result["training_time"],
                "val_metrics": result["val_metrics"],
                "test_metrics": result["test_metrics"],
                "is_interpretable": result["is_interpretable"],
                "confidence_intervals": self.confidence_intervals.get(name, {}),
                "feature_importance_available": result["feature_importance"] is not None,
            }
            if name in self.hpo_audit_by_model:
                entry["hpo"] = self.hpo_audit_by_model[name]
            rmeta = result.get("runtime_metadata")
            if rmeta:
                entry["runtime_metadata"] = rmeta
            payload["models"][name] = entry
        return payload

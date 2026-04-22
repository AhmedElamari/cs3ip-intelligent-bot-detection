"""Optional adversarial robustness analysis for the benchmark pipeline."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, f1_score

from adversarial import RealisticPerturbationEngine
from .output_formatting import format_frame_for_export, format_payload_for_export


def run_robustness_analysis(
    benchmark: Any,
    feature_names: Sequence[str],
    config: Any,
    output_dir: Path,
) -> dict:
    """Run the optional robustness audit and save result artifacts."""
    if not config.get('robustness.enabled', False):
        return {}

    analyzer = RobustnessAnalyzer(benchmark, feature_names, config)
    results = analyzer.run()
    analyzer.save_outputs(output_dir, results)
    benchmark.robustness_summary = results['summary']
    benchmark.robustness_degradation = results['degradation']
    benchmark.feature_attack_results = results['feature_attacks']
    benchmark.robustness_profile_diagnostics = results['profile_diagnostics']
    return results


class RobustnessAnalyzer:
    """Evaluate prediction robustness under realistic perturbations."""

    def __init__(self, benchmark: Any, feature_names: Sequence[str], config: Any):
        self.benchmark = benchmark
        self.feature_names = list(feature_names)
        self.config = config
        self.base_test = self._to_frame(benchmark.base_test_inputs, self.feature_names)
        self.base_train = self._to_frame(benchmark.base_train_inputs, self.feature_names)
        self.y_test = np.asarray(benchmark.y_test)
        self.y_train = np.asarray(benchmark.base_y_train)
        self.engine = RealisticPerturbationEngine(
            feature_names=self.feature_names,
            X_train=self.base_train,
            y_train=self.y_train,
            expensive_nudge_fraction=config.get('robustness.expensive_nudge_fraction', 0.05),
        )
        self.attack_population = config.get('robustness.attack_population', 'true_bots')
        self.profiles = list(config.get('robustness.profiles', ['cheap_only', 'realistic_mixed']))
        self.evaluate_single_feature_attacks = config.get('robustness.evaluate_single_feature_attacks', True)
        self.evaluate_bundle_attacks = config.get('robustness.evaluate_bundle_attacks', True)

    @staticmethod
    def _to_frame(X: Any, feature_names: Sequence[str]) -> pd.DataFrame:
        if isinstance(X, pd.DataFrame):
            return X.copy()
        return pd.DataFrame(X, columns=feature_names)

    def run(self) -> Dict[str, pd.DataFrame]:
        feature_rows: List[Dict[str, Any]] = []
        summary_rows: List[Dict[str, Any]] = []
        degradation_rows: List[Dict[str, Any]] = []

        attacked_mask = self._attack_population_mask()
        attacked_base = (
            self.base_test.loc[attacked_mask]
            .sort_values(self.feature_names, kind='mergesort')
            .reset_index(drop=True)
        )
        profile_diagnostics = self._profile_diagnostics_frame(attacked_base)

        for model_name, result in self.benchmark.results.items():
            model = result['model']
            baseline_inputs = self.benchmark.prepare_eval_inputs(model_name, attacked_base)
            baseline_preds = np.asarray(model.predict(baseline_inputs))
            baseline_proba = self._bot_probability(model, baseline_inputs)
            baseline_detected_mask = baseline_preds == 1
            baseline_detected_count = int(baseline_detected_mask.sum())
            baseline_attacked_metrics = self._attacked_population_metrics(baseline_preds, baseline_proba)

            degradation_rows.extend(self._degradation_rows(model_name, model))

            if self.evaluate_single_feature_attacks:
                for feature in self.engine.available_single_feature_attacks():
                    attack_result = self.engine.apply_single_feature_attack(attacked_base, feature)
                    row = self._single_attack_row(
                        model_name,
                        feature,
                        attack_result,
                        attacked_base,
                        baseline_detected_mask,
                        baseline_detected_count,
                        baseline_proba,
                        model,
                    )
                    feature_rows.append(row)

            if self.evaluate_bundle_attacks:
                for profile in self.profiles:
                    profile_result = self.engine.apply_profile(attacked_base, profile)
                    summary_rows.append(
                        self._profile_row(
                            model_name,
                            profile,
                            profile_result,
                            attacked_base,
                            baseline_detected_mask,
                            baseline_detected_count,
                            baseline_proba,
                            baseline_attacked_metrics,
                            model,
                        )
                    )

        return {
            'summary': pd.DataFrame(summary_rows),
            'feature_attacks': pd.DataFrame(feature_rows),
            'degradation': pd.DataFrame(degradation_rows),
            'profile_diagnostics': profile_diagnostics,
        }

    def _attack_population_mask(self) -> np.ndarray:
        if self.attack_population != 'true_bots':
            raise ValueError(
                "robustness.attack_population currently only supports 'true_bots' to avoid audit ambiguity."
            )
        return self.y_test == 1

    def _scenario_metrics(self, model_name: str, model: Any, X_frame: pd.DataFrame) -> Dict[str, float]:
        inputs = self.benchmark.prepare_eval_inputs(model_name, X_frame)
        preds = np.asarray(model.predict(inputs))
        proba = self._bot_probability(model, inputs)
        y = self.y_test
        macro_f1 = float(f1_score(y, preds, average='macro', zero_division=0))
        pr_auc = float(average_precision_score(y, proba))
        return {'macro_f1': macro_f1, 'pr_auc': pr_auc}

    def _profile_diagnostics_frame(self, attacked_base: pd.DataFrame) -> pd.DataFrame:
        rows: List[Dict[str, Any]] = []
        for profile in self.profiles:
            result = self.engine.apply_profile(attacked_base, profile)
            rows.extend(result.diagnostics)
        return pd.DataFrame(rows)

    @staticmethod
    def _attacked_population_metrics(preds: np.ndarray, proba: np.ndarray) -> Dict[str, float]:
        attacked_true_bots = len(preds)
        if attacked_true_bots == 0:
            return {
                'attacked_bot_recall': np.nan,
                'attacked_bot_mean_probability': np.nan,
            }
        return {
            'attacked_bot_recall': float(np.mean(preds == 1)),
            'attacked_bot_mean_probability': float(np.mean(proba)),
        }

    def _build_profile_perturbed_full_test(self, profile: str) -> pd.DataFrame:
        """Full test matrix with profile applied to true-bot rows only (index-aligned)."""
        X_work = self.base_test.copy()
        attacked_mask = self._attack_population_mask()
        if not attacked_mask.any():
            return X_work
        sub = (
            self.base_test.loc[attacked_mask]
            .sort_values(self.feature_names, kind='mergesort')
            .copy()
        )
        profile_result = self.engine.apply_profile(sub, profile)
        if not profile_result.applied:
            return X_work
        mutated = profile_result.data
        for col in mutated.columns:
            if col in X_work.columns:
                if pd.api.types.is_numeric_dtype(mutated[col]) and not pd.api.types.is_float_dtype(X_work[col]):
                    X_work[col] = X_work[col].astype(np.float64)
                X_work.loc[sub.index, col] = mutated.loc[sub.index, col].values
        return X_work

    def _degradation_rows(self, model_name: str, model: Any) -> List[Dict[str, Any]]:
        rows: List[Dict[str, Any]] = []
        base_m = self._scenario_metrics(model_name, model, self.base_test)
        rows.append({
            'model': model_name,
            'scenario': 'baseline',
            'macro_f1': base_m['macro_f1'],
            'pr_auc': base_m['pr_auc'],
        })
        for profile in self.profiles:
            X_prof = self._build_profile_perturbed_full_test(profile)
            m = self._scenario_metrics(model_name, model, X_prof)
            rows.append({
                'model': model_name,
                'scenario': profile,
                'macro_f1': m['macro_f1'],
                'pr_auc': m['pr_auc'],
            })
        return rows

    def _single_attack_row(
        self,
        model_name: str,
        feature: str,
        attack_result: Any,
        attacked_base: pd.DataFrame,
        baseline_detected_mask: np.ndarray,
        baseline_detected_count: int,
        baseline_proba: np.ndarray,
        model: Any,
    ) -> Dict[str, Any]:
        row = {
            'model': model_name,
            'feature': feature,
            'attack_name': attack_result.attack_name,
            'cost_tier': attack_result.cost_tier,
            'attacked_true_bots': len(attacked_base),
            'baseline_detected_bots': baseline_detected_count,
            'skip_reason': attack_result.skip_reason,
            'flips_to_human': np.nan,
            'flip_rate': np.nan,
            'confidence_drop_mean': np.nan,
            'confidence_drop_median': np.nan,
            'confidence_drop_std': np.nan,
            'confidence_drop_non_flip_mean': np.nan,
        }
        if not attack_result.applied:
            return row

        mutated_inputs = self.benchmark.prepare_eval_inputs(model_name, attack_result.data)
        mutated_preds = np.asarray(model.predict(mutated_inputs))
        mutated_proba = self._bot_probability(model, mutated_inputs)
        row.update(
            self._prediction_metrics(
                baseline_detected_mask,
                baseline_detected_count,
                baseline_proba,
                mutated_preds,
                mutated_proba,
            )
        )
        return row

    def _profile_row(
        self,
        model_name: str,
        profile: str,
        profile_result: Any,
        attacked_base: pd.DataFrame,
        baseline_detected_mask: np.ndarray,
        baseline_detected_count: int,
        baseline_proba: np.ndarray,
        baseline_attacked_metrics: Dict[str, float],
        model: Any,
    ) -> Dict[str, Any]:
        row = {
            'model': model_name,
            'profile': profile,
            'attacked_true_bots': len(attacked_base),
            'baseline_detected_bots': baseline_detected_count,
            'skip_reason': profile_result.skip_reason,
            'flips_to_human': np.nan,
            'flip_rate': np.nan,
            'confidence_drop_mean': np.nan,
            'confidence_drop_median': np.nan,
            'confidence_drop_std': np.nan,
            'confidence_drop_non_flip_mean': np.nan,
            'attacked_bot_recall_baseline': baseline_attacked_metrics['attacked_bot_recall'],
            'attacked_bot_recall': np.nan,
            'attacked_bot_recall_delta': np.nan,
            'attacked_bot_mean_probability_baseline': baseline_attacked_metrics['attacked_bot_mean_probability'],
            'attacked_bot_mean_probability': np.nan,
            'attacked_bot_mean_probability_delta': np.nan,
        }
        if not profile_result.applied:
            return row

        mutated_inputs = self.benchmark.prepare_eval_inputs(model_name, profile_result.data)
        mutated_preds = np.asarray(model.predict(mutated_inputs))
        mutated_proba = self._bot_probability(model, mutated_inputs)
        attacked_metrics = self._attacked_population_metrics(mutated_preds, mutated_proba)
        row.update(
            self._prediction_metrics(
                baseline_detected_mask,
                baseline_detected_count,
                baseline_proba,
                mutated_preds,
                mutated_proba,
            )
        )
        row.update({
            'attacked_bot_recall': attacked_metrics['attacked_bot_recall'],
            'attacked_bot_recall_delta': (
                attacked_metrics['attacked_bot_recall'] - baseline_attacked_metrics['attacked_bot_recall']
            ),
            'attacked_bot_mean_probability': attacked_metrics['attacked_bot_mean_probability'],
            'attacked_bot_mean_probability_delta': (
                attacked_metrics['attacked_bot_mean_probability'] -
                baseline_attacked_metrics['attacked_bot_mean_probability']
            ),
        })
        return row

    def _prediction_metrics(
        self,
        baseline_detected_mask: np.ndarray,
        baseline_detected_count: int,
        baseline_proba: np.ndarray,
        mutated_preds: np.ndarray,
        mutated_proba: np.ndarray,
    ) -> Dict[str, Any]:
        if baseline_detected_count == 0:
            return {
                'flips_to_human': np.nan,
                'flip_rate': np.nan,
                'confidence_drop_mean': np.nan,
                'confidence_drop_median': np.nan,
                'confidence_drop_std': np.nan,
                'confidence_drop_non_flip_mean': np.nan,
            }

        relevant_preds = mutated_preds[baseline_detected_mask]
        relevant_proba = mutated_proba[baseline_detected_mask]
        relevant_base_proba = baseline_proba[baseline_detected_mask]
        confidence_drop = relevant_base_proba - relevant_proba
        non_flip_mask = relevant_preds == 1
        flips_to_human = int(np.sum(relevant_preds == 0))
        non_flip_mean = (
            self._stable_mean(confidence_drop[non_flip_mask]) if np.any(non_flip_mask) else np.nan
        )
        return {
            'flips_to_human': flips_to_human,
            'flip_rate': flips_to_human / baseline_detected_count,
            'confidence_drop_mean': self._stable_mean(confidence_drop),
            'confidence_drop_median': self._stable_median(confidence_drop),
            'confidence_drop_std': self._stable_std(confidence_drop),
            'confidence_drop_non_flip_mean': non_flip_mean,
        }

    def _bot_probability(self, model: Any, X: Any) -> np.ndarray:
        proba = model.predict_proba(X)
        proba = np.asarray(proba)
        return proba[:, 1] if proba.ndim > 1 else proba

    def save_outputs(self, output_dir: Path, results: Dict[str, pd.DataFrame]) -> None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        sorted_results = self._sorted_results(results)
        file_map = {
            'summary': 'robustness_summary.csv',
            'feature_attacks': 'feature_attack_results.csv',
            'degradation': 'robustness_degradation.csv',
            'profile_diagnostics': 'profile_diagnostics.csv',
        }
        for key, filename in file_map.items():
            format_frame_for_export(sorted_results[key]).to_csv(output_dir / filename, index=False)

        report = {
            'artifacts': {
                key: self._artifact_manifest(filename, sorted_results[key])
                for key, filename in file_map.items()
            },
            'overview': self._report_overview(sorted_results),
            'profile_sanity': self._profile_sanity(sorted_results),
        }
        with open(output_dir / 'robustness_report.json', 'w', encoding='utf-8') as handle:
            json.dump(
                format_payload_for_export(report),
                handle,
                indent=2,
                default=self._json_default,
                sort_keys=True,
            )

    @staticmethod
    def _artifact_manifest(filename: str, frame: pd.DataFrame) -> Dict[str, Any]:
        return {
            'file': filename,
            'rows': int(len(frame)),
            'columns': frame.columns.tolist(),
        }

    @classmethod
    def _report_overview(cls, results: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        summary = results['summary']
        feature_attacks = results['feature_attacks']
        degradation = results['degradation']

        overview: Dict[str, Any] = {
            'models': cls._sorted_unique(summary, 'model') or cls._sorted_unique(degradation, 'model'),
            'profiles': cls._sorted_unique(summary, 'profile'),
            'scenarios_evaluated': cls._sorted_unique(degradation, 'scenario'),
            'features_evaluated': cls._sorted_unique(feature_attacks, 'feature'),
            'baseline_detected_bots_total': cls._sum_column(summary, 'baseline_detected_bots'),
            'attacked_true_bots_total': cls._sum_column(summary, 'attacked_true_bots'),
            'mean_flip_rate': cls._mean_column(summary, 'flip_rate'),
            'mean_confidence_drop': cls._mean_column(feature_attacks, 'confidence_drop_mean'),
            'mean_attacked_bot_recall_delta': cls._mean_column(summary, 'attacked_bot_recall_delta'),
            'mean_attacked_bot_mean_probability_delta': cls._mean_column(summary, 'attacked_bot_mean_probability_delta'),
            'mean_macro_f1': cls._mean_column(degradation, 'macro_f1'),
            'mean_pr_auc': cls._mean_column(degradation, 'pr_auc'),
        }
        return overview

    @classmethod
    def _profile_sanity(cls, results: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        summary = results.get('summary', pd.DataFrame())
        degradation = results.get('degradation', pd.DataFrame())
        diagnostics = results.get('profile_diagnostics', pd.DataFrame())
        output: Dict[str, Any] = {}
        for profile in cls._sorted_unique(summary, 'profile') or cls._sorted_unique(diagnostics, 'profile'):
            profile_summary = summary[summary.get('profile') == profile] if 'profile' in summary else pd.DataFrame()
            profile_diagnostics = (
                diagnostics[diagnostics.get('profile') == profile] if 'profile' in diagnostics else pd.DataFrame()
            )
            profile_degradation = (
                degradation[degradation.get('scenario') == profile] if 'scenario' in degradation else pd.DataFrame()
            )
            output[str(profile)] = {
                'recipe_count': int(len(profile_diagnostics)),
                'applied_recipe_count': int(
                    profile_diagnostics['recipe_applied'].fillna(False).astype(bool).sum()
                ) if 'recipe_applied' in profile_diagnostics else 0,
                'changed_expensive_recipe_count': int(
                    (
                        (profile_diagnostics.get('cost_tier') == 'expensive') &
                        profile_diagnostics.get('recipe_applied', pd.Series(dtype=bool)).fillna(False).astype(bool)
                    ).sum()
                ) if not profile_diagnostics.empty else 0,
                'changed_columns': sorted({
                    column
                    for columns in profile_diagnostics.get('changed_columns', pd.Series(dtype=str)).fillna('')
                    for column in str(columns).split(';')
                    if column
                }),
                'mean_changed_rows': cls._mean_column(profile_diagnostics, 'changed_rows'),
                'mean_relative_delta': cls._mean_column(profile_diagnostics, 'mean_relative_delta'),
                'mean_flip_rate': cls._mean_column(profile_summary, 'flip_rate'),
                'mean_attacked_bot_recall_delta': cls._mean_column(profile_summary, 'attacked_bot_recall_delta'),
                'mean_macro_f1': cls._mean_column(profile_degradation, 'macro_f1'),
                'mean_pr_auc': cls._mean_column(profile_degradation, 'pr_auc'),
            }
        return output

    @staticmethod
    def _sorted_unique(frame: pd.DataFrame, column: str) -> List[Any]:
        if column not in frame:
            return []
        values = frame[column].dropna().unique().tolist()
        return sorted(values)

    @staticmethod
    def _sum_column(frame: pd.DataFrame, column: str) -> int:
        if column not in frame or frame.empty:
            return 0
        return int(frame[column].fillna(0).sum())

    @staticmethod
    def _mean_column(frame: pd.DataFrame, column: str) -> Any:
        if column not in frame or frame.empty:
            return None
        values = frame[column].dropna().to_numpy(dtype=np.float64, copy=True)
        if values.size == 0:
            return None
        return RobustnessAnalyzer._stable_mean(values)

    @staticmethod
    def _json_default(value: Any) -> Any:
        if pd.isna(value):
            return None
        if hasattr(value, 'item'):
            return value.item()
        raise TypeError(f'Object of type {type(value).__name__} is not JSON serializable')

    @staticmethod
    def _stable_float_array(values: Any) -> np.ndarray:
        array = np.asarray(values, dtype=np.float64).reshape(-1).copy()
        if array.size == 0:
            return array
        return np.sort(array, kind='mergesort')

    @classmethod
    def _stable_mean(cls, values: Any) -> float:
        array = cls._stable_float_array(values)
        if array.size == 0:
            return np.nan
        return math.fsum(array.tolist()) / array.size

    @classmethod
    def _stable_std(cls, values: Any) -> float:
        array = cls._stable_float_array(values)
        if array.size == 0:
            return np.nan
        mean = cls._stable_mean(array)
        squared_diffs = ((value - mean) ** 2 for value in array.tolist())
        variance = math.fsum(squared_diffs) / array.size
        return math.sqrt(variance)

    @classmethod
    def _stable_median(cls, values: Any) -> float:
        array = cls._stable_float_array(values)
        if array.size == 0:
            return np.nan
        middle = array.size // 2
        if array.size % 2:
            return float(array[middle])
        return (float(array[middle - 1]) + float(array[middle])) / 2.0

    @staticmethod
    def _sorted_results(results: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        sort_columns = {
            'summary': ['model', 'profile'],
            'feature_attacks': ['model', 'feature'],
            'degradation': ['model', 'scenario'],
            'profile_diagnostics': ['profile', 'cost_tier', 'feature'],
        }
        sorted_results = {}
        for key, frame in results.items():
            columns = [column for column in sort_columns.get(key, []) if column in frame.columns]
            if columns:
                sorted_results[key] = frame.sort_values(columns).reset_index(drop=True)
            else:
                sorted_results[key] = frame.copy()
        return sorted_results

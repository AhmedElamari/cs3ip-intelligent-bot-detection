"""Optional adversarial robustness analysis for the benchmark pipeline."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from adversarial import RealisticPerturbationEngine
from explainability import FeatureResilienceAnalyzer, SHAPExplainer


SUPPORTED_SHAP_MODELS = {'random_forest', 'xgboost', 'tabnet'}
EXPECTED_SHAP_EXCEPTIONS = (
    ImportError,
    RuntimeError,
    TypeError,
    ValueError,
    AttributeError,
)


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
    return results


class RobustnessAnalyzer:
    """Evaluate prediction and explanation robustness under realistic perturbations."""

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
        self.max_shap_samples = config.get('robustness.max_shap_samples', 50)
        self.shap_top_k = config.get('robustness.shap_top_k', 5)
        self.supported_shap_models = SUPPORTED_SHAP_MODELS

    @staticmethod
    def _to_frame(X: Any, feature_names: Sequence[str]) -> pd.DataFrame:
        if isinstance(X, pd.DataFrame):
            return X.copy()
        return pd.DataFrame(X, columns=feature_names)

    def run(self) -> Dict[str, pd.DataFrame]:
        feature_rows: List[Dict[str, Any]] = []
        summary_rows: List[Dict[str, Any]] = []
        stability_rows: List[Dict[str, Any]] = []
        frs_rows: List[Dict[str, Any]] = []
        pivot_rows: List[Dict[str, Any]] = []
        diagnostic_rows: List[Dict[str, Any]] = []

        attacked_mask = self._attack_population_mask()
        attacked_base = self.base_test.loc[attacked_mask].reset_index(drop=True)

        for model_name, result in self.benchmark.results.items():
            model = result['model']
            baseline_inputs = self.benchmark.prepare_eval_inputs(model_name, attacked_base)
            baseline_preds = np.asarray(model.predict(baseline_inputs))
            baseline_proba = self._bot_probability(model, baseline_inputs)
            baseline_detected_mask = baseline_preds == 1
            baseline_detected_count = int(baseline_detected_mask.sum())

            shap_context = self._build_shap_context(model_name, model, attacked_base)
            if shap_context['diagnostic'] is not None:
                diagnostic_rows.append(shap_context['diagnostic'])

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
                        shap_context,
                        frs_rows,
                        stability_rows,
                        diagnostic_rows,
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
                            model,
                            shap_context,
                            stability_rows,
                            pivot_rows,
                            diagnostic_rows,
                        )
                    )

        return {
            'summary': pd.DataFrame(summary_rows),
            'feature_attacks': pd.DataFrame(feature_rows),
            'shap_rank_stability': pd.DataFrame(stability_rows),
            'feature_resilience': pd.DataFrame(frs_rows),
            'shap_pivots': pd.DataFrame(pivot_rows),
            'shap_diagnostics': pd.DataFrame(diagnostic_rows),
        }

    def _attack_population_mask(self) -> np.ndarray:
        if self.attack_population != 'true_bots':
            raise ValueError(
                "robustness.attack_population currently only supports 'true_bots' to avoid audit ambiguity."
            )
        return self.y_test == 1

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
        shap_context: Dict[str, Any],
        frs_rows: List[Dict[str, Any]],
        stability_rows: List[Dict[str, Any]],
        diagnostic_rows: List[Dict[str, Any]],
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
            'mean_rank_stability': np.nan,
            'shap_status': 'not_requested',
            'shap_error': None,
        }
        if not attack_result.applied:
            return row

        mutated_inputs = self.benchmark.prepare_eval_inputs(model_name, attack_result.data)
        mutated_preds = np.asarray(model.predict(mutated_inputs))
        mutated_proba = self._bot_probability(model, mutated_inputs)
        metrics = self._prediction_metrics(
            baseline_detected_mask,
            baseline_detected_count,
            baseline_proba,
            mutated_preds,
            mutated_proba,
        )
        row.update(metrics)
        shap_metrics = self._shap_metrics(
            model_name=model_name,
            scenario_name=feature,
            scenario_type='feature',
            primary_feature=feature,
            shap_context=shap_context,
            mutated_inputs=mutated_inputs,
            baseline_detected_count=baseline_detected_count,
            flips_to_human=metrics['flips_to_human'],
        )
        row.update(self._shap_columns(shap_metrics))
        stability_rows.extend(shap_metrics['stability_rows'])
        diagnostic = shap_metrics['diagnostic']
        if diagnostic is not None:
            diagnostic_rows.append(diagnostic)
        if shap_metrics['frs_row'] is not None:
            frs_rows.append(shap_metrics['frs_row'])
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
        model: Any,
        shap_context: Dict[str, Any],
        stability_rows: List[Dict[str, Any]],
        pivot_rows: List[Dict[str, Any]],
        diagnostic_rows: List[Dict[str, Any]],
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
            'mean_rank_stability': np.nan,
            'shap_status': 'not_requested',
            'shap_error': None,
        }
        if not profile_result.applied:
            return row

        mutated_inputs = self.benchmark.prepare_eval_inputs(model_name, profile_result.data)
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
        shap_metrics = self._shap_metrics(
            model_name=model_name,
            scenario_name=profile,
            scenario_type='profile',
            primary_feature=None,
            shap_context=shap_context,
            mutated_inputs=mutated_inputs,
            baseline_detected_count=baseline_detected_count,
            flips_to_human=row['flips_to_human'],
        )
        row.update(self._shap_columns(shap_metrics))
        stability_rows.extend(shap_metrics['stability_rows'])
        pivot_rows.extend(shap_metrics['pivot_rows'])
        diagnostic = shap_metrics['diagnostic']
        if diagnostic is not None:
            diagnostic_rows.append(diagnostic)
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
        non_flip_mean = float(np.mean(confidence_drop[non_flip_mask])) if np.any(non_flip_mask) else np.nan
        return {
            'flips_to_human': flips_to_human,
            'flip_rate': flips_to_human / baseline_detected_count,
            'confidence_drop_mean': float(np.mean(confidence_drop)),
            'confidence_drop_median': float(np.median(confidence_drop)),
            'confidence_drop_std': float(np.std(confidence_drop)),
            'confidence_drop_non_flip_mean': non_flip_mean,
        }

    def _bot_probability(self, model: Any, X: Any) -> np.ndarray:
        proba = model.predict_proba(X)
        proba = np.asarray(proba)
        return proba[:, 1] if proba.ndim > 1 else proba

    @staticmethod
    def _format_exception(exc: Exception) -> str:
        return f'{type(exc).__name__}: {exc}'

    def _build_shap_context(
        self,
        model_name: str,
        model: Any,
        attacked_base: pd.DataFrame,
    ) -> Dict[str, Any]:
        if model_name not in self.supported_shap_models:
            return self._shap_context_skip(
                model_name,
                stage='build_context',
                status='unsupported_model',
                message=f'SHAP robustness metrics are not configured for model {model_name!r}.',
            )
        try:
            X_train_model, _, _ = self.benchmark.get_prepared_inputs(model_name)
            sample_count = min(self.max_shap_samples, len(attacked_base))
            attacked_subset = attacked_base.iloc[:sample_count].copy()
            baseline_inputs = self.benchmark.prepare_eval_inputs(model_name, attacked_subset)
            explainer = SHAPExplainer(model, self.feature_names)
            explainer.fit(X_train_model, max_samples=self.max_shap_samples)
            baseline_values = explainer.explain(baseline_inputs)
            return {
                'explainer': explainer,
                'baseline_inputs': baseline_inputs,
                'baseline_values': baseline_values,
                'attacked_subset': attacked_subset,
                'available': True,
                'diagnostic': None,
            }
        except EXPECTED_SHAP_EXCEPTIONS as exc:
            return self._shap_context_skip(
                model_name,
                stage='build_context',
                status='build_failed',
                message=self._format_exception(exc),
            )

    @staticmethod
    def _shap_context_skip(
        model_name: str,
        stage: str,
        status: str,
        message: str,
    ) -> Dict[str, Any]:
        return {
            'available': False,
            'explainer': None,
            'baseline_inputs': None,
            'baseline_values': None,
            'attacked_subset': None,
            'diagnostic': {
                'model': model_name,
                'stage': stage,
                'status': status,
                'error': message,
            },
        }

    @staticmethod
    def _shap_columns(shap_metrics: Dict[str, Any]) -> Dict[str, Any]:
        return {
            'mean_rank_stability': shap_metrics['mean_rank_stability'],
            'shap_status': shap_metrics['status'],
            'shap_error': shap_metrics['error'],
        }

    def _shap_metrics(
        self,
        model_name: str,
        scenario_name: str,
        scenario_type: str,
        primary_feature: Optional[str],
        shap_context: Dict[str, Any],
        mutated_inputs: Any,
        baseline_detected_count: int,
        flips_to_human: Any,
    ) -> Dict[str, Any]:
        result = {
            'mean_rank_stability': np.nan,
            'stability_rows': [],
            'pivot_rows': [],
            'frs_row': None,
            'status': 'not_requested',
            'error': None,
            'diagnostic': None,
        }
        if not shap_context['available']:
            diagnostic = shap_context['diagnostic']
            result['status'] = diagnostic['status']
            result['error'] = diagnostic['error']
            return result

        try:
            subset_len = len(shap_context['attacked_subset'])
            mutated_subset = mutated_inputs[:subset_len]
            mutated_values = shap_context['explainer'].explain(mutated_subset)
        except EXPECTED_SHAP_EXCEPTIONS as exc:
            result['status'] = 'explain_failed'
            result['error'] = self._format_exception(exc)
            result['diagnostic'] = {
                'model': model_name,
                'scenario_type': scenario_type,
                'scenario_name': scenario_name,
                'stage': 'explain',
                'status': result['status'],
                'error': result['error'],
            }
            return result

        baseline_values = np.asarray(shap_context['baseline_values'])
        mutated_values = np.asarray(mutated_values)
        mean_stability = []
        feature_stabilities = []
        feature_idx = self.feature_names.index(primary_feature) if primary_feature in self.feature_names else None
        result['status'] = 'computed'

        for idx, (before_row, after_row) in enumerate(zip(baseline_values, mutated_values)):
            before_ranks = FeatureResilienceAnalyzer.rank_positions(before_row)
            after_ranks = FeatureResilienceAnalyzer.rank_positions(after_row)
            rank_stability = FeatureResilienceAnalyzer.normalized_rank_spearman(before_row, after_row)
            mean_stability.append(rank_stability)
            result['stability_rows'].append({
                'model': model_name,
                'scenario_type': scenario_type,
                'scenario_name': scenario_name,
                'row_index': idx,
                'rank_stability': rank_stability,
            })
            pivot = FeatureResilienceAnalyzer.top_k_pivot_metadata(
                self.feature_names,
                before_row,
                after_row,
                top_k=self.shap_top_k,
            )
            pivot.update({
                'model': model_name,
                'scenario_name': scenario_name,
                'scenario_type': scenario_type,
                'row_index': idx,
            })
            result['pivot_rows'].append(pivot)
            if feature_idx is not None:
                feature_stabilities.append(
                    FeatureResilienceAnalyzer.feature_rank_stability(before_ranks, after_ranks, feature_idx)
                )

        if mean_stability:
            result['mean_rank_stability'] = float(np.mean(mean_stability))

        if feature_idx is not None:
            mutable_indices = [
                self.feature_names.index(name)
                for name in self.engine.available_single_feature_attacks()
                if name in self.feature_names
            ]
            baseline_abs = np.abs(baseline_values)
            feature_importance = float(np.mean(baseline_abs[:, feature_idx]))
            max_importance = float(np.max(np.mean(baseline_abs[:, mutable_indices], axis=0))) if mutable_indices else feature_importance
            importance_norm = feature_importance / max_importance if max_importance else np.nan
            frs = FeatureResilienceAnalyzer.compute_feature_resilience(
                importance=importance_norm,
                stability=float(np.mean(feature_stabilities)) if feature_stabilities else np.nan,
                flips_to_human=int(flips_to_human) if not pd.isna(flips_to_human) else 0,
                baseline_detected_bots=baseline_detected_count,
            )
            result['frs_row'] = {
                'model': model_name,
                'feature': primary_feature,
                'importance_norm': importance_norm,
                'stability': float(np.mean(feature_stabilities)) if feature_stabilities else np.nan,
                'flip_rate': (int(flips_to_human) / baseline_detected_count) if baseline_detected_count else np.nan,
                'frs': frs,
            }

        return result

    def save_outputs(self, output_dir: Path, results: Dict[str, pd.DataFrame]) -> None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        file_map = {
            'summary': 'robustness_summary.csv',
            'feature_attacks': 'feature_attack_results.csv',
            'shap_rank_stability': 'shap_rank_stability.csv',
            'feature_resilience': 'feature_resilience_scores.csv',
            'shap_pivots': 'shap_pivot_features.csv',
            'shap_diagnostics': 'shap_diagnostics.csv',
        }
        for key, filename in file_map.items():
            results[key].to_csv(output_dir / filename, index=False)

        report = {
            'summary_rows': results['summary'].to_dict(orient='records'),
            'feature_attack_rows': results['feature_attacks'].to_dict(orient='records'),
            'shap_rank_stability_rows': results['shap_rank_stability'].to_dict(orient='records'),
            'feature_resilience_rows': results['feature_resilience'].to_dict(orient='records'),
            'shap_pivot_rows': results['shap_pivots'].to_dict(orient='records'),
            'shap_diagnostics_rows': results['shap_diagnostics'].to_dict(orient='records'),
        }
        with open(output_dir / 'robustness_report.json', 'w', encoding='utf-8') as handle:
            json.dump(report, handle, indent=2)

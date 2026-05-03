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

FRS_RESILIENCE_COLUMNS = [
    'model', 'feature', 'mean_abs_shap', 'importance_norm', 'spearman_rho', 'stability', 'flip_rate', 'frs',
]
FRS_STABILITY_COLUMNS = ['model', 'feature', 'spearman_rho', 'stability']
FRS_ABLATION_COLUMNS = ['model', 'k', 'macro_f1', 'pr_auc', 'macro_f1_drop', 'pr_auc_drop']


def _frs_empty_bundle(feature_frame: pd.DataFrame, fidelity: Dict[str, Any], markdown: str = '') -> Dict[str, Any]:
    return {
        'feature_frame': feature_frame,
        'feature_resilience': pd.DataFrame(columns=FRS_RESILIENCE_COLUMNS),
        'shap_rank_stability': pd.DataFrame(columns=FRS_STABILITY_COLUMNS),
        'shap_cumulative_ablation': pd.DataFrame(columns=FRS_ABLATION_COLUMNS),
        'fidelity': fidelity,
        'markdown': markdown,
    }


def _resolve_frs_model(config: Any) -> str:
    name = config.get('robustness.frs.model')
    if name:
        return str(name)
    return str(config.get('explainability.poster.model', 'xgboost') or 'xgboost')


def _minmax_importance(keys: Sequence[str], importance: Dict[str, float]) -> Dict[str, float]:
    if not keys:
        return {}
    vec = np.array([float(importance.get(k, 0.0)) for k in keys], dtype=np.float64)
    vmin, vmax = float(np.min(vec)), float(np.max(vec))
    if vmax <= vmin:
        return {k: 1.0 for k in keys}
    return {k: (float(importance.get(k, 0.0)) - vmin) / (vmax - vmin) for k in keys}


def _spearman_rho_pair(a: np.ndarray, b: np.ndarray) -> float:
    s1 = pd.Series(a)
    s2 = pd.Series(b)
    rho = s1.corr(s2, method='spearman')
    if rho is None or pd.isna(rho):
        return float('nan')
    return float(rho)


def compute_frs(importance_norm: float, stability: float, flip_rate: Any) -> float:
    fr = float(flip_rate) if flip_rate is not None and np.isfinite(flip_rate) else 0.0
    fr = min(1.0, max(0.0, fr))
    imp = float(importance_norm)
    stab = float(stability)
    return float(max(0.0, min(1.0, imp)) * max(0.0, min(1.0, stab)) * (1.0 - fr))


def _training_fill_value(X_train_df: pd.DataFrame, feature: str) -> float:
    col = X_train_df[feature]
    numeric = pd.to_numeric(col, errors='coerce').dropna()
    if numeric.empty:
        return 0.0
    uniq = np.unique(numeric.to_numpy())
    if len(uniq) <= 2 and np.all(np.isin(uniq, (0.0, 1.0))):
        mode = col.mode()
        return float(int(round(float(mode.iloc[0])))) if len(mode) else float(round(float(numeric.median())))
    return float(max(0.0, float(numeric.median())))


def build_feature_resilience_markdown(top_resilient: List[Tuple[str, float]], top_vulnerable: List[Tuple[str, float]]) -> str:
    lines = [
        '# Feature Resilience Score (FRS) Index',
        '',
        'The FRS quantifies the resilience of features against mutation on a scale from 0 (highly vulnerable) '
        'to 1 (highly resilient).',
        '',
        '## Top 5 Resilient Features (High FRS)',
        '',
        '| Feature | FRS |',
        '|---------|-----|',
    ]
    for name, score in top_resilient:
        lines.append(f'| `{name}` | {score:.2f} |')
    lines.extend([
        '',
        '## Top 5 Vulnerable Features (Low FRS)',
        '',
        '| Feature | FRS |',
        '|---------|-----|',
    ])
    for name, score in top_vulnerable:
        lines.append(f'| `{name}` | {score:.2f} |')
    lines.extend([
        '',
        'These measurements support evaluation of whether model reasoning is brittle under realistic mutations, '
        'alongside explainability artefacts.',
    ])
    return '\n'.join(lines) + '\n'


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
    benchmark.robustness_fidelity = results.get('fidelity', {})
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

    def run(self) -> Dict[str, Any]:
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
                for feature in self.engine.available_single_feature_attacks(builtin_only=True):
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

        feature_frame = pd.DataFrame(feature_rows)
        feature_frame['frs'] = np.nan
        frs_payload = self._run_frs_bundle(feature_frame)
        feature_frame = frs_payload['feature_frame']

        return {
            'summary': pd.DataFrame(summary_rows),
            'feature_attacks': feature_frame,
            'degradation': pd.DataFrame(degradation_rows),
            'profile_diagnostics': profile_diagnostics,
            'feature_resilience': frs_payload['feature_resilience'],
            'shap_rank_stability': frs_payload['shap_rank_stability'],
            'shap_cumulative_ablation': frs_payload['shap_cumulative_ablation'],
            'fidelity': frs_payload['fidelity'],
            '_feature_resilience_markdown': frs_payload['markdown'],
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
            result = self.engine.apply_profile(attacked_base, profile, collect_diagnostics=True)
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

    def _run_frs_bundle(self, feature_frame: pd.DataFrame) -> Dict[str, Any]:
        if not self.config.get('robustness.frs.enabled', True):
            return _frs_empty_bundle(
                feature_frame,
                {'selected_model': _resolve_frs_model(self.config), 'skip_reason': 'robustness.frs.enabled is false'},
            )

        model_name = _resolve_frs_model(self.config)
        fidelity: Dict[str, Any] = {'selected_model': model_name}
        if model_name not in self.benchmark.results:
            fidelity['skip_reason'] = 'model not in benchmark results'
            return _frs_empty_bundle(feature_frame, fidelity)

        result = self.benchmark.results[model_name]
        model = result['model']
        mnames = list(result.get('feature_names') or self.feature_names)
        try:
            probe = self.benchmark.prepare_eval_inputs(model_name, self.base_test.iloc[:1])
            model.predict_proba(probe)
        except Exception as exc:
            fidelity['skip_reason'] = f'predict_proba failed: {exc}'
            return _frs_empty_bundle(feature_frame, fidelity)

        try:
            from explainability.shap_explainer import SHAPExplainer
        except ImportError as exc:
            fidelity['skip_reason'] = f'SHAPExplainer unavailable: {exc}'
            return _frs_empty_bundle(feature_frame, fidelity)

        X_train_m, _, X_test_m = self.benchmark.get_prepared_inputs(model_name)
        max_bg = int(self.config.get('explainability.shap.max_samples', 100))
        max_shap = int(self.config.get('robustness.frs.shap_max_samples', max_bg))
        n_shap = min(max_shap, len(X_test_m))
        if n_shap <= 0:
            fidelity['skip_reason'] = 'empty test set for SHAP slice'
            return _frs_empty_bundle(feature_frame, fidelity)

        try:
            explainer = SHAPExplainer(model, mnames)
            explainer.fit(X_train_m, max_samples=max_bg)
            X_slice = X_test_m[:n_shap]
            explainer.explain(X_slice)
            baseline_imp = explainer.get_global_importance()
        except Exception as exc:
            fidelity['skip_reason'] = f'SHAP failed: {exc}'
            return _frs_empty_bundle(feature_frame, fidelity)

        top_k = max(1, int(self.config.get('robustness.frs.shap_top_k', 15)))
        sorted_imp = sorted(baseline_imp.items(), key=lambda item: item[1], reverse=True)
        top_k_feats = [name for name, _ in sorted_imp[:top_k]]
        static_feats = frozenset(self.engine.available_single_feature_attacks(builtin_only=True))
        union_features = sorted((frozenset(top_k_feats) | static_feats) & set(mnames))
        norm_imp = _minmax_importance(union_features, baseline_imp)

        baseline_vec = np.array([float(baseline_imp.get(name, 0.0)) for name in mnames], dtype=np.float64)
        X_shap_df = self._to_frame(X_slice, mnames)
        bot_mask = (self.y_test[:n_shap] == 1)

        fresh_bots = (
            self.base_test.loc[self._attack_population_mask()]
            .sort_values(self.feature_names, kind='mergesort')
            .reset_index(drop=True)
        )
        baseline_inputs_bots = self.benchmark.prepare_eval_inputs(model_name, fresh_bots)
        baseline_preds_bots = np.asarray(model.predict(baseline_inputs_bots))
        baseline_proba_bots = self._bot_probability(model, baseline_inputs_bots)
        baseline_detected_mask = baseline_preds_bots == 1
        baseline_detected_count = int(baseline_detected_mask.sum())

        resilience_rows: List[Dict[str, Any]] = []
        stability_rows: List[Dict[str, Any]] = []
        extra_rows: List[Dict[str, Any]] = []

        for feat in union_features:
            self.engine.register_dynamic_recipe(feat)
            if not bot_mask.any():
                rho = 1.0
            else:
                X_mut = X_shap_df.copy()
                sub = X_mut.loc[bot_mask].sort_values(self.feature_names, kind='mergesort')
                attack_shap = self.engine.apply_single_feature_attack(sub, feat)
                if attack_shap.applied:
                    aligned = attack_shap.data.reindex(columns=X_mut.columns)
                    X_mut.loc[sub.index, :] = aligned.loc[sub.index, :].to_numpy(dtype=np.float64, copy=False)
                explainer.explain(X_mut)
                attacked_imp = explainer.get_global_importance()
                attacked_vec = np.array([float(attacked_imp.get(name, 0.0)) for name in mnames], dtype=np.float64)
                rho = _spearman_rho_pair(baseline_vec, attacked_vec)
            spearman = rho
            stability = float(max(0.0, min(1.0, rho))) if np.isfinite(rho) else 0.0

            attack_full = self.engine.apply_single_feature_attack(fresh_bots.copy(), feat)
            attack_row = self._single_attack_row(
                model_name,
                feat,
                attack_full,
                fresh_bots,
                baseline_detected_mask,
                baseline_detected_count,
                baseline_proba_bots,
                model,
            )
            flip_rate = attack_row.get('flip_rate', np.nan)
            frs_val = compute_frs(norm_imp[feat], stability, flip_rate)

            mask = (feature_frame['model'] == model_name) & (feature_frame['feature'] == feat)
            if mask.any():
                feature_frame.loc[mask, 'frs'] = frs_val
            elif feat not in static_feats:
                attack_row['frs'] = frs_val
                extra_rows.append(attack_row)

            resilience_rows.append({
                'model': model_name,
                'feature': feat,
                'mean_abs_shap': float(baseline_imp.get(feat, 0.0)),
                'importance_norm': float(norm_imp[feat]),
                'spearman_rho': float(spearman) if np.isfinite(spearman) else np.nan,
                'stability': stability,
                'flip_rate': flip_rate,
                'frs': frs_val,
            })
            stability_rows.append({
                'model': model_name,
                'feature': feat,
                'spearman_rho': float(spearman) if np.isfinite(spearman) else np.nan,
                'stability': stability,
            })

        explainer.explain(X_slice)
        resilience_df = pd.DataFrame(resilience_rows, columns=FRS_RESILIENCE_COLUMNS)
        stability_df = pd.DataFrame(stability_rows, columns=FRS_STABILITY_COLUMNS)

        if resilience_df.empty:
            fidelity['skip_reason'] = 'no features in FRS union'
            return _frs_empty_bundle(
                feature_frame,
                fidelity,
                '# Feature Resilience Score (FRS) Index\n\n_No features evaluated._\n',
            )

        top10 = [name for name, _ in sorted_imp[:10]]
        ablation_ks = self.config.get('robustness.frs.ablation_top_ks', [1, 3, 5, 10])
        base_full = self._to_frame(self.benchmark.base_test_inputs, self.feature_names)
        train_df = self._to_frame(X_train_m, mnames)
        base_metrics = self._scenario_metrics(model_name, model, base_full)
        ablation_rows: List[Dict[str, Any]] = []
        for k in ablation_ks:
            limit = int(min(k, len(top10)))
            X_ab = base_full.copy()
            for name in top10[:limit]:
                if name in X_ab.columns and name in train_df.columns:
                    X_ab[name] = _training_fill_value(train_df, name)
            metrics_ab = self._scenario_metrics(model_name, model, X_ab)
            ablation_rows.append({
                'model': model_name,
                'k': int(k),
                'macro_f1': metrics_ab['macro_f1'],
                'pr_auc': metrics_ab['pr_auc'],
                'macro_f1_drop': base_metrics['macro_f1'] - metrics_ab['macro_f1'],
                'pr_auc_drop': base_metrics['pr_auc'] - metrics_ab['pr_auc'],
            })
        ablation_df = pd.DataFrame(ablation_rows, columns=FRS_ABLATION_COLUMNS)
        drops_ok = all(
            float(row['macro_f1_drop']) > 0.0 and float(row['pr_auc_drop']) > 0.0
            for _, row in ablation_df.iterrows()
        ) if not ablation_df.empty else False
        fidelity.update({
            'baseline_macro_f1': base_metrics['macro_f1'],
            'baseline_pr_auc': base_metrics['pr_auc'],
            'drop_at_k': {int(r['k']): {'macro_f1_drop': float(r['macro_f1_drop']), 'pr_auc_drop': float(r['pr_auc_drop'])} for _, r in ablation_df.iterrows()},
            'fidelity_passed': bool(drops_ok),
        })

        ranked = sorted(((row['feature'], row['frs']) for _, row in resilience_df.iterrows()), key=lambda x: x[1], reverse=True)
        top_resilient = ranked[:5]
        top_vulnerable = sorted(((row['feature'], row['frs']) for _, row in resilience_df.iterrows()), key=lambda x: x[1])[:5]
        md = build_feature_resilience_markdown(top_resilient, top_vulnerable)

        if extra_rows:
            feature_frame = pd.concat([feature_frame, pd.DataFrame(extra_rows)], ignore_index=True)

        return {
            'feature_frame': feature_frame,
            'feature_resilience': resilience_df,
            'shap_rank_stability': stability_df,
            'shap_cumulative_ablation': ablation_df,
            'fidelity': fidelity,
            'markdown': md,
        }

    def save_outputs(self, output_dir: Path, results: Dict[str, Any]) -> None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        payload = dict(results)
        fidelity = payload.pop('fidelity', {})
        md_text = payload.pop('_feature_resilience_markdown', '')
        sorted_results = self._sorted_results(payload)
        file_map = {
            'summary': 'robustness_summary.csv',
            'feature_attacks': 'feature_attack_results.csv',
            'degradation': 'robustness_degradation.csv',
            'profile_diagnostics': 'profile_diagnostics.csv',
            'feature_resilience': 'feature_resilience.csv',
            'shap_rank_stability': 'shap_rank_stability.csv',
            'shap_cumulative_ablation': 'shap_cumulative_ablation.csv',
        }
        for key, filename in file_map.items():
            frame = sorted_results.get(key)
            if frame is None:
                continue
            format_frame_for_export(frame).to_csv(output_dir / filename, index=False)

        (output_dir / 'feature_resilience.md').write_text(md_text or '', encoding='utf-8')

        report = {
            'artifacts': {
                key: self._artifact_manifest(filename, sorted_results[key])
                for key, filename in file_map.items()
                if key in sorted_results
            },
            'overview': self._report_overview(sorted_results),
            'profile_sanity': self._profile_sanity(sorted_results),
        }
        report['overview']['fidelity'] = format_payload_for_export(fidelity)
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
            profile_summary = summary[summary['profile'] == profile] if 'profile' in summary else pd.DataFrame()
            profile_diagnostics = (
                diagnostics[diagnostics['profile'] == profile] if 'profile' in diagnostics else pd.DataFrame()
            )
            profile_degradation = (
                degradation[degradation['scenario'] == profile] if 'scenario' in degradation else pd.DataFrame()
            )
            applied_mask = (
                profile_diagnostics['recipe_applied'].fillna(False).astype(bool)
                if 'recipe_applied' in profile_diagnostics
                else pd.Series(False, index=profile_diagnostics.index)
            )
            expensive_mask = (
                profile_diagnostics['cost_tier'].eq('expensive')
                if 'cost_tier' in profile_diagnostics
                else pd.Series(False, index=profile_diagnostics.index)
            )
            output[str(profile)] = {
                'recipe_count': int(len(profile_diagnostics)),
                'applied_recipe_count': int(applied_mask.sum()),
                'changed_expensive_recipe_count': int((expensive_mask & applied_mask).sum()),
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
    def _sorted_results(results: Dict[str, Any]) -> Dict[str, pd.DataFrame]:
        sort_columns = {
            'summary': ['model', 'profile'],
            'feature_attacks': ['model', 'feature'],
            'degradation': ['model', 'scenario'],
            'profile_diagnostics': ['profile', 'cost_tier', 'feature'],
            'feature_resilience': ['model', 'feature'],
            'shap_rank_stability': ['model', 'feature'],
            'shap_cumulative_ablation': ['model', 'k'],
        }
        sorted_results: Dict[str, pd.DataFrame] = {}
        for key, frame in results.items():
            if not isinstance(frame, pd.DataFrame):
                continue
            columns = [column for column in sort_columns.get(key, []) if column in frame.columns]
            if columns:
                sorted_results[key] = frame.sort_values(columns).reset_index(drop=True)
            else:
                sorted_results[key] = frame.copy()
        return sorted_results

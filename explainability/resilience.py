"""Resilience metrics for adversarial explainability analysis."""

from __future__ import annotations

from typing import Dict, List

import numpy as np


class FeatureResilienceAnalyzer:
    """Helper methods for SHAP rank stability and FRS calculation."""

    @staticmethod
    def rank_positions(values: np.ndarray) -> np.ndarray:
        order = np.argsort(-np.abs(np.asarray(values)))
        ranks = np.empty(len(order), dtype=int)
        ranks[order] = np.arange(len(order))
        return ranks

    @classmethod
    def feature_rank_stability(
        cls,
        before_ranks: np.ndarray,
        after_ranks: np.ndarray,
        feature_index: int,
    ) -> float:
        feature_count = max(len(before_ranks) - 1, 1)
        delta = abs(int(before_ranks[feature_index]) - int(after_ranks[feature_index]))
        return max(0.0, 1.0 - (delta / feature_count))

    @classmethod
    def normalized_rank_spearman(cls, before_values: np.ndarray, after_values: np.ndarray) -> float:
        before_ranks = cls.rank_positions(before_values)
        after_ranks = cls.rank_positions(after_values)
        if len(before_ranks) < 2:
            return 1.0
        n = len(before_ranks)
        diff = before_ranks - after_ranks
        rho = 1 - ((6 * np.sum(diff ** 2)) / (n * (n ** 2 - 1)))
        return float((rho + 1) / 2)

    @classmethod
    def top_k_pivot_metadata(
        cls,
        feature_names: List[str],
        before_values: np.ndarray,
        after_values: np.ndarray,
        top_k: int = 5,
    ) -> Dict[str, str]:
        before_order = np.argsort(-np.abs(np.asarray(before_values)))[:top_k]
        after_order = np.argsort(-np.abs(np.asarray(after_values)))[:top_k]
        before_set = [feature_names[idx] for idx in before_order]
        after_set = [feature_names[idx] for idx in after_order]
        entered = [name for name in after_set if name not in before_set]
        dropped = [name for name in before_set if name not in after_set]
        return {
            'top_feature_before': before_set[0] if before_set else '',
            'top_feature_after': after_set[0] if after_set else '',
            'entered_top_k': ','.join(entered),
            'dropped_top_k': ','.join(dropped),
        }

    @staticmethod
    def compute_feature_resilience(
        importance: float,
        stability: float,
        flips_to_human: int,
        baseline_detected_bots: int,
    ) -> float:
        if baseline_detected_bots == 0:
            return float('nan')
        flip_rate = flips_to_human / baseline_detected_bots
        return float(importance * stability * (1 - flip_rate))

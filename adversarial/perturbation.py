"""Cost-aware feature perturbation helpers for robustness analysis."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, Iterable, List, Optional, Sequence

import numpy as np
import pandas as pd


MutationRule = Callable[[pd.DataFrame, "RealisticPerturbationEngine"], bool]


@dataclass(frozen=True)
class AttackRecipe:
    feature: str
    cost_tier: str
    profile_membership: Sequence[str]
    mutation_rule: MutationRule
    preconditions: Sequence[str] = field(default_factory=tuple)
    dependent_recomputations: Sequence[str] = field(default_factory=tuple)
    skip_reason: str = 'preconditions not met or no source columns available'


@dataclass
class AttackResult:
    data: pd.DataFrame
    applied: bool
    feature: Optional[str]
    cost_tier: str
    attack_name: str
    skip_reason: str = ""


class RealisticPerturbationEngine:
    """Apply realistic, cost-aware perturbations to engineered feature matrices."""

    CHEAP_FEATURES = (
        'has_description',
        'description_length',
        'screen_name_has_digits',
        'default_profile_image',
        'default_profile',
        'has_extended_profile',
    )
    EXPENSIVE_FEATURES = ('followers_count', 'friends_count')

    def __init__(
        self,
        feature_names: Sequence[str],
        X_train: pd.DataFrame,
        y_train: Sequence[int],
        expensive_nudge_fraction: float = 0.05,
    ):
        self.feature_names = list(feature_names)
        self.expensive_nudge_fraction = expensive_nudge_fraction
        self._train_frame = self._to_frame(X_train)
        self._y_train = np.asarray(y_train)
        self._human_frame = self._train_frame.loc[self._y_train == 0]
        self._human_stats = self._build_human_stats()
        self._recipes = self._build_recipes()

    def _to_frame(self, X: pd.DataFrame) -> pd.DataFrame:
        if isinstance(X, pd.DataFrame):
            return X.copy()
        return pd.DataFrame(X, columns=self.feature_names)

    def _build_human_stats(self) -> Dict[str, float]:
        human = self._human_frame if not self._human_frame.empty else self._train_frame
        stats = {}
        desc = human.get('description_length')
        if desc is not None:
            positive_desc = desc[desc > 0]
            stats['description_length'] = float(positive_desc.median()) if not positive_desc.empty else 1.0
        for feature in ('followers_count', 'friends_count'):
            series = human.get(feature)
            if series is not None:
                stats[feature] = float(series.median())
        return stats

    def _build_recipes(self) -> Dict[str, AttackRecipe]:
        return {
            'has_description': AttackRecipe(
                feature='has_description',
                cost_tier='cheap',
                profile_membership=('cheap_only', 'realistic_mixed'),
                preconditions=('has_description available',),
                dependent_recomputations=('description_length',),
                mutation_rule=lambda frame, engine: engine._apply_has_description(frame),
            ),
            'description_length': AttackRecipe(
                feature='description_length',
                cost_tier='cheap',
                profile_membership=('cheap_only', 'realistic_mixed'),
                preconditions=('description_length available',),
                dependent_recomputations=('has_description',),
                mutation_rule=lambda frame, engine: engine._apply_description_length(frame),
            ),
            'screen_name_has_digits': AttackRecipe(
                feature='screen_name_has_digits',
                cost_tier='cheap',
                profile_membership=('cheap_only', 'realistic_mixed'),
                preconditions=('screen_name_has_digits available',),
                mutation_rule=lambda frame, engine: engine._flip_binary(frame, 'screen_name_has_digits', 0),
            ),
            'default_profile_image': AttackRecipe(
                feature='default_profile_image',
                cost_tier='cheap',
                profile_membership=('cheap_only', 'realistic_mixed'),
                preconditions=('default_profile_image available',),
                mutation_rule=lambda frame, engine: engine._flip_binary(frame, 'default_profile_image', 0),
            ),
            'default_profile': AttackRecipe(
                feature='default_profile',
                cost_tier='cheap',
                profile_membership=('cheap_only', 'realistic_mixed'),
                preconditions=('default_profile available',),
                mutation_rule=lambda frame, engine: engine._flip_binary(frame, 'default_profile', 0),
            ),
            'has_extended_profile': AttackRecipe(
                feature='has_extended_profile',
                cost_tier='cheap',
                profile_membership=('cheap_only', 'realistic_mixed'),
                preconditions=('has_extended_profile available',),
                mutation_rule=lambda frame, engine: engine._flip_binary(frame, 'has_extended_profile', 1),
            ),
            'followers_count': AttackRecipe(
                feature='followers_count',
                cost_tier='expensive',
                profile_membership=('realistic_mixed',),
                preconditions=('followers_count available', 'human training median available'),
                dependent_recomputations=('followers_to_friends_ratio', 'followers_per_day'),
                skip_reason='preconditions not met or no source columns available',
                mutation_rule=lambda frame, engine: engine._nudge_count(frame, 'followers_count'),
            ),
            'friends_count': AttackRecipe(
                feature='friends_count',
                cost_tier='expensive',
                profile_membership=('realistic_mixed',),
                preconditions=('friends_count available', 'human training median available'),
                dependent_recomputations=('followers_to_friends_ratio',),
                skip_reason='preconditions not met or no source columns available',
                mutation_rule=lambda frame, engine: engine._nudge_count(frame, 'friends_count'),
            ),
        }

    def available_single_feature_attacks(self) -> List[str]:
        return list(self._recipes.keys())

    def apply_single_feature_attack(self, X: pd.DataFrame, feature: str) -> AttackResult:
        recipe = self._recipes.get(feature)
        frame = self._to_frame(X)
        if recipe is None:
            return AttackResult(frame, False, feature, 'unknown', feature, 'unknown feature')
        if recipe.feature not in frame.columns:
            return AttackResult(frame, False, feature, recipe.cost_tier, feature, 'feature not available after preprocessing/selection')
        applied = recipe.mutation_rule(frame, self)
        if not applied:
            return AttackResult(frame, False, feature, recipe.cost_tier, feature, recipe.skip_reason)
        self._recompute_derived(frame)
        return AttackResult(frame, True, feature, recipe.cost_tier, feature)

    def apply_profile(self, X: pd.DataFrame, profile: str) -> AttackResult:
        frame = self._to_frame(X)
        applied = False
        for recipe in self._recipes.values():
            if profile not in recipe.profile_membership:
                continue
            if recipe.feature not in frame.columns:
                continue
            applied = recipe.mutation_rule(frame, self) or applied
        if applied:
            self._recompute_derived(frame)
        return AttackResult(frame, applied, None, 'profile', profile, '' if applied else 'no applicable feature recipes')

    def _apply_has_description(self, frame: pd.DataFrame) -> bool:
        changed = self._flip_binary(frame, 'has_description', 1)
        if 'description_length' in frame.columns:
            target = max(1.0, self._human_stats.get('description_length', 1.0))
            changed = self._raise_to_target(frame, 'description_length', target) or changed
        return changed

    def _apply_description_length(self, frame: pd.DataFrame) -> bool:
        target = max(1.0, self._human_stats.get('description_length', 1.0))
        changed = self._raise_to_target(frame, 'description_length', target)
        if 'has_description' in frame.columns:
            changed = self._flip_binary(frame, 'has_description', 1) or changed
        return changed

    def _flip_binary(self, frame: pd.DataFrame, feature: str, target: int) -> bool:
        if feature not in frame.columns:
            return False
        before = frame[feature].copy()
        frame[feature] = target
        return not before.equals(frame[feature])

    def _raise_to_target(self, frame: pd.DataFrame, feature: str, target: float) -> bool:
        if feature not in frame.columns:
            return False
        before = frame[feature].copy()
        frame[feature] = np.maximum(frame[feature], target)
        return not before.equals(frame[feature])

    def _nudge_count(self, frame: pd.DataFrame, feature: str) -> bool:
        if feature not in frame.columns:
            return False
        target = self._human_stats.get(feature)
        if target is None:
            return False
        current = frame[feature].astype(float)
        delta = target - current
        limit = np.maximum(1.0, current.abs()) * self.expensive_nudge_fraction
        bounded = delta.clip(lower=-limit, upper=limit)
        frame[feature] = (current + bounded).clip(lower=0)
        return bool(np.any(np.abs(bounded) > 0))

    def _recompute_derived(self, frame: pd.DataFrame) -> None:
        if {'followers_count', 'friends_count'}.issubset(frame.columns) and 'followers_to_friends_ratio' in frame.columns:
            frame['followers_to_friends_ratio'] = frame['followers_count'] / (frame['friends_count'] + 1)
        if 'account_age_days' not in frame.columns:
            return
        safe_age = frame['account_age_days'].replace(0, 1).clip(lower=1)
        if {'followers_count', 'followers_per_day'}.issubset(frame.columns):
            frame['followers_per_day'] = frame['followers_count'] / safe_age
        if {'statuses_count', 'tweets_per_day'}.issubset(frame.columns):
            frame['tweets_per_day'] = frame['statuses_count'] / safe_age
        if {'favourites_count', 'favourites_per_day'}.issubset(frame.columns):
            frame['favourites_per_day'] = frame['favourites_count'] / safe_age

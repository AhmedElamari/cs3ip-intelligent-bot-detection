"""Plausible bot evasion: cheap profile edits vs expensive follower/friend nudges."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence

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
    diagnostics: List[Dict[str, object]] = field(default_factory=list)


class RealisticPerturbationEngine:
    """Mutate only features a real account could change; recompute derived ratios/rates."""

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
        self._static_recipe_keys = frozenset(self._recipes.keys())

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
            stats['description_length'] = (
                float(positive_desc.quantile(0.25, interpolation='lower'))
                if not positive_desc.empty else 1.0
            )
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

    def available_single_feature_attacks(self, builtin_only: bool = True) -> List[str]:
        keys = sorted(self._recipes)
        if builtin_only:
            return [key for key in keys if key in self._static_recipe_keys]
        return keys

    def register_dynamic_recipe(self, feature: str) -> bool:
        """Add a single-feature probe that masks toward human-train statistics (not part of profiles)."""
        if feature in self._recipes:
            return True
        if feature not in self._train_frame.columns:
            return False
        kind = self._infer_feature_kind(feature)
        if kind == 'binary':
            self._recipes[feature] = self._build_dynamic_binary_recipe(feature)
        else:
            self._recipes[feature] = self._build_dynamic_numeric_recipe(feature)
        return True

    def _infer_feature_kind(self, feature: str) -> str:
        series = pd.to_numeric(self._train_frame[feature], errors='coerce').dropna()
        if series.empty:
            return 'numeric'
        uniq = np.unique(series.to_numpy())
        if len(uniq) <= 2 and np.all(np.isin(uniq, (0.0, 1.0))):
            return 'binary'
        return 'numeric'

    def _human_feature_series(self, feature: str) -> pd.Series:
        series = self._human_frame.get(feature)
        return series if series is not None else self._train_frame[feature]

    def _human_binary_majority(self, feature: str) -> Optional[int]:
        vals = self._human_feature_series(feature).dropna()
        if vals.empty:
            return None
        mode = vals.mode()
        if len(mode):
            return int(round(float(mode.iloc[0])))
        return int(round(float(vals.median())))

    def _human_numeric_median_for_feature(self, feature: str) -> Optional[float]:
        numeric = pd.to_numeric(
            self._human_feature_series(feature),
            errors='coerce',
        ).dropna()
        if numeric.empty:
            return None
        return float(numeric.median())

    def _build_dynamic_binary_recipe(self, feature: str) -> AttackRecipe:
        majority = self._human_binary_majority(feature)

        def mutation_rule(frame: pd.DataFrame, engine: RealisticPerturbationEngine) -> bool:
            if majority is None:
                return False
            return engine._set_feature_constant(frame, feature, int(majority))

        return AttackRecipe(
            feature=feature,
            cost_tier='generic_mask',
            profile_membership=(),
            mutation_rule=mutation_rule,
            skip_reason='binary majority unavailable',
        )

    def _build_dynamic_numeric_recipe(self, feature: str) -> AttackRecipe:
        median = self._human_numeric_median_for_feature(feature)

        def mutation_rule(frame: pd.DataFrame, engine: RealisticPerturbationEngine) -> bool:
            if median is None:
                return False
            return engine._set_feature_constant(frame, feature, float(max(0.0, median)))

        return AttackRecipe(
            feature=feature,
            cost_tier='generic_mask',
            profile_membership=(),
            mutation_rule=mutation_rule,
            skip_reason='numeric human median unavailable',
        )

    def _set_feature_constant(self, frame: pd.DataFrame, feature: str, value: float) -> bool:
        if feature not in frame.columns:
            return False
        before = frame[feature].copy()
        frame[feature] = value
        return not before.equals(frame[feature])

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

    def apply_profile(self, X: pd.DataFrame, profile: str, collect_diagnostics: bool = False) -> AttackResult:
        frame = self._to_frame(X)
        applied = False
        diagnostics = []
        recipes = [recipe for recipe in self._recipes.values() if profile in recipe.profile_membership]
        diagnostic_columns = []
        if collect_diagnostics:
            diagnostic_columns = sorted({
                column
                for recipe in recipes
                for column in (recipe.feature, *recipe.dependent_recomputations)
                if column in frame.columns
            })
        before_profile = frame[diagnostic_columns].copy() if diagnostic_columns else None
        for recipe in recipes:
            if recipe.feature not in frame.columns:
                continue
            before_feature = frame[recipe.feature].copy() if collect_diagnostics else None
            recipe_applied = recipe.mutation_rule(frame, self)
            if collect_diagnostics:
                diagnostics.append((
                    recipe,
                    recipe_applied,
                    '' if recipe_applied else recipe.skip_reason,
                    before_feature,
                    frame[recipe.feature].copy(),
                ))
            applied = recipe_applied or applied
        if applied:
            self._recompute_derived(frame)
        if not collect_diagnostics:
            return AttackResult(
                frame, applied, None, 'profile', profile, '' if applied else 'no applicable feature recipes'
            )
        after_profile = frame[diagnostic_columns].copy()
        return AttackResult(
            frame,
            applied,
            None,
            'profile',
            profile,
            '' if applied else 'no applicable feature recipes',
            diagnostics=[
                self._profile_diagnostic(
                    before_profile,
                    after_profile,
                    before_feature,
                    after_feature,
                    recipe,
                    recipe_applied,
                    profile,
                    skip_reason,
                )
                for recipe, recipe_applied, skip_reason, before_feature, after_feature in diagnostics
            ],
        )

    def _apply_has_description(self, frame: pd.DataFrame) -> bool:
        changed = self._flip_binary(frame, 'has_description', 1)
        if 'description_length' in frame.columns:
            target = max(1.0, self._human_stats.get('description_length', 1.0))
            changed = self._set_short_human_bio(frame, target) or changed
        return changed

    def _apply_description_length(self, frame: pd.DataFrame) -> bool:
        target = max(1.0, self._human_stats.get('description_length', 1.0))
        changed = self._set_short_human_bio(frame, target)
        if 'has_description' in frame.columns:
            changed = self._flip_binary(frame, 'has_description', 1) or changed
        return changed

    def _flip_binary(self, frame: pd.DataFrame, feature: str, target: int) -> bool:
        if feature not in frame.columns:
            return False
        before = frame[feature].copy()
        frame[feature] = target
        return not before.equals(frame[feature])

    def _set_short_human_bio(self, frame: pd.DataFrame, target: float) -> bool:
        if 'description_length' not in frame.columns:
            return False
        before = frame['description_length'].copy()
        current = frame['description_length'].astype(float)
        frame['description_length'] = np.where(current <= 0, target, np.minimum(current, target))
        return not before.equals(frame['description_length'])

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

    @staticmethod
    def _changed_mask(before: pd.Series, after: pd.Series) -> pd.Series:
        return ~(before.eq(after) | (before.isna() & after.isna()))

    @staticmethod
    def _stable_numeric_stats(values: pd.Series) -> tuple[float, float]:
        numeric = pd.to_numeric(values, errors='coerce')
        if numeric.notna().sum() == 0:
            return np.nan, np.nan
        return float(numeric.mean()), float(numeric.median())

    def _profile_diagnostic(
        self,
        before_profile: pd.DataFrame,
        after_profile: pd.DataFrame,
        before_feature: pd.Series,
        after_feature: pd.Series,
        recipe: AttackRecipe,
        recipe_applied: bool,
        profile: str,
        skip_reason: str,
    ) -> Dict[str, object]:
        changed_mask = self._changed_mask(before_feature, after_feature)
        changed_rows = int(changed_mask.sum())
        total_rows = len(before_feature)
        pre_mean, pre_median = self._stable_numeric_stats(before_feature)
        post_mean, post_median = self._stable_numeric_stats(after_feature)

        before_numeric = pd.to_numeric(before_feature, errors='coerce')
        after_numeric = pd.to_numeric(after_feature, errors='coerce')
        valid = before_numeric.notna() & after_numeric.notna()
        if valid.any():
            abs_delta = (after_numeric[valid] - before_numeric[valid]).abs()
            rel_delta = abs_delta / before_numeric[valid].abs().clip(lower=1.0)
            mean_abs_delta = float(abs_delta.mean())
            max_abs_delta = float(abs_delta.max())
            mean_relative_delta = float(rel_delta.mean())
        else:
            mean_abs_delta = np.nan
            max_abs_delta = np.nan
            mean_relative_delta = np.nan

        relevant_columns = [recipe.feature, *recipe.dependent_recomputations]
        changed_columns = [
            column for column in relevant_columns
            if column in before_profile.columns and column in after_profile.columns
            and (
                (
                    column == recipe.feature and changed_rows > 0
                ) or (
                    column != recipe.feature and recipe_applied and
                    bool(self._changed_mask(before_profile[column], after_profile[column]).any())
                )
            )
        ]
        return {
            'profile': profile,
            'feature': recipe.feature,
            'cost_tier': recipe.cost_tier,
            'recipe_applied': recipe_applied,
            'changed_rows': changed_rows,
            'changed_fraction': (changed_rows / total_rows) if total_rows else 0.0,
            'changed_columns': ';'.join(changed_columns),
            'pre_mean': pre_mean,
            'post_mean': post_mean,
            'pre_median': pre_median,
            'post_median': post_median,
            'mean_abs_delta': mean_abs_delta,
            'max_abs_delta': max_abs_delta,
            'mean_relative_delta': mean_relative_delta,
            'skip_reason': '' if recipe_applied else skip_reason,
        }

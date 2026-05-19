"""
XGBoost Model
=============
Gradient boosted trees via the XGBoost library.

XGBoost is the de-facto standard gradient boosting implementation, offering
regularisation, built-in missing-value handling, and first-class SHAP support
via TreeExplainer.  It replaces the sklearn GradientBoostingClassifier in this
pipeline to satisfy research objective O3 (comparative evaluation across the
interpretability spectrum: LR / RF / XGBoost).

Imbalance handling:
    XGBClassifier does not accept ``class_weight`` directly.  When
    ``class_weight='balanced'`` or a dict mapping is supplied, the wrapper
    converts it to per-sample weights passed to ``fit(sample_weight=...)``,
    mirroring the behaviour of the retired GradientBoostingModel.
"""

from typing import Any, Dict, List, Optional
import numpy as np
from .base import BaseModel


class XGBoostModel(BaseModel):
    """
    XGBoost classifier for bot detection.

    Builds an ensemble of gradient-boosted decision trees via
    ``xgboost.XGBClassifier``.  Provides feature importances
    (gain-based) and is natively supported by SHAP's TreeExplainer.

    Advantages:
        - State-of-the-art predictive performance
        - Regularisation (L1/L2) reduces overfitting
        - Built-in missing-value handling
        - Native SHAP TreeExplainer support for fast, exact explanations
        - Deterministic under fixed random_state

    Disadvantages:
        - Less interpretable than single-model approaches (LR, DT)
        - Hyperparameter-sensitive; requires tuning for best results

    Notes:
        XGBClassifier does not support ``class_weight`` directly.
        Supply ``class_weight='balanced'`` or a dict; the model converts
        it to ``sample_weight`` vectors on every ``fit`` call.
    """

    def __init__(self, random_state: int = 2112, **kwargs):
        super().__init__(name='XGBoost', random_state=random_state)
        self._params = {
            'random_state': random_state,
            'n_estimators': kwargs.get('n_estimators', 100),
            'learning_rate': kwargs.get('learning_rate', 0.1),
            'max_depth': kwargs.get('max_depth', 5),
            'subsample': kwargs.get('subsample', 0.8),
            'colsample_bytree': kwargs.get('colsample_bytree', 0.8),
            'reg_alpha': kwargs.get('reg_alpha', 0.0),
            'reg_lambda': kwargs.get('reg_lambda', 1.0),
            'n_jobs': kwargs.get('n_jobs', -1),
            'eval_metric': kwargs.get('eval_metric', 'logloss'),
            'class_weight': kwargs.get('class_weight', 'balanced'),
        }
        self.model = self._create_model(**self._params)

    # ------------------------------------------------------------------
    # BaseModel interface
    # ------------------------------------------------------------------

    def _create_model(self, **kwargs):
        try:
            from xgboost import XGBClassifier
        except ImportError:
            raise ImportError(
                "xgboost is not installed. Install with: pip install xgboost"
            )
        xgb_kwargs = {k: v for k, v in kwargs.items() if k != 'class_weight'}
        xgb_kwargs.setdefault('eval_metric', 'logloss')
        return XGBClassifier(**xgb_kwargs)

    @property
    def is_interpretable(self) -> bool:
        return False  # Strong nonlinear baseline; SHAP post-hoc, not intrinsic rules.

    @property
    def supports_feature_importance(self) -> bool:
        return True

    # ------------------------------------------------------------------
    # Imbalance handling
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_sample_weight(y: np.ndarray, class_weight) -> Optional[np.ndarray]:
        """Convert class_weight specification to per-sample weights."""
        if class_weight is None:
            return None
        y_int = np.asarray(y).astype(int)
        if class_weight == 'balanced':
            classes, counts = np.unique(y_int, return_counts=True)
            if len(classes) == 0:
                return None
            total = len(y_int)
            weight_map = {
                cls: total / (len(classes) * count)
                for cls, count in zip(classes, counts)
            }
        elif isinstance(class_weight, dict):
            weight_map = {int(k): v for k, v in class_weight.items()}
        else:
            raise ValueError(f"Unsupported class_weight value: {class_weight}")

        unknown = sorted(set(np.unique(y_int)) - set(weight_map.keys()))
        if unknown:
            raise ValueError(f"Unexpected class labels: {unknown}")
        return np.array([weight_map[int(label)] for label in y_int], dtype=float)

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        feature_names: Optional[List[str]] = None,
        **kwargs,
    ) -> 'XGBoostModel':
        sample_weight = self._compute_sample_weight(
            y_train, self._params.get('class_weight')
        )
        if sample_weight is not None:
            kwargs = dict(kwargs)
            kwargs['sample_weight'] = sample_weight
        super().fit(X_train, y_train, feature_names=feature_names, **kwargs)
        return self

    # ------------------------------------------------------------------
    # Convenience helpers
    # ------------------------------------------------------------------

    def get_top_features(self, n: int = 10) -> List[tuple]:
        """Return top N (feature_name, importance) tuples by gain."""
        self._check_fitted()
        importance_dict = self.get_feature_importance()
        return sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)[:n]

    def plot_feature_importance(
        self, n_features: int = 15, figsize: tuple = (10, 8)
    ):
        """Plot horizontal bar chart of feature importances."""
        import matplotlib.pyplot as plt

        self._check_fitted()
        importances = self.model.feature_importances_
        indices = np.argsort(importances)[::-1][:n_features]

        fig, ax = plt.subplots(figsize=figsize)
        ax.barh(
            range(len(indices)),
            importances[indices],
            align='center',
            color='steelblue',
        )
        ax.set_yticks(range(len(indices)))
        ax.set_yticklabels([self.feature_names[i] for i in indices])
        ax.invert_yaxis()
        ax.set_xlabel('Feature Importance (gain)')
        ax.set_title(f'XGBoost Feature Importances (Top {n_features})')
        plt.tight_layout()
        return fig

    def get_params(self) -> Dict[str, Any]:
        """Return current hyperparameters."""
        if self.model is not None:
            params = self.model.get_params()
            params['class_weight'] = self._params.get('class_weight')
            return params
        return self._params

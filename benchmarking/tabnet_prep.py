"""
TabNet Preprocessing Helper
===========================
Leakage-safe tabular prep for TabNet:
  - train-fitted median imputation
  - float32 cast
  - categorical index/dim metadata (for embedding layers)

All transforms are fit on training data only and applied to val/test.
"""

from dataclasses import dataclass, field
from typing import List, Tuple

import numpy as np
import pandas as pd


@dataclass
class TabNetMeta:
    """Metadata produced by TabNetPrep; consumed by TabNetModel."""
    feature_names: List[str] = field(default_factory=list)
    cat_idxs: List[int] = field(default_factory=list)
    cat_dims: List[int] = field(default_factory=list)
    # Medians are stored for reproducibility/inspection
    _medians: "pd.Series" = field(default=None, repr=False)


class TabNetPrep:
    """Fit-on-train, transform-all preprocessor for TabNet inputs.

    The current TwiBot-20 feature pipeline produces all-numeric columns;
    cat_idxs and cat_dims therefore default to empty lists.  The hook is
    preserved so categorical embeddings can be added without rework.
    """

    def __init__(self):
        self._medians: pd.Series | None = None
        self._feature_names: List[str] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit_transform(
        self, X_train: pd.DataFrame | np.ndarray
    ) -> Tuple[np.ndarray, TabNetMeta]:
        """Fit on training data and return (float32 array, metadata)."""
        X = self._to_frame(X_train)
        self._feature_names = X.columns.tolist()
        self._medians = X.median(numeric_only=True)
        meta = TabNetMeta(
            feature_names=self._feature_names,
            cat_idxs=[],
            cat_dims=[],
            _medians=self._medians,
        )
        return self._apply(X), meta

    def transform(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        """Apply fitted transforms to val/test data."""
        if self._medians is None:
            raise RuntimeError("TabNetPrep.fit_transform() must be called first.")
        return self._apply(self._to_frame(X))

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _apply(self, X: pd.DataFrame) -> np.ndarray:
        X = X.fillna(self._medians).fillna(0)
        return X.to_numpy(dtype=np.float32)

    @staticmethod
    def _to_frame(X: pd.DataFrame | np.ndarray) -> pd.DataFrame:
        if isinstance(X, pd.DataFrame):
            return X.copy()
        return pd.DataFrame(X.astype(np.float64))

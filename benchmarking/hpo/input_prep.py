"""Per-model train/val/test prep (scaling, TabNetPrep, or pass-through)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Union

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from benchmarking.tabnet_prep import TabNetMeta, TabNetPrep

SCALED_MODELS = frozenset({"logistic_regression", "svm"})


@dataclass
class PreparedModelInputs:
    X_train: np.ndarray
    X_val: np.ndarray
    X_test: np.ndarray
    scaler: Optional[StandardScaler]
    tabnet_meta: Optional[TabNetMeta]


def _to_numpy(X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
    if hasattr(X, "values"):
        return np.asarray(X.values, dtype=np.float64)
    return np.asarray(X, dtype=np.float64)


def build_model_inputs(
    model_name: str,
    X_train: Union[np.ndarray, pd.DataFrame],
    X_val: Union[np.ndarray, pd.DataFrame],
    X_test: Union[np.ndarray, pd.DataFrame],
    *,
    enable_scaling: bool,
) -> PreparedModelInputs:
    if model_name == "tabnet":
        prep = TabNetPrep()
        X_tr, meta = prep.fit_transform(X_train)
        X_v = prep.transform(X_val)
        X_te = prep.transform(X_test)
        return PreparedModelInputs(
            X_train=X_tr,
            X_val=X_v,
            X_test=X_te,
            scaler=None,
            tabnet_meta=meta,
        )

    X_tr = _to_numpy(X_train)
    X_v = _to_numpy(X_val)
    X_te = _to_numpy(X_test)

    if enable_scaling and model_name in SCALED_MODELS:
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X_tr)
        X_v = scaler.transform(X_v)
        X_te = scaler.transform(X_te)
        return PreparedModelInputs(
            X_train=X_tr,
            X_val=X_v,
            X_test=X_te,
            scaler=scaler,
            tabnet_meta=None,
        )

    return PreparedModelInputs(
        X_train=X_tr,
        X_val=X_v,
        X_test=X_te,
        scaler=None,
        tabnet_meta=None,
    )

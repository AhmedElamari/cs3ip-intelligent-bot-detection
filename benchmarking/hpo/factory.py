"""Single factory for model wrappers (HPO, training, benchmark)."""
from __future__ import annotations

from typing import Any, Dict, Optional, Union

from models import MODEL_REGISTRY, BaseModel
from benchmarking.tabnet_prep import TabNetMeta


def build_model(
    name: str,
    params: Dict[str, Any],
    *,
    class_weights: Optional[Union[str, Dict[int, float]]] = None,
    tabnet_meta: Optional[TabNetMeta] = None,
) -> BaseModel:
    if name not in MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model: {name!r}. Available: {list(MODEL_REGISTRY.keys())}"
        )

    merged = dict(params)
    if class_weights is not None:
        merged["class_weight"] = class_weights

    if name == "tabnet" and tabnet_meta is not None:
        merged["cat_idxs"] = tabnet_meta.cat_idxs
        merged["cat_dims"] = tabnet_meta.cat_dims

    cls = MODEL_REGISTRY[name]
    return cls(**merged)

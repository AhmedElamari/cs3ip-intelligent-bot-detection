"""Hyperparameter optimisation (HPO) package."""
from benchmarking.hpo.contracts import HPOEntry, HPOResultV1
from benchmarking.hpo.registry import get, iter_entries, list_registered, register
from benchmarking.hpo.service import (
    HPOCliOverrides,
    merge_hpo_into_config_params,
    optimize_model,
    require_optuna,
    require_tabnet_dl,
    resolve_hpo,
)

__all__ = [
    "HPOResultV1",
    "HPOEntry",
    "get",
    "register",
    "list_registered",
    "iter_entries",
    "optimize_model",
    "resolve_hpo",
    "merge_hpo_into_config_params",
    "HPOCliOverrides",
    "require_optuna",
    "require_tabnet_dl",
]

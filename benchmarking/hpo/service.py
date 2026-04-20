"""Optuna HPO: optimize_model, resolve_hpo."""
from __future__ import annotations

import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Union

import numpy as np

from config import Config
from benchmarking.hpo.cache import (
    atomic_write_json,
    cache_path_for_signature,
    compute_signature,
    read_hpo_json,
)
from benchmarking.hpo.factory import build_model
from benchmarking.hpo.input_prep import build_model_inputs
from benchmarking.hpo.registry import get as get_hpo_entry


def require_optuna() -> Any:
    try:
        import optuna
    except ImportError as exc:
        raise ImportError(
            "optuna is not installed. Install with: pip install -r requirements.txt"
        ) from exc
    return optuna


def require_tabnet_dl() -> None:
    try:
        from pytorch_tabnet.tab_model import TabNetClassifier  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "pytorch-tabnet is not installed. Install with: pip install -r requirements-dl.txt"
        ) from exc


def _compute_virtual_batch_size(batch_size: int, ratio: int) -> int:
    return max(8, batch_size // ratio)


def _postprocess_tabnet_best_params(best_params: dict[str, Any]) -> dict[str, Any]:
    bp = dict(best_params)
    ratio = bp.get("virtual_batch_size_ratio")
    if ratio is not None:
        bp["virtual_batch_size"] = _compute_virtual_batch_size(
            int(bp.get("batch_size", 1024)), int(ratio)
        )
    return bp


def _device_string() -> str:
    try:
        import torch
    except ImportError:
        return "cpu"
    return "cuda" if torch.cuda.is_available() else "cpu"


def _validated_trial_count(n_trials: int) -> int:
    if n_trials < 1:
        raise ValueError("HPO trial count must be at least 1.")
    return n_trials


def _fit_and_val_f1(
    model_name: str,
    merged_params: dict[str, Any],
    prep: Any,
    y_train: np.ndarray,
    y_val: np.ndarray,
    feature_names: Optional[list[str]],
    trial: Any,
    entry: Any,
    optuna_mod: Any,
) -> float:
    cw = merged_params.get("class_weight")
    model = build_model(
        model_name,
        merged_params,
        class_weights=cw,
        tabnet_meta=prep.tabnet_meta,
    )
    fit_names = feature_names
    if model_name == "tabnet" and not fit_names and prep.tabnet_meta is not None:
        fit_names = prep.tabnet_meta.feature_names

    if hasattr(model, "prepare_eval_set"):
        model.prepare_eval_set(prep.X_val, y_val)

    model.fit(prep.X_train, y_train, feature_names=fit_names)
    score = float(entry.score_fn(model, prep.X_val, y_val))
    if model_name == "tabnet":
        trial.report(score, step=0)
        if trial.should_prune():
            raise optuna_mod.TrialPruned()
    return score


def optimize_model(
    model_name: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    *,
    config: Config,
    n_trials: int,
    seed: int,
    enable_scaling: bool,
    class_weights: Optional[Union[str, Dict[int, float]]] = None,
    feature_names: Optional[list[str]] = None,
    output_path: Optional[Path] = None,
    device: str = "auto",
    entry: Optional[Any] = None,
) -> dict[str, Any]:
    """
    Run Optuna HPO for ``model_name``; return HPOResultV1 dict.
    """
    n_trials = _validated_trial_count(int(n_trials))
    optuna = require_optuna()
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    entry = entry or get_hpo_entry(model_name)
    if entry.requires_dl:
        require_tabnet_dl()

    base_params = config.get_model_params(model_name)
    resolved_device = device
    if model_name == "tabnet":
        resolved_device = _device_string() if device == "auto" else device
        base_params = {**base_params, "device_name": resolved_device}
    if class_weights is not None:
        base_params = {**base_params, "class_weight": class_weights}

    y_tr = np.asarray(y_train, dtype=int)
    y_v = np.asarray(y_val, dtype=int)
    prep = build_model_inputs(
        model_name,
        X_train,
        X_val,
        X_val,
        enable_scaling=enable_scaling,
    )

    def objective(trial: Any) -> float:
        suggested = entry.suggest_fn(trial)
        trial.set_user_attr("model_params", dict(suggested))
        merged = {**base_params, **suggested}
        return _fit_and_val_f1(
            model_name,
            merged,
            prep,
            y_tr,
            y_v,
            feature_names,
            trial,
            entry,
            optuna,
        )

    sampler = optuna.samplers.TPESampler(seed=seed)
    pruner = None
    if entry.pruner_factory is not None:
        pruner = entry.pruner_factory()

    study = optuna.create_study(
        direction="maximize",
        sampler=sampler,
        pruner=pruner,
    )

    warnings_list: list[str] = []
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
        warnings_list = [str(w.message) for w in caught]

    best = study.best_trial
    best_params = dict(best.user_attrs.get("model_params") or best.params)
    if model_name == "tabnet":
        best_params = _postprocess_tabnet_best_params(best_params)

    result: dict[str, Any] = {
        "schema_version": "HPOResultV1",
        "status": "ok",
        "best_params": best_params,
        "best_score": float(best.value),
        "trial_count": len(study.trials),
        "metric": "val_f1",
        "seed": seed,
        "warnings": warnings_list[:10],
        "model_name": model_name,
        "search_space_version": entry.search_space_version,
    }
    if model_name == "tabnet":
        result["device"] = resolved_device

    if output_path is not None:
        atomic_write_json(Path(output_path), result)

    return result


@dataclass
class HPOCliOverrides:
    no_tune: bool = False
    retune: bool = False
    hpo_trials: Optional[int] = None


def resolve_hpo(
    model_name: str,
    config: Config,
    *,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    feature_names_ordered: list[str],
    data_dir: Path,
    enable_scaling: bool,
    class_weights: Optional[Union[str, Dict[int, float]]] = None,
    cli: Optional[HPOCliOverrides] = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    cli = cli or HPOCliOverrides()
    hpo_cfg = config.get("hpo") or {}
    enabled = hpo_cfg.get("enabled", True)
    reuse_cache = hpo_cfg.get("reuse_cache", True)
    cache_dir = Path(hpo_cfg.get("cache_dir", "results/hpo_cache"))
    seed = int(hpo_cfg.get("sampler_seed", config.get("random_state", 2112)))
    fail_fast = hpo_cfg.get("fail_fast", True)

    trials_default = (hpo_cfg.get("trials_per_model") or {}).get(model_name, 30)
    n_trials = _validated_trial_count(
        int(cli.hpo_trials if cli.hpo_trials is not None else trials_default)
    )

    audit: dict[str, Any] = {
        "model_name": model_name,
        "cache_hit": False,
        "artifact": None,
        "search_space_version": None,
        "trial_count": 0,
        "best_score": None,
        "skipped": False,
    }

    if cli.no_tune and cli.retune:
        raise ValueError("Cannot combine --no-tune with --retune")

    if cli.no_tune or not enabled:
        audit["skipped"] = True
        return (
            {
                "schema_version": "HPOResultV1",
                "status": "skipped",
                "best_params": {},
                "best_score": float("nan"),
                "trial_count": 0,
                "metric": "val_f1",
                "seed": seed,
                "warnings": [],
                "model_name": model_name,
                "search_space_version": "none",
            },
            audit,
        )

    try:
        entry = get_hpo_entry(model_name)
    except KeyError as exc:
        if fail_fast:
            raise ValueError(
                f"No HPO registry entry for model {model_name!r}. "
                "Disable tuning with --no-tune or add a registry entry."
            ) from exc
        audit["skipped"] = True
        audit["search_space_version"] = "missing"
        return (
            {
                "schema_version": "HPOResultV1",
                "status": "skipped",
                "best_params": {},
                "best_score": float("nan"),
                "trial_count": 0,
                "metric": "val_f1",
                "seed": seed,
                "warnings": [f"No HPO registry entry for model {model_name!r}."],
                "model_name": model_name,
                "search_space_version": "missing",
            },
            audit,
        )
    if entry.requires_dl:
        require_tabnet_dl()

    sig = compute_signature(
        model_name,
        config,
        feature_names_ordered,
        Path(data_dir),
        entry.search_space_version,
        metric=str(hpo_cfg.get("metric", "val_f1")),
    )
    audit["search_space_version"] = entry.search_space_version
    out_path = cache_path_for_signature(cache_dir, model_name, sig)
    audit["artifact"] = str(out_path)

    if reuse_cache and not cli.retune and out_path.is_file():
        data = read_hpo_json(out_path)
        audit["cache_hit"] = True
        audit["trial_count"] = int(data.get("trial_count", 0))
        audit["best_score"] = data.get("best_score")
        return data, audit

    require_optuna()

    result = optimize_model(
        model_name,
        X_train,
        y_train,
        X_val,
        y_val,
        config=config,
        n_trials=n_trials,
        seed=seed,
        enable_scaling=enable_scaling,
        class_weights=class_weights,
        feature_names=feature_names_ordered,
        output_path=out_path,
        entry=entry,
    )
    audit["trial_count"] = int(result.get("trial_count", n_trials))
    audit["best_score"] = result.get("best_score")
    return result, audit


def merge_hpo_into_config_params(
    config: Config,
    model_name: str,
    best_params: dict[str, Any],
) -> None:
    key = f"models.{model_name}.params"
    current = config.get(key) or {}
    bp = dict(best_params)
    if model_name == "tabnet":
        from benchmarking.tabnet_optuna import best_params_for_tabnetmodel

        bp = best_params_for_tabnetmodel({"best_params": bp})
    merged = {**current, **bp}
    config.set(key, merged)

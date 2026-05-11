"""Multi-seed retraining for top scoreboard models (stability evidence)."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from config import Config
from benchmarking import ModelBenchmark
from benchmarking.dissertation_scoreboard import build_scoreboard
from benchmarking.model_factory import create_models


def run_multi_seed_retraining(
    *,
    benchmark: ModelBenchmark,
    config: Config,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    feature_names: list,
    output_dir: Path,
    seeds: list[int],
    top_k: int,
    enable_scaling: bool,
    test_metadata: pd.DataFrame | None = None,
) -> dict[str, Any]:
    """Retrain only top-K models (scoreboard order) for each seed; write CSV/JSON summaries."""
    sb = build_scoreboard(benchmark)
    if sb.empty:
        payload: dict[str, Any] = {
            "schema_version": "MultiSeedRetrainingV1",
            "status": "skipped",
            "reason": "empty_scoreboard",
            "top_models": [],
            "seeds": list(seeds),
            "rank_source": "dissertation_scoreboard",
            "rows": [],
        }
        (output_dir / "multi_seed_retraining.json").write_text(
            json.dumps(payload, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        return payload

    top_models = sb["Model"].head(int(top_k)).astype(str).tolist()
    rows: list[dict[str, Any]] = []

    for seed in seeds:
        cfg = copy.deepcopy(config)
        cfg.set("random_state", int(seed))
        for mn in list(cfg.get("models", {}).keys()):
            cfg.set(f"models.{mn}.enabled", mn in top_models)
        models = create_models(cfg)
        mb = ModelBenchmark(
            models=models,
            experiment_name=f"multi_seed_{seed}",
        )
        if test_metadata is not None and hasattr(mb, "set_test_metadata"):
            mb.set_test_metadata(test_metadata)
        mb.run_benchmark(
            X_train,
            y_train,
            X_val,
            y_val,
            X_test,
            y_test,
            feature_names=feature_names,
            verbose=False,
            compute_statistics=False,
            enable_scaling=enable_scaling,
        )
        for model_name in top_models:
            r = mb.results[model_name]
            row: dict[str, Any] = {
                "model": model_name,
                "seed": int(seed),
                "training_time": r["training_time"],
            }
            for k, v in r["val_metrics"].items():
                row[f"val_{k}"] = v
            for k, v in r["test_metrics"].items():
                row[f"test_{k}"] = v
            rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(output_dir / "multi_seed_retraining.csv", index=False)

    numeric = [c for c in df.select_dtypes(include=[np.number]).columns if c != "seed"]
    if numeric:
        grouped = df.groupby("model")[numeric].agg(["mean", "std", "min", "max"])
        grouped.columns = [f"{a}_{b}" for a, b in grouped.columns]
        summary = grouped.reset_index()
    else:
        summary = pd.DataFrame({"model": top_models})
    summary.to_csv(output_dir / "multi_seed_summary.csv", index=False)

    payload = {
        "schema_version": "MultiSeedRetrainingV1",
        "status": "ok",
        "top_models": top_models,
        "seeds": [int(s) for s in seeds],
        "rank_source": "dissertation_scoreboard",
        "rows": rows,
    }
    (output_dir / "multi_seed_retraining.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str),
        encoding="utf-8",
    )
    return payload

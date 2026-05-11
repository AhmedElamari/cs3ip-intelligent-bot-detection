"""Multi-seed utilities: --seeds aggregation and optional top-model retraining."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

from config import Config
from benchmarking import ModelBenchmark
from benchmarking.dissertation_scoreboard import build_scoreboard
from benchmarking.model_factory import create_models

METRIC_KEYS = ("f1_macro", "pr_auc", "mcc", "balanced_accuracy")

_DISPLAY = {
    "f1_macro": "F1-Macro",
    "pr_auc": "PR-AUC",
    "mcc": "MCC",
    "balanced_accuracy": "Balanced Accuracy",
}


def validate_seeds(seeds: Sequence[int]) -> list[int]:
    """Require at least 3 unique integer seeds."""
    if len(seeds) < 3:
        raise ValueError("--seeds requires at least 3 distinct integer seeds.")
    out: list[int] = []
    for s in seeds:
        if not isinstance(s, (int, np.integer)):
            raise TypeError("Each seed must be an integer.")
        out.append(int(s))
    if len(set(out)) != len(out):
        raise ValueError("--seeds must be unique (no duplicates).")
    return out


def extract_per_seed_rows(
    benchmark: ModelBenchmark, *, seed: int
) -> list[dict[str, Any]]:
    """One dict per model with test point metrics."""
    rows: list[dict[str, Any]] = []
    for model_name in sorted(benchmark.results.keys()):
        tm = benchmark.results[model_name].get("test_metrics") or {}
        row: dict[str, Any] = {"seed": seed, "model": model_name}
        for key in METRIC_KEYS:
            row[key] = float(tm.get(key, float("nan")))
        rows.append(row)
    return rows


def build_summary(per_seed_rows: list[dict[str, Any]]) -> pd.DataFrame:
    """Mean and sample std (ddof=1) per model; sort by f1_macro mean, pr_auc mean, model."""
    if not per_seed_rows:
        return pd.DataFrame()
    df = pd.DataFrame(per_seed_rows)
    groups = df.groupby("model", sort=False)
    summary_rows = []
    for model_name, g in groups:
        n = int(len(g))
        entry: dict[str, Any] = {"model": model_name, "n_seeds": n}
        for key in METRIC_KEYS:
            series = pd.to_numeric(g[key], errors="coerce")
            mu = float(series.mean(skipna=True))
            if n <= 1 or series.dropna().shape[0] <= 1:
                std_val = float("nan")
            else:
                std_val = float(series.std(ddof=1, skipna=True))
            entry[f"{key}_mean"] = mu
            entry[f"{key}_std"] = std_val
            entry[f"{key}_mean_pm_std"] = _format_mean_pm_std(mu, std_val)
        summary_rows.append(entry)
    summary = pd.DataFrame(summary_rows)
    summary["_sort_fm"] = summary["f1_macro_mean"].fillna(-np.inf)
    summary["_sort_pr"] = summary["pr_auc_mean"].fillna(-np.inf)
    summary = summary.sort_values(
        ["_sort_fm", "_sort_pr", "model"], ascending=[False, False, True]
    )
    summary = summary.drop(columns=["_sort_fm", "_sort_pr"])
    summary = summary.reset_index(drop=True)
    return summary


def _format_mean_pm_std(mu: float, sigma: float) -> str:
    if not np.isfinite(mu):
        return "—"
    if not np.isfinite(sigma):
        return f"{mu:.4f}"
    return f"{mu:.4f} ± {sigma:.4f}"


def write_multi_seed_outputs(
    per_seed_rows: list[dict[str, Any]],
    parent_dir: Path,
) -> None:
    """Write multi_seed_results.csv, multi_seed_summary.csv, multi_seed_summary.md."""
    parent_dir = Path(parent_dir)
    parent_dir.mkdir(parents=True, exist_ok=True)
    results_df = pd.DataFrame(per_seed_rows)
    if not results_df.empty:
        results_df = results_df.sort_values(["seed", "model"])
    results_df.to_csv(parent_dir / "multi_seed_results.csv", index=False)

    summary = build_summary(per_seed_rows)
    summary.to_csv(parent_dir / "multi_seed_summary.csv", index=False)

    md_lines = [
        "Mean ± sample standard deviation across independent training seeds. "
        "This summarizes **retraining / initialization randomness**, not test-set bootstrap "
        "resampling uncertainty (see `dissertation_statistical_summary.*` when run with bootstrap CIs).",
        "",
    ]
    if summary.empty:
        md_lines.append("_No rows to summarize._")
    else:
        table = pd.DataFrame(
            {
                "model": summary["model"],
                "n_seeds": summary["n_seeds"],
                **{
                    f"{_DISPLAY[key]} (mean ± std)": summary[f"{key}_mean_pm_std"]
                    for key in METRIC_KEYS
                },
            }
        )
        md_lines.extend(_dataframe_to_markdown(table))
        md_lines.append("")
    (parent_dir / "multi_seed_summary.md").write_text(
        "\n".join(md_lines), encoding="utf-8"
    )


def _dataframe_to_markdown(frame: pd.DataFrame) -> list[str]:
    header = "| " + " | ".join(str(c) for c in frame.columns) + " |"
    sep = "| " + " | ".join("---" for _ in frame.columns) + " |"
    rows = []
    for _, row in frame.iterrows():
        rows.append("| " + " | ".join(str(v) for v in row) + " |")
    return [header, sep, *rows]


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
    feature_names: list[str],
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
        if test_metadata is not None:
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

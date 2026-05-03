"""Chronological concept-drift splits and delta tables for benchmark outputs."""

from __future__ import annotations

import math
from typing import Any, Dict

import numpy as np
import pandas as pd

from benchmarking.dissertation_scoreboard import METRIC_COLUMNS
from pipeline_utils import time_stratified_split


def build_temporal_split_dict(
    raw_splits: Dict[str, pd.DataFrame],
    *,
    val_size: float,
    test_size: float,
    time_col: str,
    random_state: int,
    min_samples_per_split: int = 1,
) -> Dict[str, pd.DataFrame]:
    """Combine official splits and re-partition by time (oldest→train, newest→test)."""
    missing = [k for k in ("train", "val", "test") if k not in raw_splits]
    if missing:
        raise KeyError(f"raw_splits missing keys: {missing}")
    combined = pd.concat(
        [raw_splits["train"], raw_splits["val"], raw_splits["test"]],
        ignore_index=True,
    )
    if "label" not in combined.columns:
        raise ValueError("Combined data has no label column.")
    combined = combined.dropna(subset=["label"])
    if combined.empty:
        raise ValueError("No labeled rows after dropping missing labels.")
    train_df, val_df, test_df = time_stratified_split(
        combined,
        val_size=val_size,
        test_size=test_size,
        time_col=time_col,
        random_state=random_state,
        min_samples_per_split=min_samples_per_split,
    )
    return {"train": train_df, "val": val_df, "test": test_df}


def _label_rate(series: pd.Series) -> float:
    if series.empty:
        return float("nan")
    return float(series.astype(float).mean())


def format_protocol_note(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    *,
    time_col: str,
    reference_date_policy: str,
) -> str:
    lines = [
        "Chronological concept-drift benchmark (by account creation time, not tweet time).",
        f"Reference date policy for age features: {reference_date_policy}.",
        f"Splits: train n={len(train_df)}, val n={len(val_df)}, drift-test n={len(test_df)}.",
        (
            f"Class rate (label=1): train {_label_rate(train_df['label']):.3f}, "
            f"val {_label_rate(val_df['label']):.3f}, test {_label_rate(test_df['label']):.3f}."
        ),
    ]
    if time_col in test_df.columns:
        t_test = pd.to_datetime(test_df[time_col], errors="coerce")
        valid = t_test.dropna()
        if not valid.empty:
            lines.append(
                f"Drift-test window ({time_col}): {valid.min().date()} – {valid.max().date()}."
            )
    return "\n".join(lines)


def build_concept_drift_delta(
    main_scoreboard: pd.DataFrame,
    drift_scoreboard: pd.DataFrame,
) -> pd.DataFrame:
    """Per-model metric change: drift_test minus standard test (negative ≈ degradation)."""
    perf = [d for d, _ in METRIC_COLUMNS]
    main_i = main_scoreboard.set_index("Model")
    drift_i = drift_scoreboard.set_index("Model")
    rows = []
    for model in drift_i.index:
        if model not in main_i.index:
            continue
        row: Dict[str, Any] = {"Model": model}
        for col in perf:
            m = main_i.loc[model, col]
            d = drift_i.loc[model, col]
            key = f"{col} Δ (drift−baseline)"
            if pd.isna(m) or pd.isna(d) or not math.isfinite(float(m)) or not math.isfinite(float(d)):
                row[key] = np.nan
            else:
                row[key] = round(float(d) - float(m), 3)
        rows.append(row)
    return pd.DataFrame(rows)

"""Internal helpers for stable persisted-output formatting."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


_TIME_COLUMNS = {
    "training_time",
    "training_time_s",
    "training_time_seconds",
    "training time (s)",
}
_INTEGER_COLUMNS = {
    "rank",
    "rows",
    "attacked_true_bots",
    "baseline_detected_bots",
    "flips_to_human",
    "row_index",
    "mcnemar_b",
    "mcnemar_c",
}


def _normalize_name(name: Any) -> str:
    return str(name).strip().lower()


def _precision_for_name(name: Any) -> int:
    return 2 if _normalize_name(name) in _TIME_COLUMNS else 4


def _is_integer_column(name: Any) -> bool:
    return _normalize_name(name) in _INTEGER_COLUMNS


def _round_float(value: Any, precision: int) -> Any:
    if isinstance(value, (bool, np.bool_)) or value is None:
        return value
    if isinstance(value, (float, np.floating)):
        if not np.isfinite(value):
            return value
        return round(float(value), precision)
    return value


def format_frame_for_export(frame: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with persisted-output precision applied by column."""
    if frame is None or frame.empty:
        return frame

    export_df = frame.copy()
    for column in export_df.columns:
        if _is_integer_column(column):
            series = export_df[column]
            if pd.api.types.is_numeric_dtype(series):
                finite = series.dropna()
                if finite.empty or np.allclose(finite, np.round(finite), atol=0.0):
                    export_df[column] = series.round().astype("Int64")
            continue

        if pd.api.types.is_float_dtype(export_df[column]):
            precision = _precision_for_name(column)
            export_df[column] = export_df[column].map(
                lambda value, p=precision: _round_float(value, p)
            )

    return export_df


def format_payload_for_export(value: Any, key: str | None = None) -> Any:
    """Recursively apply persisted-output precision to JSON-safe payloads."""
    if isinstance(value, dict):
        return {
            item_key: format_payload_for_export(item_value, key=item_key)
            for item_key, item_value in value.items()
        }
    if isinstance(value, list):
        return [format_payload_for_export(item, key=key) for item in value]
    if key and _is_integer_column(key):
        return int(value) if value is not None and not pd.isna(value) else value
    if isinstance(value, (float, np.floating)):
        return _round_float(value, _precision_for_name(key or ""))
    if hasattr(value, "item") and not isinstance(value, (str, bytes)):
        return format_payload_for_export(value.item(), key=key)
    return value

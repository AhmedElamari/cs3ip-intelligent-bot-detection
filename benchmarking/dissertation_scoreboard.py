"""Table 8.2 scoreboard — ranked by F1-Macro (both classes matter under imbalance)."""

from __future__ import annotations

import math
from typing import Any, List, Tuple

import numpy as np
import pandas as pd

CAPTION = (
    "Baseline test-set performance of all benchmarked models under the standard "
    "non-adversarial evaluation condition. Higher values indicate better predictive "
    "performance; lower training times indicate greater computational efficiency."
)

METRIC_COLUMNS: List[Tuple[str, str]] = [
    ("Precision", "precision"),
    ("Recall", "recall"),
    ("F1-Macro", "f1_macro"),
    ("F1-Weighted", "f1_weighted"),
    ("PR-AUC", "pr_auc"),
    ("ROC-AUC", "roc_auc"),
    ("MCC", "mcc"),
    ("Balanced Accuracy", "balanced_accuracy"),
]

SCOREBOARD_METRIC_KEYS = tuple(k for _, k in METRIC_COLUMNS)

DISPLAY_ORDER = ["Rank", "Model"] + [d for d, _ in METRIC_COLUMNS] + ["Train Time (s)"]
_TIME = "Train Time (s)"


def _metric_key(dataset: str) -> str:
    return f"{dataset}_metrics"


def _finite_sort(v: object) -> float:
    try:
        x = float(v)
    except (TypeError, ValueError):
        return float("-inf")
    return x if math.isfinite(x) else float("-inf")


def build_scoreboard(benchmark: Any, dataset: str = "test") -> pd.DataFrame:
    mkey = _metric_key(dataset)
    rows = []
    for name, result in (getattr(benchmark, "results", None) or {}).items():
        tm = result.get(mkey, {})
        line = {"Model": name, _TIME: result.get("training_time", float("nan"))}
        for disp, src in METRIC_COLUMNS:
            line[disp] = tm.get(src, float("nan"))
        rows.append(line)

    # Primary rank: F1-Macro; ROC-AUC tie-break (class-balanced headline for thesis).
    rows.sort(
        key=lambda r: (-_finite_sort(r["F1-Macro"]), -_finite_sort(r["ROC-AUC"]), str(r["Model"]))
    )

    out = []
    for rank, row in enumerate(rows, start=1):
        entry = {"Rank": rank, "Model": row["Model"]}
        for disp, _ in METRIC_COLUMNS:
            v = row[disp]
            entry[disp] = round(float(v), 3) if _finite_sort(v) > float("-inf") else np.nan
        tt = row[_TIME]
        entry[_TIME] = round(float(tt), 2) if _finite_sort(tt) > float("-inf") else np.nan
        out.append(entry)

    return pd.DataFrame(out, columns=DISPLAY_ORDER)


def _best_mask(series: pd.Series) -> np.ndarray:
    s = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)
    if s.size == 0:
        return np.array([], dtype=bool)
    finite = np.isfinite(s)
    if not finite.any():
        return np.zeros(len(s), dtype=bool)
    mx = float(np.nanmax(s))
    return finite & np.isclose(s, mx, rtol=0, atol=1e-12)


def _format_metric_cell(v: object) -> str:
    if v is None or (isinstance(v, float) and (math.isnan(v) or not math.isfinite(v))):
        return "—"
    return f"{float(v):.3f}"


def _format_time_cell(v: object) -> str:
    if v is None or (isinstance(v, float) and (math.isnan(v) or not math.isfinite(v))):
        return "—"
    return f"{float(v):.2f}"


def to_markdown(df: pd.DataFrame, caption: str | None = None) -> str:
    perf = [d for d, _ in METRIC_COLUMNS]
    masks = {c: _best_mask(df[c]) for c in perf}
    header = "| " + " | ".join(df.columns) + " |"
    sep = "| " + " | ".join("---" for _ in df.columns) + " |"
    cap = CAPTION if caption is None else caption
    lines = [cap, "", header, sep]
    for i in range(len(df)):
        cells = []
        for col in df.columns:
            v = df.iloc[i][col]
            if col == "Rank":
                cells.append(str(int(v)) if pd.notna(v) else "—")
            elif col == "Model":
                cells.append(str(v))
            elif col in masks and masks[col][i]:
                cells.append(f"**{_format_metric_cell(v)}**")
            elif col == _TIME:
                cells.append(_format_time_cell(v))
            else:
                cells.append(_format_metric_cell(v))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def _latex_escape_model(name: str) -> str:
    return str(name).replace("_", r"\_")


def to_latex(df: pd.DataFrame) -> str:
    perf = [d for d, _ in METRIC_COLUMNS]
    masks = {c: _best_mask(df[c]) for c in perf}
    ncols = len(df.columns)
    spec = f"@{{}}{'rl' + 'r' * (ncols - 2)}@{{}}"
    caption_tex = CAPTION.replace("%", r"\%")
    out = [
        "% Requires \\usepackage{booktabs}",
        "% " + caption_tex,
        "",
        f"\\begin{{tabular}}{{{spec}}}",
        "\\toprule",
        " & ".join(str(c).replace("_", r"\_") for c in df.columns) + r"\\",
        "\\midrule",
    ]
    for i in range(len(df)):
        parts = []
        for j, col in enumerate(df.columns):
            v = df.iloc[i, j]
            if col == "Rank":
                parts.append(str(int(v)) if pd.notna(v) else "—")
            elif col == "Model":
                parts.append(_latex_escape_model(v))
            elif col in masks and masks[col][i]:
                parts.append(r"\textbf{" + _format_metric_cell(v) + "}")
            elif col == _TIME:
                parts.append(_format_time_cell(v))
            else:
                parts.append(_format_metric_cell(v))
        out.append(" & ".join(parts) + r"\\")
    out.extend(["\\bottomrule", "\\end{tabular}"])
    return "\n".join(out)

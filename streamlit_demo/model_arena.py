"""Tab 1 — Model Arena: HTML builders and Streamlit hook."""

from __future__ import annotations

import html
from typing import Any, Mapping, Sequence

METRIC_KEYS = ("f1", "pr_auc", "roc_auc", "mcc", "train_seconds")
METRIC_LABELS = {
    "f1": "F1",
    "pr_auc": "PR-AUC",
    "roc_auc": "ROC-AUC",
    "mcc": "MCC",
    "train_seconds": "Train (s)",
}

ACCENT = "#6CC8BE"
GREEN = "#4DA87A"
AMBER = "#C8A44A"
TEXT_MID = "#949bb2"


def compute_best_per_metric(rows: Sequence[Mapping[str, Any]]) -> dict[str, float]:
    best: dict[str, float] = {}
    for key in METRIC_KEYS:
        vals = [float(r[key]) for r in rows]
        best[key] = min(vals) if key == "train_seconds" else max(vals)
    return best


def bar_width_pct(value: float, metric_key: str) -> float:
    denom = 50.0 if metric_key == "train_seconds" else 1.0
    return min(100.0, max(0.0, (float(value) / denom) * 100.0))


def _metric_cell(value: float, metric_key: str, is_best: bool) -> str:
    if metric_key == "train_seconds" and is_best:
        color = GREEN
    elif is_best:
        color = ACCENT
    else:
        color = TEXT_MID
    w = bar_width_pct(value, metric_key)
    shown = f"{value:.3f}"
    weight = "600" if is_best else "400"
    return (
        '<div class="demo-metric-cell">'
        '<div class="demo-bar-track">'
        f'<div class="demo-bar-fill" style="width:{w:.1f}%;background:{color};"></div>'
        "</div>"
        f'<span class="demo-metric-val{" demo-metric-val--best" if is_best else ""}" '
        f'style="color:{color};font-weight:{weight};">{html.escape(shown)}</span>'
        "</div>"
    )


def _champion_row(rows: Sequence[Mapping[str, Any]]) -> Mapping[str, Any] | None:
    champs = [r for r in rows if r.get("champion")]
    return champs[0] if champs else None


def _winner_badge_text(champion_name: str) -> str:
    """Short badge for the winner card (uppercase mono-friendly)."""
    key = champion_name.strip().lower()
    if key == "logistic regression":
        return "LR"
    if key == "random forest":
        return "RF"
    if key == "decision tree":
        return "DT"
    if key == "xgboost":
        return "XGB"
    if key == "tabnet":
        return "TN"
    if key == "svm":
        return "SVM"
    return champion_name[:4].upper() if len(champion_name) > 4 else champion_name.upper()


def _winner_chips_from_champion(
    champion: Mapping[str, Any],
) -> tuple[tuple[str, str, str], tuple[str, str, str], tuple[str, str, str]]:
    """(label, formatted value, colour) tuples for F1 / PR-AUC / MCC."""
    f1 = float(champion["f1"])
    pr = float(champion["pr_auc"])
    mcc = float(champion["mcc"])
    return (
        ("F1", f"{f1:.3f}", ACCENT),
        ("PR-AUC", f"{pr:.3f}", GREEN),
        ("MCC", f"{mcc:.3f}", AMBER),
    )


def arena_provenance_footnote_html() -> str:
    """HTML footnote for Tab 1; lives here so `app.py` need not import a string constant from `data`."""
    from streamlit_demo.data import (
        DEMO_ARENA_BENCHMARK_REL,
        DEMO_RF_HPO_CACHE_REL,
        DEMO_RF_TUNED_XAI_REL,
    )

    return (
        f"Table metrics: <code>{DEMO_ARENA_BENCHMARK_REL}</code> "
        f"(per-model HPO audit in <code>results.json</code> there). "
        f"Tab 2 SHAP plot + Tab 3 live predictor: Optuna-tuned RF from <code>{DEMO_RF_TUNED_XAI_REL}</code> "
        f"(val F1 0.8467, 20 trials; cache <code>{DEMO_RF_HPO_CACHE_REL}</code>)."
    )


def build_model_arena_html(
    rows: Sequence[Mapping[str, Any]],
    dataset_meta: Mapping[str, str],
    *,
    footnote_html: str = "",
) -> str:
    best = compute_best_per_metric(rows)
    champ = _champion_row(rows)
    if champ is None:
        chips: tuple[tuple[str, str, str], ...] = ()
    else:
        chips = _winner_chips_from_champion(champ)
    chip_html = "".join(
        f'<div class="demo-winner-chip">'
        f'<div class="demo-winner-chip-val" style="color:{c};">{html.escape(v)}</div>'
        f'<div class="demo-winner-chip-lbl">{html.escape(lbl)}</div></div>'
        for lbl, v, c in chips
    )

    header_cells = "".join(
        f'<div class="demo-metric-th">{html.escape(METRIC_LABELS[k])}</div>' for k in METRIC_KEYS
    )

    body_rows: list[str] = []
    for row in rows:
        name = str(row["name"])
        champion = bool(row.get("champion"))
        row_cls = "demo-metric-row demo-metric-row--champion" if champion else "demo-metric-row"
        name_cls = "demo-model-name demo-model-name--strong" if champion else "demo-model-name demo-model-name--muted"
        dot = '<span class="demo-pulse-dot"></span>' if champion else ""
        tag = (
            '<span class="demo-best-tag">Best</span>'
            if champion
            else ""
        )
        metric_cells = "".join(
            _metric_cell(float(row[k]), k, float(row[k]) == best[k]) for k in METRIC_KEYS
        )
        body_rows.append(
            f'<div class="{row_cls}">'
            f'<div class="demo-model-name-wrap">{dot}'
            f'<span class="{name_cls}">{html.escape(name)}</span>{tag}</div>'
            f"{metric_cells}</div>"
        )

    meta_right = (
        '<div class="demo-meta-right">'
        f'<div>{html.escape(dataset_meta["account_count_label"])}</div>'
        f'<div class="demo-meta-ratio">{html.escape(dataset_meta["class_ratio_label"])}</div>'
        "</div>"
    )

    n_models = len(rows)
    if champ is not None:
        cname = str(champ["name"])
        badge = html.escape(_winner_badge_text(cname))
        winner_title = (
            f'<div class="demo-winner-title">{html.escape(cname)} - <span class="demo-winner-title-accent">'
            "Best performer</span></div>"
        )
        winner_sub = (
            '<div class="demo-winner-sub">XGBoost has the highest test F1 on this export; Random Forest is '
            "listed next as the dissertation interpretability anchor (best PR-AUC and ROC-AUC here, with "
            "clearer attributions than boosted trees). Pure F1 ranking still puts decision tree and SVM "
            "above Random Forest; see the full table.</div>"
        )
    else:
        badge = "-"
        winner_title = '<div class="demo-winner-title">No champion flagged</div>'
        winner_sub = '<div class="demo-winner-sub"></div>'

    return (
        '<div class="demo-page-body">'
        '<div class="demo-arena-intro">'
        "<div>"
        '<span class="demo-section-label">Section 01</span>'
        '<h1 class="demo-h1">Model Comparison</h1>'
        '<p class="demo-lede">Six models evaluated on the same held-out TwiBot-20 test set in this '
        f"benchmark export ({n_models} rows).</p>"
        f"</div>{meta_right}</div>"
        '<div class="demo-winner-card">'
        f'<div class="demo-winner-badge">{badge}</div>'
        '<div class="demo-winner-copy">'
        f"{winner_title}"
        f"{winner_sub}"
        "</div>"
        f'<div class="demo-winner-chips">{chip_html}</div></div>'
        '<div class="demo-metric-table">'
        '<div class="demo-metric-header-row">'
        '<div class="demo-metric-th">Model</div>'
        f"{header_cells}</div>"
        f'{"".join(body_rows)}</div>'
        '<div class="demo-f1-note">'
        '<div class="demo-f1-icon">?</div>'
        "<div>"
        '<span class="demo-f1-label">Why F1 over accuracy?</span>'
        '<div class="demo-f1-copy">TwiBot-20 is imbalanced — roughly 64% human, 36% bot. A naive '
        'classifier predicting "human" every time scores 64% accuracy yet detects zero bots. F1 and '
        "MCC penalise this, making them the authoritative metrics for bot detection tasks.</div>"
        "</div></div>"
        f'<p class="demo-arena-footnote" style="margin-top:16px;font-size:12px;color:{TEXT_MID};">'
        f"{footnote_html}</p>"
        "</div>"
    )


def render_model_arena() -> None:
    import streamlit as st

    from streamlit_demo.data import DEMO_DATASET_META, MODEL_ARENA_ROWS

    st.markdown(
        build_model_arena_html(
            MODEL_ARENA_ROWS,
            DEMO_DATASET_META,
            footnote_html=arena_provenance_footnote_html(),
        ),
        unsafe_allow_html=True,
    )

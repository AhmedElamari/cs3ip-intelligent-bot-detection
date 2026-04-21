"""Poster/dissertation matplotlib helpers for benchmark figures."""
from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch

from explainability.poster_shap import LABEL_MAP as FEATURE_LABELS

POSTER_RC: dict[str, Any] = {
    "font.size": 14,
    "axes.labelsize": 15,
    "axes.titlesize": 16,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 12,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
}
PALETTE = {
    "baseline": "#4C72B0",
    "cheap_only": "#DD8452",
    "realistic_mixed": "#C44E52",
    "cheap": "#DD8452",
    "expensive": "#C44E52",
}
SCENARIO_LABELS = {
    "baseline": "Baseline (clean)",
    "cheap_only": "Cheap attacks",
    "realistic_mixed": "Realistic mixed",
}
MODEL_LABELS = {
    "xgboost": "XGBoost",
    "random_forest": "Random Forest",
    "logistic_regression": "Logistic Regression",
    "svm": "SVM",
    "tabnet": "TabNet",
    "mlp": "MLP",
    "naive_bayes": "Naive Bayes",
    "decision_tree": "Decision Tree",
}


def pretty_model(name: str) -> str:
    return str(name).replace("_", " ").title()


def poster_style():
    return mpl.rc_context(POSTER_RC)


def save_poster_figure(fig: mpl.figure.Figure, output_dir: Path, stem: str) -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    png, pdf = output_dir / f"{stem}.png", output_dir / f"{stem}.pdf"
    fig.savefig(png, dpi=300, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    return png


def write_caption(output_dir: Path, stem: str, text: str) -> Path:
    path = Path(output_dir) / f"{stem}_caption.md"
    path.write_text(text.rstrip() + "\n", encoding="utf-8")
    return path


def degradation_matrix(df: pd.DataFrame, top_models: Sequence[str], scenarios: Sequence[str]):
    plot_df = df[df["model"].isin(top_models)]
    if plot_df.empty or "baseline" not in scenarios:
        return [], np.array([]), np.array([])
    f1_p = plot_df.pivot_table(index="model", columns="scenario", values="macro_f1", aggfunc="first")
    pr_p = plot_df.pivot_table(index="model", columns="scenario", values="pr_auc", aggfunc="first")
    if "baseline" not in f1_p.columns:
        return [], np.array([]), np.array([])
    models = [m for m in top_models if m in f1_p.index and pd.notna(f1_p.loc[m, "baseline"])]
    if not models:
        return [], np.array([]), np.array([])
    cols = list(scenarios)
    return list(models), f1_p.reindex(models, columns=cols).to_numpy(float), pr_p.reindex(models, columns=cols).to_numpy(float)


def plot_degradation(models: Sequence[str], scenarios: Sequence[str], f1: np.ndarray) -> mpl.figure.Figure:
    n, ns = len(models), len(scenarios)
    fig, ax = plt.subplots(figsize=(max(9.0, 2.5 * n), 5.2), constrained_layout=True)
    x, width = np.arange(n), 0.8 / max(ns, 1)
    bi = scenarios.index("baseline") if "baseline" in scenarios else 0
    for si, scen in enumerate(scenarios):
        off = (si - (ns - 1) / 2) * width
        col = f1[:, si]
        bars = ax.bar(
            x + off, col, width, color=PALETTE.get(scen, "#888888"),
            label=SCENARIO_LABELS.get(scen, scen), edgecolor="white", linewidth=0.6,
        )
        if scen == "baseline":
            for b in bars:
                b.set_hatch("//")
        ax.bar_label(bars, labels=[f"{v:.2f}" if np.isfinite(v) else "" for v in col], padding=2, fontsize=10)
        if si != bi:
            d = col - f1[:, bi]
            for i, bar in enumerate(bars):
                if np.isfinite(d[i]):
                    ax.annotate(
                        f"Δ {d[i]:+.2f}",
                        xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                        xytext=(0, 14), textcoords="offset points", ha="center", fontsize=9,
                        color=PALETTE.get(scen, "#333333"), fontweight="bold",
                    )
    ax.set(ylabel="Macro-F1 (test set)", title="Macro-F1 drops under realistic adversarial profiles (bots perturbed)",
           ylim=(0, 1.08), xticks=x, xticklabels=[MODEL_LABELS.get(m, pretty_model(m)) for m in models])
    ax.grid(True, alpha=0.25, axis="y")
    ax.legend(loc="lower center", ncol=ns, bbox_to_anchor=(0.5, -0.22), frameon=False)
    return fig


def _pr_cell(v: float) -> str:
    return "not available" if not math.isfinite(float(v)) else f"{float(v):.2f}"


def degradation_caption(
    models: Sequence[str], scenarios: Sequence[str], pr: np.ndarray, pairwise: Sequence[Mapping[str, Any]],
) -> str:
    sd = {s: SCENARIO_LABELS.get(s, s) for s in scenarios}
    pr_lines = "\n".join(
        f"- **{MODEL_LABELS.get(m, pretty_model(m))}** — "
        + ", ".join(f"{sd[s]} {_pr_cell(pr[mi, sj])}" for sj, s in enumerate(scenarios))
        + "."
        for mi, m in enumerate(models)
    )
    body = (
        "**Figure H1. Macro-F1 under clean vs adversarial conditions (top-3 models, TwiBot-20 test).**\n\n"
        "Grouped bars show Macro-F1 for the three highest-ranked models on the baseline (clean) test set, "
        "under cost-only cheap attacks, and under the realistic mixed profile. Attacks are applied only to "
        "true-bot rows. Numeric labels show point Macro-F1; Δ values above attack bars show the drop "
        "relative to that model's clean baseline.\n\n**PR-AUC by scenario:**\n\n" + pr_lines
    )
    best_r, best_p = None, float("inf")
    for r in pairwise or []:
        if str(r.get("metric", "")) != "f1":
            continue
        try:
            p = float(r.get("bootstrap_p_corrected", float("nan")))
        except (TypeError, ValueError):
            continue
        if math.isfinite(p) and p < best_p:
            best_p, best_r = p, r
    if best_r is not None and math.isfinite(best_p):
        body += (
            f"\n\nPairwise clean-baseline significance (paired-bootstrap ΔF1, Holm–Bonferroni corrected): "
            f"{best_r.get('model_a')} vs {best_r.get('model_b')}, p={best_p:.2g}. "
            "Full pairwise table in `pairwise_significance.csv`."
        )
    return body + "\n"


def vulnerability_frame(df: pd.DataFrame, best_model: str, top_n: int) -> pd.DataFrame:
    out = (
        df[(df["model"] == best_model) & df["flip_rate"].notna()]
        .sort_values("flip_rate", ascending=False).head(top_n).copy()
    )
    if out.empty:
        return out
    out["cost_tier"] = out["cost_tier"].fillna("cheap").astype(str).str.lower()
    fn = out["feature"].astype(str)
    out["display_feature"] = fn.map(FEATURE_LABELS).fillna(fn)
    return out


def plot_vulnerability(filtered: pd.DataFrame, best_model: str) -> mpl.figure.Figure:
    n = len(filtered)
    fig, ax = plt.subplots(figsize=(8.5, max(4.2, 0.55 * n)), constrained_layout=True)
    y = np.arange(n)
    rates = filtered["flip_rate"].to_numpy()
    bars = ax.barh(
        y, rates, color=[PALETTE.get(t, PALETTE["cheap"]) for t in filtered["cost_tier"]],
        alpha=0.95, edgecolor="white", linewidth=0.6,
    )
    ax.bar_label(bars, labels=[f"{100 * v:.0f}%" for v in rates], padding=3, fontsize=10)
    xmax = float(np.nanmax(rates)) if n else 0.0
    nm = MODEL_LABELS.get(best_model, pretty_model(best_model))
    ax.set(xlim=(0, min(1.0, xmax + 0.08)), xlabel="Flip rate on true bots (prediction bot → human)",
           title=f"Attack surface by profile feature — {nm}", yticks=y,
           yticklabels=filtered["display_feature"].astype(str).tolist())
    ax.invert_yaxis()
    ax.grid(True, alpha=0.25, axis="x")
    ax.legend(handles=[
        Patch(color=PALETTE["cheap"], label="Cheap (profile edits)"),
        Patch(color=PALETTE["expensive"], label="Expensive (followers/friends)"),
    ], loc="lower right", frameon=False)
    return fig


def vulnerability_caption(filtered: pd.DataFrame, best_model: str) -> str:
    nm = MODEL_LABELS.get(best_model, pretty_model(best_model))
    c = filtered[filtered["cost_tier"] == "cheap"]["display_feature"].astype(str).head(3)
    e = filtered[filtered["cost_tier"] == "expensive"]["display_feature"].astype(str).head(2)
    ct = ", ".join(c) if len(c) else "none observed"
    et = ", ".join(e) if len(e) else "none observed"
    return (
        f"**Figure V1. Single-feature attack surface — {nm}.**\n\n"
        "Horizontal bars: fraction of true bots flipping bot→human under one perturbed profile attribute. "
        "Colour = cost (cheap profile edits vs expensive follower/friend manipulation).\n\n"
        "This is an **attack surface**, not importance ranking; top rows are most gameable.\n\n"
        f"**Top cheap levers:** {ct}.\n\n**Top expensive levers:** {et}.\n"
    )

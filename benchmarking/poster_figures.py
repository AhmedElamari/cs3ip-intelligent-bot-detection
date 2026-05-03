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
    "realistic_mixed": "Mixed realistic attacks",
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


def targeted_recall_matrix(df: pd.DataFrame, top_models: Sequence[str], scenarios: Sequence[str]):
    plot_df = df[df["model"].isin(top_models)]
    if plot_df.empty or "baseline" not in scenarios:
        return [], np.array([])
    if "attacked_bot_recall_baseline" not in plot_df.columns or "attacked_bot_recall" not in plot_df.columns:
        return [], np.array([])
    baseline = plot_df.groupby("model")["attacked_bot_recall_baseline"].first()
    attacked = plot_df.pivot_table(index="model", columns="profile", values="attacked_bot_recall", aggfunc="first")
    models = [m for m in top_models if m in baseline.index and pd.notna(baseline.loc[m])]
    if not models:
        return [], np.array([])
    cols = []
    for scenario in scenarios:
        if scenario == "baseline":
            cols.append(baseline.reindex(models).to_numpy(float))
        else:
            if scenario not in attacked.columns:
                cols.append(np.full(len(models), np.nan))
            else:
                cols.append(attacked.reindex(models)[scenario].to_numpy(float))
    return list(models), np.column_stack(cols)


def _attack_deltas(f1: np.ndarray, scenarios: Sequence[str]) -> np.ndarray:
    if f1.size == 0 or "baseline" not in scenarios:
        return np.array([])
    bi = scenarios.index("baseline")
    attack_indices = [idx for idx, scenario in enumerate(scenarios) if idx != bi]
    if not attack_indices:
        return np.array([])
    baseline = f1[:, [bi]]
    return f1[:, attack_indices] - baseline


def _degradation_description(
    f1: np.ndarray,
    scenarios: Sequence[str],
    tolerance: float = 0.01,
) -> tuple[str, str]:
    deltas = _attack_deltas(f1, scenarios)
    if deltas.size == 0:
        return (
            "Macro-F1 under the tested adversarial profiles",
            "No attack scenarios were available beyond the clean baseline.",
        )
    min_delta = float(np.nanmin(deltas))
    max_delta = float(np.nanmax(deltas))
    if min_delta >= -tolerance and max_delta <= tolerance:
        return (
            "Macro-F1 remains stable under the tested adversarial profiles",
            f"Across the tested perturbations, Macro-F1 stays within +/-{tolerance:.2f} "
            "of each model's clean baseline.",
        )
    if max_delta <= tolerance and min_delta < -tolerance:
        if min_delta >= -0.05:
            return (
                "Macro-F1 is slightly lower under the tested adversarial profiles",
                "Across the tested perturbations, Macro-F1 is slightly lower than the clean baseline.",
            )
        return (
            "Macro-F1 declines under the tested adversarial profiles",
            "Across the tested perturbations, Macro-F1 is consistently lower than the clean baseline.",
        )
    if min_delta >= -tolerance and max_delta > tolerance:
        return (
            "Macro-F1 is flat to slightly higher under the tested adversarial profiles",
            "Across the tested perturbations, Macro-F1 is flat to slightly higher than the clean baseline.",
        )
    return (
        "Macro-F1 changes modestly under the tested adversarial profiles",
        "Across the tested perturbations, Macro-F1 varies by scenario but does not move in one direction.",
    )


def _plot_grouped_metric(
    ax: mpl.axes.Axes,
    models: Sequence[str],
    scenarios: Sequence[str],
    values: np.ndarray,
    *,
    ylabel: str,
    title: str,
    ylim: tuple[float, float],
    annotate_delta: bool,
    show_legend: bool,
) -> None:
    n, ns = len(models), len(scenarios)
    x, width = np.arange(n), 0.8 / max(ns, 1)
    bi = scenarios.index("baseline") if "baseline" in scenarios else 0
    for si, scen in enumerate(scenarios):
        off = (si - (ns - 1) / 2) * width
        col = values[:, si]
        bars = ax.bar(
            x + off, col, width, color=PALETTE.get(scen, "#888888"),
            label=SCENARIO_LABELS.get(scen, scen), edgecolor="white", linewidth=0.6,
        )
        if scen == "baseline":
            for bar in bars:
                bar.set_hatch("//")
        ax.bar_label(bars, labels=[f"{v:.2f}" if np.isfinite(v) else "" for v in col], padding=2, fontsize=10)
        if si == bi or not annotate_delta:
            continue
        delta = col - values[:, bi]
        attack_rank = sum(1 for prev_idx, scenario in enumerate(scenarios[:si]) if prev_idx != bi)
        for idx, bar in enumerate(bars):
            if np.isfinite(delta[idx]):
                ax.annotate(
                    f"Change {delta[idx]:+.2f}",
                    xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                    xytext=(0, 14 + 12 * attack_rank),
                    textcoords="offset points",
                    ha="center",
                    fontsize=9,
                    color=PALETTE.get(scen, "#333333"),
                    fontweight="bold",
                )
    ax.set(
        xlabel="Model",
        ylabel=ylabel,
        title=title,
        ylim=ylim,
        xticks=x,
        xticklabels=[MODEL_LABELS.get(m, pretty_model(m)) for m in models],
    )
    ax.grid(True, alpha=0.25, axis="y")
    if show_legend:
        ax.legend(loc="lower center", ncol=ns, bbox_to_anchor=(0.5, -0.28), frameon=False)


def plot_degradation(
    models: Sequence[str],
    scenarios: Sequence[str],
    f1: np.ndarray,
    attacked_recall: np.ndarray | None = None,
) -> mpl.figure.Figure:
    if attacked_recall is None:
        fig, ax = plt.subplots(figsize=(max(9.0, 2.5 * len(models)), 5.2), constrained_layout=True)
        _plot_grouped_metric(
            ax,
            models,
            scenarios,
            f1,
            ylabel="Macro-F1 (test set)",
            title=_degradation_description(f1, scenarios)[0],
            ylim=(0, 1.08),
            annotate_delta=True,
            show_legend=True,
        )
        return fig

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(max(10.0, 3.25 * len(models)), 4.6),
        constrained_layout=False,
    )
    fig.subplots_adjust(left=0.07, right=0.98, bottom=0.17, top=0.82, wspace=0.16)
    _plot_grouped_metric(
        axes[0],
        models,
        scenarios,
        attacked_recall,
        ylabel="Recall on attacked true bots",
        title="Attacked true-bot recall",
        ylim=(0, 1.08),
        annotate_delta=False,
        show_legend=False,
    )
    _plot_grouped_metric(
        axes[1],
        models,
        scenarios,
        f1,
        ylabel="Macro-F1 (test set)",
        title="Full-test Macro-F1",
        ylim=(0, 1.08),
        annotate_delta=False,
        show_legend=False,
    )
    axes[0].yaxis.labelpad = 8
    axes[1].yaxis.labelpad = 8
    for ax in axes:
        ax.title.set_fontsize(12)
        ax.xaxis.label.set_fontsize(11)
        ax.yaxis.label.set_fontsize(11)
        ax.tick_params(axis="both", labelsize=10)
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(
            handles,
            labels,
            loc="upper center",
            ncol=len(scenarios),
            bbox_to_anchor=(0.5, 0.98),
            frameon=False,
            fontsize=9,
        )
    return fig


def _pr_cell(value: float) -> str:
    return "not available" if not math.isfinite(float(value)) else f"{float(value):.2f}"


def _attack_targeted_summary_sentence(attacked_recall: np.ndarray, scenarios: Sequence[str]) -> str:
    deltas = _attack_deltas(attacked_recall, scenarios)
    if deltas.size == 0 or not np.isfinite(deltas).any():
        return "Attacked true-bot recall is unavailable for the selected scenarios."
    bi = scenarios.index("baseline")
    attack_scenarios = [scenario for idx, scenario in enumerate(scenarios) if idx != bi]
    mean_deltas = np.array([
        np.nanmean(column) if np.isfinite(column).any() else np.nan
        for column in deltas.T
    ])
    worst_idx = int(np.nanargmin(mean_deltas))
    worst_scenario = attack_scenarios[worst_idx]
    worst_delta = float(mean_deltas[worst_idx])
    return (
        "Because perturbations are applied only to true-bot rows, attacked true-bot recall isolates "
        "the targeted evasion effect; "
        f"the largest mean recall delta occurs under {SCENARIO_LABELS.get(worst_scenario, worst_scenario)} "
        f"({worst_delta:+.02f} vs baseline)."
    )


def _delta(value: float, baseline: float) -> float:
    return float(value) - float(baseline)


def _delta_pct(value: float, baseline: float) -> float:
    baseline = float(baseline)
    if not math.isfinite(baseline) or baseline == 0:
        return float("nan")
    return 100.0 * _delta(value, baseline) / baseline


def _round_table_value(value: Any) -> Any:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return value
    return round(number, 4) if math.isfinite(number) else np.nan


def robustness_delta_table(
    degradation: pd.DataFrame,
    summary: pd.DataFrame,
    models: Sequence[str],
    scenarios: Sequence[str],
) -> pd.DataFrame:
    """Companion table with exact baseline-vs-attacked degradation statistics."""
    rows = []
    degradation = degradation[degradation["model"].isin(models)].copy()
    summary = summary[summary["model"].isin(models)].copy()
    for model in models:
        model_degradation = degradation[degradation["model"] == model].set_index("scenario")
        if "baseline" not in model_degradation.index:
            continue
        baseline_macro_f1 = float(model_degradation.loc["baseline", "macro_f1"])
        baseline_pr_auc = float(model_degradation.loc["baseline", "pr_auc"])
        for scenario in scenarios:
            if scenario == "baseline" or scenario not in model_degradation.index:
                continue
            profile_rows = summary[(summary["model"] == model) & (summary["profile"] == scenario)]
            if profile_rows.empty:
                continue
            profile = profile_rows.iloc[0]
            attacked_macro_f1 = float(model_degradation.loc[scenario, "macro_f1"])
            attacked_pr_auc = float(model_degradation.loc[scenario, "pr_auc"])
            baseline_recall = float(profile["attacked_bot_recall_baseline"])
            attacked_recall = float(profile["attacked_bot_recall"])
            rows.append({
                "Model": MODEL_LABELS.get(model, pretty_model(model)),
                "Attack Profile": SCENARIO_LABELS.get(scenario, scenario),
                "Attacked True Bots": int(profile["attacked_true_bots"]),
                "Baseline Recall": baseline_recall,
                "Attacked Recall": attacked_recall,
                "Recall Delta": _delta(attacked_recall, baseline_recall),
                "Recall Delta %": _delta_pct(attacked_recall, baseline_recall),
                "Baseline Macro-F1": baseline_macro_f1,
                "Attacked Macro-F1": attacked_macro_f1,
                "Macro-F1 Delta": _delta(attacked_macro_f1, baseline_macro_f1),
                "Macro-F1 Delta %": _delta_pct(attacked_macro_f1, baseline_macro_f1),
                "Baseline PR-AUC": baseline_pr_auc,
                "Attacked PR-AUC": attacked_pr_auc,
                "PR-AUC Delta": _delta(attacked_pr_auc, baseline_pr_auc),
                "PR-AUC Delta %": _delta_pct(attacked_pr_auc, baseline_pr_auc),
                "Flip Rate": float(profile["flip_rate"]),
                "Mean Confidence Drop": float(profile["confidence_drop_mean"]),
                "Mean Non-Flip Confidence Drop": float(profile["confidence_drop_non_flip_mean"]),
            })
    table = pd.DataFrame(rows)
    if table.empty:
        return table
    for column in table.columns:
        if column not in {"Model", "Attack Profile", "Attacked True Bots"}:
            table[column] = table[column].map(_round_table_value)
    return table


def write_robustness_delta_table(output_dir: Path, table: pd.DataFrame) -> None:
    if table.empty:
        return
    stem = Path(output_dir) / "robustness_profile_degradation_table"
    table.to_csv(stem.with_suffix(".csv"), index=False)
    lines = [
        "Exact robustness deltas for the plotted attack profiles.",
        "",
        "| " + " | ".join(table.columns) + " |",
        "| " + " | ".join("---" for _ in table.columns) + " |",
    ]
    for _, row in table.iterrows():
        lines.append("| " + " | ".join(str(row[column]) for column in table.columns) + " |")
    stem.with_suffix(".md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def degradation_caption(
    models: Sequence[str],
    scenarios: Sequence[str],
    f1: np.ndarray,
    pr: np.ndarray,
    pairwise: Sequence[Mapping[str, Any]],
    attacked_recall: np.ndarray | None = None,
) -> str:
    count = len(models)
    model_phrase = f"top-{count} model{'s' if count != 1 else ''}" if count else "selected models"
    ranked_phrase = (
        "the highest-ranked model" if count == 1 else f"the {count} highest-ranked models"
    ) if count else "the selected models"
    scenario_labels = {scenario: SCENARIO_LABELS.get(scenario, scenario) for scenario in scenarios}
    scenario_text = ", ".join(scenario_labels.values())
    takeaway, summary = _degradation_description(f1, scenarios)
    pr_lines = "\n".join(
        f"- **{MODEL_LABELS.get(model, pretty_model(model))}** - "
        + ", ".join(f"{scenario_labels[scenario]} {_pr_cell(pr[mi, sj])}" for sj, scenario in enumerate(scenarios))
        + "."
        for mi, model in enumerate(models)
    )
    if attacked_recall is None:
        body = (
            f"**Figure H1. {takeaway} ({model_phrase}, TwiBot-20 test).**\n\n"
            f"Grouped bars show Macro-F1 for {ranked_phrase} across {scenario_text}. Attacks are applied "
            "only to true-bot rows for non-baseline scenarios. Numeric labels show point Macro-F1; "
            "Change values above attack bars show the "
            "difference relative to that model's clean baseline. "
            + summary
            + "\n\n**PR-AUC by scenario:**\n\n"
            + pr_lines
        )
    else:
        body = (
            f"**Figure H1. Test-set robustness under selected adversarial profile perturbations "
            f"({model_phrase}).**\n\n"
            f"Bars report single-run values on the TwiBot-20 held-out test split for {ranked_phrase} "
            f"across {scenario_text}. Non-baseline evasion profiles perturb only true-bot rows using the "
            "configured profile perturbations from the robustness analysis. "
            "The top panel reports recall on the attacked true-bot subset; the bottom panel reports full-test "
            "Macro-F1 so the targeted effect is shown alongside overall test-set performance. "
            + _attack_targeted_summary_sentence(attacked_recall, scenarios)
            + " "
            + summary
            + "\n\nExact baseline-vs-attack deltas, percentage changes, PR-AUC changes, flip rates, and confidence-drop "
            "statistics are reported "
            "in `robustness_profile_degradation_table.csv`.\n\n"
            + "**Attacked true-bot recall:** targeted-subset recall after profile perturbation.\n\n"
            + f"**Full-test Macro-F1:** {takeaway}.\n\n"
            + "**PR-AUC by scenario:**\n\n"
            + pr_lines
        )
    best_row, best_p = None, float("inf")
    plotted_models = {str(model) for model in models}
    for row in pairwise or []:
        if str(row.get("metric", "")) != "f1_macro":
            continue
        if {str(row.get("model_a")), str(row.get("model_b"))} - plotted_models:
            continue
        try:
            p_value = float(row.get("bootstrap_p_corrected", float("nan")))
        except (TypeError, ValueError):
            continue
        if math.isfinite(p_value) and p_value < best_p:
            best_p, best_row = p_value, row
    if best_row is not None and math.isfinite(best_p):
        body += (
            f"\n\nPairwise clean-baseline significance (paired-bootstrap Delta F1-macro, Holm-Bonferroni corrected): "
            f"{best_row.get('model_a')} vs {best_row.get('model_b')}, p={best_p:.2g}. "
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
    feature_names = out["feature"].astype(str)
    out["display_feature"] = feature_names.map(FEATURE_LABELS).fillna(feature_names)
    return out


def plot_vulnerability(filtered: pd.DataFrame, best_model: str) -> mpl.figure.Figure:
    n = len(filtered)
    fig, ax = plt.subplots(figsize=(8.5, max(4.2, 0.55 * n)), constrained_layout=True)
    y = np.arange(n)
    rates = filtered["flip_rate"].to_numpy()
    bars = ax.barh(
        y,
        rates,
        color=[PALETTE.get(tier, PALETTE["cheap"]) for tier in filtered["cost_tier"]],
        alpha=0.95,
        edgecolor="white",
        linewidth=0.6,
    )
    ax.bar_label(bars, labels=[f"{100 * value:.0f}%" for value in rates], padding=3, fontsize=10)
    xmax = float(np.nanmax(rates)) if n else 0.0
    model_label = MODEL_LABELS.get(best_model, pretty_model(best_model))
    ax.set(
        xlim=(0, min(1.0, xmax + 0.08)),
        xlabel="Flip rate on true bots (prediction bot -> human)",
        title=f"Attack surface by profile feature - {model_label}",
        yticks=y,
        yticklabels=filtered["display_feature"].astype(str).tolist(),
    )
    ax.invert_yaxis()
    ax.grid(True, alpha=0.25, axis="x")
    ax.legend(
        handles=[
            Patch(color=PALETTE["cheap"], label="Cheap (profile edits)"),
            Patch(color=PALETTE["expensive"], label="Expensive (followers/friends)"),
        ],
        loc="lower right",
        frameon=False,
    )
    return fig


def vulnerability_caption(filtered: pd.DataFrame, best_model: str) -> str:
    model_label = MODEL_LABELS.get(best_model, pretty_model(best_model))
    cheap = filtered[filtered["cost_tier"] == "cheap"]["display_feature"].astype(str).head(3)
    expensive = filtered[filtered["cost_tier"] == "expensive"]["display_feature"].astype(str).head(2)
    cheap_text = ", ".join(cheap) if len(cheap) else "none observed"
    expensive_text = ", ".join(expensive) if len(expensive) else "none observed"
    return (
        f"**Figure V1. Single-feature attack surface - {model_label}.**\n\n"
        "Horizontal bars: fraction of true bots flipping bot->human under one perturbed profile attribute. "
        "Colour = cost (cheap profile edits vs expensive follower/friend manipulation).\n\n"
        "This is an **attack surface**, not importance ranking; top rows are most gameable.\n\n"
        f"**Top cheap levers:** {cheap_text}.\n\n**Top expensive levers:** {expensive_text}.\n"
    )

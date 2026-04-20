"""Output helpers for the benchmark pipeline: comparison tables, plots, and final results."""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from config import Config
from .model_benchmark import ModelBenchmark
from benchmarking.dissertation_scoreboard import build_scoreboard, to_latex, to_markdown
from benchmarking.run_metadata import BenchmarkRunContext, write_run_metadata


def _save_plot(fig, output_path: Path) -> None:
    """Save and close a matplotlib figure."""
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def save_dissertation_figures(benchmark: ModelBenchmark, output_dir: Path) -> None:
    """Save PR comparison and best-model confusion matrices (test set); warn on individual failures."""
    try:
        fig = benchmark.plot_pr_curves_top(top_n=3)
        if fig is not None:
            _save_plot(fig, output_dir / "pr_curves_comparison.png")
    except (KeyError, AttributeError, ImportError, ValueError) as e:
        print(f"Warning: Could not save PR curve comparison: {e}")

    for normalize, filename in (
        ("true", "confusion_matrix_best_model_normalized.png"),
        (None, "confusion_matrix_best_model_raw.png"),
    ):
        try:
            fig_cm = benchmark.plot_best_confusion_matrix(normalize=normalize)
            if fig_cm is None:
                continue
            _save_plot(fig_cm, output_dir / filename)
        except (KeyError, AttributeError, ImportError, ValueError) as e:
            print(f"Warning: Could not save confusion matrix ({filename}): {e}")


def save_robustness_degradation_figure(benchmark: ModelBenchmark, output_dir: Path) -> None:
    """Grouped bar chart: Macro-F1 per scenario for top-3 scoreboard models."""
    df = getattr(benchmark, "robustness_degradation", None)
    if df is None or df.empty:
        return

    scoreboard = build_scoreboard(benchmark)
    if scoreboard.empty:
        return

    top_models = scoreboard["Model"].head(3).tolist()
    plot_df = df[df["model"].isin(top_models)].copy()
    if plot_df.empty:
        return

    scenario_set = set(plot_df["scenario"].astype(str))
    scenarios = ["baseline"] + [p for p in ("cheap_only", "realistic_mixed") if p in scenario_set]
    models = [m for m in top_models if m in plot_df["model"].values]
    if not models:
        return

    n_scen = len(scenarios)
    fig_w = max(8.0, 2.0 * len(models))
    fig, ax = plt.subplots(figsize=(fig_w, 5))
    x = np.arange(len(models))
    width = 0.8 / max(n_scen, 1)
    colors = plt.cm.Set1(np.linspace(0, 1, max(n_scen, 1)))

    for si, scen in enumerate(scenarios):
        offset = (si - n_scen / 2 + 0.5) * width
        vals = []
        for model in models:
            sub = plot_df[(plot_df["model"] == model) & (plot_df["scenario"] == scen)]
            vals.append(float(sub["macro_f1"].iloc[0]) if len(sub) else float("nan"))
        ax.bar(
            x + offset,
            vals,
            width,
            label=scen.replace("_", " "),
            color=colors[si % len(colors)],
            alpha=0.85,
        )

    ax.set_ylabel("Macro-F1 (test set)")
    ax.set_title("Robustness: Macro-F1 under adversarial profiles (bots perturbed)")
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=15, ha="right")
    ax.set_ylim(0, 1.05)
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    _save_plot(fig, output_dir / "robustness_profile_degradation.png")


def save_feature_vulnerability_outputs(
    benchmark: ModelBenchmark,
    output_dir: Path,
    top_n: int = 10,
) -> None:
    """Top single-feature attack flip rates for best scoreboard model: CSV + horizontal bar chart."""
    df = getattr(benchmark, "feature_attack_results", None)
    if df is None or df.empty:
        return

    scoreboard = build_scoreboard(benchmark)
    if scoreboard.empty:
        return

    best_model = str(scoreboard.iloc[0]["Model"])
    filtered = (
        df[(df["model"] == best_model) & df["flip_rate"].notna()]
        .sort_values("flip_rate", ascending=False)
        .head(top_n)
    )
    if filtered.empty:
        return

    out_csv = filtered[
        ["feature", "attack_name", "cost_tier", "flip_rate", "confidence_drop_mean"]
    ].rename(
        columns={
            "feature": "Feature",
            "attack_name": "Attack Name",
            "cost_tier": "Cost Tier",
            "flip_rate": "Flip Rate",
            "confidence_drop_mean": "Confidence Drop Mean",
        }
    )
    out_csv.to_csv(output_dir / "top_feature_vulnerabilities.csv", index=False)

    fig_h = max(4.0, 0.45 * len(filtered))
    fig, ax = plt.subplots(figsize=(7, fig_h))
    y_pos = np.arange(len(filtered))
    ax.barh(y_pos, filtered["flip_rate"].to_numpy(), color="steelblue", alpha=0.85)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(filtered["feature"].astype(str).tolist())
    ax.invert_yaxis()
    ax.set_xlabel("Flip rate (true bots predicted bot → human)")
    ax.set_title(f"Single-feature attack vulnerability — {best_model}")
    ax.grid(True, alpha=0.3, axis="x")
    plt.tight_layout()
    _save_plot(fig, output_dir / "feature_attack_flip_rates_best_model.png")


def save_final_outputs(
    benchmark: ModelBenchmark,
    output_dir: Path,
    config: Config,
    run_context: BenchmarkRunContext,
) -> None:
    """Save required benchmark artifacts; raise if any required write fails."""
    benchmark.save_results(output_dir)

    scoreboard_df = build_scoreboard(benchmark)
    if not scoreboard_df.empty:
        scoreboard_df.to_csv(output_dir / "dissertation_scoreboard.csv", index=False)
        (output_dir / "dissertation_scoreboard.md").write_text(
            to_markdown(scoreboard_df), encoding="utf-8"
        )
        (output_dir / "dissertation_scoreboard.tex").write_text(
            to_latex(scoreboard_df), encoding="utf-8"
        )

    report = benchmark.generate_report()
    report_path_md = output_dir / 'benchmark_report.md'
    report_path_md.write_text(report, encoding='utf-8')

    print(f"Saved benchmark report to {report_path_md}")

    config.to_json(output_dir / 'config.json')
    save_dissertation_figures(benchmark, output_dir)
    save_robustness_degradation_figure(benchmark, output_dir)
    save_feature_vulnerability_outputs(benchmark, output_dir)
    metadata_path = write_run_metadata(run_context)
    print(f"Saved run metadata to {metadata_path}")

"""Output helpers for the benchmark pipeline: comparison tables, plots, and final results."""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from config import Config
from .model_benchmark import ModelBenchmark
from benchmarking.dissertation_scoreboard import build_scoreboard, to_latex, to_markdown
from benchmarking.poster_figures import (
    degradation_caption,
    degradation_matrix,
    plot_degradation,
    plot_vulnerability,
    poster_style,
    save_poster_figure,
    targeted_recall_matrix,
    vulnerability_caption,
    vulnerability_frame,
    write_caption,
)
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
    """Poster robustness figure with attack-targeted recall plus full-test Macro-F1 context."""
    df = getattr(benchmark, "robustness_degradation", None)
    if df is None or df.empty:
        return
    summary = getattr(benchmark, "robustness_summary", None)
    scoreboard = build_scoreboard(benchmark)
    if scoreboard.empty:
        return
    top = scoreboard["Model"].head(3).tolist()
    want = ("baseline", "cheap_only", "realistic_mixed")
    scenarios = [s for s in want if s in set(df["scenario"].astype(str))]
    models, f1, pr = degradation_matrix(df, top, scenarios)
    if not models or "baseline" not in scenarios:
        return
    recall_models, attacked_recall = targeted_recall_matrix(summary, models, scenarios) if summary is not None else ([], np.array([]))
    attacked_recall = attacked_recall if recall_models == models and attacked_recall.size > 0 else None
    with poster_style():
        fig = plot_degradation(models, scenarios, f1, attacked_recall)
        save_poster_figure(fig, output_dir, "robustness_profile_degradation")
    write_caption(
        output_dir,
        "robustness_profile_degradation",
        degradation_caption(models, scenarios, f1, pr, getattr(benchmark, "pairwise_significance", []), attacked_recall),
    )


def save_feature_vulnerability_outputs(
    benchmark: ModelBenchmark,
    output_dir: Path,
    top_n: int = 8,
) -> None:
    """Top single-feature flip rates for best model: CSV + poster PNG/PDF/caption."""
    df = getattr(benchmark, "feature_attack_results", None)
    if df is None or df.empty:
        return
    scoreboard = build_scoreboard(benchmark)
    if scoreboard.empty:
        return
    best_model = str(scoreboard.iloc[0]["Model"])
    filtered = vulnerability_frame(df, best_model, top_n)
    if filtered.empty:
        return
    filtered[
        ["feature", "attack_name", "cost_tier", "flip_rate", "confidence_drop_mean"]
    ].rename(
        columns={
            "feature": "Feature",
            "attack_name": "Attack Name",
            "cost_tier": "Cost Tier",
            "flip_rate": "Flip Rate",
            "confidence_drop_mean": "Confidence Drop Mean",
        }
    ).to_csv(output_dir / "top_feature_vulnerabilities.csv", index=False)
    with poster_style():
        fig = plot_vulnerability(filtered, best_model)
        save_poster_figure(fig, output_dir, "feature_attack_flip_rates_best_model")
    write_caption(
        output_dir,
        "feature_attack_flip_rates_best_model",
        vulnerability_caption(filtered, best_model),
    )


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

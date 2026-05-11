"""Output helpers for the benchmark pipeline: comparison tables, plots, and final results."""

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from config import Config
from .model_benchmark import ModelBenchmark
from benchmarking.dissertation_scoreboard import METRIC_COLUMNS, build_scoreboard, to_latex, to_markdown
from benchmarking.poster_figures import (
    degradation_caption,
    degradation_matrix,
    plot_degradation,
    plot_vulnerability,
    poster_style,
    robustness_delta_table,
    save_poster_figure,
    targeted_recall_matrix,
    vulnerability_caption,
    vulnerability_frame,
    write_robustness_delta_table,
    write_caption,
)
from benchmarking.run_metadata import BenchmarkRunContext, write_run_metadata
from benchmarking.threshold_analysis import save_threshold_analysis_outputs
from benchmarking.time_stratified_results import build_concept_drift_delta

DRIFT_SCOREBOARD_INTRO = (
    "Concept-drift test set: models trained on oldest accounts (chronological split over combined "
    "labelled data), validated on the middle window, evaluated on the newest window."
)


def _model_runtime_from_benchmark(benchmark: ModelBenchmark) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for name, res in benchmark.results.items():
        model = res.get("model")
        get_rt = getattr(model, "get_runtime_metadata", None) if model is not None else None
        if callable(get_rt):
            meta = get_rt()
            if meta:
                out[name] = meta
    return out


def _save_plot(fig, output_path: Path) -> None:
    """Save and close a matplotlib figure."""
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def _format_ci_cell(point, lower, upper) -> str:
    if pd.isna(point):
        return "—"
    if pd.isna(lower) or pd.isna(upper):
        return f"{float(point):.3f}"
    return f"{float(point):.3f} [{float(lower):.3f}, {float(upper):.3f}]"


def _markdown_table(frame: pd.DataFrame) -> str:
    header = "| " + " | ".join(frame.columns) + " |"
    separator = "| " + " | ".join("---" for _ in frame.columns) + " |"
    rows = [
        "| " + " | ".join(str(value) for value in row) + " |"
        for row in frame.itertuples(index=False, name=None)
    ]
    return "\n".join([header, separator, *rows])


def save_dissertation_statistical_summary(
    benchmark: ModelBenchmark,
    output_dir: Path,
) -> None:
    """Write dissertation-facing metrics with bootstrap confidence intervals."""
    if not hasattr(benchmark, "get_confidence_intervals"):
        return
    ci_df = benchmark.get_confidence_intervals()
    scoreboard_df = build_scoreboard(benchmark)
    if scoreboard_df.empty or ci_df.empty:
        return

    metrics = [key for _, key in METRIC_COLUMNS if key in set(ci_df["metric"].astype(str))]
    metric_display = {key: display for display, key in METRIC_COLUMNS}
    summary_rows = []
    markdown_rows = []
    for _, score_row in scoreboard_df.iterrows():
        model = str(score_row["Model"])
        out_row = {"Rank": int(score_row["Rank"]), "Model": model}
        md_row = {"Rank": int(score_row["Rank"]), "Model": model}
        for metric in metrics:
            ci_match = ci_df[(ci_df["model"].astype(str) == model) & (ci_df["metric"].astype(str) == metric)]
            display_name = metric_display.get(metric, metric)
            if ci_match.empty:
                out_row[f"{display_name} Point"] = np.nan
                out_row[f"{display_name} CI Lower"] = np.nan
                out_row[f"{display_name} CI Upper"] = np.nan
                md_row[display_name] = "—"
                continue
            ci = ci_match.iloc[0]
            out_row[f"{display_name} Point"] = ci["point"]
            out_row[f"{display_name} CI Lower"] = ci["lower"]
            out_row[f"{display_name} CI Upper"] = ci["upper"]
            md_row[display_name] = _format_ci_cell(ci["point"], ci["lower"], ci["upper"])
        summary_rows.append(out_row)
        markdown_rows.append(md_row)

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(output_dir / "dissertation_statistical_summary.csv", index=False)

    markdown_df = pd.DataFrame(markdown_rows)
    for display_name in [metric_display.get(metric, metric) for metric in metrics]:
        if display_name not in markdown_df:
            continue
        point_col = f"{display_name} Point"
        if point_col not in summary_df:
            continue
        values = pd.to_numeric(summary_df[point_col], errors="coerce")
        if values.dropna().empty:
            continue
        best = values.max()
        best_mask = values.eq(best)
        markdown_df.loc[best_mask, display_name] = markdown_df.loc[best_mask, display_name].map(
            lambda value: f"**{value}**"
        )

    caption = (
        "Test-set performance with 95% bootstrap confidence intervals. "
        "Intervals are single-run resampling summaries, not repeated-seed stability estimates."
    )
    (output_dir / "dissertation_statistical_summary.md").write_text(
        caption + "\n\n" + _markdown_table(markdown_df) + "\n",
        encoding="utf-8",
    )


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
    plot_df = df[df["model"].isin(top)]
    scenarios = [s for s in want if s in set(plot_df["scenario"].astype(str))]
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
    if summary is not None and not summary.empty:
        write_robustness_delta_table(
            output_dir,
            robustness_delta_table(df, summary, models, scenarios),
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


def save_attack_feature_space_tsne(benchmark: ModelBenchmark, output_dir: Path) -> None:
    """Save qualitative t-SNE attack-space artifacts when robustness data exists."""
    summary = getattr(benchmark, "robustness_summary", None)
    if summary is None or summary.empty:
        return
    try:
        from benchmarking.feature_space_tsne import save_attack_feature_space_tsne as save_tsne

        save_tsne(benchmark, benchmark.base_feature_names, output_dir)
    except (AttributeError, ImportError, RuntimeError, ValueError) as e:
        print(f"Warning: Could not save attack feature-space t-SNE: {e}")


def save_concept_drift_outputs(
    main_benchmark: ModelBenchmark,
    drift_benchmark: ModelBenchmark,
    output_dir: Path,
    *,
    protocol_note: str,
) -> None:
    """Write time-stratified scoreboard and baseline-vs-drift deltas (minimal drift bundle)."""
    drift_sb = build_scoreboard(drift_benchmark)
    if drift_sb.empty:
        return
    main_sb = build_scoreboard(main_benchmark)
    drift_sb.to_csv(output_dir / "time_stratified_scoreboard.csv", index=False)
    caption = f"{DRIFT_SCOREBOARD_INTRO}\n\n{protocol_note}"
    (output_dir / "time_stratified_scoreboard.md").write_text(
        to_markdown(drift_sb, caption=caption) + "\n", encoding="utf-8"
    )
    delta = build_concept_drift_delta(main_sb, drift_sb)
    delta.to_csv(output_dir / "concept_drift_delta.csv", index=False)
    delta_intro = (
        "Δ = drift-test metric minus standard split test metric (negative usually means degradation)."
    )
    (output_dir / "concept_drift_delta.md").write_text(
        delta_intro + "\n\n" + _markdown_table(delta) + "\n", encoding="utf-8"
    )


def save_final_outputs(
    benchmark: ModelBenchmark,
    output_dir: Path,
    config: Config,
    run_context: BenchmarkRunContext,
    *,
    threshold_analysis_enabled: bool = False,
    threshold_precision_floor: float = 0.80,
    drift_benchmark: ModelBenchmark | None = None,
    drift_protocol_note: str = "",
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
        save_dissertation_statistical_summary(benchmark, output_dir)

    report = benchmark.generate_report()
    report_path_md = output_dir / 'benchmark_report.md'
    report_path_md.write_text(report, encoding='utf-8')

    print(f"Saved benchmark report to {report_path_md}")

    config.to_json(output_dir / 'config.json')
    save_dissertation_figures(benchmark, output_dir)
    save_robustness_degradation_figure(benchmark, output_dir)
    save_feature_vulnerability_outputs(benchmark, output_dir)
    save_attack_feature_space_tsne(benchmark, output_dir)
    if threshold_analysis_enabled:
        save_threshold_analysis_outputs(
            benchmark,
            output_dir,
            precision_floor=threshold_precision_floor,
        )
    if drift_benchmark is not None and getattr(drift_benchmark, "results", None):
        save_concept_drift_outputs(
            benchmark,
            drift_benchmark,
            output_dir,
            protocol_note=drift_protocol_note or DRIFT_SCOREBOARD_INTRO,
        )
    meta_map = _model_runtime_from_benchmark(benchmark)
    run_context.model_runtime_metadata = meta_map if meta_map else None
    metadata_path = write_run_metadata(run_context)
    print(f"Saved run metadata to {metadata_path}")

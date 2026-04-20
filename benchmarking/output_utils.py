"""Output helpers for the benchmark pipeline: comparison tables, plots, and final results."""
from pathlib import Path

import matplotlib.pyplot as plt

from config import Config
from benchmarking import ModelBenchmark
from benchmarking.dissertation_scoreboard import build_scoreboard, to_latex, to_markdown
from benchmarking.run_metadata import BenchmarkRunContext, write_run_metadata


def _save_plot(fig, output_path: Path) -> None:
    """Save and close a matplotlib figure."""
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def save_comparison_outputs(benchmark: ModelBenchmark, output_dir: Path, config: Config) -> None:
    """Save optional plots for the benchmark pipeline."""
    if not config.get('output.save_plots'):
        return

    try:
        _save_plot(benchmark.plot_comparison(), output_dir / 'performance_comparison.png')
        _save_plot(benchmark.plot_training_times(), output_dir / 'training_times.png')
        print(f"Saved performance plots to {output_dir}")
    except Exception as e:
        print(f"Warning: Could not save plots: {e}")


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
    for report_path in (report_path_md, output_dir / 'benchmark_report.txt'):
        report_path.write_text(report, encoding='utf-8')

    print(f"Saved benchmark report to {report_path_md}")

    config.to_json(output_dir / 'config.json')
    metadata_path = write_run_metadata(run_context)
    print(f"Saved run metadata to {metadata_path}")

"""Output helpers for the benchmark pipeline: comparison tables, plots, and final results."""
import matplotlib.pyplot as plt
from pathlib import Path
from config import Config
from benchmarking import ModelBenchmark


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


def save_final_outputs(benchmark: ModelBenchmark, output_dir: Path, config: Config) -> None:
    """Save required benchmark artifacts; raise if any required write fails.

    Generates all artifacts in memory before writing to avoid partial output on
    disk if any step raises.
    """
    report = benchmark.generate_report()

    benchmark.save_results(output_dir)

    report_path_md = output_dir / 'benchmark_report.md'
    report_path_txt = output_dir / 'benchmark_report.txt'
    config_path = output_dir / 'config.json'
    report_path_md.write_text(report, encoding='utf-8')
    report_path_txt.write_text(report, encoding='utf-8')
    print(f"Saved benchmark report to {report_path_md}")

    config.to_json(config_path)

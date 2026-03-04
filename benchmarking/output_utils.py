"""
Output helpers for the benchmark pipeline.
This module contains functions to save the comparison tables and plots for the benchmark pipeline.
It also contains a function to save the final outputs of the benchmark pipeline.

Args:
    benchmark: The benchmark pipeline object holding trained models and results.
    output_dir: Path to directory where comparison tables and plots will be saved
    config: Configuration object holding output settings

Returns:
    None
"""
import matplotlib.pyplot as plt
from pathlib import Path
from config import Config
from benchmarking import ModelBenchmark


def _save_plot(fig, output_path: Path) -> None:
    """Save and close a matplotlib figure."""
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def save_comparison_outputs(benchmark: ModelBenchmark, output_dir: Path, config: Config) -> None:
    """Save comparison tables and plots.
    This function saves the comparison table and plots for the benchmark pipeline.

    Args:
        benchmark: The benchmark pipeline object holding trained models and results.
        output_dir: Path to directory where comparison tables and plots will be saved
        config: Configuration object holding output settings
    """
    
    try:
        comparison_df = benchmark.get_comparison_table()
        comparison_df.to_csv(output_dir / 'model_comparison.csv', index=False)
        print(f"\nSaved comparison table to {output_dir / 'model_comparison.csv'}")
    except (KeyError, ValueError, RuntimeError, OSError) as e:
        print(f"Warning: Could not save comparison table: {e}")
        return

    if not config.get('output.save_plots'):
        return

    try:
        _save_plot(benchmark.plot_comparison(), output_dir / 'performance_comparison.png')
        _save_plot(benchmark.plot_training_times(), output_dir / 'training_times.png')
        print(f"Saved performance plots to {output_dir}")
    except (KeyError, ValueError, RuntimeError, OSError) as e:
        print(f"Warning: Could not save plots: {e}")


def save_final_outputs(benchmark: ModelBenchmark, output_dir: Path, config: Config) -> None:
    """Save benchmark results, report, and config."""
    benchmark.save_results(output_dir)

    report_path = output_dir / 'benchmark_report.txt'
    config_path = output_dir / 'config.json'
    report = benchmark.generate_report()
    report_path.write_text(report, encoding='utf-8')

    config.to_json(config_path)
    print(f"Saved benchmark report to {report_path}")
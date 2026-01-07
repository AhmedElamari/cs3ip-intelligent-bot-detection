"""
Output helpers for the benchmark pipeline.
"""

from pathlib import Path

from config import Config


def save_comparison_outputs(benchmark, output_dir: Path, config: Config) -> None:
    """Save comparison tables and plots.
    This function saves the comparison table and plots for the benchmark pipeline.

    Args:
        benchmark: The benchmark pipeline object holding trained models and results.
        output_dir: Path to directory where comparison tables and plots will be saved
        config: Configuration object holding output settings
    """
    
    comparison_df = benchmark.get_comparison_table()
    comparison_df.to_csv(output_dir / 'model_comparison.csv', index=False)
    print(f"\nSaved comparison table to {output_dir / 'model_comparison.csv'}")

    if config.get('output.save_plots'):
        try:
            # Performance comparison
            fig = benchmark.plot_comparison()
            fig.savefig(
                output_dir / 'performance_comparison.png',
                dpi=150,
                bbox_inches='tight'
            )

            # Training times
            fig = benchmark.plot_training_times()
            fig.savefig(
                output_dir / 'training_times.png',
                dpi=150,
                bbox_inches='tight'
            )

            print(f"Saved performance plots to {output_dir}")
        except Exception as e:
            print(f"Warning: Could not save plots: {e}")


def save_final_outputs(benchmark, output_dir: Path, config: Config) -> None:
    """Save benchmark results, report, and config."""
    benchmark.save_results(output_dir)

    report = benchmark.generate_report()
    with open(output_dir / 'benchmark_report.txt', 'w') as f:
        f.write(report)

    config.to_json(output_dir / 'config.json')

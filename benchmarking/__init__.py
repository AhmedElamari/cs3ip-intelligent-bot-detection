"""
Benchmarking Module
===================
Tools for comparing and evaluating multiple ML models.
"""

try:
    import matplotlib
except ImportError:  # pragma: no cover - plotting is optional
    pass
else:
    matplotlib.use("Agg", force=True)

from .model_benchmark import ModelBenchmark
from .metrics import MetricsCalculator

__all__ = [
    'ModelBenchmark',
    'MetricsCalculator',
]

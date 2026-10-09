"""
Evaluation and benchmarking package for CrunchDAO Synth.
"""

from .local_evaluator import LocalBacktestHarness
from .benchmark_comparator import BenchmarkComparator
from .visualization import (
    plot_mixture_density,
    plot_crps_timeline,
    plot_leaderboard_ranks,
)

__all__ = [
    "LocalBacktestHarness",
    "BenchmarkComparator",
    "plot_mixture_density",
    "plot_crps_timeline",
    "plot_leaderboard_ranks",
]

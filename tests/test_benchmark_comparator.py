"""
Unit tests for BenchmarkComparator and leaderboard ranking logic.
"""

from datetime import datetime, timezone
from pathlib import Path
import tempfile
import unittest
import numpy as np
import pandas as pd

from src.evaluation.benchmark_comparator import BenchmarkComparator


class TestBenchmarkComparator(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.comparator = BenchmarkComparator(cache_dir=Path(self.temp_dir.name))

        # Create a mock historical trackers DataFrame
        self.mock_history = pd.DataFrame({
            "tracker": ["model_A", "model_B", "benchmark", "model_C"] * 5,
            "horizon": [3600] * 20,
            "asset": ["BTC"] * 10 + ["ETH"] * 10,
            "performed_at": pd.date_range("2026-03-20", periods=20, freq="h", tz="UTC"),
            "resolvable_at": pd.date_range("2026-03-20 01:00", periods=20, freq="h", tz="UTC"),
            "score": [
                # BTC scores
                1.50, 1.80, 2.00, 2.20,
                1.52, 1.81, 2.02, 2.19,
                1.48, 1.79,
                # ETH scores
                0.80, 0.90, 1.10, 1.30,
                0.82, 0.89, 1.11, 1.28,
                0.81, 0.91,
            ],
        })

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_compute_ranks_superior_to_benchmark(self):
        # Tracker score 1.60 is better than benchmark (2.01) and model_B (1.80), but worse than model_A (1.50)
        btc_history = self.mock_history[self.mock_history["asset"] == "BTC"]
        stats = self.comparator.compute_ranks_for_asset(btc_history, my_score=1.60, asset="BTC")

        self.assertEqual(stats["asset"], "BTC")
        self.assertEqual(stats["my_rank"], 2)  # model_A (1.50) < 1.60 -> rank 2
        self.assertEqual(stats["total_trackers"], 5)  # 4 models + my model
        self.assertTrue(stats["beats_benchmark"])
        self.assertLess(stats["delta_vs_benchmark_pct"], 0.0)  # Lower is better

    def test_compute_ranks_inferior_to_benchmark(self):
        btc_history = self.mock_history[self.mock_history["asset"] == "BTC"]
        stats = self.comparator.compute_ranks_for_asset(btc_history, my_score=2.50, asset="BTC")

        self.assertEqual(stats["my_rank"], 5)  # Worst
        self.assertFalse(stats["beats_benchmark"])
        self.assertGreater(stats["delta_vs_benchmark_pct"], 0.0)

    def test_filter_history(self):
        filtered = self.comparator.filter_history(
            self.mock_history,
            horizon=3600,
            assets=["BTC"],
        )
        self.assertEqual(len(filtered), 10)
        self.assertTrue((filtered["asset"] == "BTC").all())
        self.assertTrue((filtered["horizon"] == 3600).all())

    def test_format_leaderboard_table(self):
        df_comp = pd.DataFrame([{
            "asset": "BTC",
            "my_score": 1.6500,
            "benchmark_score": 2.0100,
            "my_rank": 2,
            "benchmark_rank": 3,
            "total_trackers": 5,
            "percentile": 40.0,
            "delta_vs_benchmark_pct": -17.91,
            "beats_benchmark": True,
        }])
        table = self.comparator.format_leaderboard_table(df_comp)
        self.assertIn("CRUNCHDAO SYNTH: LEADERBOARD", table)
        self.assertIn("BTC", table)
        self.assertIn("YES [✓]", table)
        self.assertIn("-17.91%", table)


if __name__ == "__main__":
    unittest.main()

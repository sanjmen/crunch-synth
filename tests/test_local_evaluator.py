"""
Unit tests for LocalBacktestHarness and event-driven timeline simulation.
"""

from datetime import datetime, timezone, timedelta
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import pandas as pd

from src.trackers.gaussian_baseline import GaussianBaselineTracker
from src.evaluation.local_evaluator import LocalBacktestHarness
from src.data.price_manager import PriceManager


class TestLocalBacktestHarness(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.price_manager = PriceManager(cache_dir=self.temp_dir.name)
        self.tracker = GaussianBaselineTracker()

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_invalid_horizon_profile(self):
        with self.assertRaises(ValueError):
            LocalBacktestHarness(
                tracker=self.tracker,
                horizon_profile="invalid_profile_123",
                price_manager=self.price_manager,
            )

    def test_harness_initialization(self):
        harness = LocalBacktestHarness(
            tracker=self.tracker,
            assets=["BTC", "ETH"],
            horizon_profile="24h",
            days_test=2,
            days_warmup=5,
            price_manager=self.price_manager,
        )
        self.assertEqual(harness.assets, ["BTC", "ETH"])
        self.assertEqual(harness.horizon, 86400)
        self.assertEqual(harness.interval, 3600)
        self.assertEqual(harness.days_test, 2)
        self.assertEqual(harness.days_warmup, 5)

    def test_harness_simulation_with_mock_prices(self):
        # Create 12 hours of synthetic data at 60s intervals
        end_time = datetime(2026, 1, 15, 12, 0, 0, tzinfo=timezone.utc)
        start_time = end_time - timedelta(hours=12)

        start_ts = int(start_time.timestamp())
        end_ts = int(end_time.timestamp())

        ticks = []
        p = 50000.0
        for ts in range(start_ts, end_ts, 60):
            p += np.random.normal(0, 10.0)
            ticks.append((ts, p))

        # Populate cache
        df_ticks = pd.DataFrame(ticks, columns=["timestamp", "price"])
        self.price_manager.write_cached_prices("BTC", df_ticks)

        # Run 1h profile harness (horizon=3600, interval=300)
        harness = LocalBacktestHarness(
            tracker=self.tracker,
            assets=["BTC"],
            horizon_profile="1h",
            evaluation_end=end_time,
            days_test=1,
            days_warmup=1,
            price_manager=self.price_manager,
            use_cache=True,
        )

        results = harness.run(verbose=False)

        self.assertIn("BTC", results["assets"])
        self.assertIn("crps_mean", results["assets"]["BTC"])
        self.assertIn("evaluations_count", results["assets"]["BTC"])

        summary = harness.summary_table()
        self.assertIsInstance(summary, pd.DataFrame)
        self.assertEqual(len(summary), 1)
        self.assertEqual(summary.iloc[0]["Asset"], "BTC")

        series = harness.get_score_series("BTC")
        self.assertIsInstance(series, pd.DataFrame)
        self.assertIn("score", series.columns)


if __name__ == "__main__":
    unittest.main()

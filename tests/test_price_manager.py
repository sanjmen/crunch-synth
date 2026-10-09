"""
Unit tests for PriceManager and caching functionality.
"""

from datetime import datetime, timezone, timedelta
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch, MagicMock
import numpy as np
import pandas as pd

from src.data.price_manager import PriceManager, ALL_ASSETS, ASSET_CATEGORIES


class TestPriceManager(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.manager = PriceManager(cache_dir=self.temp_dir.name)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_all_supported_assets_in_categories(self):
        """Verify that every asset in ALL_ASSETS belongs to a defined category."""
        categorized = []
        for cat, assets in ASSET_CATEGORIES.items():
            categorized.extend(assets)

        self.assertEqual(len(ALL_ASSETS), 12)
        self.assertEqual(set(ALL_ASSETS), set(categorized))

    def test_empty_cache_read(self):
        """Reading non-existent cache should return empty dataframe with expected schema."""
        df = self.manager.read_cached_prices("BTC")
        self.assertTrue(df.empty)
        self.assertListEqual(list(df.columns), ["timestamp", "price"])
        self.assertEqual(df["timestamp"].dtype, "int64")
        self.assertEqual(df["price"].dtype, "float64")

    def test_write_and_read_roundtrip(self):
        """Writing prices to cache and reading back should preserve order and types and remove duplicates."""
        raw_df = pd.DataFrame({
            "timestamp": [1700000120, 1700000060, 1700000060, 1700000180],
            "price": [102.5, 100.0, 100.0, 105.0],
        })
        self.manager.write_cached_prices("TEST", raw_df)

        read_df = self.manager.read_cached_prices("TEST")
        self.assertEqual(len(read_df), 3)  # Duplicate removed
        self.assertListEqual(read_df["timestamp"].tolist(), [1700000060, 1700000120, 1700000180])
        self.assertListEqual(read_df["price"].tolist(), [100.0, 102.5, 105.0])

    @patch("crunch_synth.pricedb.get_price_history")
    def test_get_prices_cache_hit_and_miss(self, mock_get_price_history):
        """Test that get_prices calls API on cache miss and uses cache on subsequent calls."""
        base_ts = 1700000000
        mock_data = [(base_ts + i * 60, 100.0 + i) for i in range(10)]
        mock_get_price_history.return_value = mock_data

        from_dt = datetime.fromtimestamp(base_ts, tz=timezone.utc)
        to_dt = datetime.fromtimestamp(base_ts + 9 * 60, tz=timezone.utc)

        # First call: miss -> queries API
        res1 = self.manager.get_prices("ETH", from_dt, to_dt, use_cache=True)
        self.assertEqual(len(res1), 10)
        self.assertEqual(mock_get_price_history.call_count, 1)

        # Second call: hit -> should not query API again
        res2 = self.manager.get_prices("ETH", from_dt, to_dt, use_cache=True)
        self.assertEqual(len(res2), 10)
        self.assertEqual(mock_get_price_history.call_count, 1)

    @patch("crunch_synth.pricedb.get_price_history")
    def test_get_test_and_warmup_prices_split(self, mock_get_price_history):
        """Verify proper partitioning into warmup and test time windows."""
        now = datetime(2026, 1, 20, 0, 0, 0, tzinfo=timezone.utc)
        # Create 10 days of minute ticks (represented sparsely for test speed)
        ticks = []
        start_ts = int((now - timedelta(days=5)).timestamp())
        end_ts = int(now.timestamp())
        for ts in range(start_ts, end_ts, 3600):  # Hourly ticks
            ticks.append((ts, 50000.0 + (ts - start_ts) * 0.01))

        mock_get_price_history.return_value = ticks

        test_prices, warmup_prices = self.manager.get_test_and_warmup_prices(
            assets=["BTC"],
            evaluation_end=now,
            days_test=2,
            days_warmup=3,
            use_cache=False,
        )

        cutoff_ts = int((now - timedelta(days=2)).timestamp())

        self.assertIn("BTC", test_prices)
        self.assertIn("BTC", warmup_prices)

        # Every tick in warmup must be < cutoff
        for ts, _ in warmup_prices["BTC"]:
            self.assertLess(ts, cutoff_ts)

        # Every tick in test must be >= cutoff
        for ts, _ in test_prices["BTC"]:
            self.assertGreaterEqual(ts, cutoff_ts)


if __name__ == "__main__":
    unittest.main()

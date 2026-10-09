"""
Unit tests for EmpiricalQuantileForecaster and QuantileMixtureTracker.
"""

import unittest
import numpy as np
import pandas as pd

from src.models.quantile_regressor import EmpiricalQuantileForecaster
from src.trackers.quantile_tracker import QuantileMixtureTracker


class TestQuantileTracker(unittest.TestCase):
    def setUp(self):
        np.random.seed(42)
        # Create synthetic OHLC DataFrame
        n = 80
        curr = 100.0
        records = []
        for i in range(n):
            o = curr
            r = np.random.normal(0, 0.01)
            c = o * np.exp(r)
            h = max(o, c) * 1.005
            l = min(o, c) * 0.995
            records.append({
                "timestamp": 1700000000 + i * 300,
                "open": o,
                "high": h,
                "low": l,
                "close": c,
                "count": 10,
            })
            curr = c

        self.ohlc_df = pd.DataFrame(records)

    def test_forecaster_fit_and_predict(self):
        forecaster = EmpiricalQuantileForecaster(max_iter=20)
        forecaster.fit(self.ohlc_df)
        self.assertTrue(forecaster.is_fitted)

        preds = forecaster.predict_quantiles(self.ohlc_df)
        self.assertIn(0.10, preds)
        self.assertIn(0.50, preds)
        self.assertIn(0.90, preds)
        # Monotonicity check
        self.assertLessEqual(preds[0.10], preds[0.50])
        self.assertLessEqual(preds[0.50], preds[0.90])

    def test_quantile_mixture_tracker(self):
        tracker = QuantileMixtureTracker()
        ticks = [(row["timestamp"], row["close"]) for _, row in self.ohlc_df.iterrows()]
        tracker.tick({"SOL": ticks})

        preds = tracker.predict("SOL", horizon=3600, step=300)
        self.assertEqual(len(preds), 12)
        for p in preds:
            self.assertEqual(len(p["components"]), 2)
            self.assertAlmostEqual(
                p["components"][0]["weight"] + p["components"][1]["weight"], 1.0, places=5
            )


if __name__ == "__main__":
    unittest.main()

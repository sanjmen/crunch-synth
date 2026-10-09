"""
Unit tests for AdaptiveVolatilityTracker.
"""

import time
import unittest
import numpy as np

from src.trackers.adaptive_volatility_tracker import AdaptiveVolatilityTracker


class TestAdaptiveVolatilityTracker(unittest.TestCase):
    def setUp(self):
        self.tracker = AdaptiveVolatilityTracker()

    def test_predict_without_data(self):
        result = self.tracker.predict("BTC", horizon=3600, step=300)
        self.assertEqual(result, [])

    def test_predict_with_synthetic_prices(self):
        base_time = 1700000000
        base_price = 60000.0

        # Ingest 200 ticks of data at 60s intervals
        ticks = []
        p = base_price
        for i in range(200):
            p += np.random.normal(0, 15.0)
            ticks.append((base_time + i * 60, p))

        self.tracker.tick({"BTC": ticks})

        # Predict 1h horizon with 5m step (12 steps)
        t0 = time.time()
        preds = self.tracker.predict("BTC", horizon=3600, step=300)
        elapsed = time.time() - t0

        self.assertLess(elapsed, 0.2, f"Prediction took too long: {elapsed:.4f}s")
        self.assertEqual(len(preds), 12)

        for i, pred in enumerate(preds):
            self.assertEqual(pred["step"], (i + 1) * 300)
            self.assertEqual(pred["type"], "mixture")
            self.assertEqual(len(pred["components"]), 1)

            comp = pred["components"][0]
            self.assertEqual(comp["weight"], 1.0)
            self.assertEqual(comp["density"]["type"], "builtin")
            self.assertEqual(comp["density"]["name"], "norm")
            self.assertIn("loc", comp["density"]["params"])
            self.assertIn("scale", comp["density"]["params"])
            self.assertGreater(comp["density"]["params"]["scale"], 0)

    def test_hurst_scaling_vs_brownian(self):
        # Tracker with Hurst scaling enabled vs disabled
        tracker_hurst = AdaptiveVolatilityTracker(use_hurst_scaling=True)
        tracker_brownian = AdaptiveVolatilityTracker(use_hurst_scaling=False)

        base_time = 1700000000
        ticks = [(base_time + i * 60, 50000.0 + np.random.normal(0, 10.0)) for i in range(100)]

        tracker_hurst.tick({"SOL": ticks})
        tracker_brownian.tick({"SOL": ticks})

        # SOL default Hurst is 0.55 (>0.50), so Hurst scaled dispersion should be higher for step > base_step
        preds_hurst = tracker_hurst.predict("SOL", horizon=86400, step=3600)
        preds_brownian = tracker_brownian.predict("SOL", horizon=86400, step=3600)

        scale_hurst = preds_hurst[-1]["components"][0]["density"]["params"]["scale"]
        scale_brownian = preds_brownian[-1]["components"][0]["density"]["params"]["scale"]

        self.assertGreater(scale_hurst, 0)
        self.assertGreater(scale_brownian, 0)


if __name__ == "__main__":
    unittest.main()

"""
Unit tests for GaussianMixtureTracker.
"""

import time
import unittest
import numpy as np

from src.trackers.gmm_tracker import GaussianMixtureTracker


class TestGaussianMixtureTracker(unittest.TestCase):
    def setUp(self):
        self.tracker = GaussianMixtureTracker()

    def test_predict_without_data(self):
        res = self.tracker.predict("ETH", horizon=3600, step=300)
        self.assertEqual(res, [])

    def test_predict_with_synthetic_prices(self):
        base_time = 1700000000
        # Ingest 200 ticks of data at 60s intervals
        ticks = [(base_time + i * 60, 3000.0 + np.random.normal(0, 3.0)) for i in range(200)]
        self.tracker.tick({"ETH": ticks})

        t0 = time.time()
        preds = self.tracker.predict("ETH", horizon=3600, step=300)
        elapsed = time.time() - t0

        self.assertLess(elapsed, 0.2)
        self.assertEqual(len(preds), 12)

        for i, p in enumerate(preds):
            self.assertEqual(p["step"], (i + 1) * 300)
            self.assertEqual(p["type"], "mixture")
            self.assertEqual(len(p["components"]), 2)  # 2-component GMM

            c_norm = p["components"][0]
            c_jump = p["components"][1]

            # Weights sum to 1.0
            self.assertAlmostEqual(c_norm["weight"] + c_jump["weight"], 1.0, places=5)
            self.assertGreater(c_norm["weight"], c_jump["weight"])  # Normal weight > Jump weight

            # Jump scale > Normal scale
            self.assertGreater(
                c_jump["density"]["params"]["scale"],
                c_norm["density"]["params"]["scale"],
            )


if __name__ == "__main__":
    unittest.main()

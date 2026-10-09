"""
Unit tests for MultiHorizonMetaEnsemble and SubTracker routing architecture.
"""

import time
import unittest
import numpy as np

from src.trackers.multi_horizon_router import (
    MultiHorizonMetaEnsemble,
    HorizonSpecializedSubTracker,
)


class TestMultiHorizonRouter(unittest.TestCase):
    def setUp(self):
        self.meta_tracker = MultiHorizonMetaEnsemble()

    def test_routing_initialization(self):
        # Verify subtrackers are registered
        self.assertIsNotNone(self.meta_tracker.sub_1h)
        self.assertIsNotNone(self.meta_tracker.sub_24h)
        self.assertEqual(len(self.meta_tracker._routes), 2)

    def test_tick_broadcasting(self):
        base_time = 1700000000
        ticks = [(base_time + i * 60, 50000.0 + np.random.normal(0, 10.0)) for i in range(100)]
        self.meta_tracker.tick({"BTC": ticks})

        # Verify prices reached meta tracker PriceStore
        last_meta = self.meta_tracker.prices.get_last_price("BTC")
        self.assertIsNotNone(last_meta)

        # Verify prices reached SubTrackers
        last_1h = self.meta_tracker.sub_1h.prices.get_last_price("BTC")
        self.assertIsNotNone(last_1h)
        self.assertEqual(last_meta[1], last_1h[1])

        # Verify prices reached underlying expert models
        last_expert = self.meta_tracker.sub_1h.expert_student_t.prices.get_last_price("BTC")
        self.assertIsNotNone(last_expert)
        self.assertEqual(last_meta[1], last_expert[1])

    def test_predict_routing_1h_and_24h(self):
        base_time = 1700000000
        ticks = [(base_time + i * 60, 50000.0 + np.random.normal(0, 10.0)) for i in range(100)]
        self.meta_tracker.tick({"BTC": ticks})

        # 1-hour horizon (3600s, step=300s -> 12 steps)
        t0 = time.time()
        preds_1h = self.meta_tracker.predict("BTC", horizon=3600, step=300)
        elapsed_1h = time.time() - t0

        self.assertEqual(len(preds_1h), 12)
        self.assertLess(elapsed_1h, 0.5)

        # 24-hour horizon (86400s, step=3600s -> 24 steps)
        t1 = time.time()
        preds_24h = self.meta_tracker.predict("BTC", horizon=86400, step=3600)
        elapsed_24h = time.time() - t1

        self.assertEqual(len(preds_24h), 24)
        self.assertLess(elapsed_24h, 0.5)

        # Components inside each blended prediction
        for p in preds_1h:
            self.assertEqual(p["type"], "mixture")
            self.assertGreater(len(p["components"]), 1)
            total_w = sum(c["weight"] for c in p["components"])
            self.assertAlmostEqual(total_w, 1.0, places=4)


if __name__ == "__main__":
    unittest.main()

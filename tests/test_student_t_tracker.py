"""
Unit tests for StudentTTracker.
"""

import time
import unittest
import numpy as np

from src.trackers.student_t_tracker import StudentTTracker


class TestStudentTTracker(unittest.TestCase):
    def setUp(self):
        self.tracker = StudentTTracker()

    def test_predict_without_data(self):
        res = self.tracker.predict("BTC", horizon=3600, step=300)
        self.assertEqual(res, [])

    def test_predict_with_synthetic_prices(self):
        base_time = 1700000000
        # Ingest 200 ticks of data at 60s intervals
        ticks = [(base_time + i * 60, 50000.0 + np.random.normal(0, 10.0)) for i in range(200)]
        self.tracker.tick({"BTC": ticks})

        t0 = time.time()
        preds = self.tracker.predict("BTC", horizon=3600, step=300)
        elapsed = time.time() - t0

        self.assertLess(elapsed, 0.2)
        self.assertEqual(len(preds), 12)

        for i, p in enumerate(preds):
            self.assertEqual(p["step"], (i + 1) * 300)
            self.assertEqual(p["type"], "mixture")
            self.assertEqual(len(p["components"]), 1)

            comp = p["components"][0]
            self.assertEqual(comp["weight"], 1.0)
            self.assertEqual(comp["density"]["type"], "builtin")
            self.assertEqual(comp["density"]["name"], "t")

            params = comp["density"]["params"]
            self.assertIn("df", params)
            self.assertIn("loc", params)
            self.assertIn("scale", params)
            self.assertGreater(params["df"], 2.0)
            self.assertGreater(params["scale"], 0)

    def test_calibrated_df_per_asset(self):
        self.assertEqual(self.tracker.get_calibrated_df("BTC"), 4.2)
        self.assertEqual(self.tracker.get_calibrated_df("SP500"), 7.0)
        self.assertEqual(self.tracker.get_calibrated_df("XAUT"), 6.0)


if __name__ == "__main__":
    unittest.main()

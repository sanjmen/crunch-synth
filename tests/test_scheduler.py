"""
Unit tests for ScheduledRetrainer and AtomicModelContainer.
"""

from datetime import timedelta
import threading
import unittest
import numpy as np

from src.retraining.scheduler import AtomicModelContainer, ScheduledRetrainer
from src.trackers.multi_horizon_router import MultiHorizonMetaEnsemble


class TestRetrainingScheduler(unittest.TestCase):
    def test_atomic_model_container(self):
        container = AtomicModelContainer({"param": 10.0})
        self.assertEqual(container.get(), {"param": 10.0})

        # Test atomic swap across threads
        def worker():
            for i in range(100):
                container.swap({"param": float(i)})

        threads = [threading.Thread(target=worker) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        final = container.get()
        self.assertIn("param", final)
        self.assertIsInstance(final["param"], float)

    def test_scheduled_retrainer_initialization_and_tasks(self):
        tracker = MultiHorizonMetaEnsemble()
        retrainer = ScheduledRetrainer(
            tracker=tracker,
            hurst_interval=timedelta(hours=1),
            volatility_interval=timedelta(minutes=30),
            regime_interval=timedelta(minutes=15),
        )

        # Ingest some prices
        base_time = 1700000000
        ticks = [(base_time + i * 60, 50000.0 + np.random.normal(0, 10.0)) for i in range(120)]
        tracker.tick({"BTC": ticks})

        # Test running tasks manually
        retrainer.recalibrate_hurst_task()
        hurst_map = retrainer.hurst_container.get()
        self.assertIn("BTC", hurst_map)

        retrainer.recalibrate_volatility_task()
        vol_map = retrainer.vol_container.get()
        self.assertIsInstance(vol_map, dict)

        retrainer.recalibrate_regime_task()
        regime_map = retrainer.regime_container.get()
        self.assertIsInstance(regime_map, dict)


if __name__ == "__main__":
    unittest.main()

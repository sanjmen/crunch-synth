"""
Unit tests for Hurst exponent estimation and anomalous diffusion scaling in src/features/hurst.py.
"""

import unittest
import numpy as np

from src.features.hurst import (
    estimate_hurst_variance_time,
    estimate_hurst_rs,
    AssetHurstCalibrator,
)


class TestHurst(unittest.TestCase):
    def setUp(self):
        np.random.seed(42)
        # 1. Standard Brownian Motion (random walk): expected H ~ 0.5
        n = 500
        steps = np.random.normal(0, 1.0, size=n)
        self.random_walk = np.exp(np.cumsum(steps * 0.01))

        # 2. Trending / Persistent series: H > 0.5
        trend_steps = np.random.normal(0.02, 0.01, size=n)
        self.trending_series = np.exp(np.cumsum(trend_steps))

    def test_estimate_hurst_variance_time(self):
        h = estimate_hurst_variance_time(self.random_walk)
        self.assertGreater(h, 0.35)
        self.assertLess(h, 0.65)

    def test_estimate_hurst_rs(self):
        rets = np.diff(np.log(self.random_walk))
        h = estimate_hurst_rs(rets)
        self.assertGreater(h, 0.35)
        self.assertLess(h, 0.70)

    def test_asset_hurst_calibrator_defaults(self):
        calibrator = AssetHurstCalibrator()

        # Check default asset mappings
        self.assertEqual(calibrator.get_hurst("BTC"), 0.52)
        self.assertEqual(calibrator.get_hurst("SP500"), 0.47)
        self.assertEqual(calibrator.get_hurst("XAUT"), 0.46)
        self.assertEqual(calibrator.get_hurst("UNKNOWN_ASSET"), 0.50)

    def test_scale_dispersion(self):
        calibrator = AssetHurstCalibrator()

        base_sigma = 10.0
        # 1-hour step (3600s) from 5-minute base (300s): ratio = 12
        # For BTC (H=0.52): 10 * 12^0.52
        scaled_btc = calibrator.scale_dispersion(base_sigma, step_seconds=3600, base_step_seconds=300, asset="BTC")
        # For SP500 (H=0.47): 10 * 12^0.47
        scaled_sp = calibrator.scale_dispersion(base_sigma, step_seconds=3600, base_step_seconds=300, asset="SP500")

        self.assertGreater(scaled_btc, base_sigma)
        self.assertGreater(scaled_sp, base_sigma)
        # Because BTC has higher H (0.52 > 0.47), scaled dispersion must be higher
        self.assertGreater(scaled_btc, scaled_sp)


if __name__ == "__main__":
    unittest.main()

"""
Unit tests for microstructure and spread estimators in src/features/microstructure.py.
"""

import unittest
import numpy as np
import pandas as pd

from src.features.microstructure import (
    corwin_schultz_spread,
    roll_effective_spread,
    amihud_illiquidity,
    LiquidityRegimeDetector,
)


class TestMicrostructure(unittest.TestCase):
    def setUp(self):
        np.random.seed(123)
        self.highs = [100.0, 102.0, 101.5, 103.0, 104.0, 103.5]
        self.lows = [98.0, 99.0, 99.5, 101.0, 102.0, 101.0]
        self.closes = [99.0, 101.0, 100.5, 102.5, 103.0, 102.0]
        self.volumes = [1000.0, 1200.0, 900.0, 5000.0, 1100.0, 1300.0]

    def test_corwin_schultz_spread(self):
        spreads = corwin_schultz_spread(self.highs, self.lows)
        self.assertEqual(len(spreads), len(self.highs))
        self.assertTrue((spreads >= 0.0).all())
        self.assertTrue(np.isfinite(spreads).all())

    def test_roll_effective_spread(self):
        # Generate mean-reverting price series with bid-ask bounce
        prices = [100.0, 100.5, 100.0, 100.5, 100.0, 100.5]
        s = roll_effective_spread(prices)
        self.assertGreaterEqual(s, 0.0)

    def test_amihud_illiquidity(self):
        rets = np.diff(np.log(self.closes))
        vols = self.volumes[1:]
        illiq = amihud_illiquidity(rets, vols)
        self.assertGreaterEqual(illiq, 0.0)
        self.assertTrue(np.isfinite(illiq))

    def test_liquidity_regime_detector(self):
        detector = LiquidityRegimeDetector(volume_z_threshold=2.0)
        recent_vols = [1000, 1100, 1050, 980, 1020, 1010]
        recent_spreads = [0.001, 0.0012, 0.0011, 0.0009, 0.0010]

        # Normal condition
        mult_normal, shock_normal = detector.compute_shock_multiplier(
            current_volume=1050,
            recent_volumes=recent_vols,
            current_spread=0.0011,
            recent_spreads=recent_spreads,
        )
        self.assertEqual(mult_normal, 1.0)
        self.assertFalse(shock_normal)

        # Volume shock condition (5000 is 100x std)
        mult_shock, shock_detected = detector.compute_shock_multiplier(
            current_volume=5000,
            recent_volumes=recent_vols,
            current_spread=0.0011,
            recent_spreads=recent_spreads,
        )
        self.assertGreater(mult_shock, 1.0)
        self.assertTrue(shock_detected)


if __name__ == "__main__":
    unittest.main()

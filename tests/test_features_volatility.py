"""
Unit tests for volatility estimators and online filters in src/features/volatility.py.
"""

import unittest
import numpy as np
import pandas as pd

from src.features.volatility import (
    ticks_to_ohlc,
    realized_volatility,
    parkinson_volatility,
    garman_klass_volatility,
    rogers_satchell_volatility,
    OnlineEWMAVolatility,
    OnlineGARCH11,
)


class TestVolatilityEstimators(unittest.TestCase):
    def setUp(self):
        np.random.seed(42)
        # Generate 200 synthetic 5-min bars
        base_price = 100.0
        n_bars = 100

        self.opens = []
        self.highs = []
        self.lows = []
        self.closes = []

        curr = base_price
        for _ in range(n_bars):
            o = curr
            ret = np.random.normal(0, 0.01)
            c = o * np.exp(ret)
            intra_std = abs(ret) + 0.005
            h = max(o, c) * np.exp(abs(np.random.normal(0, intra_std)))
            l = min(o, c) * np.exp(-abs(np.random.normal(0, intra_std)))

            self.opens.append(o)
            self.highs.append(h)
            self.lows.append(l)
            self.closes.append(c)
            curr = c

    def test_ticks_to_ohlc(self):
        # 1700000100 is an exact multiple of 300 (5666667 * 300)
        base = 1700000100
        ticks = [
            (base + 0, 100.0),
            (base + 60, 105.0),
            (base + 120, 95.0),
            (base + 290, 102.0),
            # Next bar (at base + 300)
            (base + 300, 103.0),
            (base + 360, 108.0),
        ]
        df_ohlc = ticks_to_ohlc(ticks, bar_seconds=300)
        self.assertEqual(len(df_ohlc), 2)
        # Bar 1
        b1 = df_ohlc.iloc[0]
        self.assertEqual(b1["open"], 100.0)
        self.assertEqual(b1["high"], 105.0)
        self.assertEqual(b1["low"], 95.0)
        self.assertEqual(b1["close"], 102.0)
        self.assertEqual(b1["count"], 4)

        # Bar 2
        b2 = df_ohlc.iloc[1]
        self.assertEqual(b2["open"], 103.0)
        self.assertEqual(b2["high"], 108.0)
        self.assertEqual(b2["low"], 103.0)
        self.assertEqual(b2["close"], 108.0)
        self.assertEqual(b2["count"], 2)

    def test_realized_volatility(self):
        rv = realized_volatility(self.closes)
        self.assertGreater(rv, 0.0)
        self.assertLess(rv, 0.5)

        rv_ann = realized_volatility(self.closes, annualize=True)
        self.assertGreater(rv_ann, rv)

    def test_parkinson_volatility(self):
        pv = parkinson_volatility(self.highs, self.lows)
        self.assertGreater(pv, 0.0)
        self.assertLess(pv, 0.5)

    def test_garman_klass_volatility(self):
        gk = garman_klass_volatility(self.opens, self.highs, self.lows, self.closes)
        self.assertGreater(gk, 0.0)
        self.assertLess(gk, 0.5)

    def test_rogers_satchell_volatility(self):
        rs = rogers_satchell_volatility(self.opens, self.highs, self.lows, self.closes)
        self.assertGreater(rs, 0.0)
        self.assertLess(rs, 0.5)

    def test_online_ewma_filter(self):
        ewma = OnlineEWMAVolatility(decay=0.94)
        rets = np.random.normal(0, 0.02, size=50)

        ewma.warm_up(rets[:30])
        self.assertTrue(ewma.initialized)
        vol_before = ewma.current_volatility
        self.assertGreater(vol_before, 0.0)

        # Update with a large shock
        vol_after = ewma.update(0.10)
        self.assertGreater(vol_after, vol_before)

    def test_online_garch_filter_and_projection(self):
        garch = OnlineGARCH11(omega=1e-5, alpha=0.10, beta=0.85)
        rets = np.random.normal(0, 0.015, size=50)

        garch.warm_up(rets)
        vol = garch.current_volatility
        self.assertGreater(vol, 0.0)

        # Forecast forward step variance
        step_var_1 = garch.forecast_step_variance(1)
        step_var_10 = garch.forecast_step_variance(10)
        self.assertGreater(step_var_1, 0.0)
        self.assertGreater(step_var_10, 0.0)

        # Forecast cumulative volatility
        cum_vol_12 = garch.forecast_cumulative_volatility(12)
        cum_vol_1 = garch.forecast_cumulative_volatility(1)
        self.assertGreater(cum_vol_12, cum_vol_1)


if __name__ == "__main__":
    unittest.main()

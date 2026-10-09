"""
Unit tests for MarketRegimeClassifier in src/features/regime_classifier.py.
"""

import unittest
import numpy as np
import pandas as pd

from src.features.regime_classifier import MarketRegimeClassifier, MarketRegime


class TestRegimeClassifier(unittest.TestCase):
    def setUp(self):
        self.classifier = MarketRegimeClassifier()

    def test_trending_regime(self):
        # Generate monotonic upward trend
        n = 50
        prices = [100.0 + i * 2.0 + np.random.normal(0, 0.1) for i in range(n)]
        ohlc = pd.DataFrame({
            "open": prices,
            "high": [p + 0.5 for p in prices],
            "low": [p - 0.5 for p in prices],
            "close": prices,
        })

        res = self.classifier.classify_from_ohlc(ohlc, asset="BTC")
        self.assertIn("regime", res)
        self.assertEqual(res["regime"], MarketRegime.TRENDING)
        self.assertGreater(res["weights"][MarketRegime.TRENDING.value], 0.5)

    def test_volatile_regime(self):
        # Generate stable history followed by extreme volatility explosion
        n = 50
        base = [100.0 + np.random.normal(0, 0.1) for _ in range(40)]
        jump = [100.0 + np.random.normal(0, 5.0) for _ in range(10)]
        prices = base + jump

        highs = [p + 0.1 for p in base] + [p + 10.0 for p in jump]
        lows = [p - 0.1 for p in base] + [p - 10.0 for p in jump]

        ohlc = pd.DataFrame({
            "open": prices,
            "high": highs,
            "low": lows,
            "close": prices,
        })

        res = self.classifier.classify_from_ohlc(ohlc, asset="BTC")
        self.assertEqual(res["regime"], MarketRegime.VOLATILE)
        self.assertGreater(res["weights"][MarketRegime.VOLATILE.value], 0.5)

    def test_ranging_regime(self):
        # Constant quiescent prices
        n = 50
        prices = [100.0 + np.random.normal(0, 0.05) for _ in range(n)]
        ohlc = pd.DataFrame({
            "open": prices,
            "high": [p + 0.1 for p in prices],
            "low": [p - 0.1 for p in prices],
            "close": prices,
        })

        res = self.classifier.classify_from_ohlc(ohlc, asset="SP500")
        self.assertEqual(res["regime"], MarketRegime.RANGING)


if __name__ == "__main__":
    unittest.main()

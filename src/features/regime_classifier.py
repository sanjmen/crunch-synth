"""
Dynamic Market Regime Classification (Trending vs Ranging vs Volatile) for CrunchDAO Synth.

Classifies incoming asset price dynamics into discrete or continuous regime probabilities:
1. TRENDING: Strong directional drift and persistent order flow (favors directional Hurst scaling).
2. RANGING: Low volatility and mean-reverting microstructure (favors tight quiescent densities).
3. VOLATILE: High volatility z-score and spread expansion (favors multi-modal GMM jump components).
"""

from enum import Enum
from typing import Dict, List, Optional, Tuple, Any, Union
import numpy as np
import pandas as pd

from src.features.volatility import garman_klass_volatility, parkinson_volatility
from src.features.microstructure import corwin_schultz_spread


class MarketRegime(str, Enum):
    TRENDING = "TRENDING"
    RANGING = "RANGING"
    VOLATILE = "VOLATILE"


class MarketRegimeClassifier:
    """
    Classifies market state into TRENDING, RANGING, or VOLATILE regimes
    using rolling volatility z-scores, trend signal-to-noise ratios, and spread proxies.
    """

    def __init__(
        self,
        vol_z_threshold: float = 1.25,
        trend_snr_threshold: float = 0.65,
        rolling_window: int = 24,
    ):
        self.vol_z_threshold = vol_z_threshold
        self.trend_snr_threshold = trend_snr_threshold
        self.rolling_window = rolling_window

    def classify_from_ohlc(
        self,
        ohlc_df: pd.DataFrame,
        asset: str = "BTC",
    ) -> Dict[str, Any]:
        """
        Computes regime classification and continuous regime probability weights.

        Returns
        -------
        dict with:
            'regime': MarketRegime
            'trend_snr': float
            'vol_zscore': float
            'weights': Dict[str, float]
        """
        if len(ohlc_df) < 12:
            return {
                "regime": MarketRegime.RANGING,
                "trend_snr": 0.0,
                "vol_zscore": 0.0,
                "weights": {
                    MarketRegime.TRENDING.value: 0.20,
                    MarketRegime.RANGING.value: 0.60,
                    MarketRegime.VOLATILE.value: 0.20,
                },
            }

        closes = ohlc_df["close"].to_numpy()
        highs = ohlc_df["high"].to_numpy()
        lows = ohlc_df["low"].to_numpy()
        opens = ohlc_df["open"].to_numpy()

        n = len(closes)
        recent_n = min(n, self.rolling_window)

        sub_c = closes[-recent_n:]
        sub_h = highs[-recent_n:]
        sub_l = lows[-recent_n:]
        sub_o = opens[-recent_n:]

        # 1. Volatility estimation & z-score: compare recent 6 bars vs prior history
        if n >= 12:
            curr_gk = garman_klass_volatility(opens[-6:], highs[-6:], lows[-6:], closes[-6:])
            hist_gk = garman_klass_volatility(opens[:-6], highs[:-6], lows[:-6], closes[:-6])
        else:
            curr_gk = garman_klass_volatility(opens, highs, lows, closes)
            hist_gk = curr_gk

        if hist_gk > 1e-8:
            vol_ratio = curr_gk / hist_gk
            vol_zscore = float((vol_ratio - 1.0) / 0.35)
        else:
            vol_zscore = 0.0

        # 2. Trend Signal-to-Noise Ratio (SNR)
        # Linear slope over recent closes normalized by volatility
        x = np.arange(recent_n, dtype=np.float64)
        poly = np.polyfit(x, sub_c, 1)
        slope = poly[0]
        mean_p = np.mean(sub_c)
        pct_slope = slope / max(mean_p, 1e-4)

        trend_snr = float(abs(pct_slope) / max(hist_gk, 1e-4))

        # 3. Decision logic
        if vol_zscore > self.vol_z_threshold:
            regime = MarketRegime.VOLATILE
            weights = {
                MarketRegime.VOLATILE.value: 0.65,
                MarketRegime.TRENDING.value: 0.20,
                MarketRegime.RANGING.value: 0.15,
            }
        elif trend_snr > self.trend_snr_threshold:
            regime = MarketRegime.TRENDING
            weights = {
                MarketRegime.TRENDING.value: 0.65,
                MarketRegime.RANGING.value: 0.20,
                MarketRegime.VOLATILE.value: 0.15,
            }
        else:
            regime = MarketRegime.RANGING
            weights = {
                MarketRegime.RANGING.value: 0.65,
                MarketRegime.TRENDING.value: 0.20,
                MarketRegime.VOLATILE.value: 0.15,
            }

        return {
            "regime": regime,
            "trend_snr": round(trend_snr, 4),
            "vol_zscore": round(vol_zscore, 4),
            "weights": weights,
        }

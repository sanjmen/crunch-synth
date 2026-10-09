"""
Microstructure, Spread, and Liquidity Shock Indicators for CrunchDAO Synth.

Implements:
1. Corwin-Schultz (2012) High-Low Bid-Ask Spread Estimator.
2. Roll (1984) Effective Spread from serial covariance of returns.
3. Amihud Illiquidity proxy from return and volume.
4. Liquidity Shock & Volatility Multiplier detection.
"""

from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd


def corwin_schultz_spread(
    high: Union[np.ndarray, pd.Series, List[float]],
    low: Union[np.ndarray, pd.Series, List[float]],
) -> np.ndarray:
    """
    Computes Corwin-Schultz (2012) Bid-Ask Spread from consecutive high-low pairs.

    Parameters
    ----------
    high : array-like
        High price series.
    low : array-like
        Low price series.

    Returns
    -------
    np.ndarray
        Estimated effective percentage bid-ask spread per bar (floored at 0).
    """
    h = np.asarray(high, dtype=np.float64)
    l = np.asarray(low, dtype=np.float64)

    n = len(h)
    if n < 2 or len(l) != n:
        return np.zeros(n, dtype=np.float64)

    spreads = np.zeros(n, dtype=np.float64)
    sqrt2 = np.sqrt(2.0)
    denom = 3.0 - 2.0 * sqrt2  # ~0.171572875

    for t in range(1, n):
        h0, l0 = h[t - 1], l[t - 1]
        h1, l1 = h[t], l[t]

        if h0 <= 0 or l0 <= 0 or h1 <= 0 or l1 <= 0 or h0 < l0 or h1 < l1:
            spreads[t] = 0.0
            continue

        hl0 = np.log(h0 / l0)
        hl1 = np.log(h1 / l1)
        beta = hl0**2 + hl1**2

        h2 = max(h0, h1)
        l2 = min(l0, l1)
        gamma = (np.log(h2 / l2)) ** 2

        alpha = (sqrt2 * np.sqrt(beta) - np.sqrt(beta)) / denom - np.sqrt(gamma / denom)
        if alpha < 0:
            spreads[t] = 0.0
        else:
            spread = 2.0 * (np.exp(alpha) - 1.0) / (1.0 + np.exp(alpha))
            spreads[t] = max(0.0, float(spread))

    return spreads


def roll_effective_spread(prices: Union[np.ndarray, pd.Series, List[float]]) -> float:
    """
    Computes Roll (1984) Effective Spread from serial covariance of price increments.
    S = 2 * sqrt(-cov(dr_t, dr_{t-1})) when cov < 0, else 0.
    """
    p = np.asarray(prices, dtype=np.float64)
    if len(p) < 3:
        return 0.0

    rets = np.diff(np.log(p))
    cov = np.cov(rets[1:], rets[:-1])[0, 1] if len(rets) > 2 else 0.0

    if cov < 0:
        return float(2.0 * np.sqrt(-cov))
    return 0.0


def amihud_illiquidity(
    returns: Union[np.ndarray, pd.Series, List[float]],
    volumes: Union[np.ndarray, pd.Series, List[float]],
) -> float:
    """
    Computes Amihud (2002) Illiquidity ratio:
    ILLIQ = mean( |r_t| / Volume_t )
    """
    r = np.asarray(returns, dtype=np.float64)
    v = np.asarray(volumes, dtype=np.float64)

    if len(r) == 0 or len(v) != len(r):
        return 0.0

    valid = (v > 0) & np.isfinite(r)
    if not np.any(valid):
        return 0.0

    r_abs = np.abs(r[valid])
    v_val = v[valid]
    return float(np.mean(r_abs / v_val))


class LiquidityRegimeDetector:
    """
    Detects market liquidity shocks using volume anomalies and spread expansions.
    Provides a dynamic volatility multiplier to adjust forecast dispersions during shocks.
    """

    def __init__(
        self,
        volume_z_threshold: float = 2.0,
        spread_quantile_threshold: float = 0.85,
        max_multiplier: float = 2.5,
    ):
        self.vol_z_thresh = volume_z_threshold
        self.spread_thresh = spread_quantile_threshold
        self.max_mult = max_multiplier

    def compute_shock_multiplier(
        self,
        current_volume: float,
        recent_volumes: Union[np.ndarray, List[float]],
        current_spread: float,
        recent_spreads: Union[np.ndarray, List[float]],
    ) -> Tuple[float, bool]:
        """
        Calculates a scaling multiplier >= 1.0 based on current volume and spread anomalies.

        Returns
        -------
        (multiplier, is_shock)
        """
        vols = np.asarray(recent_volumes, dtype=np.float64)
        spreads = np.asarray(recent_spreads, dtype=np.float64)

        vol_shock = False
        spread_shock = False
        mult = 1.0

        if len(vols) >= 5 and np.std(vols) > 0:
            mean_v = np.mean(vols)
            std_v = np.std(vols)
            z_vol = (current_volume - mean_v) / std_v
            if z_vol > self.vol_z_thresh:
                vol_shock = True
                mult += min(0.5, (z_vol - self.vol_z_thresh) * 0.25)

        if len(spreads) >= 5:
            thresh_val = np.percentile(spreads, self.spread_thresh * 100)
            if current_spread > thresh_val and thresh_val > 0:
                spread_shock = True
                ratio = current_spread / thresh_val
                mult += min(1.0, (ratio - 1.0) * 0.5)

        is_shock = vol_shock or spread_shock
        final_mult = float(min(self.max_mult, max(1.0, mult)))
        return final_mult, is_shock

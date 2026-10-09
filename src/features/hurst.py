"""
Hurst Exponent Estimation and Anomalous Temporal Diffusion Scaling for CrunchDAO Synth.

Replaces standard Brownian sqrt(step) diffusion scaling with empirical
fractal scaling:
    sigma(step) = sigma_base * (step / base_step) ** H

Where:
- H < 0.5: Sub-diffusive (mean-reverting / anti-persistent)
- H = 0.5: Standard Brownian motion (random walk)
- H > 0.5: Super-diffusive (trending / persistent)
"""

from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd


def estimate_hurst_variance_time(
    prices: Union[np.ndarray, pd.Series, List[float]],
    lags: Optional[List[int]] = None,
) -> float:
    """
    Estimates the Hurst exponent using log-log Variance-Time regression:
        ln(Std(r_tau)) = H * ln(tau) + C

    Parameters
    ----------
    prices : array-like
        Chronological price series.
    lags : Optional[List[int]]
        List of integer bar lags over which to measure return dispersion.
        Defaults to [1, 2, 4, 8, 16, 32, 64].

    Returns
    -------
    float
        Estimated Hurst exponent H, clipped to [0.05, 0.95].
    """
    p = np.asarray(prices, dtype=np.float64)
    if len(p) < 64:
        return 0.5  # Fallback to standard Brownian motion

    log_p = np.log(p)
    lags = lags or [1, 2, 4, 8, 16, 32, 64]
    valid_lags = [lag for lag in lags if lag < len(log_p) // 4]

    if len(valid_lags) < 3:
        return 0.5

    log_lags = []
    log_stds = []

    for lag in valid_lags:
        # Lagged log returns
        rets = log_p[lag:] - log_p[:-lag]
        std_ret = np.std(rets)
        if std_ret > 1e-12:
            log_lags.append(np.log(lag))
            log_stds.append(np.log(std_ret))

    if len(log_lags) < 3:
        return 0.5

    # Linear regression: slope = H
    poly = np.polyfit(log_lags, log_stds, 1)
    hurst_val = float(poly[0])

    # Guard bounds for physical validity
    return float(np.clip(hurst_val, 0.05, 0.95))


def estimate_hurst_rs(
    series: Union[np.ndarray, pd.Series, List[float]],
    min_chunk: int = 16,
    max_chunk: Optional[int] = None,
) -> float:
    """
    Estimates the Hurst exponent via Classical Rescaled Range (R/S) Analysis.

    Parameters
    ----------
    series : array-like
        Return or increment series.
    min_chunk : int
        Minimum window chunk size.
    max_chunk : Optional[int]
        Maximum chunk size (defaults to len(series) // 2).

    Returns
    -------
    float
        Estimated Hurst exponent H in [0.05, 0.95].
    """
    x = np.asarray(series, dtype=np.float64)
    n = len(x)
    if n < min_chunk * 4:
        return 0.5

    max_chunk = max_chunk or (n // 2)
    # Generate powers of 2 for chunk sizes
    chunks = []
    curr = min_chunk
    while curr <= max_chunk:
        chunks.append(curr)
        curr = int(curr * 1.5)

    rs_values = []
    chunk_sizes = []

    for chunk in chunks:
        num_splits = n // chunk
        if num_splits < 2:
            continue

        rs_chunk_list = []
        for i in range(num_splits):
            segment = x[i * chunk : (i + 1) * chunk]
            m = np.mean(segment)
            s = np.std(segment)
            if s <= 1e-12:
                continue

            y = np.cumsum(segment - m)
            r = np.max(y) - np.min(y)
            rs_chunk_list.append(r / s)

        if rs_chunk_list:
            rs_values.append(np.log(np.mean(rs_chunk_list)))
            chunk_sizes.append(np.log(chunk))

    if len(chunk_sizes) < 3:
        return 0.5

    poly = np.polyfit(chunk_sizes, rs_values, 1)
    return float(np.clip(poly[0], 0.05, 0.95))


class AssetHurstCalibrator:
    """
    Maintains calibrated Hurst scaling exponents for all Synth assets.
    Provides anomalous diffusion step scaling:
        scale_step(sigma_base, step, base_step) = sigma_base * (step / base_step) ** H
    """

    # Empirical calibrated default baselines for Synth competition assets
    DEFAULT_HURST: Dict[str, float] = {
        # Crypto: slightly super-diffusive / trending on intraday horizons
        "BTC": 0.52,
        "ETH": 0.53,
        "SOL": 0.55,
        "XRP": 0.51,
        "HYPE": 0.54,
        # Equities: mean-reverting microstructure noise / sub-diffusive intraday
        "SP500": 0.47,
        "NVDAX": 0.51,
        "TSLAX": 0.52,
        "AAPLX": 0.48,
        "GOOGLX": 0.48,
        # Commodities: sub-diffusive / mean-reverting ranges
        "XAUT": 0.46,
        "WTIOIL": 0.49,
    }

    def __init__(self, asset_hurst: Optional[Dict[str, float]] = None):
        self.hurst_map = dict(self.DEFAULT_HURST)
        if asset_hurst:
            self.hurst_map.update(asset_hurst)

    def get_hurst(self, asset: str) -> float:
        """Returns the calibrated Hurst exponent for the given asset."""
        return self.hurst_map.get(asset.upper(), 0.50)

    def calibrate_from_prices(
        self,
        asset: str,
        prices: Union[np.ndarray, List[float]],
    ) -> float:
        """
        Calibrates H dynamically from historical price series and stores the parameter.
        """
        h = estimate_hurst_variance_time(prices)
        self.hurst_map[asset.upper()] = h
        return h

    def scale_dispersion(
        self,
        base_sigma: float,
        step_seconds: int,
        base_step_seconds: int = 300,
        asset: str = "BTC",
    ) -> float:
        """
        Applies fractal anomalous diffusion scaling to base standard deviation:
            sigma(step) = base_sigma * (step / base_step) ** H
        """
        if step_seconds <= 0 or base_step_seconds <= 0:
            return base_sigma

        h = self.get_hurst(asset)
        ratio = step_seconds / float(base_step_seconds)
        scale_factor = ratio**h
        return float(base_sigma * scale_factor)

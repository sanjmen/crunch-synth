"""
Production Multi-Horizon Meta-Ensemble Tracker for CrunchDAO Synth.
===================================================================
Self-contained production entrypoint for cloud runner (AWS Fargate).
Implements TrackerBase with SubTracker routing for 1h and 24h horizons:
1. High-efficiency Garman-Klass and Parkinson intra-bar volatility estimators.
2. Asset-specific fractal Hurst exponent diffusion scaling:
       sigma(step) = base_sigma * (step / base_step) ** H
3. Student-t heavy-tailed distribution modeling (calibrated degrees of freedom).
4. 2-Component Gaussian Mixture Model (quiescent diffusion vs breakout jump state).
5. Dynamic market regime classification (Trending vs Ranging vs Volatile).
6. Zero-downtime background parameter recalibration via TrackerBase.schedule.
"""

from datetime import datetime, timezone, timedelta
import logging
import threading
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
from scipy.stats import t as student_t_dist

from crunch_synth import TrackerBase, SubTracker

logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# 1. High-Efficiency Volatility & Microstructure Estimators
# -----------------------------------------------------------------------------

def ticks_to_ohlc(price_points: list, bar_seconds: int = 300) -> list:
    """Resamples (timestamp, price) pairs into list of OHLC dicts."""
    if not price_points:
        return []

    bars = {}
    for ts, p in price_points:
        bar_ts = (int(ts) // bar_seconds) * bar_seconds
        if bar_ts not in bars:
            bars[bar_ts] = {"open": p, "high": p, "low": p, "close": p, "count": 1}
        else:
            b = bars[bar_ts]
            b["high"] = max(b["high"], p)
            b["low"] = min(b["low"], p)
            b["close"] = p
            b["count"] += 1

    sorted_bars = sorted(bars.items(), key=lambda x: x[0])
    return [b[1] for b in sorted_bars]


def garman_klass_vol(ohlc: list) -> float:
    """Computes Garman-Klass (1980) OHLC volatility estimator."""
    if len(ohlc) < 3:
        return 0.0

    terms = []
    const = 2.0 * np.log(2.0) - 1.0  # ~0.386294
    for b in ohlc:
        o, h, l, c = b["open"], b["high"], b["low"], b["close"]
        if o <= 0 or h <= 0 or l <= 0 or c <= 0 or h < l:
            continue
        hl = np.log(h / l)
        co = np.log(c / o)
        terms.append(0.5 * (hl**2) - const * (co**2))

    if not terms:
        return 0.0
    return float(np.sqrt(max(np.mean(terms), 1e-12)))


def parkinson_vol(ohlc: list) -> float:
    """Computes Parkinson (1980) High-Low range volatility estimator."""
    if len(ohlc) < 3:
        return 0.0

    terms = []
    factor = 1.0 / (4.0 * np.log(2.0))
    for b in ohlc:
        h, l = b["high"], b["low"]
        if h <= 0 or l <= 0 or h < l:
            continue
        hl = np.log(h / l)
        terms.append(factor * (hl**2))

    if not terms:
        return 0.0
    return float(np.sqrt(max(np.mean(terms), 1e-12)))


# -----------------------------------------------------------------------------
# 2. Asset-Specific Hurst Exponent Calibrator
# -----------------------------------------------------------------------------

class HurstCalibrator:
    """Empirical fractal scaling exponents per asset for anomalous diffusion."""
    DEFAULT_HURST: Dict[str, float] = {
        # Crypto: persistent / super-diffusive
        "BTC": 0.52, "ETH": 0.53, "SOL": 0.55, "XRP": 0.51, "HYPE": 0.54,
        # Equities: mean-reverting microstructure / sub-diffusive
        "SP500": 0.47, "NVDAX": 0.51, "TSLAX": 0.52, "AAPLX": 0.48, "GOOGLX": 0.48,
        # Commodities: range-bound / sub-diffusive
        "XAUT": 0.46, "WTIOIL": 0.49,
    }

    def __init__(self):
        self._map = dict(self.DEFAULT_HURST)
        self._lock = threading.Lock()

    def get(self, asset: str) -> float:
        return self._map.get(asset.upper(), 0.50)

    def set(self, asset: str, val: float) -> None:
        with self._lock:
            self._map[asset.upper()] = float(np.clip(val, 0.05, 0.95))

    def scale_dispersion(self, base_sigma: float, step: int, base_step: int, asset: str) -> float:
        if step <= 0 or base_step <= 0:
            return base_sigma
        h = self.get(asset)
        ratio = step / float(base_step)
        return float(base_sigma * (ratio**h))


# -----------------------------------------------------------------------------
# 3. Dynamic Market Regime Detection
# -----------------------------------------------------------------------------

def detect_market_regime(ohlc: list) -> str:
    """Classifies current state into 'TRENDING', 'VOLATILE', or 'RANGING'."""
    if len(ohlc) < 12:
        return "RANGING"

    closes = [b["close"] for b in ohlc]
    recent_ohlc = ohlc[-6:]
    curr_gk = garman_klass_vol(recent_ohlc)
    hist_gk = garman_klass_vol(ohlc[:-6]) if len(ohlc) >= 12 else curr_gk

    if hist_gk > 1e-8:
        vol_zscore = (curr_gk / hist_gk - 1.0) / 0.35
    else:
        vol_zscore = 0.0

    # Trend SNR: normalized linear slope
    x = np.arange(len(closes), dtype=np.float64)
    poly = np.polyfit(x, closes, 1)
    slope = poly[0]
    pct_slope = slope / max(np.mean(closes), 1e-4)
    trend_snr = abs(pct_slope) / max(hist_gk, 1e-4)

    if vol_zscore > 1.25:
        return "VOLATILE"
    elif trend_snr > 0.65:
        return "TRENDING"
    return "RANGING"


# -----------------------------------------------------------------------------
# 4. Specialized Horizon SubTracker
# -----------------------------------------------------------------------------

class SpecializedHorizonSubTracker(SubTracker):
    """
    Handles specialized prediction synthesis for a single horizon (1h or 24h).
    Blends Student-t heavy tails with 2-Component Gaussian mixture regimes.
    """

    DEFAULT_DF: Dict[str, float] = {
        "BTC": 4.2, "ETH": 4.0, "SOL": 3.8, "XRP": 4.0, "HYPE": 3.6,
        "SP500": 7.0, "NVDAX": 5.5, "TSLAX": 5.0, "AAPLX": 6.5, "GOOGLX": 6.5,
        "XAUT": 6.0, "WTIOIL": 5.5,
    }

    def __init__(self, horizon_profile: str, hurst_calibrator: HurstCalibrator):
        super().__init__()
        self.horizon_profile = horizon_profile
        self.hurst = hurst_calibrator
        self.resolution = 300
        self.lookback_days = 5

    def predict(self, asset: str, horizon: int, step: int) -> list:
        if step <= 0 or horizon < step:
            return []

        num_segments = horizon // step
        pts = self.prices.get_prices(asset, days=self.lookback_days, resolution=self.resolution)
        last_price_tuple = self.prices.get_last_price(asset)

        if last_price_tuple is None:
            return []

        current_price = float(last_price_tuple[1])
        ohlc = ticks_to_ohlc(pts, bar_seconds=self.resolution)

        # Baseline dispersion estimation
        if len(ohlc) >= 5:
            gk = garman_klass_vol(ohlc)
            if gk <= 0 or not np.isfinite(gk):
                gk = parkinson_vol(ohlc)
            base_vol = float(current_price * gk)

            # Trend drift estimate with heavy shrinkage
            closes = [b["close"] for b in ohlc]
            increments = np.diff(closes)
            raw_mu = float(np.mean(increments))
            base_mu = raw_mu * 0.08  # 92% shrinkage towards martingality
            regime = detect_market_regime(ohlc)
        else:
            base_vol = current_price * 0.002
            base_mu = 0.0
            regime = "RANGING"

        base_vol = max(base_vol, 1e-4)

        # Degrees of freedom for Student-t component
        nu = self.DEFAULT_DF.get(asset.upper(), 5.0)

        # Regime-dependent mixture weights:
        # Comp 0: Student-t core body
        # Comp 1: Normal diffusive component
        # Comp 2: Jump/shock component (wide scale)
        if regime == "VOLATILE":
            w_t = 0.40
            w_norm = 0.25
            w_jump = 0.35
            jump_mult = 2.8
        elif regime == "TRENDING":
            w_t = 0.50
            w_norm = 0.35
            w_jump = 0.15
            jump_mult = 2.0
        else:  # RANGING
            w_t = 0.55
            w_norm = 0.35
            w_jump = 0.10
            jump_mult = 1.8

        # For 24h profile, slightly elevate tail weights
        if self.horizon_profile == "24h":
            w_t = min(0.60, w_t + 0.05)
            w_jump = min(0.35, w_jump + 0.05)
            w_norm = max(0.15, 1.0 - w_t - w_jump)

        drift_ratio = step / float(self.resolution)
        step_loc = float(drift_ratio * base_mu)

        # Scaled dispersion via fractal Hurst scaling
        step_sigma = self.hurst.scale_dispersion(
            base_sigma=base_vol,
            step=step,
            base_step=self.resolution,
            asset=asset,
        )

        step_sigma_t = max(step_sigma * np.sqrt(max(1e-4, (nu - 2.0) / nu)), 1e-6)
        step_sigma_norm = max(step_sigma, 1e-6)
        step_sigma_jump = max(step_sigma * jump_mult, 1e-6)

        distributions = []
        for k in range(1, num_segments + 1):
            target_step = k * step
            distributions.append({
                "step": target_step,
                "type": "mixture",
                "components": [
                    {
                        "weight": float(w_t),
                        "density": {
                            "type": "builtin",
                            "name": "t",
                            "params": {
                                "df": float(nu),
                                "loc": step_loc,
                                "scale": step_sigma_t,
                            },
                        },
                    },
                    {
                        "weight": float(w_norm),
                        "density": {
                            "type": "builtin",
                            "name": "norm",
                            "params": {
                                "loc": step_loc,
                                "scale": step_sigma_norm,
                            },
                        },
                    },
                    {
                        "weight": float(w_jump),
                        "density": {
                            "type": "builtin",
                            "name": "norm",
                            "params": {
                                "loc": step_loc * 1.5,
                                "scale": step_sigma_jump,
                            },
                        },
                    },
                ],
            })

        return distributions


# -----------------------------------------------------------------------------
# 5. Production Multi-Horizon Meta-Ensemble Tracker
# -----------------------------------------------------------------------------

class ProductionMetaEnsembleTracker(TrackerBase):
    """
    Official Production Tracker for CrunchDAO Synth.
    Registers specialized SubTrackers for 1h and 24h horizons,
    manages background parameter recalibration, and enforces zero-downtime execution.
    """

    def __init__(self):
        super().__init__()
        self.hurst = HurstCalibrator()

        # Specialized SubTrackers
        self.sub_1h = SpecializedHorizonSubTracker("1h", self.hurst)
        self.sub_24h = SpecializedHorizonSubTracker("24h", self.hurst)

        # Register official CrunchDAO routes
        self.track(3600, "*", self.sub_1h)
        self.track(86400, "*", self.sub_24h)

        # Register background cron jobs via TrackerBase.schedule
        self.schedule(
            name="recalibrate_hurst",
            func=self._cron_recalibrate_hurst,
            interval=timedelta(hours=6),
            immediate=False,
        )

    def _cron_recalibrate_hurst(self):
        """Periodic background task: re-estimates Hurst exponents across assets."""
        assets = list(self.prices._prices.keys()) if hasattr(self.prices, "_prices") else []
        for asset in assets:
            pts = self.prices.get_prices(asset, days=7, resolution=300)
            if len(pts) >= 100:
                p = [x[1] for x in pts]
                log_p = np.log(p)
                lags = [1, 2, 4, 8, 16, 32]
                log_lags = []
                log_stds = []
                for lag in lags:
                    rets = log_p[lag:] - log_p[:-lag]
                    std_r = np.std(rets)
                    if std_r > 1e-12:
                        log_lags.append(np.log(lag))
                        log_stds.append(np.log(std_r))
                if len(log_lags) >= 3:
                    h_val = float(np.polyfit(log_lags, log_stds, 1)[0])
                    self.hurst.set(asset, h_val)


# Production Aliases for CrunchDAO Coordinator
Tracker = ProductionMetaEnsembleTracker
ProductionTracker = ProductionMetaEnsembleTracker


if __name__ == "__main__":
    import sys
    print("Testing ProductionMetaEnsembleTracker initialization...")
    tracker = ProductionMetaEnsembleTracker()
    print("ProductionMetaEnsembleTracker successfully initialized.")
    sys.exit(0)

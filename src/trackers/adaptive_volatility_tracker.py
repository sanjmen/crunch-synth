"""
Adaptive Volatility Tracker for CrunchDAO Synth.

Upgrades the Gaussian baseline by incorporating Milestone 3 quantitative features:
1. High-efficiency Garman-Klass / Parkinson intra-bar volatility estimators on resampled OHLC candles.
2. Online recursive GARCH(1,1) forward term-structure projection across future steps.
3. Asset-specific fractal Hurst exponent scaling (replacing standard sqrt(step) Brownian motion).
4. Corwin-Schultz spread & volume shock liquidity adjustments.
"""

from datetime import datetime, timezone
import logging
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
from crunch_synth import TrackerBase

from src.features.volatility import (
    ticks_to_ohlc,
    garman_klass_volatility,
    parkinson_volatility,
    OnlineGARCH11,
)
from src.features.hurst import AssetHurstCalibrator
from src.features.microstructure import (
    corwin_schultz_spread,
    LiquidityRegimeDetector,
)

logger = logging.getLogger(__name__)


class AdaptiveVolatilityTracker(TrackerBase):
    """
    Adaptive volatility predictive density tracker.
    Models future price increments using high-efficiency volatility estimators,
    GARCH forward term structure, and asset-specific Hurst exponent scaling.
    """

    def __init__(
        self,
        lookback_days: int = 5,
        resolution_seconds: int = 300,
        min_samples: int = 10,
        default_scale: float = 1.0,
        use_hurst_scaling: bool = True,
        use_garch_term_structure: bool = True,
        use_microstructure_adjustment: bool = True,
        garch_weight: float = 0.40,
    ):
        super().__init__()
        self.lookback_days = lookback_days
        self.resolution_seconds = resolution_seconds
        self.min_samples = min_samples
        self.default_scale = default_scale

        self.use_hurst_scaling = use_hurst_scaling
        self.use_garch_term_structure = use_garch_term_structure
        self.use_microstructure_adjustment = use_microstructure_adjustment
        self.garch_weight = garch_weight

        self.hurst_calibrator = AssetHurstCalibrator()
        self.liquidity_detector = LiquidityRegimeDetector()

        # Cache of online GARCH filters per asset
        self.garch_filters: Dict[str, OnlineGARCH11] = {}

    def _get_or_init_garch(self, asset: str, returns: np.ndarray) -> OnlineGARCH11:
        if asset not in self.garch_filters:
            filt = OnlineGARCH11(omega=1e-5, alpha=0.10, beta=0.85)
            if len(returns) > 5:
                filt.warm_up(returns)
            self.garch_filters[asset] = filt
        return self.garch_filters[asset]

    def predict(self, asset: str, horizon: int, step: int) -> List[Dict[str, Any]]:
        """
        Generate predictive density functions for incremental price returns.

        :param asset: Asset symbol (e.g. 'BTC', 'ETH', 'SOL').
        :param horizon: Total forecast horizon in seconds.
        :param step: Step size in seconds.
        :return: List of mixture density specifications.
        """
        if step <= 0 or horizon < step:
            logger.warning(f"Invalid horizon/step for {asset}: horizon={horizon}, step={step}")
            return []

        num_segments = horizon // step

        # Retrieve recent price observations
        price_points = self.prices.get_prices(
            asset, days=self.lookback_days, resolution=self.resolution_seconds
        )

        last_price_data = self.prices.get_last_price(asset)
        if last_price_data is None:
            logger.warning(f"No price data available in PriceStore for {asset}")
            return []

        # last_price_data is tuple (timestamp, price)
        current_price = float(last_price_data[1]) if isinstance(last_price_data, (tuple, list)) else float(last_price_data)

        if not price_points or len(price_points) < self.min_samples:
            # Fallback to conservative estimate
            mu = 0.0
            base_vol_price = max(current_price * 0.002, self.default_scale)
        else:
            # Resample into OHLC candles
            ohlc_df = ticks_to_ohlc(price_points, bar_seconds=self.resolution_seconds)

            if len(ohlc_df) >= 5:
                # High-efficiency Garman-Klass volatility (in log-return space)
                gk_vol = garman_klass_volatility(
                    ohlc_df["open"].to_numpy(),
                    ohlc_df["high"].to_numpy(),
                    ohlc_df["low"].to_numpy(),
                    ohlc_df["close"].to_numpy(),
                )
                if gk_vol <= 0 or not np.isfinite(gk_vol):
                    gk_vol = parkinson_volatility(ohlc_df["high"].to_numpy(), ohlc_df["low"].to_numpy())

                # Convert to price space for one base bar
                gk_vol_price = float(current_price * gk_vol)

                # Online GARCH filter
                log_rets = np.diff(np.log(ohlc_df["close"].to_numpy()))
                garch = self._get_or_init_garch(asset, log_rets)
                garch_vol_price = float(current_price * garch.current_volatility)

                if self.use_garch_term_structure:
                    w_garch = self.garch_weight
                    base_vol_price = (1.0 - w_garch) * gk_vol_price + w_garch * garch_vol_price
                else:
                    base_vol_price = gk_vol_price

                # Microstructure spread & liquidity shock adjustment
                if self.use_microstructure_adjustment and len(ohlc_df) >= 10:
                    spreads = corwin_schultz_spread(ohlc_df["high"].to_numpy(), ohlc_df["low"].to_numpy())
                    curr_spread = float(spreads[-1]) if len(spreads) > 0 else 0.0
                    recent_counts = ohlc_df["count"].to_numpy()
                    curr_count = float(recent_counts[-1])

                    mult, _ = self.liquidity_detector.compute_shock_multiplier(
                        current_volume=curr_count,
                        recent_volumes=recent_counts[:-1],
                        current_spread=curr_spread,
                        recent_spreads=spreads[:-1],
                    )
                    base_vol_price *= mult

                # Robust drift estimate with shrinkage towards 0
                price_increments = np.diff(ohlc_df["close"].to_numpy())
                raw_mu = float(np.mean(price_increments))
                mu = raw_mu * 0.1  # Heavy shrinkage towards zero-drift martingality
            else:
                increments = np.diff([p for _, p in price_points])
                base_vol_price = float(np.std(increments)) if len(increments) > 1 else current_price * 0.002
                mu = 0.0

        base_vol_price = max(base_vol_price, 1e-4)

        # Scale dispersion across forecast steps
        distributions: List[Dict[str, Any]] = []

        for k in range(1, num_segments + 1):
            target_step = k * step

            if self.use_hurst_scaling:
                # Apply asset-specific fractal diffusion scaling
                step_vol = self.hurst_calibrator.scale_dispersion(
                    base_sigma=base_vol_price,
                    step_seconds=step,
                    base_step_seconds=self.resolution_seconds,
                    asset=asset,
                )
            else:
                ratio = step / self.resolution_seconds
                step_vol = float(np.sqrt(ratio) * base_vol_price)

            # Step drift
            drift_ratio = step / self.resolution_seconds
            step_mean = float(drift_ratio * mu)

            step_vol = max(step_vol, 1e-6)

            distributions.append({
                "step": target_step,
                "type": "mixture",
                "components": [
                    {
                        "weight": 1.0,
                        "density": {
                            "type": "builtin",
                            "name": "norm",
                            "params": {
                                "loc": step_mean,
                                "scale": step_vol,
                            },
                        },
                    }
                ],
            })

        return distributions

"""
Student-t Heavy-Tailed Predictive Density Tracker for CrunchDAO Synth.

Models future incremental price returns using Student-t distributions ('t')
to capture leptokurtosis, power-law tails, and extreme market moves in crypto and volatile assets:
    r_{t,k} ~ t_{nu}( loc_k, scale_k )
"""

from datetime import datetime, timezone
import logging
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
from scipy.stats import t as student_t_dist
from crunch_synth import TrackerBase

from src.features.volatility import ticks_to_ohlc, garman_klass_volatility, parkinson_volatility
from src.features.hurst import AssetHurstCalibrator

logger = logging.getLogger(__name__)


class StudentTTracker(TrackerBase):
    """
    Heavy-tailed predictive density tracker using Student-t distributions.
    Calibrates degrees of freedom (nu) per asset to capture empirical tail heaviness.
    """

    DEFAULT_DEGREES_OF_FREEDOM: Dict[str, float] = {
        # Crypto: heavy tails (nu ~ 3.5 - 4.5)
        "BTC": 4.2,
        "ETH": 4.0,
        "SOL": 3.8,
        "XRP": 4.0,
        "HYPE": 3.6,
        # Equities: moderate tails (nu ~ 5.5 - 7.5)
        "SP500": 7.0,
        "NVDAX": 5.5,
        "TSLAX": 5.0,
        "AAPLX": 6.5,
        "GOOGLX": 6.5,
        # Commodities: moderate tails (nu ~ 5.5 - 6.5)
        "XAUT": 6.0,
        "WTIOIL": 5.5,
    }

    def __init__(
        self,
        lookback_days: int = 5,
        resolution_seconds: int = 300,
        min_samples: int = 15,
        default_scale: float = 1.0,
        fit_df_dynamically: bool = True,
        min_df: float = 2.5,
        max_df: float = 30.0,
    ):
        super().__init__()
        self.lookback_days = lookback_days
        self.resolution_seconds = resolution_seconds
        self.min_samples = min_samples
        self.default_scale = default_scale
        self.fit_df_dynamically = fit_df_dynamically
        self.min_df = min_df
        self.max_df = max_df

        self.hurst_calibrator = AssetHurstCalibrator()

    def get_calibrated_df(self, asset: str) -> float:
        """Returns the default calibrated degrees of freedom for the asset."""
        return self.DEFAULT_DEGREES_OF_FREEDOM.get(asset.upper(), 5.0)

    def estimate_student_t_params(
        self,
        asset: str,
        returns: np.ndarray,
    ) -> Tuple[float, float, float]:
        """
        Estimates (df, loc, scale) for Student-t distribution.
        """
        default_nu = self.get_calibrated_df(asset)

        if not self.fit_df_dynamically or len(returns) < 30:
            loc = float(np.mean(returns)) if len(returns) > 0 else 0.0
            scale = float(np.std(returns)) if len(returns) > 1 else self.default_scale
            return default_nu, loc, max(scale, 1e-6)

        try:
            # Fit Student-t via Maximum Likelihood
            df_fit, loc_fit, scale_fit = student_t_dist.fit(returns)
            nu = float(np.clip(df_fit, self.min_df, self.max_df))
            return nu, float(loc_fit), max(float(scale_fit), 1e-6)
        except Exception as e:
            logger.warning(f"Student-t fit failed for {asset}: {e}. Using calibrated defaults.")
            loc = float(np.mean(returns))
            scale = float(np.std(returns)) if len(returns) > 1 else self.default_scale
            return default_nu, loc, max(scale, 1e-6)

    def predict(self, asset: str, horizon: int, step: int) -> List[Dict[str, Any]]:
        """
        Generates predictive Student-t mixtures for each forecast step.
        """
        if step <= 0 or horizon < step:
            logger.warning(f"Invalid horizon/step for {asset}: horizon={horizon}, step={step}")
            return []

        num_segments = horizon // step

        price_points = self.prices.get_prices(
            asset, days=self.lookback_days, resolution=self.resolution_seconds
        )

        last_price_data = self.prices.get_last_price(asset)
        if last_price_data is None:
            logger.warning(f"No price data available in PriceStore for {asset}")
            return []

        current_price = float(last_price_data[1]) if isinstance(last_price_data, (tuple, list)) else float(last_price_data)

        if not price_points or len(price_points) < self.min_samples:
            nu = self.get_calibrated_df(asset)
            base_loc = 0.0
            base_scale = max(current_price * 0.002, self.default_scale)
        else:
            ohlc_df = ticks_to_ohlc(price_points, bar_seconds=self.resolution_seconds)

            if len(ohlc_df) >= 5:
                # Price increments in dollar space
                price_increments = np.diff(ohlc_df["close"].to_numpy())
                nu, raw_loc, _ = self.estimate_student_t_params(asset, price_increments)

                # High efficiency Garman-Klass volatility in price space
                gk_vol = garman_klass_volatility(
                    ohlc_df["open"].to_numpy(),
                    ohlc_df["high"].to_numpy(),
                    ohlc_df["low"].to_numpy(),
                    ohlc_df["close"].to_numpy(),
                )
                if gk_vol <= 0 or not np.isfinite(gk_vol):
                    gk_vol = parkinson_volatility(ohlc_df["high"].to_numpy(), ohlc_df["low"].to_numpy())

                base_scale = float(current_price * gk_vol)

                # For Student-t, the variance is Var = scale^2 * (nu / (nu - 2)) for nu > 2.
                # To align the scale parameter so that the distribution variance matches Garman-Klass:
                if nu > 2.0:
                    base_scale = float(base_scale * np.sqrt((nu - 2.0) / nu))

                base_loc = raw_loc * 0.1  # Shrink drift towards zero
            else:
                increments = np.diff([p for _, p in price_points])
                nu = self.get_calibrated_df(asset)
                base_scale = float(np.std(increments)) if len(increments) > 1 else current_price * 0.002
                base_loc = 0.0

        base_scale = max(base_scale, 1e-4)

        distributions: List[Dict[str, Any]] = []
        for k in range(1, num_segments + 1):
            target_step = k * step

            # Fractal Hurst scaling for Student-t scale parameter
            step_scale = self.hurst_calibrator.scale_dispersion(
                base_sigma=base_scale,
                step_seconds=step,
                base_step_seconds=self.resolution_seconds,
                asset=asset,
            )

            drift_ratio = step / self.resolution_seconds
            step_loc = float(drift_ratio * base_loc)
            step_scale = max(step_scale, 1e-6)

            distributions.append({
                "step": target_step,
                "type": "mixture",
                "components": [
                    {
                        "weight": 1.0,
                        "density": {
                            "type": "builtin",
                            "name": "t",
                            "params": {
                                "df": float(nu),
                                "loc": step_loc,
                                "scale": step_scale,
                            },
                        },
                    }
                ],
            })

        return distributions

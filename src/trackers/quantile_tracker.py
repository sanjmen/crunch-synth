"""
Quantile Regression Mixture Tracker for CrunchDAO Synth.

Employs gradient boosted trees trained on asymmetric pinball loss to forecast
empirical quantiles and translates them into calibrated predictive density mixtures.
"""

from datetime import datetime, timezone
import logging
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
from crunch_synth import TrackerBase

from src.features.volatility import ticks_to_ohlc, garman_klass_volatility, parkinson_volatility
from src.features.hurst import AssetHurstCalibrator
from src.models.quantile_regressor import EmpiricalQuantileForecaster

logger = logging.getLogger(__name__)


class QuantileMixtureTracker(TrackerBase):
    """
    Quantile Regression predictive density tracker.
    Models return dispersion through non-parametric quantiles synthesized into
    multi-component Gaussian mixtures.
    """

    def __init__(
        self,
        lookback_days: int = 5,
        resolution_seconds: int = 300,
        min_samples: int = 25,
        default_scale: float = 1.0,
    ):
        super().__init__()
        self.lookback_days = lookback_days
        self.resolution_seconds = resolution_seconds
        self.min_samples = min_samples
        self.default_scale = default_scale

        self.hurst_calibrator = AssetHurstCalibrator()
        self.forecasters: Dict[str, EmpiricalQuantileForecaster] = {}

    def predict(self, asset: str, horizon: int, step: int) -> List[Dict[str, Any]]:
        """
        Generates predictive mixtures informed by empirical quantile regressions.
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
            base_vol = max(current_price * 0.002, self.default_scale)
            w_core, w_tail = 0.75, 0.25
            mu_core, mu_tail = 0.0, 0.0
            sigma_core, sigma_tail = base_vol, base_vol * 2.2
        else:
            ohlc_df = ticks_to_ohlc(price_points, bar_seconds=self.resolution_seconds)

            if len(ohlc_df) >= 30:
                gk_vol = garman_klass_volatility(
                    ohlc_df["open"].to_numpy(),
                    ohlc_df["high"].to_numpy(),
                    ohlc_df["low"].to_numpy(),
                    ohlc_df["close"].to_numpy(),
                )
                if gk_vol <= 0 or not np.isfinite(gk_vol):
                    gk_vol = parkinson_volatility(ohlc_df["high"].to_numpy(), ohlc_df["low"].to_numpy())

                base_vol = float(current_price * gk_vol)

                # Initialize or fit forecaster
                if asset not in self.forecasters:
                    self.forecasters[asset] = EmpiricalQuantileForecaster()

                forecaster = self.forecasters[asset]
                forecaster.fit(ohlc_df)
                pred_quantiles = forecaster.predict_quantiles(ohlc_df)

                w_core, mu_core, sigma_core, w_tail, mu_tail, sigma_tail = (
                    forecaster.quantiles_to_mixture_parameters(pred_quantiles, base_vol)
                )
            else:
                increments = np.diff([p for _, p in price_points])
                base_vol = float(np.std(increments)) if len(increments) > 1 else current_price * 0.002
                w_core, w_tail = 0.75, 0.25
                mu_core, mu_tail = 0.0, 0.0
                sigma_core, sigma_tail = base_vol, base_vol * 2.2

        distributions: List[Dict[str, Any]] = []
        drift_ratio = step / self.resolution_seconds

        for k in range(1, num_segments + 1):
            target_step = k * step

            step_sigma_core = self.hurst_calibrator.scale_dispersion(
                base_sigma=sigma_core,
                step_seconds=step,
                base_step_seconds=self.resolution_seconds,
                asset=asset,
            )
            step_sigma_tail = self.hurst_calibrator.scale_dispersion(
                base_sigma=sigma_tail,
                step_seconds=step,
                base_step_seconds=self.resolution_seconds,
                asset=asset,
            )

            step_loc_core = float(drift_ratio * mu_core)
            step_loc_tail = float(drift_ratio * mu_tail)

            step_sigma_core = max(step_sigma_core, 1e-6)
            step_sigma_tail = max(step_sigma_tail, 1e-6)

            distributions.append({
                "step": target_step,
                "type": "mixture",
                "components": [
                    {
                        "weight": float(w_core),
                        "density": {
                            "type": "builtin",
                            "name": "norm",
                            "params": {
                                "loc": step_loc_core,
                                "scale": step_sigma_core,
                            },
                        },
                    },
                    {
                        "weight": float(w_tail),
                        "density": {
                            "type": "builtin",
                            "name": "norm",
                            "params": {
                                "loc": step_loc_tail,
                                "scale": step_sigma_tail,
                            },
                        },
                    },
                ],
            })

        return distributions

"""
2-Component Gaussian Mixture Model Tracker for Regime-Switching.

Models incremental returns as a multi-modal mixture of:
1. Normal State (High probability, low-to-moderate variance diffusive regime).
2. Jump / Breakout State (Lower probability, wide variance regime capturing tail events).
"""

from datetime import datetime, timezone
import logging
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
from sklearn.mixture import GaussianMixture
from crunch_synth import TrackerBase

from src.features.volatility import ticks_to_ohlc, garman_klass_volatility, parkinson_volatility
from src.features.hurst import AssetHurstCalibrator

logger = logging.getLogger(__name__)


class GaussianMixtureTracker(TrackerBase):
    """
    2-Component Gaussian Mixture Model (GMM) predictive density tracker.
    Captures regime-switching behavior (quiescent diffusion vs breakout jumps).
    """

    def __init__(
        self,
        lookback_days: int = 5,
        resolution_seconds: int = 300,
        min_samples: int = 20,
        default_scale: float = 1.0,
        normal_weight_prior: float = 0.80,
        jump_scale_multiplier: float = 2.5,
        random_state: int = 42,
    ):
        super().__init__()
        self.lookback_days = lookback_days
        self.resolution_seconds = resolution_seconds
        self.min_samples = min_samples
        self.default_scale = default_scale
        self.normal_weight_prior = normal_weight_prior
        self.jump_scale_multiplier = jump_scale_multiplier
        self.random_state = random_state

        self.hurst_calibrator = AssetHurstCalibrator()

    def fit_2component_gmm(
        self,
        returns: np.ndarray,
        base_vol: float,
    ) -> Tuple[float, float, float, float, float, float]:
        """
        Fits or initializes 2 Gaussian mixture components:
        Returns: (w_norm, mu_norm, sigma_norm, w_jump, mu_jump, sigma_jump)
        """
        if len(returns) < 30:
            # Fallback prior
            w_norm = self.normal_weight_prior
            w_jump = 1.0 - w_norm
            mu_norm = 0.0
            sigma_norm = max(base_vol, 1e-4)
            mu_jump = 0.0
            sigma_jump = max(base_vol * self.jump_scale_multiplier, 1e-4)
            return w_norm, mu_norm, sigma_norm, w_jump, mu_jump, sigma_jump

        try:
            X = returns.reshape(-1, 1)
            gmm = GaussianMixture(
                n_components=2,
                covariance_type="spherical",
                max_iter=100,
                random_state=self.random_state,
            )
            gmm.fit(X)

            weights = gmm.weights_
            means = gmm.means_.flatten()
            variances = gmm.covariances_.flatten()
            sigmas = np.sqrt(np.maximum(variances, 1e-8))

            # Order components: index 0 is normal (smaller sigma), index 1 is jump (larger sigma)
            if sigmas[0] > sigmas[1]:
                idx_norm, idx_jump = 1, 0
            else:
                idx_norm, idx_jump = 0, 1

            w_norm = float(weights[idx_norm])
            w_jump = float(weights[idx_jump])
            mu_norm = float(means[idx_norm]) * 0.1  # Shrink drift
            mu_jump = float(means[idx_jump]) * 0.2
            sigma_norm = float(sigmas[idx_norm])
            sigma_jump = float(sigmas[idx_jump])

            # Enforce regularity bounds
            w_norm = float(np.clip(w_norm, 0.55, 0.90))
            w_jump = 1.0 - w_norm
            sigma_norm = max(sigma_norm, base_vol * 0.5, 1e-4)
            sigma_jump = max(sigma_jump, sigma_norm * 1.8, 1e-4)

            return w_norm, mu_norm, sigma_norm, w_jump, mu_jump, sigma_jump

        except Exception as e:
            logger.warning(f"GMM fitting failed: {e}. Falling back to prior.")
            w_norm = self.normal_weight_prior
            w_jump = 1.0 - w_norm
            return w_norm, 0.0, base_vol, w_jump, 0.0, base_vol * self.jump_scale_multiplier

    def predict(self, asset: str, horizon: int, step: int) -> List[Dict[str, Any]]:
        """
        Generates 2-component Gaussian mixture predictive distributions.
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
            w_norm = self.normal_weight_prior
            w_jump = 1.0 - w_norm
            mu_norm, mu_jump = 0.0, 0.0
            sigma_norm, sigma_jump = base_vol, base_vol * self.jump_scale_multiplier
        else:
            ohlc_df = ticks_to_ohlc(price_points, bar_seconds=self.resolution_seconds)

            if len(ohlc_df) >= 5:
                gk_vol = garman_klass_volatility(
                    ohlc_df["open"].to_numpy(),
                    ohlc_df["high"].to_numpy(),
                    ohlc_df["low"].to_numpy(),
                    ohlc_df["close"].to_numpy(),
                )
                if gk_vol <= 0 or not np.isfinite(gk_vol):
                    gk_vol = parkinson_volatility(ohlc_df["high"].to_numpy(), ohlc_df["low"].to_numpy())

                base_vol = float(current_price * gk_vol)
                price_increments = np.diff(ohlc_df["close"].to_numpy())

                w_norm, mu_norm, sigma_norm, w_jump, mu_jump, sigma_jump = self.fit_2component_gmm(
                    price_increments, base_vol
                )
            else:
                increments = np.diff([p for _, p in price_points])
                base_vol = float(np.std(increments)) if len(increments) > 1 else current_price * 0.002
                w_norm = self.normal_weight_prior
                w_jump = 1.0 - w_norm
                mu_norm, mu_jump = 0.0, 0.0
                sigma_norm, sigma_jump = base_vol, base_vol * self.jump_scale_multiplier

        distributions: List[Dict[str, Any]] = []
        drift_ratio = step / self.resolution_seconds

        for k in range(1, num_segments + 1):
            target_step = k * step

            # Fractal Hurst scaling for both regimes
            step_sigma_norm = self.hurst_calibrator.scale_dispersion(
                base_sigma=sigma_norm,
                step_seconds=step,
                base_step_seconds=self.resolution_seconds,
                asset=asset,
            )
            step_sigma_jump = self.hurst_calibrator.scale_dispersion(
                base_sigma=sigma_jump,
                step_seconds=step,
                base_step_seconds=self.resolution_seconds,
                asset=asset,
            )

            step_loc_norm = float(drift_ratio * mu_norm)
            step_loc_jump = float(drift_ratio * mu_jump)

            step_sigma_norm = max(step_sigma_norm, 1e-6)
            step_sigma_jump = max(step_sigma_jump, 1e-6)

            distributions.append({
                "step": target_step,
                "type": "mixture",
                "components": [
                    {
                        "weight": float(w_norm),
                        "density": {
                            "type": "builtin",
                            "name": "norm",
                            "params": {
                                "loc": step_loc_norm,
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
                                "loc": step_loc_jump,
                                "scale": step_sigma_jump,
                            },
                        },
                    },
                ],
            })

        return distributions

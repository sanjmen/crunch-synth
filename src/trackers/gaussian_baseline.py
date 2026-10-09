"""
Gaussian Baseline Tracker for CrunchDAO Synth.

Models future incremental price returns r_{t,k} = P_{t+k} - P_{t+(k-1)} as a
Gaussian distribution scaled by sqrt(step / dt) according to Brownian diffusion.
"""

from datetime import datetime, timezone
import logging
from typing import List, Dict, Any, Optional
import numpy as np

from crunch_synth import TrackerBase

logger = logging.getLogger(__name__)


class GaussianBaselineTracker(TrackerBase):
    """
    Gaussian predictive density tracker.
    
    Generates continuous predictive density distributions for incremental returns:
        r_{t,k} ~ N( (step / tau) * mu, sqrt(step / tau) * sigma )
        
    Where:
        mu = empirical mean incremental return over lookback window
        sigma = empirical standard deviation over lookback window
        tau = sampling resolution of historical observations (e.g., 300s)
    """

    def __init__(
        self,
        lookback_days: int = 5,
        resolution_seconds: int = 300,
        min_samples: int = 5,
        default_scale: float = 1.0,
    ):
        """
        Initialize the GaussianBaselineTracker.

        :param lookback_days: Number of days of historical prices to use for statistics.
        :param resolution_seconds: Resolution in seconds to downsample prices from PriceStore.
        :param min_samples: Minimum price observations required to compute statistics.
        :param default_scale: Fallback standard deviation if computed sigma <= 0.
        """
        super().__init__()
        self.lookback_days = lookback_days
        self.resolution_seconds = resolution_seconds
        self.min_samples = min_samples
        self.default_scale = default_scale

    def predict(self, asset: str, horizon: int, step: int) -> List[Dict[str, Any]]:
        """
        Generate predictive density functions for incremental returns.

        :param asset: Asset symbol (e.g. 'BTC', 'ETH', 'SOL').
        :param horizon: Total forecast horizon in seconds (e.g. 86400 or 3600).
        :param step: Step size in seconds (e.g. 60, 300, 3600).
        :return: List of density specifications adhering to `density_pdf` format.
        """
        if step <= 0 or horizon < step:
            logger.warning(f"Invalid horizon/step for {asset}: horizon={horizon}, step={step}")
            return []

        num_segments = horizon // step

        # Fetch recent historical prices from the built-in PriceStore
        price_points = self.prices.get_prices(
            asset, days=self.lookback_days, resolution=self.resolution_seconds
        )

        if not price_points or len(price_points) < self.min_samples:
            # Fallback to last known price or uninformative Gaussian if history is shallow
            last_price = self.prices.get_last_price(asset)
            if last_price is None:
                logger.warning(f"No price data available in PriceStore for {asset}")
                return []
            mu = 0.0
            sigma = max(last_price * 0.001, self.default_scale)
        else:
            _, past_prices = zip(*price_points)
            returns = np.diff(past_prices)

            mu = float(np.mean(returns))
            sigma = float(np.std(returns))

            if sigma <= 0.0 or not np.isfinite(sigma):
                sigma = self.default_scale

        scaling_ratio = step / self.resolution_seconds
        step_drift = scaling_ratio * mu
        step_volatility = float(np.sqrt(scaling_ratio) * sigma)

        # Ensure volatility is positive and strictly finite
        step_volatility = max(step_volatility, 1e-6)

        distributions: List[Dict[str, Any]] = []
        for k in range(1, num_segments + 1):
            distributions.append({
                "step": k * step,
                "type": "mixture",
                "components": [
                    {
                        "density": {
                            "type": "builtin",
                            "name": "norm",
                            "params": {
                                "loc": float(step_drift),
                                "scale": float(step_volatility),
                            },
                        },
                        "weight": 1.0,
                    }
                ],
            })

        return distributions

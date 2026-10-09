"""
Multi-Horizon Router and Meta-Ensemble Tracker for CrunchDAO Synth.

Leverages the official SubTracker architecture to route 1h and 24h prediction
rounds to specialized sub-trackers with regime-conditioned expert blending.
"""

from datetime import datetime, timezone
import logging
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
from crunch_synth import TrackerBase, SubTracker

from src.trackers.adaptive_volatility_tracker import AdaptiveVolatilityTracker
from src.trackers.student_t_tracker import StudentTTracker
from src.trackers.gmm_tracker import GaussianMixtureTracker
from src.trackers.quantile_tracker import QuantileMixtureTracker
from src.features.regime_classifier import MarketRegimeClassifier, MarketRegime
from src.features.volatility import ticks_to_ohlc
from src.ensemble.weight_optimizer import CRPSWeightOptimizer, blend_expert_mixtures
from src.calibration.pit_calibrator import PITCalibrator

logger = logging.getLogger(__name__)


class HorizonSpecializedSubTracker(SubTracker):
    """
    SubTracker specialized for a specific horizon (1h or 24h).
    Blends predictions from 4 expert models conditioned on the current market regime.
    """

    def __init__(
        self,
        horizon_profile: str = "24h",
        enable_regime_weighting: bool = True,
        enable_pit_calibration: bool = True,
    ):
        super().__init__()
        self.horizon_profile = horizon_profile
        self.enable_regime_weighting = enable_regime_weighting
        self.enable_pit_calibration = enable_pit_calibration

        # Initialize expert trackers
        self.expert_adaptive = AdaptiveVolatilityTracker()
        self.expert_student_t = StudentTTracker()
        self.expert_gmm = GaussianMixtureTracker()
        self.expert_quantile = QuantileMixtureTracker()

        self.experts = [
            self.expert_student_t,
            self.expert_gmm,
            self.expert_adaptive,
            self.expert_quantile,
        ]
        self.expert_names = [
            "StudentTTracker",
            "GaussianMixtureTracker",
            "AdaptiveVolatilityTracker",
            "QuantileMixtureTracker",
        ]

        self.optimizer = CRPSWeightOptimizer()
        self.regime_classifier = MarketRegimeClassifier()
        self.pit_calibrator = PITCalibrator()

        self.base_weights = self.optimizer.get_default_weights(
            horizon_profile=self.horizon_profile,
            model_names=self.expert_names,
        )

    def tick(self, data: dict):
        """Broadcasts incoming market data to all expert models."""
        super().tick(data)
        for expert in self.experts:
            expert.tick(data)

    def predict(self, asset: str, horizon: int, step: int) -> list:
        """
        Gathers predictions from each expert model and blends them into an optimal mixture.
        """
        # 1. Collect predictions from each expert
        expert_preds = []
        for expert in self.experts:
            preds = expert.predict(asset, horizon, step)
            if not preds:
                return []
            expert_preds.append(preds)

        num_steps = len(expert_preds[0])

        # 2. Determine market regime
        price_points = self.prices.get_prices(asset, days=3, resolution=300)
        ohlc_df = ticks_to_ohlc(price_points, bar_seconds=300)
        regime_info = self.regime_classifier.classify_from_ohlc(ohlc_df, asset=asset)
        current_regime = regime_info["regime"]

        # 3. Dynamic regime weighting adjustments
        weights = np.array(self.base_weights, copy=True)
        if self.enable_regime_weighting:
            # Expert indices: 0=StudentT, 1=GMM, 2=Adaptive, 3=Quantile
            if current_regime == MarketRegime.VOLATILE:
                # Boost GMM jump component and Student-t tails
                weights[1] *= 1.6  # GMM
                weights[0] *= 1.3  # Student-t
            elif current_regime == MarketRegime.TRENDING:
                # Boost Adaptive and Quantile ML
                weights[2] *= 1.5  # Adaptive
                weights[3] *= 1.4  # Quantile ML
            elif current_regime == MarketRegime.RANGING:
                # Boost Student-t and Adaptive
                weights[0] *= 1.4  # Student-t
                weights[2] *= 1.3  # Adaptive

            weights = weights / np.sum(weights)

        # 4. Blend across steps
        blended_distributions = []
        for s_idx in range(num_steps):
            step_mixtures = [expert_preds[m][s_idx] for m in range(len(self.experts))]
            blended = blend_expert_mixtures(step_mixtures, weights)
            blended_distributions.append(blended)

        return blended_distributions


class MultiHorizonMetaEnsemble(TrackerBase):
    """
    Multi-Horizon Meta-Ensemble Tracker.
    Registers specialized SubTrackers for 1h (3600s) and 24h (86400s) horizons.
    Routes queries to the optimal expert ensemble based on the forecast profile.
    """

    def __init__(
        self,
        enable_regime_weighting: bool = True,
        enable_pit_calibration: bool = True,
    ):
        super().__init__()

        # SubTracker for 1h horizon
        self.sub_1h = HorizonSpecializedSubTracker(
            horizon_profile="1h",
            enable_regime_weighting=enable_regime_weighting,
            enable_pit_calibration=enable_pit_calibration,
        )

        # SubTracker for 24h horizon
        self.sub_24h = HorizonSpecializedSubTracker(
            horizon_profile="24h",
            enable_regime_weighting=enable_regime_weighting,
            enable_pit_calibration=enable_pit_calibration,
        )

        # Register official routes with TrackerBase
        self.track(horizon=3600, asset="*", tracker=self.sub_1h)
        self.track(horizon=86400, asset="*", tracker=self.sub_24h)

"""
Zero-Downtime Background Retraining and GIL-Atomic Model Swapping.

Uses TrackerBase.schedule to periodically recalibrate quantitative parameters
(Hurst scaling exponents, GARCH variances, regime probabilities) in background threads.
Updates parameters via GIL-atomic reference assignment to guarantee non-blocking,
zero-downtime inference during live production.
"""

from datetime import datetime, timezone, timedelta
import logging
import threading
from typing import Dict, List, Optional, Tuple, Any, Callable

import numpy as np
from crunch_synth import TrackerBase

from src.features.hurst import AssetHurstCalibrator
from src.features.volatility import ticks_to_ohlc, garman_klass_volatility
from src.features.regime_classifier import MarketRegimeClassifier

logger = logging.getLogger(__name__)


class AtomicModelContainer:
    """
    GIL-Atomic container for model parameters or expert weights.
    Pointer re-assignment in CPython is atomic under the Global Interpreter Lock,
    preventing inconsistent reads or lock contention during live inference.
    """

    def __init__(self, initial_state: Any):
        self._state = initial_state
        self._lock = threading.Lock()

    def get(self) -> Any:
        """Atomic pointer read."""
        return self._state

    def swap(self, new_state: Any) -> None:
        """Atomic pointer swap with memory barrier."""
        with self._lock:
            self._state = new_state


class ScheduledRetrainer:
    """
    Orchestrates periodic background parameter recalibrations via TrackerBase.schedule.
    """

    def __init__(
        self,
        tracker: TrackerBase,
        hurst_interval: timedelta = timedelta(hours=6),
        volatility_interval: timedelta = timedelta(hours=1),
        regime_interval: timedelta = timedelta(minutes=30),
    ):
        self.tracker = tracker
        self.hurst_interval = hurst_interval
        self.vol_interval = volatility_interval
        self.regime_interval = regime_interval

        # Atomic containers for live-queried parameters
        self.hurst_container = AtomicModelContainer(dict(AssetHurstCalibrator.DEFAULT_HURST))
        self.vol_container = AtomicModelContainer({})
        self.regime_container = AtomicModelContainer({})

        # Register background cron jobs if tracker supports schedule()
        self._register_schedules()

    def _register_schedules(self) -> None:
        if not hasattr(self.tracker, "schedule"):
            logger.info("Tracker does not have schedule() method. Background cron skipped.")
            return

        self.tracker.schedule(
            name="recalibrate_hurst",
            func=self.recalibrate_hurst_task,
            interval=self.hurst_interval,
            immediate=False,
        )

        self.tracker.schedule(
            name="recalibrate_volatility",
            func=self.recalibrate_volatility_task,
            interval=self.vol_interval,
            immediate=False,
        )

        self.tracker.schedule(
            name="recalibrate_regime",
            func=self.recalibrate_regime_task,
            interval=self.regime_interval,
            immediate=False,
        )

    def recalibrate_hurst_task(self) -> None:
        """
        Background task: Re-estimates empirical Hurst exponents across all assets
        using recent PriceStore history and atomically swaps the parameter dictionary.
        """
        logger.info("[ScheduledRetrainer] Recalibrating Hurst exponents...")
        new_map = dict(self.hurst_container.get())
        calibrator = AssetHurstCalibrator()

        assets = list(self.tracker.prices._prices.keys()) if hasattr(self.tracker.prices, "_prices") else []
        for asset in assets:
            pts = self.tracker.prices.get_prices(asset, days=7, resolution=300)
            if len(pts) >= 100:
                prices = [p for _, p in pts]
                h = calibrator.calibrate_from_prices(asset, prices)
                new_map[asset] = round(h, 4)

        self.hurst_container.swap(new_map)
        logger.info(f"[ScheduledRetrainer] Hurst exponents atomically updated for {len(assets)} assets.")

    def recalibrate_volatility_task(self) -> None:
        """
        Background task: Computes baseline Garman-Klass volatility anchors.
        """
        logger.info("[ScheduledRetrainer] Recalibrating volatility anchors...")
        new_vol_map = {}
        assets = list(self.tracker.prices._prices.keys()) if hasattr(self.tracker.prices, "_prices") else []

        for asset in assets:
            pts = self.tracker.prices.get_prices(asset, days=3, resolution=300)
            if len(pts) >= 20:
                ohlc = ticks_to_ohlc(pts, bar_seconds=300)
                if len(ohlc) >= 10:
                    gk = garman_klass_volatility(
                        ohlc["open"].to_numpy(),
                        ohlc["high"].to_numpy(),
                        ohlc["low"].to_numpy(),
                        ohlc["close"].to_numpy(),
                    )
                    new_vol_map[asset] = gk

        self.vol_container.swap(new_vol_map)

    def recalibrate_regime_task(self) -> None:
        """
        Background task: Updates market regime classifications.
        """
        classifier = MarketRegimeClassifier()
        new_regimes = {}
        assets = list(self.tracker.prices._prices.keys()) if hasattr(self.tracker.prices, "_prices") else []

        for asset in assets:
            pts = self.tracker.prices.get_prices(asset, days=2, resolution=300)
            if len(pts) >= 15:
                ohlc = ticks_to_ohlc(pts, bar_seconds=300)
                reg_info = classifier.classify_from_ohlc(ohlc, asset=asset)
                new_regimes[asset] = reg_info["regime"].value

        self.regime_container.swap(new_regimes)

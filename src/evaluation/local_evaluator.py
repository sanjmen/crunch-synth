"""
Local Evaluator module for benchmarking trackers on historical test sets.
"""

from datetime import datetime, timezone
import logging
import time
from typing import Dict, List, Optional, Tuple, Any

from tqdm.auto import tqdm
from crunch_synth import (
    TrackerBase,
    TrackerEvaluator,
    FORECAST_PROFILES,
    load_test_prices_once,
    load_initial_price_histories_once,
    build_events,
    compute_ranks,
)

logger = logging.getLogger(__name__)


class LocalBacktestHarness:
    """
    Backtesting harness that mirrors the CrunchDAO Synth live execution environment.
    """

    def __init__(
        self,
        tracker: TrackerBase,
        assets: Optional[List[str]] = None,
        horizon_profile: str = "24h",
        evaluation_end: Optional[datetime] = None,
        days_test: int = 3,
        days_warmup: int = 15,
    ):
        self.tracker = tracker
        self.assets = assets or ["BTC", "SOL", "ETH"]
        self.horizon_profile = horizon_profile

        if horizon_profile not in FORECAST_PROFILES:
            raise ValueError(f"Unknown profile '{horizon_profile}'. Choose from {list(FORECAST_PROFILES.keys())}")

        self.horizon = FORECAST_PROFILES[horizon_profile]["horizon"]
        self.steps = FORECAST_PROFILES[horizon_profile]["steps"]
        self.interval = FORECAST_PROFILES[horizon_profile]["interval"]

        self.evaluation_end = evaluation_end or datetime(2025, 11, 15, 0, 0, 0, tzinfo=timezone.utc)
        self.days_test = days_test
        self.days_warmup = days_warmup

        self.evaluator = TrackerEvaluator(self.tracker)

    def run(self, verbose: bool = True) -> Dict[str, Any]:
        """
        Execute the backtest over the configured assets and return performance metrics.
        """
        start_time = time.time()
        if verbose:
            print(f"Loading data for {len(self.assets)} assets ({', '.join(self.assets)})...")

        test_prices = load_test_prices_once(
            self.assets, self.evaluation_end, days=self.days_test
        )
        warmup_prices = load_initial_price_histories_once(
            self.assets, self.evaluation_end, days_history=self.days_warmup, days_offset=self.days_test
        )

        results: Dict[str, Any] = {
            "assets": {},
            "overall_score": None,
            "horizon_profile": self.horizon_profile,
            "elapsed_seconds": 0.0,
        }

        for asset in self.assets:
            if asset not in test_prices:
                logger.warning(f"No test prices found for {asset}")
                continue

            asset_test = test_prices[asset]
            asset_warmup = warmup_prices.get(asset, [])

            # Warm-up phase: initialize historical state
            self.evaluator.tick({asset: asset_warmup})

            # Build event timeline
            performed_ts, resolvable_ts = build_events(
                asset_test, None, asset, self.horizon, self.interval
            )

            next_perf = 0
            next_res = 0

            pbar = tqdm(
                desc=f"Evaluating {asset}",
                total=int((resolvable_ts < self.evaluation_end.timestamp()).sum()),
                unit="eval",
                disable=not verbose,
            )

            for ts, price in asset_test:
                self.evaluator.tick({asset: [(ts, price)]})

                if next_perf < len(performed_ts) and ts >= performed_ts[next_perf]:
                    next_perf += 1
                    self.evaluator.enqueue_predictions(asset, self.horizon, self.steps)

                if next_res < len(resolvable_ts) and ts >= resolvable_ts[next_res]:
                    eval_ts = int(resolvable_ts[next_res])
                    evaluated = self.evaluator.evaluate_quarantine(asset, ts=eval_ts)
                    if evaluated:
                        pbar.update(1)
                    next_res += 1

            pbar.close()

            asset_score = self.evaluator.overall_score_asset(asset)
            recent_score = self.evaluator.recent_score_asset(asset)
            results["assets"][asset] = {
                "overall_score": asset_score,
                "recent_score": recent_score,
            }

            if verbose:
                print(f"[{asset}] Overall CRPS: {asset_score:.4f} | Recent: {recent_score:.4f}")

        results["overall_score"] = self.evaluator.overall_score()
        results["elapsed_seconds"] = time.time() - start_time

        if verbose:
            print(f"\nFinal Overall CRPS Score: {results['overall_score']:.4f} (Elapsed: {results['elapsed_seconds']:.1f}s)")

        return results

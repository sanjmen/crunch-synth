"""
Local Simulation Runner and Backtest Harness for CrunchDAO Synth Trackers.

Simulates the live production environment using discrete timeline events:
- Ingests chronological tick data
- Emits forecast mixtures at scheduled query timestamps
- Evaluates quarantined probability densities against realized price moves
- Calculates continuous ranked probability scores (CRPS) per asset and overall
"""

from datetime import datetime, timezone
import logging
import time
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from crunch_synth import (
    TrackerBase,
    TrackerEvaluator,
    FORECAST_PROFILES,
    build_events,
)

from src.data.price_manager import PriceManager, ALL_ASSETS

logger = logging.getLogger(__name__)


class LocalBacktestHarness:
    """
    Simulation harness that mirrors the CrunchDAO Synth live execution container.
    Wraps TrackerEvaluator and orchestrates tick feeds, prediction enqueuing,
    and quarantine scoring.
    """

    def __init__(
        self,
        tracker: TrackerBase,
        assets: Optional[List[str]] = None,
        horizon_profile: str = "24h",
        evaluation_end: Optional[datetime] = None,
        days_test: int = 3,
        days_warmup: int = 15,
        price_manager: Optional[PriceManager] = None,
        use_cache: bool = True,
    ):
        self.tracker = tracker
        self.assets = assets or ["BTC", "SOL", "ETH"]
        self.horizon_profile = horizon_profile

        if horizon_profile not in FORECAST_PROFILES:
            raise ValueError(
                f"Unknown profile '{horizon_profile}'. Choose from {list(FORECAST_PROFILES.keys())}"
            )

        self.horizon = FORECAST_PROFILES[horizon_profile]["horizon"]
        self.steps = FORECAST_PROFILES[horizon_profile]["steps"]
        self.interval = FORECAST_PROFILES[horizon_profile]["interval"]

        self.evaluation_end = evaluation_end or datetime.now(timezone.utc)
        if self.evaluation_end.tzinfo is None:
            self.evaluation_end = self.evaluation_end.replace(tzinfo=timezone.utc)

        self.days_test = days_test
        self.days_warmup = days_warmup
        self.price_manager = price_manager or PriceManager()
        self.use_cache = use_cache

        self.evaluator = TrackerEvaluator(self.tracker)
        self.timeline_events: Dict[str, Dict[str, Any]] = {}
        self.results: Dict[str, Any] = {}

    def run(self, verbose: bool = True) -> Dict[str, Any]:
        """
        Execute the event-driven backtest simulation across all configured assets.
        """
        start_time = time.time()
        if verbose:
            print("=" * 70)
            print("CRUNCHDAO SYNTH: LOCAL SIMULATION HARNESS")
            print(f"Tracker: {self.tracker.__class__.__name__}")
            print(f"Horizon Profile: {self.horizon_profile} (H={self.horizon}s, Int={self.interval}s)")
            print(f"Window: Test {self.days_test}d | Warm-up {self.days_warmup}d | End: {self.evaluation_end.isoformat()}")
            print(f"Assets ({len(self.assets)}): {', '.join(self.assets)}")
            print("=" * 70)

        # Retrieve test and warm-up tick datasets via PriceManager
        test_prices, warmup_prices = self.price_manager.get_test_and_warmup_prices(
            assets=self.assets,
            evaluation_end=self.evaluation_end,
            days_test=self.days_test,
            days_warmup=self.days_warmup,
            use_cache=self.use_cache,
        )

        self.results = {
            "tracker": self.tracker.__class__.__name__,
            "assets": {},
            "overall_score": None,
            "horizon_profile": self.horizon_profile,
            "horizon_seconds": self.horizon,
            "interval_seconds": self.interval,
            "days_test": self.days_test,
            "days_warmup": self.days_warmup,
            "evaluation_end": self.evaluation_end.isoformat(),
            "elapsed_seconds": 0.0,
        }

        for asset in self.assets:
            asset_test = test_prices.get(asset, [])
            asset_warmup = warmup_prices.get(asset, [])

            if not asset_test:
                logger.warning(f"No test prices found for {asset}, skipping.")
                continue

            if verbose:
                print(f"\nEvaluating asset: {asset} (warmup={len(asset_warmup)} ticks, test={len(asset_test)} ticks)")

            # Phase 1: Ingest warm-up history to populate rolling statistics
            if asset_warmup:
                self.evaluator.tick({asset: asset_warmup})

            # Phase 2: Build discrete event timeline
            performed_ts, resolvable_ts = build_events(
                asset_test, None, asset, self.horizon, self.interval
            )

            self.timeline_events[asset] = {
                "performed_count": len(performed_ts),
                "resolvable_count": len(resolvable_ts),
                "performed_ts": performed_ts,
                "resolvable_ts": resolvable_ts,
            }

            next_perf = 0
            next_res = 0
            total_evals_expected = int((resolvable_ts <= self.evaluation_end.timestamp()).sum())

            pbar = tqdm(
                desc=f"Simulation {asset}",
                total=total_evals_expected,
                unit="eval",
                disable=not verbose,
            )

            # Phase 3: Event simulation loop
            for ts, price in asset_test:
                # Ingest tick
                self.evaluator.tick({asset: [(ts, price)]})

                # Trigger forecast emission event
                if next_perf < len(performed_ts) and ts >= performed_ts[next_perf]:
                    next_perf += 1
                    self.evaluator.enqueue_predictions(asset, self.horizon, self.steps)

                # Trigger quarantine resolution event
                if next_res < len(resolvable_ts) and ts >= resolvable_ts[next_res]:
                    eval_ts = int(resolvable_ts[next_res])
                    evaluated = self.evaluator.evaluate_quarantine(asset, ts=eval_ts)
                    if evaluated:
                        pbar.update(1)
                    next_res += 1

            pbar.close()

            # Phase 4: Extract and compute metrics for this asset
            asset_score = self.evaluator.overall_score_asset(asset)
            recent_score = self.evaluator.recent_score_asset(asset)
            raw_scores = [s for _, s in self.evaluator.scores.get(asset, [])]

            if raw_scores:
                scores_arr = np.array(raw_scores, dtype=np.float64)
                stat_mean = float(np.mean(scores_arr))
                stat_std = float(np.std(scores_arr))
                stat_median = float(np.median(scores_arr))
                stat_min = float(np.min(scores_arr))
                stat_max = float(np.max(scores_arr))
                q25, q75 = float(np.percentile(scores_arr, 25)), float(np.percentile(scores_arr, 75))
            else:
                stat_mean = asset_score
                stat_std = 0.0
                stat_median = asset_score
                stat_min = asset_score
                stat_max = asset_score
                q25, q75 = 0.0, 0.0

            self.results["assets"][asset] = {
                "overall_score": asset_score,
                "recent_score": recent_score,
                "evaluations_count": len(raw_scores),
                "crps_mean": stat_mean,
                "crps_std": stat_std,
                "crps_median": stat_median,
                "crps_min": stat_min,
                "crps_max": stat_max,
                "crps_q25": q25,
                "crps_q75": q75,
            }

            if verbose:
                print(
                    f"[{asset}] CRPS Mean: {stat_mean:.4f} ± {stat_std:.4f} | "
                    f"Median: {stat_median:.4f} | Range: [{stat_min:.4f}, {stat_max:.4f}] | "
                    f"N={len(raw_scores)}"
                )

        # Compute global weighted score
        self.results["overall_score"] = self.evaluator.overall_score()
        self.results["elapsed_seconds"] = time.time() - start_time

        if verbose:
            print("\n" + "=" * 70)
            print(f"SIMULATION COMPLETE in {self.results['elapsed_seconds']:.2f}s")
            overall_str = f"{self.results['overall_score']:.4f}" if self.results['overall_score'] is not None else "N/A"
            print(f"Overall Weighted CRPS: {overall_str}")
            print("=" * 70)

        return self.results

    def get_score_series(self, asset: str) -> pd.DataFrame:
        """
        Returns a DataFrame containing the timestamped CRPS score series for an asset.
        """
        records = self.evaluator.scores.get(asset, [])
        if not records:
            return pd.DataFrame(columns=["timestamp", "datetime", "score"])

        df = pd.DataFrame(records, columns=["timestamp", "score"])
        df["datetime"] = pd.to_datetime(df["timestamp"], unit="s", utc=True)
        return df[["timestamp", "datetime", "score"]].sort_values("timestamp").reset_index(drop=True)

    def summary_table(self) -> pd.DataFrame:
        """
        Generates a summary DataFrame of performance metrics across all evaluated assets.
        """
        rows = []
        for asset, stats in self.results.get("assets", {}).items():
            rows.append({
                "Asset": asset,
                "CRPS Mean": round(stats["crps_mean"], 4),
                "CRPS Std": round(stats["crps_std"], 4),
                "Median": round(stats["crps_median"], 4),
                "Min": round(stats["crps_min"], 4),
                "Max": round(stats["crps_max"], 4),
                "Evals": stats["evaluations_count"],
            })

        df = pd.DataFrame(rows)
        if not df.empty:
            df = df.sort_values("CRPS Mean").reset_index(drop=True)
        return df

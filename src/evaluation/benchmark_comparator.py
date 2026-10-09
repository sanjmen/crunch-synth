"""
Leaderboard and Benchmark Comparator for CrunchDAO Synth.

Compares tracker CRPS scores against historical competitor submissions
and official CrunchDAO benchmark models.
"""

from datetime import datetime, timezone, timedelta
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd
from crunch_synth import TrackerEvaluator

logger = logging.getLogger(__name__)

TRACKERS_HISTORY_URL = (
    "https://raw.githubusercontent.com/crunchdao/crunch-synth/master/"
    "crunch_synth/examples/trackers_history/df_trackers_history.csv"
)


class BenchmarkComparator:
    """
    Ranks TrackerEvaluator results against historical live competitor models
    and official benchmarks using historical leaderboard data.
    """

    def __init__(self, cache_dir: Optional[Path] = None):
        if cache_dir is None:
            base_dir = Path(__file__).resolve().parent.parent.parent
            self.cache_dir = base_dir / "data" / "cache"
        else:
            self.cache_dir = Path(cache_dir)

        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache_parquet = self.cache_dir / "trackers_history.parquet"

    def get_full_history(self) -> pd.DataFrame:
        """
        Retrieves the full historical competition leaderboard dataset.
        Caches locally in Parquet to avoid recurring large network downloads.
        """
        if self.cache_parquet.exists():
            try:
                df = pd.read_parquet(self.cache_parquet)
                return df
            except Exception as e:
                logger.warning(f"Error reading cache {self.cache_parquet}: {e}. Re-fetching.")

        print(f"Downloading historical competitor database from GitHub...")
        df = pd.read_csv(TRACKERS_HISTORY_URL)
        df["performed_at"] = pd.to_datetime(df["performed_at"], utc=True)
        df["resolvable_at"] = pd.to_datetime(df["resolvable_at"], utc=True)

        # Save to parquet
        df.to_parquet(self.cache_parquet, index=False, engine="pyarrow")
        print(f"Cached {len(df):,} competitor evaluation rows to {self.cache_parquet}")
        return df

    def filter_history(
        self,
        df_history: pd.DataFrame,
        horizon: int,
        assets: List[str],
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
    ) -> pd.DataFrame:
        """
        Filters competitor evaluations matching horizon, assets, and temporal window.
        """
        df_filtered = df_history[
            (df_history["horizon"] == horizon) & (df_history["asset"].isin(assets))
        ].copy()

        if start_time is not None and end_time is not None:
            window_df = df_filtered[
                (df_filtered["performed_at"] >= start_time)
                & (df_filtered["performed_at"] <= end_time)
            ]
            if not window_df.empty:
                return window_df
            else:
                logger.warning(
                    f"No history overlapping [{start_time}, {end_time}]. "
                    f"Falling back to asset-level aggregate across full historical span."
                )

        return df_filtered

    def compute_ranks_for_asset(
        self,
        df_asset_history: pd.DataFrame,
        my_score: float,
        asset: str,
    ) -> Dict[str, Any]:
        """
        Computes ranking and benchmark comparison for an individual asset.
        Lower CRPS is better.
        """
        if df_asset_history.empty:
            return {
                "asset": asset,
                "my_score": my_score,
                "benchmark_score": np.nan,
                "my_rank": 1,
                "benchmark_rank": np.nan,
                "total_trackers": 1,
                "percentile": 0.0,
                "delta_vs_benchmark_pct": np.nan,
                "beats_benchmark": False,
            }

        # Average CRPS per competitor tracker
        mean_scores = df_asset_history.groupby("tracker")["score"].mean().sort_values()

        # Rank (1-based, lower score is rank 1)
        my_rank = int((mean_scores < my_score).sum() + 1)
        total_trackers = len(mean_scores) + 1
        percentile = round((my_rank / total_trackers) * 100, 2)

        benchmark_score = mean_scores.get("benchmark", np.nan)
        if not np.isnan(benchmark_score):
            benchmark_rank = int((mean_scores < benchmark_score).sum() + 1)
            delta_pct = round(((my_score - benchmark_score) / benchmark_score) * 100, 2)
            beats_benchmark = bool(my_score < benchmark_score)
        else:
            benchmark_rank = np.nan
            delta_pct = np.nan
            beats_benchmark = False

        return {
            "asset": asset,
            "my_score": my_score,
            "benchmark_score": benchmark_score,
            "my_rank": my_rank,
            "benchmark_rank": benchmark_rank,
            "total_trackers": total_trackers,
            "percentile": percentile,
            "delta_vs_benchmark_pct": delta_pct,
            "beats_benchmark": beats_benchmark,
        }

    def compare_evaluator(
        self,
        evaluator: TrackerEvaluator,
        horizon: int,
        assets: List[str],
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
    ) -> pd.DataFrame:
        """
        Compares all evaluated assets in TrackerEvaluator against historical trackers.
        Returns a structured leaderboard comparison DataFrame.
        """
        df_history = self.get_full_history()
        filtered = self.filter_history(df_history, horizon, assets, start_time, end_time)

        records = []
        for asset in assets:
            my_score = evaluator.overall_score_asset(asset)
            if my_score is None or my_score == 0.0:
                continue

            asset_history = filtered[filtered["asset"] == asset]
            stats = self.compute_ranks_for_asset(asset_history, my_score, asset)
            records.append(stats)

        df_comparison = pd.DataFrame(records)
        return df_comparison

    def format_leaderboard_table(self, df_comparison: pd.DataFrame) -> str:
        """
        Renders an ASCII / Markdown table summarizing rank vs benchmark.
        """
        if df_comparison.empty:
            return "No comparative evaluation records available."

        lines = [
            "=" * 85,
            "CRUNCHDAO SYNTH: LEADERBOARD & BENCHMARK COMPARATOR",
            "=" * 85,
            f"{'Asset':<8} | {'My CRPS':<10} | {'Benchmark':<10} | {'Delta (%)':<10} | {'Rank / Total':<14} | {'Top %':<8} | {'Beats BM'}",
            "-" * 85,
        ]

        for _, row in df_comparison.iterrows():
            bm_str = f"{row['benchmark_score']:.4f}" if not np.isnan(row["benchmark_score"]) else "N/A"
            delta_str = f"{row['delta_vs_benchmark_pct']:+.2f}%" if not np.isnan(row["delta_vs_benchmark_pct"]) else "N/A"
            rank_str = f"{int(row['my_rank'])} / {int(row['total_trackers'])}"
            pct_str = f"{row['percentile']:.1f}%"
            beats_str = "YES [✓]" if row["beats_benchmark"] else "NO  [✗]"

            lines.append(
                f"{row['asset']:<8} | {row['my_score']:<10.4f} | {bm_str:<10} | {delta_str:<10} | {rank_str:<14} | {pct_str:<8} | {beats_str}"
            )

        lines.append("=" * 85)
        return "\n".join(lines)

"""
Historical Price Manager and On-Disk Cache for CrunchDAO Synth.

Handles retrieval, local parquet persistence, deduplication, and
integrity verification for all 12 Synth competition assets.
"""

from datetime import datetime, timezone, timedelta
import logging
import os
from pathlib import Path
import time
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from crunch_synth import pricedb, SUPPORTED_ASSETS

logger = logging.getLogger(__name__)

ALL_ASSETS: List[str] = list(SUPPORTED_ASSETS)

ASSET_CATEGORIES: Dict[str, List[str]] = {
    "crypto": ["BTC", "ETH", "SOL", "XRP", "HYPE"],
    "equities": ["SP500", "NVDAX", "TSLAX", "AAPLX", "GOOGLX"],
    "commodities": ["XAUT", "WTIOIL"],
}


class PriceManager:
    """
    Manages historical tick data ingestion, on-disk Parquet caching,
    and preparation of warm-up and test splits for all Synth assets.
    """

    def __init__(self, cache_dir: Optional[Union[str, Path]] = None):
        if cache_dir is None:
            # Default to data/cache/prices in project root
            base_dir = Path(__file__).resolve().parent.parent.parent
            self.cache_dir = base_dir / "data" / "cache" / "prices"
        else:
            self.cache_dir = Path(cache_dir)

        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _get_cache_path(self, asset: str) -> Path:
        return self.cache_dir / f"{asset.upper()}.parquet"

    def read_cached_prices(self, asset: str) -> pd.DataFrame:
        """
        Reads cached parquet file for the given asset.
        Returns an empty DataFrame with ['timestamp', 'price'] if not found.
        """
        cache_path = self._get_cache_path(asset)
        if not cache_path.exists():
            return pd.DataFrame(columns=["timestamp", "price"]).astype({
                "timestamp": "int64",
                "price": "float64",
            })

        try:
            df = pd.read_parquet(cache_path)
            if "timestamp" not in df.columns or "price" not in df.columns:
                raise ValueError(f"Corrupted cache file {cache_path}")
            return df.sort_values("timestamp").drop_duplicates(subset=["timestamp"]).reset_index(drop=True)
        except Exception as e:
            logger.warning(f"Error reading cache for {asset}: {e}. Initializing empty.")
            return pd.DataFrame(columns=["timestamp", "price"]).astype({
                "timestamp": "int64",
                "price": "float64",
            })

    def write_cached_prices(self, asset: str, df: pd.DataFrame) -> None:
        """Saves deduplicated, sorted price history to on-disk Parquet."""
        if df.empty:
            return
        df_clean = (
            df[["timestamp", "price"]]
            .dropna()
            .drop_duplicates(subset=["timestamp"])
            .sort_values("timestamp")
            .reset_index(drop=True)
        )
        df_clean["timestamp"] = df_clean["timestamp"].astype("int64")
        df_clean["price"] = df_clean["price"].astype("float64")

        cache_path = self._get_cache_path(asset)
        df_clean.to_parquet(cache_path, index=False, engine="pyarrow")

    def fetch_prices_from_api(
        self,
        asset: str,
        from_dt: datetime,
        to_dt: datetime,
        max_retries: int = 3,
        backoff_sec: float = 2.0,
    ) -> List[Tuple[int, float]]:
        """
        Retrieves tick data directly from pricedb with automatic retries.
        """
        # Ensure UTC timezone
        if from_dt.tzinfo is None:
            from_dt = from_dt.replace(tzinfo=timezone.utc)
        if to_dt.tzinfo is None:
            to_dt = to_dt.replace(tzinfo=timezone.utc)

        last_error = None
        for attempt in range(1, max_retries + 1):
            try:
                data = pricedb.get_price_history(
                    asset=asset,
                    from_=from_dt,
                    to=to_dt,
                )
                return [(int(ts), float(pr)) for ts, pr in data if pr is not None and not np.isnan(pr)]
            except Exception as e:
                last_error = e
                logger.warning(
                    f"Attempt {attempt}/{max_retries} failed for {asset} [{from_dt} to {to_dt}]: {e}"
                )
                if attempt < max_retries:
                    time.sleep(backoff_sec * attempt)

        raise RuntimeError(f"Failed to fetch prices for {asset} after {max_retries} attempts: {last_error}")

    def get_prices(
        self,
        asset: str,
        from_dt: datetime,
        to_dt: datetime,
        use_cache: bool = True,
    ) -> List[Tuple[int, float]]:
        """
        Retrieves price history for an asset over [from_dt, to_dt].
        Uses local Parquet cache if available, fetching and updating any missing intervals.
        """
        if from_dt.tzinfo is None:
            from_dt = from_dt.replace(tzinfo=timezone.utc)
        if to_dt.tzinfo is None:
            to_dt = to_dt.replace(tzinfo=timezone.utc)

        target_start_ts = int(from_dt.timestamp())
        target_end_ts = int(to_dt.timestamp())

        if not use_cache:
            raw = self.fetch_prices_from_api(asset, from_dt, to_dt)
            return raw

        cached_df = self.read_cached_prices(asset)

        # Check coverage
        needs_fetch = False
        fetch_from = from_dt
        fetch_to = to_dt

        if cached_df.empty:
            needs_fetch = True
        else:
            cached_min_ts = cached_df["timestamp"].min()
            cached_max_ts = cached_df["timestamp"].max()

            # If cache does not fully cover the target interval, expand it
            if target_start_ts < cached_min_ts or target_end_ts > cached_max_ts:
                needs_fetch = True
                fetch_from = min(from_dt, datetime.fromtimestamp(cached_min_ts, tz=timezone.utc))
                fetch_to = max(to_dt, datetime.fromtimestamp(cached_max_ts, tz=timezone.utc))

        if needs_fetch:
            new_data = self.fetch_prices_from_api(asset, fetch_from, fetch_to)
            new_df = pd.DataFrame(new_data, columns=["timestamp", "price"])
            combined_df = pd.concat([cached_df, new_df], ignore_index=True)
            self.write_cached_prices(asset, combined_df)
            cached_df = self.read_cached_prices(asset)

        # Slice the requested window
        mask = (cached_df["timestamp"] >= target_start_ts) & (cached_df["timestamp"] <= target_end_ts)
        sliced = cached_df[mask]
        return list(zip(sliced["timestamp"].tolist(), sliced["price"].tolist()))

    def get_test_and_warmup_prices(
        self,
        assets: Optional[List[str]] = None,
        evaluation_end: Optional[datetime] = None,
        days_test: int = 3,
        days_warmup: int = 15,
        use_cache: bool = True,
    ) -> Tuple[Dict[str, List[Tuple[int, float]]], Dict[str, List[Tuple[int, float]]]]:
        """
        Prepares the test and warm-up price datasets for TrackerEvaluator.

        Parameters
        ----------
        assets : Optional[List[str]]
            List of assets (defaults to ALL_ASSETS).
        evaluation_end : Optional[datetime]
            Timestamp at which evaluation ends (defaults to current UTC time).
        days_test : int
            Length of the out-of-sample test window in days.
        days_warmup : int
            Length of the historical warm-up window immediately preceding the test window.
        use_cache : bool
            Whether to use local Parquet caching.

        Returns
        -------
        (test_prices, warmup_prices)
            Tuple of dictionaries mapping asset symbol to list of (timestamp, price) tuples.
        """
        assets = assets or ALL_ASSETS
        to_test = evaluation_end if evaluation_end else datetime.now(timezone.utc)
        if to_test.tzinfo is None:
            to_test = to_test.replace(tzinfo=timezone.utc)

        from_test = to_test - timedelta(days=days_test)
        from_warmup = from_test - timedelta(days=days_warmup)

        test_prices: Dict[str, List[Tuple[int, float]]] = {}
        warmup_prices: Dict[str, List[Tuple[int, float]]] = {}

        for asset in assets:
            # Single fetch for the continuous span [from_warmup, to_test]
            full_series = self.get_prices(asset, from_warmup, to_test, use_cache=use_cache)
            cutoff_ts = int(from_test.timestamp())

            warmup_ticks = [item for item in full_series if item[0] < cutoff_ts]
            test_ticks = [item for item in full_series if item[0] >= cutoff_ts]

            warmup_prices[asset] = warmup_ticks
            test_prices[asset] = test_ticks

        return test_prices, warmup_prices

    def get_cache_stats(self) -> pd.DataFrame:
        """Returns diagnostic metadata about cached files on disk."""
        records = []
        for asset in ALL_ASSETS:
            path = self._get_cache_path(asset)
            if not path.exists():
                records.append({
                    "asset": asset,
                    "cached": False,
                    "rows": 0,
                    "min_date": None,
                    "max_date": None,
                    "size_kb": 0.0,
                })
                continue

            df = self.read_cached_prices(asset)
            size_kb = os.path.getsize(path) / 1024.0
            if df.empty:
                records.append({
                    "asset": asset,
                    "cached": True,
                    "rows": 0,
                    "min_date": None,
                    "max_date": None,
                    "size_kb": round(size_kb, 2),
                })
            else:
                min_dt = datetime.fromtimestamp(df["timestamp"].min(), tz=timezone.utc).isoformat()
                max_dt = datetime.fromtimestamp(df["timestamp"].max(), tz=timezone.utc).isoformat()
                records.append({
                    "asset": asset,
                    "cached": True,
                    "rows": len(df),
                    "min_date": min_dt,
                    "max_date": max_dt,
                    "size_kb": round(size_kb, 2),
                })

        return pd.DataFrame(records)

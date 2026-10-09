#!/usr/bin/env python3
"""
CLI tool to download and cache historical price data for CrunchDAO Synth assets.
Supports individual assets, categories ('crypto', 'equities', 'commodities'), or 'all'.
"""

import argparse
from datetime import datetime, timezone, timedelta
from pathlib import Path
import sys

# Ensure repository root is on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data.price_manager import PriceManager, ALL_ASSETS, ASSET_CATEGORIES


def main():
    parser = argparse.ArgumentParser(description="Download & Cache Historical Price Data for Synth Assets")
    parser.add_argument(
        "--assets",
        nargs="+",
        default=["all"],
        help="List of asset symbols (e.g. BTC ETH SP500), category ('crypto', 'equities', 'commodities'), or 'all'",
    )
    parser.add_argument(
        "--days",
        type=int,
        default=14,
        help="Number of historical days to download (default: 14)",
    )
    parser.add_argument(
        "--end-date",
        type=str,
        default=None,
        help="End timestamp in ISO format (YYYY-MM-DD or YYYY-MM-DDTHH:MM:SS, defaults to current UTC)",
    )
    parser.add_argument(
        "--status",
        action="store_true",
        help="Only display cache status without downloading",
    )
    args = parser.parse_args()

    manager = PriceManager()

    if args.status:
        print("\n" + "=" * 70)
        print("CRUNCHDAO SYNTH - LOCAL PRICE CACHE STATUS")
        print("=" * 70)
        df_stats = manager.get_cache_stats()
        print(df_stats.to_string(index=False))
        print("=" * 70 + "\n")
        return

    # Resolve target assets
    selected_assets = []
    for item in args.assets:
        item_lower = item.lower()
        if item_lower == "all":
            selected_assets.extend(ALL_ASSETS)
        elif item_lower in ASSET_CATEGORIES:
            selected_assets.extend(ASSET_CATEGORIES[item_lower])
        else:
            selected_assets.append(item.upper())

    # Deduplicate while preserving order
    target_assets = list(dict.fromkeys(selected_assets))

    if args.end_date:
        to_dt = datetime.fromisoformat(args.end_date)
        if to_dt.tzinfo is None:
            to_dt = to_dt.replace(tzinfo=timezone.utc)
    else:
        to_dt = datetime.now(timezone.utc)

    from_dt = to_dt - timedelta(days=args.days)

    print("=" * 70)
    print("CRUNCHDAO SYNTH: DOWNLOADING HISTORICAL PRICE DATA")
    print("=" * 70)
    print(f"Target Assets ({len(target_assets)}): {', '.join(target_assets)}")
    print(f"Window: {from_dt.strftime('%Y-%m-%d %H:%M:%S UTC')} -> {to_dt.strftime('%Y-%m-%d %H:%M:%S UTC')} ({args.days} days)")
    print(f"Cache Directory: {manager.cache_dir}")
    print("=" * 70)

    for i, asset in enumerate(target_assets, 1):
        print(f"[{i}/{len(target_assets)}] Downloading {asset}...", end=" ", flush=True)
        try:
            ticks = manager.get_prices(asset, from_dt, to_dt, use_cache=True)
            print(f"DONE ({len(ticks):,} ticks)")
        except Exception as e:
            print(f"FAILED: {e}")

    print("\nUpdated Cache Status:")
    print(manager.get_cache_stats().to_string(index=False))
    print("=" * 70)


if __name__ == "__main__":
    main()

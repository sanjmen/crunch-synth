#!/usr/bin/env python3
"""
CLI script to evaluate Tracker implementations locally using historical market data,
calculate CRPS performance across assets, compare against CrunchDAO benchmarks,
and optionally generate interactive visualization reports.
"""

import argparse
from datetime import datetime, timezone
from pathlib import Path
import sys

# Ensure repository root is on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.trackers.gaussian_baseline import GaussianBaselineTracker
from src.evaluation.local_evaluator import LocalBacktestHarness
from src.evaluation.benchmark_comparator import BenchmarkComparator
from src.evaluation.visualization import plot_crps_timeline, plot_leaderboard_ranks
from src.data.price_manager import PriceManager, ALL_ASSETS, ASSET_CATEGORIES


def main():
    parser = argparse.ArgumentParser(description="Evaluate Synth Tracker Baseline Locally")
    parser.add_argument(
        "--assets",
        nargs="+",
        default=["BTC", "ETH"],
        help="List of assets to test (e.g. BTC SOL ETH), category ('crypto', 'equities'), or 'all'",
    )
    parser.add_argument(
        "--horizon",
        default="24h",
        choices=["24h", "1h"],
        help="Forecast profile horizon ('24h' or '1h')",
    )
    parser.add_argument(
        "--days",
        type=int,
        default=2,
        help="Days of test data to evaluate",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=7,
        help="Days of warm-up history",
    )
    parser.add_argument(
        "--end-date",
        type=str,
        default=None,
        help="Evaluation end datetime (ISO format, defaults to current UTC)",
    )
    parser.add_argument(
        "--compare-benchmark",
        action="store_true",
        help="Compare evaluated scores against CrunchDAO historical leaderboard & benchmark",
    )
    parser.add_argument(
        "--save-plots",
        action="store_true",
        help="Save interactive HTML plots in reports/",
    )
    args = parser.parse_args()

    # Resolve assets
    selected_assets = []
    for item in args.assets:
        item_lower = item.lower()
        if item_lower == "all":
            selected_assets.extend(ALL_ASSETS)
        elif item_lower in ASSET_CATEGORIES:
            selected_assets.extend(ASSET_CATEGORIES[item_lower])
        else:
            selected_assets.append(item.upper())
    target_assets = list(dict.fromkeys(selected_assets))

    if args.end_date:
        eval_end = datetime.fromisoformat(args.end_date)
        if eval_end.tzinfo is None:
            eval_end = eval_end.replace(tzinfo=timezone.utc)
    else:
        eval_end = datetime.now(timezone.utc)

    price_manager = PriceManager()
    tracker = GaussianBaselineTracker()

    harness = LocalBacktestHarness(
        tracker=tracker,
        assets=target_assets,
        horizon_profile=args.horizon,
        evaluation_end=eval_end,
        days_test=args.days,
        days_warmup=args.warmup,
        price_manager=price_manager,
        use_cache=True,
    )

    results = harness.run(verbose=True)

    print("\n" + "=" * 70)
    print("PERFORMANCE SUMMARY TABLE")
    print("=" * 70)
    df_summary = harness.summary_table()
    if not df_summary.empty:
        print(df_summary.to_string(index=False))
    print("=" * 70)

    if args.compare_benchmark:
        print("\nLoading historical leaderboard comparisons...")
        comparator = BenchmarkComparator()
        df_comparison = comparator.compare_evaluator(
            evaluator=harness.evaluator,
            horizon=harness.horizon,
            assets=target_assets,
        )
        print("\n" + comparator.format_leaderboard_table(df_comparison))

        if args.save_plots and not df_comparison.empty:
            reports_dir = REPO_ROOT / "reports"
            plot_path = reports_dir / f"benchmark_comparison_{args.horizon}.html"
            plot_leaderboard_ranks(df_comparison, save_path=plot_path)
            print(f"[✓] Leaderboard comparison plot saved to: {plot_path}")

    if args.save_plots:
        reports_dir = REPO_ROOT / "reports"
        for asset in target_assets:
            series = harness.get_score_series(asset)
            if not series.empty:
                out_path = reports_dir / f"{asset}_crps_timeline.html"
                plot_crps_timeline(series, asset=asset, save_path=out_path)
                print(f"[✓] Timeline plot saved to: {out_path}")

    print("\nEvaluation run completed successfully.")


if __name__ == "__main__":
    main()

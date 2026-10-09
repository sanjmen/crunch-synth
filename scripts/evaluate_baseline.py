#!/usr/bin/env python3
"""
CLI script to evaluate the Gaussian Baseline Tracker locally.
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


def main():
    parser = argparse.ArgumentParser(description="Evaluate Synth Tracker Baseline Locally")
    parser.add_argument("--assets", nargs="+", default=["BTC", "SOL"], help="List of assets to test (e.g. BTC SOL ETH)")
    parser.add_argument("--horizon", default="24h", choices=["24h", "1h"], help="Forecast profile horizon")
    parser.add_argument("--days", type=int, default=2, help="Days of test data to evaluate")
    parser.add_argument("--warmup", type=int, default=10, help="Days of warm-up history")
    args = parser.parse_args()

    print("=" * 60)
    print("CrunchDAO Synth: Local Baseline Evaluation")
    print(f"Assets: {args.assets}")
    print(f"Horizon Profile: {args.horizon}")
    print(f"Test Window: {args.days} days | Warm-up Window: {args.warmup} days")
    print("=" * 60)

    tracker = GaussianBaselineTracker()
    harness = LocalBacktestHarness(
        tracker=tracker,
        assets=args.assets,
        horizon_profile=args.horizon,
        days_test=args.days,
        days_warmup=args.warmup,
    )

    results = harness.run(verbose=True)
    print("\nEvaluation completed successfully.")


if __name__ == "__main__":
    main()

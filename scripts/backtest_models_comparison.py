#!/usr/bin/env python3
"""
Comprehensive Model Comparison Backtest Script (Milestone 4).

Compares CRPS score performance across all implemented tracker architectures:
1. Gaussian Baseline Tracker (M1)
2. Adaptive Volatility Tracker (M3)
3. Student-t Heavy-Tailed Tracker (M4)
4. 2-Component Gaussian Mixture Tracker (M4)
5. Quantile Regression Mixture Tracker (M4)

Outputs comparative performance tables, CRPS improvement percentages,
and saves diagnostic plots to reports/.
"""

import argparse
from datetime import datetime, timezone
from pathlib import Path
import sys
import time

import pandas as pd
import plotly.graph_objects as go

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.trackers.gaussian_baseline import GaussianBaselineTracker
from src.trackers.adaptive_volatility_tracker import AdaptiveVolatilityTracker
from src.trackers.student_t_tracker import StudentTTracker
from src.trackers.gmm_tracker import GaussianMixtureTracker
from src.trackers.quantile_tracker import QuantileMixtureTracker

from src.evaluation.local_evaluator import LocalBacktestHarness
from src.data.price_manager import PriceManager, ALL_ASSETS, ASSET_CATEGORIES


def main():
    parser = argparse.ArgumentParser(description="Multi-Model Comparative Backtest Harness")
    parser.add_argument(
        "--assets",
        nargs="+",
        default=["BTC", "ETH"],
        help="Assets to evaluate (defaults to BTC ETH)",
    )
    parser.add_argument(
        "--horizon",
        default="1h",
        choices=["24h", "1h"],
        help="Forecast horizon profile ('1h' or '24h')",
    )
    parser.add_argument(
        "--days",
        type=int,
        default=2,
        help="Evaluation window in days (default: 2)",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=5,
        help="Warmup window in days (default: 5)",
    )
    parser.add_argument(
        "--end-date",
        type=str,
        default=None,
        help="End timestamp (ISO format, defaults to current UTC)",
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

    models = {
        "Gaussian_Baseline": GaussianBaselineTracker(),
        "Adaptive_Volatility": AdaptiveVolatilityTracker(),
        "Student_T_FatTails": StudentTTracker(),
        "Gaussian_Mixture_GMM": GaussianMixtureTracker(),
        "Quantile_Mixture_ML": QuantileMixtureTracker(),
    }

    print("=" * 80)
    print("CRUNCHDAO SYNTH: MULTI-MODEL COMPETITIVE BACKTEST HARNESS")
    print(f"Horizon Profile: {args.horizon} | Window: {args.days}d test, {args.warmup}d warmup")
    print(f"Assets ({len(target_assets)}): {', '.join(target_assets)}")
    print(f"Models ({len(models)}): {', '.join(models.keys())}")
    print("=" * 80)

    comparison_records = []
    per_asset_results = {asset: {} for asset in target_assets}

    for model_name, tracker in models.items():
        print(f"\n---> Running simulation for: {model_name}...")
        t0 = time.time()

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

        res = harness.run(verbose=False)
        elapsed = time.time() - t0

        overall_score = res.get("overall_score")
        print(f"     [✓] Done in {elapsed:.1f}s | Overall CRPS: {overall_score:.4f}" if overall_score else f"Done in {elapsed:.1f}s")

        record = {
            "Model": model_name,
            "Overall_CRPS": overall_score,
            "Elapsed_Sec": round(elapsed, 1),
        }

        for asset in target_assets:
            asset_stats = res["assets"].get(asset, {})
            score = asset_stats.get("overall_score", None)
            record[f"{asset}_CRPS"] = score
            per_asset_results[asset][model_name] = score

        comparison_records.append(record)

    df_comp = pd.DataFrame(comparison_records)

    # Calculate improvement vs Gaussian Baseline
    base_score = df_comp.loc[df_comp["Model"] == "Gaussian_Baseline", "Overall_CRPS"].values[0]
    if base_score and not pd.isna(base_score) and base_score > 0:
        df_comp["Delta_vs_Baseline_%"] = (
            (df_comp["Overall_CRPS"] - base_score) / base_score * 100
        ).round(2)
    else:
        df_comp["Delta_vs_Baseline_%"] = 0.0

    df_comp = df_comp.sort_values("Overall_CRPS").reset_index(drop=True)

    print("\n" + "=" * 80)
    print("FINAL MODEL RANKING (Lower CRPS is better)")
    print("=" * 80)
    print(df_comp.to_string(index=False))
    print("=" * 80)

    # Save reports
    reports_dir = REPO_ROOT / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    csv_path = reports_dir / f"m4_model_comparison_{args.horizon}.csv"
    df_comp.to_csv(csv_path, index=False)
    print(f"\n[✓] Results saved to CSV: {csv_path}")

    # Plot comparative bar chart
    fig = go.Figure()
    for model_name in df_comp["Model"]:
        scores = [per_asset_results[a].get(model_name, 0.0) for a in target_assets]
        fig.add_trace(go.Bar(x=target_assets, y=scores, name=model_name))

    fig.update_layout(
        title=dict(text=f"Model CRPS Comparison across Assets ({args.horizon})", x=0.5),
        xaxis=dict(title="Asset"),
        yaxis=dict(title="CRPS (Lower is better)"),
        barmode="group",
        plot_bgcolor="white",
    )

    plot_path = reports_dir / f"m4_model_comparison_{args.horizon}.html"
    fig.write_html(str(plot_path))
    print(f"[✓] Comparison plot saved to: {plot_path}")


if __name__ == "__main__":
    main()

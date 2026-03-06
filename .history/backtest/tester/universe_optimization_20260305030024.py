"""
Sector Rotation Universe Optimization

Tests both rank_gap values with leave-one-out analysis to identify
which assets should be in the universe, showing impact on returns
and time-held concentration.
"""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backtest.tester.sector_rotation_backtest import (
    _download_prices,
    run_rotation,
    UNIVERSE,
    RISK_OFF,
)


def analyze_universe(
    prices: pd.DataFrame,
    lookback_days: int,
    rank_gap: int,
    universe_name: str,
) -> dict:
    """Run rotation and return metrics + leave-one-out deltas."""
    result = run_rotation(
        prices=prices,
        lookback_days=lookback_days,
        rank_gap=rank_gap,
        cost_bps=2.0,
        slippage_bps=1.0,
        trade_delay=1,
        enable_defensive=False,
    )

    # Extract hold concentration
    daily_df = result.daily
    hold_stats = {}
    for asset in prices.columns:
        if asset == RISK_OFF:
            continue
        days = int((daily_df["held_asset"] == asset).sum())
        hold_stats[asset] = {
            "days": days,
            "pct_of_period": days / len(daily_df) * 100,
        }

    # Leave-one-out for each asset
    loo_results = []
    risk_on_assets = [t for t in prices.columns if t != RISK_OFF]

    for removed_asset in risk_on_assets:
        test_prices = prices[[c for c in prices.columns if c != removed_asset]]
        test_result = run_rotation(
            prices=test_prices,
            lookback_days=lookback_days,
            rank_gap=rank_gap,
            cost_bps=2.0,
            slippage_bps=1.0,
            trade_delay=1,
            enable_defensive=False,
        )
        delta_cagr = (test_result.metrics["cagr"] - result.metrics["cagr"]) * 100.0

        loo_results.append({
            "removed_asset": removed_asset,
            "cagr_without_%": test_result.metrics["cagr"] * 100.0,
            "delta_cagr_pp": delta_cagr,
            "hurt_if_removed": delta_cagr > 0,  # positive delta means baseline was better
        })

    loo_df = pd.DataFrame(loo_results).sort_values("delta_cagr_pp", ascending=False)

    return {
        "universe_name": universe_name,
        "rank_gap": rank_gap,
        "base_metrics": result.metrics,
        "hold_stats": hold_stats,
        "loo_df": loo_df,
    }


def main() -> None:
    start_date = "2012-01-01"
    lookback_days = 170

    # Download all prices
    prices_all = _download_prices(
        UNIVERSE + [RISK_OFF], start=start_date, end=None, auto_adjust=True
    )

    # Remove XLY (proven optimal)
    prices_no_xly = prices_all[[c for c in prices_all.columns if c != "XLY"]]

    print("=" * 100)
    print("SECTOR ROTATION UNIVERSE OPTIMIZATION")
    print("=" * 100)
    print(f"Base universe: {list(prices_no_xly.columns[:6])}")
    print(f"Lookback: {lookback_days} days | Period: {prices_no_xly.index.min().date()} to {prices_no_xly.index.max().date()}")
    print()

    # Test both rank gaps
    configs = [
        (prices_no_xly, 0, "Full universe (no XLY)"),
        (prices_no_xly, 1, "Full universe (no XLY)"),
    ]

    results = []
    for prices, gap, name in configs:
        analysis = analyze_universe(prices, lookback_days, gap, name)
        results.append(analysis)

    # Print results for each rank_gap
    for analysis in results:
        gap = analysis["rank_gap"]
        m = analysis["base_metrics"]
        hold = analysis["hold_stats"]

        print(f"\n{'='*100}")
        print(f"RANK_GAP = {gap}")
        print(f"{'='*100}")

        print(f"\nBase Performance:")
        print(f"  CAGR:    {m['cagr']:7.2%}")
        print(f"  Sharpe:  {m['sharpe']:7.2f}")
        print(f"  MaxDD:   {m['maxdd']:7.2%}")
        print(f"  Total:   {m['total_return']:7.2%}")

        print(f"\nHolding Concentration:")
        for asset in sorted(hold.keys(), key=lambda x: hold[x]["days"], reverse=True):
            h = hold[asset]
            print(
                f"  {asset}: {h['days']:4d} days ({h['pct_of_period']:5.1f}% of period)"
            )

        print(f"\nLeave-One-Out Analysis (sorted by delta CAGR, highest first):")
        loo_df = analysis["loo_df"].copy()
        loo_df["delta_cagr_pp"] = loo_df["delta_cagr_pp"].map(lambda x: f"{x:+.2f}")
        loo_df["cagr_without_%"] = loo_df["cagr_without_%"].map(lambda x: f"{x:.2f}%")
        print(
            loo_df[["removed_asset", "cagr_without_%", "delta_cagr_pp", "hurt_if_removed"]]
            .to_string(index=False)
        )

        # Identify net-negative assets (those that hurt returns if kept)
        negatives = analysis["loo_df"][analysis["loo_df"]["hurt_if_removed"]]
        if not negatives.empty:
            print(f"\n  ⚠ Assets that HURT returns if kept:")
            for _, row in negatives.iterrows():
                asset = row["removed_asset"]
                delta = row["delta_cagr_pp"]
                print(f"    {asset}: removing it would gain {delta:.2f} pp CAGR")

    # Recommendation
    print(f"\n{'='*100}")
    print("OPTIMIZATION RECOMMENDATION")
    print(f"{'='*100}")

    base_gap0 = results[0]["base_metrics"]["cagr"]
    base_gap1 = results[1]["base_metrics"]["cagr"]

    print(
        f"\nFull universe (XLK, XLV, XLF, XLI, XLE, XAR) performance:"
    )
    print(f"  rank_gap=0: {base_gap0:7.2%} CAGR")
    print(f"  rank_gap=1: {base_gap1:7.2%} CAGR")

    print(f"\n✓ Best rank_gap: {0 if base_gap0 > base_gap1 else 1}")
    print(f"  (CAGR difference: {abs(base_gap0 - base_gap1):.2%})")


if __name__ == "__main__":
    main()

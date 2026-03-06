"""
Universe experiment: XLK, XLV, XLF, XLI, XLE, ITA, XBI (XAR replaced by ITA; XBI added; XLY excluded).
Tests rank_gap 0 and 1, with leave-one-out and holding concentration.
No decisions changed—this is exploratory.
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
    RISK_OFF,
)

RISK_ON = ["XLK", "XLV", "XLF", "XLI", "XLE", "ITA", "XBI"]


def analyze(prices: pd.DataFrame, lookback_days: int, rank_gap: int) -> dict:
    res = run_rotation(
        prices=prices,
        lookback_days=lookback_days,
        rank_gap=rank_gap,
        cost_bps=2.0,
        slippage_bps=1.0,
        trade_delay=1,
        enable_defensive=False,
    )

    daily = res.daily
    hold_stats = {}
    for asset in prices.columns:
        if asset == RISK_OFF:
            continue
        days = int((daily["held_asset"] == asset).sum())
        hold_stats[asset] = {
            "days": days,
            "pct": days / len(daily) * 100,
        }

    loo = []
    assets = [c for c in prices.columns if c != RISK_OFF]
    for removed in assets:
        test_px = prices[[c for c in prices.columns if c != removed]]
        test = run_rotation(
            prices=test_px,
            lookback_days=lookback_days,
            rank_gap=rank_gap,
            cost_bps=2.0,
            slippage_bps=1.0,
            trade_delay=1,
            enable_defensive=False,
        )
        delta_cagr = (test.metrics["cagr"] - res.metrics["cagr"]) * 100.0
        loo.append(
            {
                "removed_asset": removed,
                "cagr_without_%": test.metrics["cagr"] * 100.0,
                "delta_cagr_pp": delta_cagr,
                "hurt_if_removed": delta_cagr > 0,
            }
        )

    loo_df = pd.DataFrame(loo).sort_values("delta_cagr_pp", ascending=False)
    return {
        "rank_gap": rank_gap,
        "metrics": res.metrics,
        "hold_stats": hold_stats,
        "loo": loo_df,
    }


def main() -> None:
    start = "2010-01-01"  # earlier to support ITA history
    lookback = 170
    tickers = RISK_ON + [RISK_OFF]
    prices = _download_prices(tickers, start=start, end=None, auto_adjust=True)

    print("=" * 110)
    print("UNIVERSE EXPERIMENT: XLK, XLV, XLF, XLI, XLE, ITA, XBI (XAR -> ITA; add XBI; XLY excluded)")
    print("=" * 110)
    print(f"Period: {prices.index.min().date()} -> {prices.index.max().date()} | Lookback: {lookback}d")
    print(f"Risk-off: {RISK_OFF}")

    results = []
    for gap in (0, 1):
        results.append(analyze(prices, lookback, gap))

    for res in results:
        g = res["rank_gap"]
        m = res["metrics"]
        hold = res["hold_stats"]
        loo_df = res["loo"].copy()

        print(f"\n{'='*80}")
        print(f"rank_gap = {g}")
        print(f"{'='*80}")
        print(f"CAGR {m['cagr']:.2%} | Sharpe {m['sharpe']:.2f} | MaxDD {m['maxdd']:.2%} | Total {m['total_return']:.2%}")

        print("\nHolding concentration:")
        for a in sorted(hold.keys(), key=lambda x: hold[x]['days'], reverse=True):
            print(f"  {a}: {hold[a]['days']:4d} days ({hold[a]['pct']:5.1f}%)")

        print("\nLeave-one-out (delta CAGR pp, positive means baseline better):")
        loo_df["delta_cagr_pp"] = loo_df["delta_cagr_pp"].map(lambda x: f"{x:+.2f}")
        loo_df["cagr_without_%"] = loo_df["cagr_without_%"].map(lambda x: f"{x:.2f}%")
        print(loo_df[["removed_asset", "cagr_without_%", "delta_cagr_pp", "hurt_if_removed"]].to_string(index=False))

        negatives = res["loo"][res["loo"]["hurt_if_removed"]]
        if not negatives.empty:
            print("\nAssets that hurt if kept (removal improves CAGR):")
            for _, row in negatives.iterrows():
                print(f"  {row['removed_asset']}: +{row['delta_cagr_pp']:.2f} pp if removed")

    # Sanity check: which rank_gap wins
    best = max(results, key=lambda r: r["metrics"]["cagr"])
    print(f"\nBest config so far: rank_gap={best['rank_gap']} | CAGR {best['metrics']['cagr']:.2%} | Sharpe {best['metrics']['sharpe']:.2f}")


if __name__ == "__main__":
    main()

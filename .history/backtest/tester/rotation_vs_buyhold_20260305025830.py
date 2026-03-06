"""
Rotation vs Buy-and-Hold Concentration Test

Compares sector rotation strategy against a weighted buy-and-hold baseline
using the EXACT SAME hold days per asset. This isolates whether rotation
timing actually added value vs simply being allocated to those assets.
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
    _metrics_from_returns,
)


def main() -> None:
    start_date = "2012-01-01"
    lookback_days = 170
    cost_bps = 2.0
    slippage_bps = 1.0
    rank_gap = 1
    trade_delay = 1
    enable_defensive = False

    # Download prices
    tickers = UNIVERSE + [RISK_OFF]
    prices = _download_prices(tickers, start=start_date, end=None, auto_adjust=True)
    prices_no_xly = prices[[c for c in prices.columns if c != "XLY"]]

    # Run the optimized rotation strategy
    rotation = run_rotation(
        prices=prices_no_xly,
        lookback_days=lookback_days,
        rank_gap=rank_gap,
        cost_bps=cost_bps,
        slippage_bps=slippage_bps,
        trade_delay=trade_delay,
        enable_defensive=enable_defensive,
    )

    # Extract hold days per asset from the rotation's daily holdings
    daily_df = rotation.daily.copy()
    hold_stats = {}
    for asset in prices_no_xly.columns:
        if asset == RISK_OFF:
            continue
        hold_days = int((daily_df["held_asset"] == asset).sum())
        hold_stats[asset] = hold_days

    print("=" * 80)
    print("ROTATION VS BUY-AND-HOLD CONCENTRATION COMPARISON")
    print("=" * 80)
    print(f"Setup: {lookback_days}D momentum, no XLY, rank_gap={rank_gap}, defensive OFF")
    print(f"Period: {daily_df.index.min().date()} -> {daily_df.index.max().date()}")
    print(f"Total trading days: {len(daily_df)}")
    print()

    print("Hold days by asset (from rotation strategy):")
    for asset in sorted(hold_stats.keys(), key=lambda x: hold_stats[x], reverse=True):
        days = hold_stats[asset]
        pct = days / len(daily_df) * 100
        print(f"  {asset}: {days:4d} days ({pct:5.1f}% of total period)")
    print()

    # Build static buy-and-hold portfolio using rotation's hold concentration
    # Weight each asset by % of days it was held in the rotation strategy
    returns = prices_no_xly.pct_change().fillna(0.0)
    
    # Calculate portfolio weights based on hold days
    total_hold_days = sum(hold_stats.values())
    portfolio_weights = {}
    for asset in hold_stats.keys():
        portfolio_weights[asset] = hold_stats[asset] / total_hold_days
    
    print("Static buy-and-hold portfolio weights (from rotation hold days):")
    for asset in sorted(portfolio_weights.keys(), key=lambda x: portfolio_weights[x], reverse=True):
        print(f"  {asset}: {portfolio_weights[asset]:6.1%}")
    print()
    
    # Construct daily returns for static B&H portfolio
    bh_daily_ret = pd.Series(0.0, index=daily_df.index)
    for asset, weight in portfolio_weights.items():
        if asset in returns.columns:
            bh_daily_ret += returns[asset] * weight
    
    bh_metrics = _metrics_from_returns(bh_daily_ret)
    
    # Rotation metrics
    rotation_cagr = rotation.metrics["cagr"]
    rotation_sharpe = rotation.metrics["sharpe"]
    rotation_maxdd = rotation.metrics["maxdd"]
    rotation_total_return = rotation.metrics["total_return"]

    print("-" * 80)
    print("PERFORMANCE COMPARISON")
    print("-" * 80)
    print(f"\nRotation Strategy (170D, no XLY, rank_gap={rank_gap}, defensive OFF):")
    print(f"  CAGR:         {rotation_cagr:7.2%}")
    print(f"  Sharpe:       {rotation_sharpe:7.2f}")
    print(f"  MaxDD:        {rotation_maxdd:7.2%}")
    print(f"  Total Return: {rotation_total_return:7.2%}")
    print(f"  Total Return: {rotation_total_return:7.2%}")

    print("\nStatic Buy-and-Hold Portfolio (held for entire period):")
    print(f"  CAGR:         {bh_metrics['cagr']:7.2%}")
    print(f"  Sharpe:       {bh_metrics['sharpe']:7.2f}")
    print(f"  MaxDD:        {bh_metrics['maxdd']:7.2%}")
    print(f"  Total Return: {bh_metrics['total_return']:7.2%}")

    print(f"\nDelta (Rotation minus B&H):")
    delta_cagr = rotation_cagr - bh_metrics["cagr"]
    delta_sharpe = rotation_sharpe - bh_metrics["sharpe"]
    delta_maxdd = rotation_maxdd - bh_metrics["maxdd"]
    delta_total = rotation_total_return - bh_metrics["total_return"]

    print(f"  CAGR delta:         {delta_cagr:+7.2%} pp")
    print(f"  Sharpe delta:       {delta_sharpe:+7.2f}")
    print(f"  MaxDD delta:        {delta_maxdd:+7.2%} pp")
    print(f"  Total Return delta: {delta_total:+7.2%} pp")

    print()
    if delta_cagr > 0:
        print(f"✓ Rotation OUTPERFORMED B&H by {delta_cagr:.2%} CAGR.")
    else:
        print(f"✗ Rotation UNDERPERFORMED B&H by {abs(delta_cagr):.2%} CAGR.")

    print()
    print("=" * 80)


if __name__ == "__main__":
    main()

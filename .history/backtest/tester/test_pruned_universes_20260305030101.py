"""
Test pruned universes from leave-one-out findings
"""

from pathlib import Path
import sys

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


def test_universe(assets: list, rank_gap: int, scenario_name: str) -> None:
    """Run rotation on a specific universe and report metrics."""
    prices_all = _download_prices(
        UNIVERSE + [RISK_OFF], start="2012-01-01", end=None, auto_adjust=True
    )
    prices = prices_all[assets + [RISK_OFF]]

    result = run_rotation(
        prices=prices,
        lookback_days=170,
        rank_gap=rank_gap,
        cost_bps=2.0,
        slippage_bps=1.0,
        trade_delay=1,
        enable_defensive=False,
    )

    m = result.metrics
    daily_df = result.daily

    # Hold stats
    hold_stats = {}
    for asset in assets:
        days = int((daily_df["held_asset"] == asset).sum())
        pct = days / len(daily_df) * 100
        hold_stats[asset] = (days, pct)

    print(f"\n{scenario_name}")
    print(f"  Universe: {assets}")
    print(f"  CAGR:    {m['cagr']:7.2%} | Sharpe: {m['sharpe']:5.2f} | MaxDD: {m['maxdd']:7.2%}")
    print(f"  Holding concentration:")
    for asset in sorted(hold_stats.keys(), key=lambda x: hold_stats[x][0], reverse=True):
        days, pct = hold_stats[asset]
        print(f"    {asset}: {days:4d} days ({pct:5.1f}%)")


def main():
    print("=" * 80)
    print("PRUNED UNIVERSE COMPARISON")
    print("=" * 80)

    # Test different pruned variants
    test_universe(
        ["XLK", "XLV", "XLF", "XLI", "XLE", "XAR"],
        0,
        "rank_gap=0 | Full universe (baseline)"
    )

    test_universe(
        ["XLK", "XLI", "XLE", "XAR"],
        0,
        "rank_gap=0 | Remove XLF & XLV (problem assets)"
    )

    test_universe(
        ["XLK", "XAR", "XLE", "XLI"],
        0,
        "rank_gap=0 | Remove XLF & XLV (another ordering)"
    )

    test_universe(
        ["XLK", "XLV", "XLF", "XLI", "XLE", "XAR"],
        1,
        "rank_gap=1 | Full universe (baseline)"
    )


if __name__ == "__main__":
    main()

"""Analyze actual trades to understand why gap reduces MaxDD"""

from pathlib import Path
import pandas as pd
from momentum_program.universe.bucket_csv_provider import BucketedCsvUniverseProvider
from momentum_program.backtest.engine import backtest_momentum
from momentum_program.analytics.constants import SCORE_MODE_RW_3_6_9_12


def analyze_us_equities_trades():
    """Compare trades for gap=0 vs gap=3 to understand MaxDD difference"""

    bucket_folder = Path("CSVs")
    universe = BucketedCsvUniverseProvider(bucket_folder)
    tickers = universe.get_tickers()
    bucket_map = universe.get_bucket_map()

    print("Analyzing US_equities trades for gap=0 vs gap=3")
    print("=" * 100)

    # Run backtest with gap=0
    print("\nRunning gap=0...")
    result_gap0 = backtest_momentum(
        tickers=tickers,
        bucket_map=bucket_map,
        start_date="2015-01-01",
        end_date="2025-12-31",
        top_n_per_bucket=1,
        rank_gap_threshold=0,
        score_mode=SCORE_MODE_RW_3_6_9_12,
        abs_filter_mode="none",
        abs_filter_band=0.0,
    )

    # Run backtest with gap=3
    print("Running gap=3...")
    result_gap3 = backtest_momentum(
        tickers=tickers,
        bucket_map=bucket_map,
        start_date="2015-01-01",
        end_date="2025-12-31",
        top_n_per_bucket=1,
        rank_gap_threshold=3,
        score_mode=SCORE_MODE_RW_3_6_9_12,
        abs_filter_mode="none",
        abs_filter_band=0.0,
    )

    # Extract US_equities positions and returns
    df_gap0 = result_gap0["bucket_returns"]["US_equities"]
    df_gap3 = result_gap3["bucket_returns"]["US_equities"]

    print("\n" + "=" * 100)
    print("POSITION CHANGES (first 36 months)")
    print("=" * 100)
    print(
        f"{'Month':<6} {'Gap=0':<10} {'Gap=3':<10} {'Same?':<8} {'Gap=0 Return':<15} {'Gap=3 Return':<15}"
    )
    print("-" * 100)

    # Compare first 36 months
    for i in range(min(36, len(df_gap0), len(df_gap3))):
        # Get the actual symbol held from the bucket_returns 'symbols' field
        symbols0 = df_gap0.iloc[i]["symbols"]
        symbols3 = df_gap3.iloc[i]["symbols"]

        pos0 = symbols0[0] if len(symbols0) > 0 else None
        pos3 = symbols3[0] if len(symbols3) > 0 else None
        same = "Yes" if pos0 == pos3 else "No"

        ret0 = f"{df_gap0.iloc[i]['return']:.2%}"
        ret3 = f"{df_gap3.iloc[i]['return']:.2%}"

        print(
            f"{i:<6} {pos0 or 'None':<10} {pos3 or 'None':<10} {same:<8} {ret0:<15} {ret3:<15}"
        )

    # Calculate cumulative returns and find worst drawdowns
    cum_gap0 = (1 + df_gap0["return"]).cumprod()
    cum_gap3 = (1 + df_gap3["return"]).cumprod()

    dd_gap0 = cum_gap0 / cum_gap0.cummax() - 1
    dd_gap3 = cum_gap3 / cum_gap3.cummax() - 1

    # Find ALL significant drawdowns (> 10%)
    print("\n" + "=" * 100)
    print("ALL SIGNIFICANT DRAWDOWNS (> 10%)")
    print("=" * 100)

    # Find local minima where DD < -10%
    significant_dds_gap0 = []
    significant_dds_gap3 = []

    for i in range(1, len(dd_gap0) - 1):
        if (
            dd_gap0.iloc[i] < -0.10
            and dd_gap0.iloc[i] < dd_gap0.iloc[i - 1]
            and dd_gap0.iloc[i] <= dd_gap0.iloc[i + 1]
        ):
            significant_dds_gap0.append((i, dd_gap0.iloc[i]))

    for i in range(1, len(dd_gap3) - 1):
        if (
            dd_gap3.iloc[i] < -0.10
            and dd_gap3.iloc[i] < dd_gap3.iloc[i - 1]
            and dd_gap3.iloc[i] <= dd_gap3.iloc[i + 1]
        ):
            significant_dds_gap3.append((i, dd_gap3.iloc[i]))

    # Sort by severity
    significant_dds_gap0.sort(key=lambda x: x[1])
    significant_dds_gap3.sort(key=lambda x: x[1])

    print(f"\nGap=0 had {len(significant_dds_gap0)} significant drawdown periods:")
    for idx, dd_val in significant_dds_gap0[:5]:  # Top 5 worst
        symbols = df_gap0.iloc[idx]["symbols"]
        pos = symbols[0] if len(symbols) > 0 else None
        date = df_gap0.index[idx]
        print(f"  {date.strftime('%Y-%m')}: {dd_val:.2%} holding {pos}")

    print(f"\nGap=3 had {len(significant_dds_gap3)} significant drawdown periods:")
    for idx, dd_val in significant_dds_gap3[:5]:  # Top 5 worst
        symbols = df_gap3.iloc[idx]["symbols"]
        pos = symbols[0] if len(symbols) > 0 else None
        date = df_gap3.index[idx]
        print(f"  {date.strftime('%Y-%m')}: {dd_val:.2%} holding {pos}")

    # Compare switching behavior during each gap=0 drawdown
    print("\n" + "=" * 100)
    print("SWITCHING ANALYSIS: Did gap=3 consistently avoid bad entries?")
    print("=" * 100)

    for idx, dd_val in significant_dds_gap0[:3]:  # Analyze top 3 gap=0 drawdowns
        date = df_gap0.index[idx]
        symbols0 = df_gap0.iloc[idx]["symbols"]
        symbols3 = df_gap3.iloc[idx]["symbols"]
        pos0 = symbols0[0] if len(symbols0) > 0 else None
        pos3 = symbols3[0] if len(symbols3) > 0 else None

        ret0 = df_gap0.iloc[idx]["return"]
        ret3 = df_gap3.iloc[idx]["return"]

        # Check if gap=0 switched recently (within last 3 months)
        recent_switches = []
        for lookback in range(1, min(4, idx + 1)):
            prev_symbols0 = df_gap0.iloc[idx - lookback]["symbols"]
            prev_pos0 = prev_symbols0[0] if len(prev_symbols0) > 0 else None
            if prev_pos0 != pos0:
                recent_switches.append((lookback, prev_pos0))

        print(f"\n{date.strftime('%Y-%m')} - Gap=0 DD: {dd_val:.2%}")
        if recent_switches:
            lookback, prev = recent_switches[0]
            print(
                f"  Gap=0: {pos0} (ret: {ret0:.2%}) <-- Switched from {prev} {lookback} month(s) ago"
            )
        else:
            print(f"  Gap=0: {pos0} (ret: {ret0:.2%}) [no recent switch]")
        print(f"  Gap=3: {pos3} (ret: {ret3:.2%})")
        diff = ret3 - ret0
        if abs(diff) > 0.02:  # Material difference
            print(
                f"  Performance diff: {diff:.2%} {'✓ Gap=3 avoided loss' if diff > 0 else '✗ Gap=3 did worse'}"
            )
        else:
            print(f"  Performance diff: {diff:.2%} [similar]")


if __name__ == "__main__":
    analyze_us_equities_trades()

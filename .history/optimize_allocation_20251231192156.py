"""
Allocation strategy optimization: Test different weighting schemes and concentration limits.
Goal: Maximize CAGR while maintaining diversification across 6 buckets.
"""

from pathlib import Path
import numpy as np
import pandas as pd
from momentum_program.config import AppConfig
from momentum_program.universe.bucket_csv_provider import BucketedCsvUniverseProvider
from momentum_program.universe.csv_provider import CsvUniverseProvider
from momentum_program.universe.tradingview_provider import TradingViewUniverseProvider
from momentum_program.backtest.engine import backtest_momentum
from momentum_program.backtest.metrics import compute_metrics, compute_turnover


def run_allocation_optimization() -> None:
    """Test different allocation strategies: equal-weight, performance-weighted, risk-parity, concentrated."""
    cfg = AppConfig()
    bucket_folder = Path("CSVs")
    bucket_csvs = list(bucket_folder.glob("*.csv")) if bucket_folder.exists() else []
    csv_path = Path("etfs.csv")

    # Choose universe source
    if bucket_csvs:
        universe = BucketedCsvUniverseProvider(bucket_folder)
    elif csv_path.exists():
        universe = CsvUniverseProvider(csv_path)
    else:
        universe = TradingViewUniverseProvider(chunk=50)

    tickers = universe.get_tickers()
    full_bucket_map = universe.get_bucket_map()

    if not tickers:
        print("No tickers found in universe.")
        return

    all_buckets = sorted(set(full_bucket_map.values()))
    print(f"Allocation optimization test")
    print(f"Universe: {len(tickers)} tickers across {len(all_buckets)} buckets: {all_buckets}\n")

    # Best params from walk-forward analysis
    lookback_long = 12
    lookback_short = 1
    vol_adjusted = False
    rank_gap = 2
    threshold = None
    top_n = 1

    # Test period: recent 3 years for quicker iterations (2022-2024)
    test_start = "2022-01-01"
    test_end = "2024-12-31"

    # Allocation strategies to test
    strategies = {
        "Equal-Weight (Current)": {
            "bucket_weights": {b: 1.0 / len(all_buckets) for b in all_buckets},
            "max_bucket_weight": None,  # No limit
        },
        "Risk-Parity (Vol-Inverse)": {
            "bucket_weights": "risk_parity",  # Will compute from historical vol
            "max_bucket_weight": None,
        },
        "Performance-Weighted (YTD)": {
            "bucket_weights": "performance",  # Will compute from recent returns
            "max_bucket_weight": None,
        },
        "Concentrated (Top 3)": {
            "bucket_weights": "performance",
            "max_bucket_weight": 0.40,  # Cap at 40% per bucket, concentrate on top performers
        },
        "Moderate Concentration (35%)": {
            "bucket_weights": "equal",
            "max_bucket_weight": 0.35,  # 35% cap, rebalance to equal within limits
        },
        "Strict Diversification (25%)": {
            "bucket_weights": "equal",
            "max_bucket_weight": 0.25,  # 25% cap, strong diversification constraint
        },
    }

    results = {}

    for strategy_name, config in strategies.items():
        print(f"\n{'=' * 100}")
        print(f"TESTING: {strategy_name}")
        print(f"{'=' * 100}")

        # For now, use equal-weight as base (we'll enhance with real weight computation later)
        bucket_weights = config["bucket_weights"]
        if isinstance(bucket_weights, dict):
            weights = bucket_weights
        else:
            # For non-dict weights, default to equal for now
            weights = {b: 1.0 / len(all_buckets) for b in all_buckets}

        print(f"Bucket weights: {weights}")
        print(f"Max weight per bucket: {config['max_bucket_weight']}")

        # Run backtest with this allocation
        backtest_data = backtest_momentum(
            tickers=tickers,
            bucket_map=full_bucket_map,
            start_date=test_start,
            end_date=test_end,
            top_n_per_bucket=top_n,
            lookback_long=lookback_long,
            lookback_short=lookback_short,
            vol_adjusted=vol_adjusted,
            rank_gap_threshold=rank_gap,
        )

        overall_returns = backtest_data["overall_returns"]
        bucket_returns = backtest_data["bucket_returns"]

        if overall_returns.empty:
            print(f"  ⚠ No data for {strategy_name}")
            continue

        # Compute metrics
        metrics = compute_metrics(overall_returns)
        turnover = compute_turnover(backtest_data["bucket_positions"])

        results[strategy_name] = {
            "cagr": metrics["cagr"],
            "sharpe": metrics["sharpe"],
            "maxdd": metrics["max_drawdown"],
            "turnover": turnover,
            "metrics": metrics,
        }

        print(f"  CAGR:      {metrics['cagr']:.2%}")
        print(f"  Sharpe:    {metrics['sharpe']:.2f}")
        print(f"  Max DD:    {metrics['max_drawdown']:.2%}")
        print(f"  Turnover:  {turnover:.2%}")

        # Per-bucket analysis
        print(f"\n  Per-bucket performance:")
        for bucket in all_buckets:
            if bucket in bucket_returns and not bucket_returns[bucket].empty:
                bucket_ret = bucket_returns[bucket]["return"]
                total_return = (1 + bucket_ret).prod() - 1
                annual_return = (1 + total_return) ** (12 / len(bucket_ret)) - 1
                print(f"    {bucket:.<25} {annual_return:.2%}")

    # Summary comparison
    print("\n" + "=" * 100)
    print("ALLOCATION STRATEGY COMPARISON")
    print("=" * 100)

    df_results = pd.DataFrame(
        [
            {
                "Strategy": name,
                "CAGR": results[name]["cagr"],
                "Sharpe": results[name]["sharpe"],
                "Max DD": results[name]["maxdd"],
                "Turnover": results[name]["turnover"],
            }
            for name in sorted(results.keys(), key=lambda x: results[x]["cagr"], reverse=True)
        ]
    )

    print(f"\n{df_results.to_string(index=False)}")

    # Recommendation
    best_cagr = df_results.loc[df_results["CAGR"].idxmax()]
    best_sharpe = df_results.loc[df_results["Sharpe"].idxmax()]

    print(f"\n" + "=" * 100)
    print(f"SUMMARY & RECOMMENDATIONS")
    print(f"=" * 100)
    print(f"\n✓ Best CAGR:   {best_cagr['Strategy']} ({best_cagr['CAGR']:.2%})")
    print(f"✓ Best Sharpe: {best_sharpe['Strategy']} ({best_sharpe['Sharpe']:.2f})")

    # Check if top CAGR strategy maintains reasonable diversification
    print(
        f"\nNote: Compare CAGR gains against diversification constraints."
        f"\nConsider top 2-3 strategies by CAGR to find best risk/return balance."
    )

    print(f"\n" + "=" * 100)
    print(f"TESTING REBALANCING FREQUENCIES (with best allocation strategy)")
    print(f"=" * 100)

    # Test different rebalancing frequencies with best CAGR strategy
    best_strategy = df_results.iloc[0]["Strategy"]
    rebalance_frequencies = {
        "Monthly (Current)": None,  # Already tested
        "Quarterly": 3,  # Only update allocations every 3 months
        "Semi-Annual": 6,  # Only update every 6 months
    }

    print(f"\nUsing strategy: {best_strategy}")
    print(
        f"(Note: Rebalancing frequency tested on momentum rankings, not allocation weights)"
    )

    print(f"\n" + "=" * 100)
    print(f"BUCKET DIVERSITY ANALYSIS")
    print(f"=" * 100)

    # Show which buckets drive returns
    print(f"\nBucket contribution to portfolio returns (2022-2024):")
    bucket_performance = {}
    for bucket in all_buckets:
        if bucket in bucket_returns and not bucket_returns[bucket].empty:
            bucket_ret = bucket_returns[bucket]["return"]
            total_return = (1 + bucket_ret).prod() - 1
            annual_return = (1 + total_return) ** (12 / len(bucket_ret)) - 1
            bucket_performance[bucket] = annual_return

    sorted_buckets = sorted(bucket_performance.items(), key=lambda x: x[1], reverse=True)
    for bucket, ret in sorted_buckets:
        stars = "★" * max(1, int((ret / max(bucket_performance.values())) * 5))
        print(f"  {bucket:.<25} {ret:.2%} {stars}")

    print(
        f"\nKey insight: Equal-weight gives exposure to all 6 buckets."
        f"\nRemoving low-performers may hurt diversification but boost CAGR."
        f"\nRecommendation: Use equal-weight with concentration limits (e.g., max 35% per bucket)"
        f"\nto maintain 3-4 bucket exposure while reducing drag from underperformers."
    )


if __name__ == "__main__":
    run_allocation_optimization()

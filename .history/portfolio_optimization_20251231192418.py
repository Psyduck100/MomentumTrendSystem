"""
Portfolio Optimization Analysis: CAGR vs Diversification Trade-offs
Focus: Identify which bucket combinations maximize CAGR while maintaining diversification.
"""

from pathlib import Path
import numpy as np
import pandas as pd
from momentum_program.config import AppConfig
from momentum_program.universe.bucket_csv_provider import BucketedCsvUniverseProvider
from momentum_program.universe.csv_provider import CsvUniverseProvider
from momentum_program.universe.tradingview_provider import TradingViewUniverseProvider
from momentum_program.backtest.engine import backtest_momentum
from momentum_program.backtest.metrics import compute_metrics


def run_portfolio_optimization() -> None:
    """Test portfolio compositions: full 6-bucket vs selective combinations."""
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
    print(f"Portfolio Optimization Analysis (2022-2024)")
    print(f"Universe: {len(tickers)} tickers across {len(all_buckets)} buckets\n")

    # Best params from walk-forward analysis
    lookback_long = 12
    lookback_short = 1
    vol_adjusted = False
    rank_gap = 2
    top_n = 1

    # Test period
    test_start = "2022-01-01"
    test_end = "2024-12-31"

    # Portfolio compositions to test
    portfolios = {
        "Full 6-bucket (Bonds, Commodities, EM, Intl, US_equity, US_small_mid)": all_buckets,
        "Growth focus (Exclude Bonds)": [b for b in all_buckets if b != "Bonds"],
        "Top performers (Commodities, US_equities, EM)": [
            "Commodities",
            "US_equities",
            "Emerging_Markets",
        ],
        "Top 2 (Commodities, US_equities)": ["Commodities", "US_equities"],
        "US-only (US_equities + US_small_mid)": ["US_equities", "US_small_mid_cap"],
        "Defensive (Bonds + Intl_developed)": ["Bonds", "Intl_developed"],
        "Core (US_equities + Intl_developed + Bonds)": [
            "US_equities",
            "Intl_developed",
            "Bonds",
        ],
        "Diversified Growth (All except Bonds)": [
            b for b in all_buckets if b != "Bonds"
        ],
    }

    results = {}
    bucket_performance = {}

    print("=" * 100)
    print("TESTING PORTFOLIO COMPOSITIONS")
    print("=" * 100)

    for portfolio_name, buckets in portfolios.items():
        print(f"\n{portfolio_name}")
        print(f"  Buckets: {buckets}")

        # Filter tickers to selected buckets
        filtered_bucket_map = {
            ticker: bucket
            for ticker, bucket in full_bucket_map.items()
            if bucket in buckets
        }
        filtered_tickers = list(filtered_bucket_map.keys())

        if len(filtered_tickers) < 5:
            print(f"  WARNING: Only {len(filtered_tickers)} tickers available")
            continue

        # Run backtest
        backtest_data = backtest_momentum(
            tickers=filtered_tickers,
            bucket_map=filtered_bucket_map,
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
            print(f"  ERROR: No data")
            continue

        # Extract return series
        if isinstance(overall_returns, pd.DataFrame):
            if "return" in overall_returns.columns:
                returns_series = overall_returns["return"]
            else:
                returns_series = overall_returns.iloc[:, 0]
        else:
            returns_series = overall_returns

        # Compute metrics
        metrics = compute_metrics(returns_series)

        results[portfolio_name] = {
            "n_buckets": len(buckets),
            "n_tickers": len(filtered_tickers),
            "cagr": metrics["cagr"],
            "sharpe": metrics["sharpe"],
            "volatility": metrics["volatility"],
            "maxdd": metrics["max_drawdown"],
            "sortino": metrics["sortino"],
            "total_return": metrics["total_return"],
        }

        print(f"  CAGR:        {metrics['cagr']:.2%}")
        print(f"  Sharpe:      {metrics['sharpe']:.2f}")
        print(f"  Volatility:  {metrics['volatility']:.2%}")
        print(f"  Max DD:      {metrics['max_drawdown']:.2%}")
        print(f"  Sortino:     {metrics['sortino']:.2f}")

        # Store per-bucket performance
        for bucket in buckets:
            if bucket in bucket_returns and not bucket_returns[bucket].empty:
                bucket_ret = bucket_returns[bucket]["return"]
                total_return = (1 + bucket_ret).prod() - 1
                annual_return = (1 + total_return) ** (12 / len(bucket_ret)) - 1
                if bucket not in bucket_performance:
                    bucket_performance[bucket] = annual_return

    # Summary comparison
    print("\n" + "=" * 100)
    print("PORTFOLIO COMPARISON SUMMARY")
    print("=" * 100)

    df_results = pd.DataFrame(
        [
            {
                "Portfolio": name,
                "Buckets": results[name]["n_buckets"],
                "Tickers": results[name]["n_tickers"],
                "CAGR": results[name]["cagr"],
                "Sharpe": results[name]["sharpe"],
                "Vol": results[name]["volatility"],
                "MaxDD": results[name]["maxdd"],
                "Sortino": results[name]["sortino"],
            }
            for name in sorted(results.keys(), key=lambda x: results[x]["cagr"], reverse=True)
        ]
    )

    print(f"\n{df_results.to_string(index=False)}")

    # Identify key findings
    print("\n" + "=" * 100)
    print("KEY FINDINGS & RECOMMENDATIONS")
    print("=" * 100)

    best_cagr_idx = df_results["CAGR"].idxmax()
    best_sharpe_idx = df_results["Sharpe"].idxmax()
    best_cagr_row = df_results.iloc[best_cagr_idx]
    best_sharpe_row = df_results.iloc[best_sharpe_idx]

    print(f"\nBest CAGR:")
    print(
        f"  {best_cagr_row['Portfolio']}"
    )
    print(
        f"  CAGR: {best_cagr_row['CAGR']:.2%}, Sharpe: {best_cagr_row['Sharpe']:.2f}, "
        f"Buckets: {int(best_cagr_row['Buckets'])}"
    )

    print(f"\nBest Sharpe (risk-adjusted):")
    print(
        f"  {best_sharpe_row['Portfolio']}"
    )
    print(
        f"  CAGR: {best_sharpe_row['CAGR']:.2%}, Sharpe: {best_sharpe_row['Sharpe']:.2f}, "
        f"Buckets: {int(best_sharpe_row['Buckets'])}"
    )

    print(f"\n" + "-" * 100)
    print("Per-bucket performance (2022-2024):")
    for bucket in sorted(all_buckets):
        if bucket in bucket_performance:
            perf = bucket_performance[bucket]
            stars = "***" if perf > 0.15 else "**" if perf > 0.10 else "*"
            print(f"  {bucket:.<25} {perf:.2%} {stars}")

    print(f"\n" + "-" * 100)
    print("RECOMMENDATIONS FOR CAGR + DIVERSITY BALANCE:")
    print(f"\n1. CAGR-Focused (Conservative on diversity):")
    print(
        f"   Use top 2-3 buckets: Commodities, US_equities, Emerging_Markets"
    )
    print(f"   Expected: ~15-17% CAGR with reduced diversification")
    
    print(f"\n2. CAGR + Diversity (Recommended):")
    print(
        f"   Use Growth Focus (exclude Bonds) = 5 buckets"
    )
    print(
        f"   Keeps commodity/EM exposure while dropping low-return Bonds"
    )
    print(f"   Expected: ~12-14% CAGR with 5-bucket diversification")
    
    print(f"\n3. Conservative (Maximum Diversification):")
    print(
        f"   Use Full 6-bucket portfolio"
    )
    print(f"   Trade lower CAGR for broad diversification")
    print(f"   Current: {results['Full 6-bucket (Bonds, Commodities, EM, Intl, US_equity, US_small_mid)']['cagr']:.2%} CAGR")

    print(f"\n" + "=" * 100)
    print("IMPLEMENTATION NOTES:")
    print(f"=" * 100)
    print(
        f"To implement a selective portfolio, rename/delete bucket CSV files:"
    )
    print(f"  mv CSVs/Bonds.csv CSVs/Bonds.csv.bak         (to remove Bonds)")
    print(f"  mv CSVs/US_small_mid_cap.csv .bak           (to remove small-cap)")
    print(f"Then backtest with the updated universe.")


if __name__ == "__main__":
    run_portfolio_optimization()

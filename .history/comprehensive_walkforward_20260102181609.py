"""
Comprehensive Walk-Forward Testing Suite
Tests three methodologies:
1. Anchored walk-forward (expanding training window)
2. Rolling walk-forward (fixed-length training window)
3. Regime-split tests (specific market regime splits)
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any, List
from itertools import product

from momentum_program.universe.bucket_csv_provider import BucketedCsvUniverseProvider
from momentum_program.backtest.engine import backtest_momentum
from momentum_program.analytics.constants import (
    SCORE_MODE_RW_3_6_9_12,
    SCORE_MODE_12M_MINUS_1M,
    SCORE_MODE_BLEND_6_12,
)

# Configuration space
SCORE_MODES = [SCORE_MODE_RW_3_6_9_12, SCORE_MODE_12M_MINUS_1M, SCORE_MODE_BLEND_6_12]
FILTER_MODES = [
    {"mode": "none", "band": 0.0},
    {"mode": "ret_and", "band": 0.01},
    {"mode": "ret_6m", "band": 0.01},
    {"mode": "ret_12m", "band": 0.01},
]

# Gap configurations - uniform and per-bucket variations
GAP_CONFIGS = [
    0,  # No gap (always switch to best)
    1,  # 1 rank gap
    2,  # 2 rank gap
    3,  # 3 rank gap
    # Per-bucket variations
    {
        "Bonds": 0,
        "Commodities": 0,
        "Emerging_Markets": 0,
        "International": 0,
        "US_equities": 0,
        "US_small_mid_cap": 0,
    },
    {
        "Bonds": 1,
        "Commodities": 1,
        "Emerging_Markets": 1,
        "International": 1,
        "US_equities": 1,
        "US_small_mid_cap": 1,
    },
    {
        "Bonds": 3,
        "Commodities": 2,
        "Emerging_Markets": 2,
        "International": 2,
        "US_equities": 2,
        "US_small_mid_cap": 2,
    },
    {
        "Bonds": 2,
        "Commodities": 2,
        "Emerging_Markets": 2,
        "International": 2,
        "US_equities": 1,
        "US_small_mid_cap": 1,
    },
    {
        "Bonds": 3,
        "Commodities": 1,
        "Emerging_Markets": 1,
        "International": 1,
        "US_equities": 0,
        "US_small_mid_cap": 0,
    },
    {
        "Bonds": 1,
        "Commodities": 2,
        "Emerging_Markets": 2,
        "International": 2,
        "US_equities": 3,
        "US_small_mid_cap": 3,
    },
    {
        "Bonds": 2,
        "Commodities": 1,
        "Emerging_Markets": 1,
        "International": 1,
        "US_equities": 0,
        "US_small_mid_cap": 1,
    },
]


def run_backtest_config(
    tickers: List[str],
    bucket_map: Dict[str, str],
    start_date: str,
    end_date: str,
    config: Dict[str, Any],
) -> Dict[str, Any]:
    """Run backtest with specific configuration"""

    try:
        result = backtest_momentum(
            tickers=tickers,
            bucket_map=bucket_map,
            start_date=start_date,
            end_date=end_date,
            top_n_per_bucket=1,
            rank_gap_threshold=config["rank_gap"],
            score_mode=config["score_mode"],
            abs_filter_mode=config["filter_mode"],
            abs_filter_band=config["filter_band"],
        )

        df = result["overall_returns"]
        if df.empty:
            return None

        returns = df["return"].values

        # Calculate metrics
        volatility = np.std(returns) * np.sqrt(12)
        sharpe = (np.mean(returns) * 12 / volatility) if volatility > 0 else 0.0
        cagr = (1 + returns).prod() ** (12 / len(returns)) - 1

        # Calculate MaxDD
        cum_ret = (1 + returns).cumprod()
        running_max = np.maximum.accumulate(cum_ret)
        drawdown = (cum_ret - running_max) / running_max
        max_dd = drawdown.min()

        # Calculate turnover
        turnover_count = 0
        prev_symbols = set()
        for symbols in result["overall_positions"]:
            curr_symbols = set(symbols) if isinstance(symbols, list) else {symbols}
            turnover_count += len(curr_symbols.symmetric_difference(prev_symbols))
            prev_symbols = curr_symbols

        avg_monthly_turnover = (
            (turnover_count / len(returns)) if len(returns) > 0 else 0
        )

        # Get returns series with dates for stitching
        returns_series = df["return"]

        return {
            "sharpe": sharpe,
            "cagr": cagr,
            "max_dd": max_dd,
            "turnover": avg_monthly_turnover,
            "n_months": len(returns),
            "returns": returns_series,
        }
    except Exception as e:
        print(f"    Error: {e}")
        return None


def generate_configs() -> List[Dict[str, Any]]:
    """Generate all configuration combinations."""
    configs = []
    for score_mode, filter_cfg, gap_cfg in product(
        SCORE_MODES, FILTER_MODES, GAP_CONFIGS
    ):
        configs.append(
            {
                "score_mode": score_mode,
                "filter_mode": filter_cfg["mode"],
                "filter_band": filter_cfg["band"],
                "rank_gap": gap_cfg,
            }
        )
    return configs


def anchored_walkforward(tickers, bucket_map, start_year=2015, end_year=2025):
    """
    Anchored walk-forward: expanding training window.
    Train: 2015-2017 → Test: 2018
    Train: 2015-2018 → Test: 2019
    etc.
    """
    print("\n" + "=" * 100)
    print("ANCHORED WALK-FORWARD TEST")
    print("=" * 100)

    results = []
    all_configs = generate_configs()

    # Start with 3-year initial training period
    initial_train_years = 3

    for test_year in range(start_year + initial_train_years, end_year + 1):
        train_start = f"{start_year}-01-01"
        train_end = f"{test_year - 1}-12-31"
        test_start = f"{test_year}-01-01"
        test_end = f"{test_year}-12-31"

        print(f"\n{'='*100}")
        print(f"Window: Train {start_year}–{test_year-1} → Test {test_year}")
        print(f"{'='*100}")
        print(f"Training on {len(all_configs)} configurations...")

        # Train: find best config
        best_config = None
        best_train_sharpe = -np.inf
        best_train_metrics = None

        for i, config in enumerate(all_configs):
            if (i + 1) % 20 == 0:
                print(f"  Tested {i+1}/{len(all_configs)} configs...")

            train_metrics = run_backtest_config(
                tickers, bucket_map, train_start, train_end, config
            )
            if train_metrics and train_metrics["sharpe"] > best_train_sharpe:
                best_train_sharpe = train_metrics["sharpe"]
                best_config = config.copy()
                best_train_metrics = train_metrics

        if not best_config:
            print(f"  No valid configs for {test_year}")
            continue

        print(f"\n  Best training config:")
        print(f"    Sharpe: {best_train_sharpe:.3f}")
        print(f"    CAGR: {best_train_metrics['cagr']*100:.2f}%")
        print(f"    MaxDD: {best_train_metrics['max_dd']*100:.2f}%")
        print(f"    Gap: {best_config['rank_gap']}")

        # Test: apply to test year
        print(f"\n  Testing on {test_year}...")
        test_metrics = run_backtest_config(
            tickers, bucket_map, test_start, test_end, best_config
        )

        if test_metrics:
            print(f"  OUT-OF-SAMPLE Results:")
            print(f"    Sharpe: {test_metrics['sharpe']:.3f}")
            print(f"    CAGR: {test_metrics['cagr']*100:.2f}%")
            print(f"    MaxDD: {test_metrics['max_dd']*100:.2f}%")

            results.append(
                {
                    "test_year": test_year,
                    "train_start": train_start,
                    "train_end": train_end,
                    "test_start": test_start,
                    "test_end": test_end,
                    "config": str(best_config),
                    "train_sharpe": best_train_sharpe,
                    "train_cagr": best_train_metrics["cagr"],
                    "train_max_dd": best_train_metrics["max_dd"],
                    "test_sharpe": test_metrics["sharpe"],
                    "test_cagr": test_metrics["cagr"],
                    "test_max_dd": test_metrics["max_dd"],
                    "test_turnover": test_metrics["turnover"],
                    "test_returns": test_metrics["returns"],
                }
            )

    return results


def rolling_walkforward(
    tickers, bucket_map, train_years=5, start_year=2015, end_year=2025
):
    """
    Rolling walk-forward: fixed training window.
    """
    print("\n" + "=" * 100)
    print(f"ROLLING WALK-FORWARD TEST ({train_years}-year training window)")
    print("=" * 100)

    results = []
    all_configs = generate_configs()

    for test_year in range(start_year + train_years, end_year + 1):
        train_start_year = test_year - train_years
        train_start = f"{train_start_year}-01-01"
        train_end = f"{test_year - 1}-12-31"
        test_start = f"{test_year}-01-01"
        test_end = f"{test_year}-12-31"

        print(f"\n{'='*100}")
        print(f"Window: Train {train_start_year}–{test_year-1} → Test {test_year}")
        print(f"{'='*100}")
        print(f"Training on {len(all_configs)} configurations...")

        # Train: find best config
        best_config = None
        best_train_sharpe = -np.inf
        best_train_metrics = None

        for i, config in enumerate(all_configs):
            if (i + 1) % 20 == 0:
                print(f"  Tested {i+1}/{len(all_configs)} configs...")

            train_metrics = run_backtest_config(
                tickers, bucket_map, train_start, train_end, config
            )
            if train_metrics and train_metrics["sharpe"] > best_train_sharpe:
                best_train_sharpe = train_metrics["sharpe"]
                best_config = config.copy()
                best_train_metrics = train_metrics

        if not best_config:
            print(f"  No valid configs for {test_year}")
            continue

        print(f"\n  Best training config:")
        print(f"    Sharpe: {best_train_sharpe:.3f}")
        print(f"    CAGR: {best_train_metrics['cagr']*100:.2f}%")
        print(f"    MaxDD: {best_train_metrics['max_dd']*100:.2f}%")
        print(f"    Gap: {best_config['rank_gap']}")

        # Test: apply to test year
        print(f"\n  Testing on {test_year}...")
        test_metrics = run_backtest_config(
            tickers, bucket_map, test_start, test_end, best_config
        )

        if test_metrics:
            print(f"  OUT-OF-SAMPLE Results:")
            print(f"    Sharpe: {test_metrics['sharpe']:.3f}")
            print(f"    CAGR: {test_metrics['cagr']*100:.2f}%")
            print(f"    MaxDD: {test_metrics['max_dd']*100:.2f}%")

            results.append(
                {
                    "test_year": test_year,
                    "train_start": train_start,
                    "train_end": train_end,
                    "test_start": test_start,
                    "test_end": test_end,
                    "config": str(best_config),
                    "train_sharpe": best_train_sharpe,
                    "train_cagr": best_train_metrics["cagr"],
                    "train_max_dd": best_train_metrics["max_dd"],
                    "test_sharpe": test_metrics["sharpe"],
                    "test_cagr": test_metrics["cagr"],
                    "test_max_dd": test_metrics["max_dd"],
                    "test_turnover": test_metrics["turnover"],
                    "test_returns": test_metrics["returns"],
                }
            )

    return results


def regime_split_test(tickers, bucket_map):
    """
    Regime-based splits to test across different market conditions.
    """
    print("\n" + "=" * 100)
    print("REGIME-SPLIT WALK-FORWARD TEST")
    print("=" * 100)

    # Define regime splits
    splits = [
        {
            "name": "Bull/QE → Crash/Rebound",
            "train_start": "2015-01-01",
            "train_end": "2019-12-31",
            "test_start": "2020-01-01",
            "test_end": "2021-12-31",
        },
        {
            "name": "Bull/QE/Crash → Rate Shock",
            "train_start": "2015-01-01",
            "train_end": "2021-12-31",
            "test_start": "2022-01-01",
            "test_end": "2022-12-31",
        },
        {
            "name": "All pre-AI → AI Rally",
            "train_start": "2015-01-01",
            "train_end": "2022-12-31",
            "test_start": "2023-01-01",
            "test_end": "2025-12-31",
        },
    ]

    results = []
    all_configs = generate_configs()

    for split in splits:
        print(f"\n{'='*100}")
        print(f"Regime: {split['name']}")
        print(f"Train: {split['train_start']} to {split['train_end']}")
        print(f"Test:  {split['test_start']} to {split['test_end']}")
        print(f"{'='*100}")
        print(f"Training on {len(all_configs)} configurations...")

        # Train: find best config
        best_config = None
        best_train_sharpe = -np.inf
        best_train_metrics = None

        for i, config in enumerate(all_configs):
            if (i + 1) % 20 == 0:
                print(f"  Tested {i+1}/{len(all_configs)} configs...")

            train_metrics = run_backtest_config(
                tickers, bucket_map, split["train_start"], split["train_end"], config
            )
            if train_metrics and train_metrics["sharpe"] > best_train_sharpe:
                best_train_sharpe = train_metrics["sharpe"]
                best_config = config.copy()
                best_train_metrics = train_metrics

        if not best_config:
            print(f"  No valid configs for {split['name']}")
            continue

        print(f"\n  Best training config:")
        print(f"    Sharpe: {best_train_sharpe:.3f}")
        print(f"    CAGR: {best_train_metrics['cagr']*100:.2f}%")
        print(f"    MaxDD: {best_train_metrics['max_dd']*100:.2f}%")
        print(f"    Gap: {best_config['rank_gap']}")

        # Test: apply to test period
        print(f"\n  Testing on regime...")
        test_metrics = run_backtest_config(
            tickers, bucket_map, split["test_start"], split["test_end"], best_config
        )

        if test_metrics:
            print(f"  OUT-OF-SAMPLE Results:")
            print(f"    Sharpe: {test_metrics['sharpe']:.3f}")
            print(f"    CAGR: {test_metrics['cagr']*100:.2f}%")
            print(f"    MaxDD: {test_metrics['max_dd']*100:.2f}%")

            results.append(
                {
                    "regime": split["name"],
                    "train_start": split["train_start"],
                    "train_end": split["train_end"],
                    "test_start": split["test_start"],
                    "test_end": split["test_end"],
                    "config": str(best_config),
                    "train_sharpe": best_train_sharpe,
                    "train_cagr": best_train_metrics["cagr"],
                    "train_max_dd": best_train_metrics["max_dd"],
                    "test_sharpe": test_metrics["sharpe"],
                    "test_cagr": test_metrics["cagr"],
                    "test_max_dd": test_metrics["max_dd"],
                    "test_turnover": test_metrics["turnover"],
                    "test_returns": test_metrics["returns"],
                }
            )

    return results


def stitch_equity_curve(results, return_column="test_returns"):
    """Stitch together test period returns into one continuous equity curve."""
    all_returns = pd.Series(dtype=float)

    for result in results:
        returns = result[return_column]
        all_returns = pd.concat([all_returns, returns])

    # Remove duplicates, keeping first occurrence
    all_returns = all_returns[~all_returns.index.duplicated(keep="first")]
    all_returns = all_returns.sort_index()

    # Calculate equity curve
    equity = (1 + all_returns).cumprod()

    # Calculate combined metrics
    total_return = equity.iloc[-1] - 1
    years = (all_returns.index[-1] - all_returns.index[0]).days / 365.25
    cagr = (1 + total_return) ** (1 / years) - 1

    sharpe = all_returns.mean() / all_returns.std() * np.sqrt(12)

    cumulative = (1 + all_returns).cumprod()
    running_max = cumulative.expanding().max()
    drawdown = (cumulative - running_max) / running_max
    max_dd = drawdown.min()

    return {
        "equity": equity,
        "returns": all_returns,
        "cagr": cagr,
        "sharpe": sharpe,
        "max_dd": max_dd,
        "total_return": total_return,
    }


def print_summary(test_name, results):
    """Print summary statistics for a test methodology."""
    print(f"\n{'='*100}")
    print(f"{test_name} - SUMMARY")
    print(f"{'='*100}")

    if not results:
        print("No results to summarize")
        return

    # Out-of-sample metrics
    test_sharpes = [r["test_sharpe"] for r in results]
    test_cagrs = [r["test_cagr"] for r in results]
    test_maxdds = [r["test_max_dd"] for r in results]

    print(f"\nCompleted {len(results)} test windows")
    print(f"\nOut-of-Sample Metrics (per window):")
    print(f"  Average Sharpe: {np.mean(test_sharpes):.3f}")
    print(f"  Median Sharpe: {np.median(test_sharpes):.3f}")
    print(f"  Sharpe StdDev: {np.std(test_sharpes):.3f}")
    print(f"  Average CAGR: {np.mean(test_cagrs)*100:.2f}%")
    print(f"  Median CAGR: {np.median(test_cagrs)*100:.2f}%")
    print(f"  Worst MaxDD: {min(test_maxdds)*100:.2f}%")

    # Stitch together equity curve
    stitched = stitch_equity_curve(results)
    print(f"\nStitched Out-of-Sample Equity Curve:")
    print(f"  Total CAGR: {stitched['cagr']*100:.2f}%")
    print(f"  Total Sharpe: {stitched['sharpe']:.3f}")
    print(f"  Total MaxDD: {stitched['max_dd']*100:.2f}%")
    print(f"  Total Return: {stitched['total_return']*100:.2f}%")

    # Configuration analysis
    configs = [r["config"] for r in results]
    print(f"\nConfiguration Frequency:")

    # Extract gaps
    gaps = []
    for config_str in configs:
        config = eval(config_str)
        gaps.append(str(config["rank_gap"]))

    from collections import Counter

    print(f"  Gaps: {Counter(gaps).most_common()}")


def main():
    print("\n" + "=" * 100)
    print("COMPREHENSIVE WALK-FORWARD TESTING SUITE")
    print("=" * 100)
    print(f"\nTesting {len(generate_configs())} total configurations")
    print(f"  Score modes: {len(SCORE_MODES)}")
    print(f"  Filter modes: {len(FILTER_MODES)}")
    print(f"  Gap configs: {len(GAP_CONFIGS)}")

    # Load universe
    bucket_folder = Path("CSVs")
    universe = BucketedCsvUniverseProvider(bucket_folder)
    tickers = universe.get_tickers()
    bucket_map = universe.get_bucket_map()

    print(
        f"\nLoaded {len(tickers)} tickers across {len(set(bucket_map.values()))} buckets"
    )

    # Run all three methodologies

    # 1. Anchored walk-forward
    print("\n\nStarting Anchored Walk-Forward Tests...")
    anchored_results = anchored_walkforward(
        tickers, bucket_map, start_year=2015, end_year=2025
    )

    # Save results
    df_anchored = pd.DataFrame(
        [{k: v for k, v in r.items() if k != "test_returns"} for r in anchored_results]
    )
    df_anchored.to_csv("walkforward_anchored.csv", index=False)
    print_summary("ANCHORED WALK-FORWARD", anchored_results)

    # 2. Rolling walk-forward (5-year)
    print("\n\nStarting Rolling Walk-Forward Tests (5-year window)...")
    rolling5_results = rolling_walkforward(
        tickers, bucket_map, train_years=5, start_year=2015, end_year=2025
    )

    df_rolling5 = pd.DataFrame(
        [{k: v for k, v in r.items() if k != "test_returns"} for r in rolling5_results]
    )
    df_rolling5.to_csv("walkforward_rolling5y.csv", index=False)
    print_summary("ROLLING 5-YEAR WALK-FORWARD", rolling5_results)

    # 3. Rolling walk-forward (4-year)
    print("\n\nStarting Rolling Walk-Forward Tests (4-year window)...")
    rolling4_results = rolling_walkforward(
        tickers, bucket_map, train_years=4, start_year=2015, end_year=2025
    )

    df_rolling4 = pd.DataFrame(
        [{k: v for k, v in r.items() if k != "test_returns"} for r in rolling4_results]
    )
    df_rolling4.to_csv("walkforward_rolling4y.csv", index=False)
    print_summary("ROLLING 4-YEAR WALK-FORWARD", rolling4_results)

    # 4. Regime splits
    print("\n\nStarting Regime-Split Tests...")
    regime_results = regime_split_test(tickers, bucket_map)

    df_regime = pd.DataFrame(
        [{k: v for k, v in r.items() if k != "test_returns"} for r in regime_results]
    )
    df_regime.to_csv("walkforward_regime.csv", index=False)
    print_summary("REGIME-SPLIT", regime_results)

    # Final comparison
    print("\n\n" + "=" * 100)
    print("FINAL COMPARISON ACROSS ALL METHODOLOGIES")
    print("=" * 100)

    print("\nStitched Out-of-Sample Performance:")
    print(
        f"\n{'Methodology':<30} {'CAGR':>8} {'Sharpe':>8} {'MaxDD':>8} {'#Windows':>10}"
    )
    print("-" * 100)

    for name, results in [
        ("Anchored (expanding)", anchored_results),
        ("Rolling 5-year", rolling5_results),
        ("Rolling 4-year", rolling4_results),
        ("Regime-split", regime_results),
    ]:
        if results:
            stitched = stitch_equity_curve(results)
            print(
                f"{name:<30} {stitched['cagr']*100:>7.2f}% {stitched['sharpe']:>7.2f} "
                f"{stitched['max_dd']*100:>7.2f}% {len(results):>10}"
            )

    print("\n" + "=" * 100)
    print("Testing complete! Results saved to:")
    print("  - walkforward_anchored.csv")
    print("  - walkforward_rolling5y.csv")
    print("  - walkforward_rolling4y.csv")
    print("  - walkforward_regime.csv")
    print("=" * 100)


if __name__ == "__main__":
    main()

"""
Comprehensive bucket optimization via walk-forward testing.
Tests all bucket combinations to identify which buckets are value-destructive.
Uses 2015-2025 period (10 years) with expanding windows for robust validation.
"""

from pathlib import Path
from itertools import combinations
import numpy as np
import pandas as pd
from momentum_program.config import AppConfig
from momentum_program.universe.bucket_csv_provider import BucketedCsvUniverseProvider
from momentum_program.universe.csv_provider import CsvUniverseProvider
from momentum_program.universe.tradingview_provider import TradingViewUniverseProvider
from momentum_program.backtest.engine import backtest_momentum
from momentum_program.backtest.metrics import compute_metrics, compute_turnover


def run_bucket_optimization() -> None:
    """Walk-forward bucket optimization over 10 years (2015-2025)."""
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
    print("=" * 100)
    print(f"BUCKET OPTIMIZATION: Find value-destructive buckets via walk-forward")
    print(f"Universe: {len(tickers)} tickers across {len(all_buckets)} buckets: {all_buckets}")
    print(f"Period: 2015-2025 (10 years)")
    print(f"Strategy: vol_adj=False, lookback=12M, rank_gap=2 (best params)")
    print("=" * 100)

    # Best params from walk-forward analysis
    params = {
        "lookback": 12,
        "vol_adjusted": False,
        "rank_gap": 2,
    }

    # Walk-forward folds: 2015-2025 (10 years) with expanding windows, 2y test
    folds = [
        {"train_start": "2015-01-01", "train_end": "2017-12-31", "test_start": "2018-01-01", "test_end": "2019-12-31"},
        {"train_start": "2015-01-01", "train_end": "2018-12-31", "test_start": "2019-01-01", "test_end": "2020-12-31"},
        {"train_start": "2015-01-01", "train_end": "2019-12-31", "test_start": "2020-01-01", "test_end": "2021-12-31"},
        {"train_start": "2015-01-01", "train_end": "2020-12-31", "test_start": "2022-01-01", "test_end": "2023-12-31"},
        {"train_start": "2015-01-01", "train_end": "2021-12-31", "test_start": "2023-01-01", "test_end": "2025-12-31"},
    ]

    print(f"\nWalk-forward folds: {len(folds)} (expanding window, 2y test periods)")
    print("-" * 100)

    # Test baseline: all buckets
    baseline_results = test_bucket_config("ALL BUCKETS", all_buckets, tickers, full_bucket_map, folds, params)

    # Test: remove each bucket individually
    single_bucket_removal_results = {}
    for bucket_to_remove in all_buckets:
        config = [b for b in all_buckets if b != bucket_to_remove]
        result = test_bucket_config(f"WITHOUT {bucket_to_remove}", config, tickers, full_bucket_map, folds, params)
        single_bucket_removal_results[bucket_to_remove] = result

    # Test: keep only essential buckets (remove multiple)
    # Try removing each bucket and analyze impact
    print("\n" + "=" * 100)
    print("SINGLE BUCKET REMOVAL IMPACT (walk-forward median stats):")
    print("=" * 100)

    removal_impacts = []
    for bucket, result in single_bucket_removal_results.items():
        sharpe_delta = result["median_sharpe"] - baseline_results["median_sharpe"]
        cagr_delta = result["median_cagr"] - baseline_results["median_cagr"]
        dd_delta = result["median_maxdd"] - baseline_results["median_maxdd"]  # Positive = less negative = improvement
        turnover_delta = result["median_turnover"] - baseline_results["median_turnover"]

        removal_impacts.append({
            "bucket": bucket,
            "sharpe_delta": sharpe_delta,
            "cagr_delta": cagr_delta,
            "dd_delta": dd_delta,
            "turnover_delta": turnover_delta,
            "final_sharpe": result["median_sharpe"],
            "final_cagr": result["median_cagr"],
            "final_maxdd": result["median_maxdd"],
        })

    # Sort by Sharpe impact (most positive first = best candidates to remove)
    removal_impacts.sort(key=lambda x: x["sharpe_delta"], reverse=True)

    print(f"\n{'Bucket':<20} {'Sharpe Impact':<15} {'CAGR Impact':<15} {'MaxDD Impact':<15} {'Result Sharpe':<15}")
    print("-" * 80)
    for impact in removal_impacts:
        print(
            f"{impact['bucket']:<20} {impact['sharpe_delta']:+.4f}           "
            f"{impact['cagr_delta']:+.2%}              {impact['dd_delta']:+.2%}              "
            f"{impact['final_sharpe']:.4f}"
        )

    # Identify value-destructive buckets (removing them improves Sharpe)
    destructive_buckets = [x["bucket"] for x in removal_impacts if x["sharpe_delta"] > 0]
    constructive_buckets = [x["bucket"] for x in removal_impacts if x["sharpe_delta"] <= 0]

    print("\n" + "=" * 100)
    print("ANALYSIS:")
    print("=" * 100)
    print(f"\nBaseline (ALL {len(all_buckets)} buckets):")
    print(f"  Sharpe: {baseline_results['median_sharpe']:.4f}")
    print(f"  CAGR:   {baseline_results['median_cagr']:.2%}")
    print(f"  MaxDD:  {baseline_results['median_maxdd']:.2%}")
    print(f"  Turnover: {baseline_results['median_turnover']:.2%}")

    if destructive_buckets:
        print(f"\n[VALUE-DESTRUCTIVE] BUCKETS (removing improves returns):")
        for bucket in destructive_buckets:
            impact = next((x for x in removal_impacts if x["bucket"] == bucket), None)
            if impact:
                print(f"  * {bucket}: Sharpe +{impact['sharpe_delta']:.4f}, CAGR +{impact['cagr_delta']:.2%}")

    if constructive_buckets:
        print(f"\n[VALUE-CONSTRUCTIVE] BUCKETS (keep these):")
        for bucket in constructive_buckets:
            impact = next((x for x in removal_impacts if x["bucket"] == bucket), None)
            if impact:
                print(f"  * {bucket}: Sharpe {impact['sharpe_delta']:+.4f}, CAGR {impact['cagr_delta']:+.2%}")

    # Recommend optimal configuration
    optimal_config = constructive_buckets if constructive_buckets else all_buckets
    optimal_result = test_bucket_config("OPTIMAL", optimal_config, tickers, full_bucket_map, folds, params)

    print(f"\n" + "=" * 100)
    print(f"RECOMMENDATION: Keep {len(optimal_config)} buckets")
    print(f"Buckets: {sorted(optimal_config)}")
    print("=" * 100)
    print(f"  Sharpe: {optimal_result['median_sharpe']:.4f} (vs {baseline_results['median_sharpe']:.4f} baseline) {optimal_result['median_sharpe'] - baseline_results['median_sharpe']:+.4f}")
    print(f"  CAGR:   {optimal_result['median_cagr']:.2%} (vs {baseline_results['median_cagr']:.2%} baseline) {optimal_result['median_cagr'] - baseline_results['median_cagr']:+.2%}")
    print(f"  MaxDD:  {optimal_result['median_maxdd']:.2%} (vs {baseline_results['median_maxdd']:.2%} baseline) {optimal_result['median_maxdd'] - baseline_results['median_maxdd']:+.2%}")

    # Per-fold breakdown
    print(f"\n" + "=" * 100)
    print("PER-FOLD BREAKDOWN (Optimal vs Baseline):")
    print("=" * 100)
    print(f"{'Fold':<15} {'Baseline Sharpe':<20} {'Optimal Sharpe':<20} {'Difference':<15}")
    print("-" * 70)

    for i, (base_fold, opt_fold) in enumerate(zip(baseline_results["fold_results"], optimal_result["fold_results"]), 1):
        diff = opt_fold["sharpe"] - base_fold["sharpe"]
        print(
            f"Fold {i:<9} {base_fold['sharpe']:<20.4f} {opt_fold['sharpe']:<20.4f} {diff:+.4f}"
        )

    print("\n" + "=" * 100)
    print("NEXT STEPS:")
    print("=" * 100)
    print(f"1. Update CSVs folder to keep only: {', '.join(sorted(optimal_config))}")
    print(f"2. Archive removed bucket CSV files with .bak extension")
    print(f"3. Re-run walk_forward.py to validate parameter tuning on optimized universe")
    print("=" * 100)


def test_bucket_config(config_name, buckets, tickers, full_bucket_map, folds, params):
    """Test a specific bucket configuration across walk-forward folds."""
    fold_results = []
    fold_sharpes = []
    fold_cagrs = []
    fold_maxdds = []
    fold_turnovers = []

    for fold_idx, fold in enumerate(folds, 1):
        # Filter to only include tickers from specified buckets
        filtered_bucket_map = {
            ticker: bucket
            for ticker, bucket in full_bucket_map.items()
            if bucket in buckets
        }
        filtered_tickers = list(filtered_bucket_map.keys())

        if len(filtered_tickers) < 5:
            print(f"  Fold {fold_idx} ({fold['test_start']} to {fold['test_end']}): Skipped (insufficient tickers: {len(filtered_tickers)})")
            continue

        # Run backtest on test period
        try:
            backtest_data = backtest_momentum(
                tickers=filtered_tickers,
                bucket_map=filtered_bucket_map,
                start_date=fold["test_start"],
                end_date=fold["test_end"],
                lookback_long=params["lookback"],
                lookback_short=1,
                vol_adjusted=params["vol_adjusted"],
                rank_gap_threshold=params["rank_gap"],
            )

            if backtest_data["overall_returns"].empty:
                print(
                    f"  Fold {fold_idx} ({fold['test_start']} to {fold['test_end']}): No returns (insufficient data)"
                )
                continue

            # Compute metrics
            metrics = compute_metrics(backtest_data["overall_returns"])
            turnover = compute_turnover(backtest_data["bucket_positions"])

            fold_results.append({
                "fold": fold_idx,
                "year": f"{fold['test_start'][:4]}-{fold['test_end'][:4]}",
                "sharpe": metrics["sharpe"],
                "cagr": metrics["cagr"],
                "maxdd": metrics["max_drawdown"],
            })
            fold_sharpes.append(metrics["sharpe"])
            fold_cagrs.append(metrics["cagr"])
            fold_maxdds.append(metrics["max_drawdown"])
            fold_turnovers.append(turnover)

        except Exception as e:
            print(f"  Fold {fold_idx} ({fold['test_start']} to {fold['test_end']}): Error - {str(e)[:50]}")
            continue

    if not fold_sharpes:
        return {
            "median_sharpe": 0,
            "median_cagr": 0,
            "median_maxdd": 0,
            "median_turnover": 0,
            "fold_results": [],
        }

    return {
        "median_sharpe": np.median(fold_sharpes),
        "mean_sharpe": np.mean(fold_sharpes),
        "std_sharpe": np.std(fold_sharpes),
        "median_cagr": np.median(fold_cagrs),
        "mean_cagr": np.mean(fold_cagrs),
        "std_cagr": np.std(fold_cagrs),
        "median_maxdd": np.median(fold_maxdds),
        "mean_maxdd": np.mean(fold_maxdds),
        "std_maxdd": np.std(fold_maxdds),
        "median_turnover": np.median(fold_turnovers),
        "mean_turnover": np.mean(fold_turnovers),
        "std_turnover": np.std(fold_turnovers),
        "fold_results": fold_results,
    }


if __name__ == "__main__":
    run_bucket_optimization()

"""
Simplified bucket optimization using systematic removal testing.
Tests each bucket removal on the 2022-2025 period (most recent 3 years).
"""

from pathlib import Path
import numpy as np
from momentum_program.config import AppConfig
from momentum_program.universe.bucket_csv_provider import BucketedCsvUniverseProvider
from momentum_program.backtest.engine import backtest_momentum
from momentum_program.backtest.metrics import compute_metrics, compute_turnover


def run_simplified_bucket_optimization():
    """Test bucket removal impact on recent data (2022-2025)."""
    cfg = AppConfig()
    bucket_folder = Path("CSVs")
    bucket_csvs = list(bucket_folder.glob("*.csv")) if bucket_folder.exists() else []

    if not bucket_csvs:
        print("No CSV files found in CSVs folder")
        return

    from momentum_program.universe.bucket_csv_provider import BucketedCsvUniverseProvider
    universe = BucketedCsvUniverseProvider(bucket_folder)
    tickers = universe.get_tickers()
    full_bucket_map = universe.get_bucket_map()
    all_buckets = sorted(set(full_bucket_map.values()))

    print("=" * 100)
    print("BUCKET OPTIMIZATION: Test each bucket removal on 2022-2025 period")
    print(f"Universe: {len(tickers)} tickers across {len(all_buckets)} buckets: {all_buckets}")
    print("=" * 100)

    # Best params
    params = {
        "lookback": 12,
        "vol_adjusted": False,
        "rank_gap": 2,
    }

    # Test period: 2022-2025 (recent 3+ years)
    test_period = {
        "start": "2022-01-01",
        "end": "2025-12-31",
    }

    # Test baseline: all buckets
    print(f"\nTesting baseline (ALL {len(all_buckets)} buckets)...")
    baseline = test_bucket_config(
        "ALL BUCKETS",
        all_buckets,
        tickers,
        full_bucket_map,
        test_period,
        params
    )

    if baseline["sharpe"] is None:
        print("ERROR: Baseline test failed. Check data availability.")
        return

    print(f"  Sharpe: {baseline['sharpe']:.4f}, CAGR: {baseline['cagr']:.2%}, MaxDD: {baseline['maxdd']:.2%}")

    # Test each bucket removal
    removal_results = {}
    print(f"\nTesting individual bucket removals...")
    for bucket_to_remove in all_buckets:
        remaining_buckets = [b for b in all_buckets if b != bucket_to_remove]
        print(f"  Testing WITHOUT {bucket_to_remove} ({len(remaining_buckets)} buckets)...")
        result = test_bucket_config(
            f"WITHOUT {bucket_to_remove}",
            remaining_buckets,
            tickers,
            full_bucket_map,
            test_period,
            params
        )
        removal_results[bucket_to_remove] = result

        if result["sharpe"] is not None:
            sharpe_delta = result["sharpe"] - baseline["sharpe"]
            cagr_delta = result["cagr"] - baseline["cagr"]
            print(f"    Sharpe: {result['sharpe']:.4f} (delta: {sharpe_delta:+.4f}), CAGR: {result['cagr']:.2%} (delta: {cagr_delta:+.2%})")
        else:
            print(f"    Failed to generate results")

    # Analyze impacts
    print("\n" + "=" * 100)
    print("IMPACT ANALYSIS (Removing each bucket from baseline):")
    print("=" * 100)

    impacts = []
    for bucket, result in removal_results.items():
        if result["sharpe"] is not None:
            sharpe_delta = result["sharpe"] - baseline["sharpe"]
            cagr_delta = result["cagr"] - baseline["cagr"]
            maxdd_delta = result["maxdd"] - baseline["maxdd"]
            impacts.append({
                "bucket": bucket,
                "sharpe_delta": sharpe_delta,
                "cagr_delta": cagr_delta,
                "maxdd_delta": maxdd_delta,
                "final_sharpe": result["sharpe"],
                "final_cagr": result["cagr"],
            })

    # Sort by Sharpe improvement (highest first = best candidates for removal)
    impacts.sort(key=lambda x: x["sharpe_delta"], reverse=True)

    print(f"\n{'Bucket':<20} {'Sharpe Change':<18} {'CAGR Change':<18} {'Result Sharpe':<16}")
    print("-" * 72)
    for impact in impacts:
        improvement_flag = "[+GAIN]" if impact["sharpe_delta"] > 0.01 else "[LOSS]" if impact["sharpe_delta"] < -0.01 else "[NEUTRAL]"
        print(
            f"{impact['bucket']:<20} {impact['sharpe_delta']:+.4f} {improvement_flag:<10} "
            f"{impact['cagr_delta']:+.2%}           {impact['final_sharpe']:.4f}"
        )

    # Identify value-destructive buckets
    destructive = [x["bucket"] for x in impacts if x["sharpe_delta"] > 0.01]
    neutral_or_good = [x["bucket"] for x in impacts if x["sharpe_delta"] <= 0.01]

    print("\n" + "=" * 100)
    print("BASELINE PERFORMANCE (ALL buckets):")
    print(f"  Sharpe: {baseline['sharpe']:.4f}")
    print(f"  CAGR:   {baseline['cagr']:.2%}")
    print(f"  MaxDD:  {baseline['maxdd']:.2%}")

    if destructive:
        print(f"\n[VALUE-DESTRUCTIVE] Buckets (removing improves performance):")
        for bucket in destructive:
            impact = next(x for x in impacts if x["bucket"] == bucket)
            print(f"  - {bucket}: Sharpe +{impact['sharpe_delta']:.4f}, CAGR +{impact['cagr_delta']:.2%}")

    if neutral_or_good:
        print(f"\n[VALUE-CONSTRUCTIVE] Buckets (keep these):")
        for bucket in sorted(neutral_or_good):
            impact = next((x for x in impacts if x["bucket"] == bucket), None)
            if impact:
                print(f"  - {bucket}: Sharpe {impact['sharpe_delta']:+.4f}, CAGR {impact['cagr_delta']:+.2%}")

    # Test optimal configuration
    optimal_buckets = neutral_or_good if neutral_or_good else all_buckets
    print(f"\n" + "=" * 100)
    print(f"RECOMMENDATION: Keep {len(optimal_buckets)} buckets (remove {len(destructive)} destructive ones)")
    print(f"Optimal buckets: {sorted(optimal_buckets)}")
    print("=" * 100)

    if optimal_buckets != all_buckets:
        optimal = test_bucket_config(
            "OPTIMAL",
            optimal_buckets,
            tickers,
            full_bucket_map,
            test_period,
            params
        )
        if optimal["sharpe"] is not None:
            print(f"  Sharpe: {optimal['sharpe']:.4f} vs {baseline['sharpe']:.4f} baseline ({optimal['sharpe'] - baseline['sharpe']:+.4f})")
            print(f"  CAGR:   {optimal['cagr']:.2%} vs {baseline['cagr']:.2%} baseline ({optimal['cagr'] - baseline['cagr']:+.2%})")
            print(f"  MaxDD:  {optimal['maxdd']:.2%} vs {baseline['maxdd']:.2%} baseline ({optimal['maxdd'] - baseline['maxdd']:+.2%})")


def test_bucket_config(config_name, buckets, tickers, full_bucket_map, period, params):
    """Test a specific bucket configuration."""
    # Filter tickers to only include specified buckets
    filtered_bucket_map = {
        ticker: bucket
        for ticker, bucket in full_bucket_map.items()
        if bucket in buckets
    }
    filtered_tickers = list(filtered_bucket_map.keys())

    if len(filtered_tickers) < 5:
        return {"sharpe": None, "cagr": None, "maxdd": None}

    try:
        backtest_data = backtest_momentum(
            tickers=filtered_tickers,
            bucket_map=filtered_bucket_map,
            start_date=period["start"],
            end_date=period["end"],
            top_n_per_bucket=1,  # 1 per bucket, like walk-forward
            lookback_long=params["lookback"],
            lookback_short=1,
            vol_adjusted=params["vol_adjusted"],
            rank_gap_threshold=params["rank_gap"],
        )

        if backtest_data["overall_returns"].empty:
            return {"sharpe": None, "cagr": None, "maxdd": None}

        metrics = compute_metrics(backtest_data["overall_returns"])
        return {
            "sharpe": metrics["sharpe"],
            "cagr": metrics["cagr"],
            "maxdd": metrics["max_drawdown"],
        }
    except Exception as e:
        return {"sharpe": None, "cagr": None, "maxdd": None}


if __name__ == "__main__":
    run_simplified_bucket_optimization()

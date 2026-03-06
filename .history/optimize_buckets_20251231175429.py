"""Optimize bucket composition: test performance with different bucket combinations."""
from pathlib import Path
import numpy as np
from itertools import combinations
from momentum_program.config import AppConfig
from momentum_program.universe.bucket_csv_provider import BucketedCsvUniverseProvider
from momentum_program.universe.csv_provider import CsvUniverseProvider
from momentum_program.universe.tradingview_provider import TradingViewUniverseProvider
from momentum_program.backtest.engine import backtest_momentum
from momentum_program.backtest.metrics import compute_metrics, compute_turnover


def test_bucket_combinations() -> None:
    """Test removing individual buckets and combinations to find optimal basket."""
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

    # Get all unique buckets
    all_buckets = sorted(set(full_bucket_map.values()))
    print(f"Found {len(all_buckets)} buckets: {all_buckets}")
    print(f"Total tickers: {len(tickers)}\n")

    # Count tickers per bucket
    bucket_counts = {}
    for bucket in all_buckets:
        bucket_counts[bucket] = sum(1 for b in full_bucket_map.values() if b == bucket)
    
    print("Tickers per bucket:")
    for bucket, count in sorted(bucket_counts.items()):
        print(f"  {bucket}: {count} tickers")
    print()

    # Best params from walk-forward analysis
    best_params = {
        "lookback": 12,
        "vol_adjusted": False,
        "rank_gap": 2,
        "threshold": None,
    }

    # Test period: use recent data (2022-2024) for faster testing
    test_start = "2022-01-01"
    test_end = "2024-12-31"

    print(f"Testing bucket combinations with params: lookback={best_params['lookback']}M, "
          f"vol_adj={best_params['vol_adjusted']}, rank_gap={best_params['rank_gap']}")
    print(f"Test period: {test_start} to {test_end}\n")
    print("=" * 100)

    results = []

    # 1. Baseline: All buckets
    print("\n1. BASELINE: All buckets")
    baseline_result = run_backtest_with_buckets(
        tickers, full_bucket_map, all_buckets, 
        test_start, test_end, best_params, cfg
    )
    results.append({
        "config": "All buckets",
        "buckets": all_buckets,
        "n_buckets": len(all_buckets),
        **baseline_result
    })
    print(f"   CAGR: {baseline_result['cagr']:.2%}, Sharpe: {baseline_result['sharpe']:.2f}, "
          f"MaxDD: {baseline_result['max_drawdown']:.2%}, Turnover: {baseline_result['turnover']:.2%}")

    # 2. Remove each bucket individually
    print("\n2. REMOVE EACH BUCKET (one at a time):")
    for bucket_to_remove in all_buckets:
        remaining_buckets = [b for b in all_buckets if b != bucket_to_remove]
        print(f"\n   Without {bucket_to_remove}:")
        
        result = run_backtest_with_buckets(
            tickers, full_bucket_map, remaining_buckets,
            test_start, test_end, best_params, cfg
        )
        
        results.append({
            "config": f"Without {bucket_to_remove}",
            "buckets": remaining_buckets,
            "n_buckets": len(remaining_buckets),
            **result
        })
        
        # Compare to baseline
        cagr_diff = result['cagr'] - baseline_result['cagr']
        sharpe_diff = result['sharpe'] - baseline_result['sharpe']
        turnover_diff = result['turnover'] - baseline_result['turnover']
        
        print(f"      CAGR: {result['cagr']:.2%} ({cagr_diff:+.2%}), "
              f"Sharpe: {result['sharpe']:.2f} ({sharpe_diff:+.2f}), "
              f"MaxDD: {result['max_drawdown']:.2%}, "
              f"Turnover: {result['turnover']:.2%} ({turnover_diff:+.2%})")

    # 3. Test keeping only top-performing asset classes (if we have enough buckets)
    if len(all_buckets) >= 5:
        print("\n3. CORE BUCKET COMBINATIONS (common strategies):")
        
        # US + Intl + Bonds only
        core_combos = [
            {
                "name": "Equities Only (US + Intl)",
                "buckets": [b for b in all_buckets if any(x in b for x in ["US", "International", "Developed", "Emerging"])]
            },
            {
                "name": "Equities + Bonds",
                "buckets": [b for b in all_buckets if any(x in b for x in ["US", "International", "Developed", "Emerging", "Bonds"])]
            },
            {
                "name": "No Commodities",
                "buckets": [b for b in all_buckets if "Commodities" not in b]
            },
        ]
        
        for combo in core_combos:
            combo_buckets = [b for b in combo["buckets"] if b in all_buckets]
            if not combo_buckets or combo_buckets == all_buckets:
                continue
                
            print(f"\n   {combo['name']}: {combo_buckets}")
            result = run_backtest_with_buckets(
                tickers, full_bucket_map, combo_buckets,
                test_start, test_end, best_params, cfg
            )
            
            results.append({
                "config": combo['name'],
                "buckets": combo_buckets,
                "n_buckets": len(combo_buckets),
                **result
            })
            
            cagr_diff = result['cagr'] - baseline_result['cagr']
            sharpe_diff = result['sharpe'] - baseline_result['sharpe']
            
            print(f"      CAGR: {result['cagr']:.2%} ({cagr_diff:+.2%}), "
                  f"Sharpe: {result['sharpe']:.2f} ({sharpe_diff:+.2f}), "
                  f"MaxDD: {result['max_drawdown']:.2%}, "
                  f"Turnover: {result['turnover']:.2%}")

    # Summary: Top configurations by Sharpe
    print("\n" + "=" * 100)
    print("SUMMARY: Top 10 bucket configurations by Sharpe ratio")
    print("=" * 100)
    
    results.sort(key=lambda x: x['sharpe'], reverse=True)
    
    for i, res in enumerate(results[:10], 1):
        print(f"{i:2d}. {res['config']:40s} | "
              f"Sharpe: {res['sharpe']:.2f}, CAGR: {res['cagr']:.2%}, "
              f"MaxDD: {res['max_drawdown']:.2%}, Turnover: {res['turnover']:.2%}")
    
    # Find biggest improvements
    print("\n" + "=" * 100)
    print("BIGGEST IMPROVEMENTS over baseline:")
    print("=" * 100)
    
    improvements = [r for r in results if r['sharpe'] > baseline_result['sharpe']]
    improvements.sort(key=lambda x: x['sharpe'] - baseline_result['sharpe'], reverse=True)
    
    if improvements:
        for i, res in enumerate(improvements[:5], 1):
            sharpe_gain = res['sharpe'] - baseline_result['sharpe']
            cagr_gain = res['cagr'] - baseline_result['cagr']
            print(f"{i}. {res['config']:40s} | "
                  f"Sharpe: +{sharpe_gain:.2f}, CAGR: {cagr_gain:+.2%}")
    else:
        print("No configurations improved over baseline.")


def run_backtest_with_buckets(
    tickers: list[str],
    full_bucket_map: dict[str, str],
    buckets_to_include: list[str],
    start_date: str,
    end_date: str,
    params: dict,
    cfg: AppConfig,
) -> dict:
    """Run backtest with only specified buckets."""
    # Filter to only include tickers from specified buckets
    filtered_bucket_map = {
        ticker: bucket
        for ticker, bucket in full_bucket_map.items()
        if bucket in buckets_to_include
    }
    
    filtered_tickers = list(filtered_bucket_map.keys())
    
    if not filtered_tickers:
        return {
            "cagr": 0.0,
            "sharpe": 0.0,
            "max_drawdown": 0.0,
            "turnover": 0.0,
            "n_tickers": 0,
        }
    
    # Run backtest
    backtest_data = backtest_momentum(
        tickers=filtered_tickers,
        bucket_map=filtered_bucket_map,
        start_date=start_date,
        end_date=end_date,
        top_n_per_bucket=cfg.strategy.top_n_per_bucket or cfg.strategy.top_n,
        lookback_long=params["lookback"],
        lookback_short=1,
        vol_adjusted=params["vol_adjusted"],
        vol_lookback=6,
        market_filter=params["threshold"] is not None,
        market_ticker="SPY",
        defensive_bucket="Bonds" if "Bonds" in buckets_to_include else None,
        market_threshold=params["threshold"] or 0.0,
        rank_gap_threshold=params["rank_gap"],
    )
    
    if backtest_data["overall_returns"].empty:
        return {
            "cagr": 0.0,
            "sharpe": 0.0,
            "max_drawdown": 0.0,
            "turnover": 0.0,
            "n_tickers": 0,
        }
    
    metrics = compute_metrics(backtest_data["overall_returns"]["return"])
    turnover = compute_turnover(backtest_data["overall_positions"])
    
    return {
        "cagr": metrics["cagr"],
        "sharpe": metrics["sharpe"],
        "max_drawdown": metrics["max_drawdown"],
        "turnover": turnover,
        "n_tickers": len(filtered_tickers),
    }


if __name__ == "__main__":
    test_bucket_combinations()

"""
Walk-forward testing to find optimal strategy configuration

Methodology:
1. Split data into rolling windows (e.g., 3 years train, 1 year test)
2. Optimize on training period
3. Test on out-of-sample period
4. Roll forward and repeat
5. Aggregate out-of-sample results
"""

from pathlib import Path
from datetime import datetime, timedelta
from itertools import product
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Tuple

from momentum_program.universe.bucket_csv_provider import BucketedCsvUniverseProvider
from momentum_program.backtest.engine import backtest_momentum
from momentum_program.analytics.constants import (
    SCORE_MODE_RW_3_6_9_12,
    SCORE_MODE_12M_MINUS_1M,
    SCORE_MODE_BLEND_6_12,
)


def run_backtest_config(
    tickers: List[str],
    bucket_map: Dict[str, str],
    start_date: str,
    end_date: str,
    config: Dict[str, Any],
) -> Dict[str, Any]:
    """Run backtest with specific configuration"""
    
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
    sharpe = sharpe_ratio(returns)
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
    
    avg_monthly_turnover = (turnover_count / len(returns)) if len(returns) > 0 else 0
    
    return {
        "sharpe": sharpe,
        "cagr": cagr,
        "max_dd": max_dd,
        "turnover": avg_monthly_turnover,
        "n_months": len(returns),
    }


def generate_configs() -> List[Dict[str, Any]]:
    """Generate all configuration combinations to test"""
    
    # Scoring modes to test
    score_modes = [
        SCORE_MODE_RW_3_6_9_12,
        SCORE_MODE_12M_MINUS_1M,
        SCORE_MODE_BLEND_6_12,
    ]
    
    # Filters to test
    filters = [
        {"mode": "none", "band": 0.0},
        {"mode": "ret_and", "band": 0.01},  # 1%
        {"mode": "ret_6m", "band": 0.01},
        {"mode": "ret_12m", "band": 0.01},
    ]
    
    # Rank gaps to test - both uniform and per-bucket
    gap_configs = [
        0,  # Always switch
        1,  # Standard momentum
        2,  # More sticky
        3,  # Very sticky
        # Custom per-bucket examples (can add more)
        {"Bonds": 2, "Commodities": 1, "Emerging_Markets": 1, "International": 1, "US_equities": 0, "US_small_mid_cap": 1},
        {"Bonds": 3, "Commodities": 1, "Emerging_Markets": 1, "International": 1, "US_equities": 1, "US_small_mid_cap": 1},
    ]
    
    configs = []
    for score_mode, filter_cfg, gap_cfg in product(score_modes, filters, gap_configs):
        configs.append({
            "score_mode": score_mode,
            "filter_mode": filter_cfg["mode"],
            "filter_band": filter_cfg["band"],
            "rank_gap": gap_cfg,
        })
    
    return configs


def walk_forward_test(
    train_years: int = 3,
    test_years: int = 1,
    start_year: int = 2015,
    end_year: int = 2025,
) -> pd.DataFrame:
    """
    Perform walk-forward optimization
    
    Args:
        train_years: Years to use for training/optimization
        test_years: Years to use for out-of-sample testing
        start_year: First year of available data
        end_year: Last year of available data
    """
    
    # Load universe
    bucket_folder = Path("CSVs")
    universe = BucketedCsvUniverseProvider(bucket_folder)
    tickers = universe.get_tickers()
    bucket_map = universe.get_bucket_map()
    
    # Generate all configs to test
    configs = generate_configs()
    print(f"Generated {len(configs)} configurations to test\n")
    
    # Walk-forward windows
    windows = []
    current_year = start_year
    
    while current_year + train_years + test_years <= end_year:
        train_start = f"{current_year}-01-01"
        train_end = f"{current_year + train_years - 1}-12-31"
        test_start = f"{current_year + train_years}-01-01"
        test_end = f"{current_year + train_years + test_years - 1}-12-31"
        
        windows.append({
            "train_start": train_start,
            "train_end": train_end,
            "test_start": test_start,
            "test_end": test_end,
        })
        
        current_year += test_years
    
    print(f"Created {len(windows)} walk-forward windows:")
    for i, w in enumerate(windows):
        print(f"  Window {i+1}: Train {w['train_start']} to {w['train_end']}, "
              f"Test {w['test_start']} to {w['test_end']}")
    print()
    
    # Results storage
    all_results = []
    
    # Process each window
    for window_idx, window in enumerate(windows):
        print(f"\n{'='*100}")
        print(f"WINDOW {window_idx + 1}: Train on {window['train_start']} to {window['train_end']}")
        print(f"{'='*100}")
        
        # Optimization phase: test all configs on training data
        train_results = []
        
        for config_idx, config in enumerate(configs):
            if config_idx % 10 == 0:
                print(f"  Training config {config_idx + 1}/{len(configs)}...", end="\r")
            
            try:
                metrics = run_backtest_config(
                    tickers, bucket_map,
                    window["train_start"], window["train_end"],
                    config
                )
                
                if metrics:
                    train_results.append({
                        "config": config,
                        "sharpe": metrics["sharpe"],
                        "cagr": metrics["cagr"],
                        "max_dd": metrics["max_dd"],
                        "turnover": metrics["turnover"],
                    })
            except Exception as e:
                print(f"\n  Error with config {config_idx}: {e}")
                continue
        
        print(f"  Completed training on {len(train_results)} configs                    ")
        
        if not train_results:
            print("  No valid results in training period!")
            continue
        
        # Find best config by Sharpe ratio
        best_config = max(train_results, key=lambda x: x["sharpe"])
        print(f"\n  Best training config:")
        print(f"    Sharpe: {best_config['sharpe']:.3f}")
        print(f"    CAGR: {best_config['cagr']:.2%}")
        print(f"    MaxDD: {best_config['max_dd']:.2%}")
        print(f"    Turnover: {best_config['turnover']:.2f}")
        print(f"    Score: {best_config['config']['score_mode']}")
        print(f"    Filter: {best_config['config']['filter_mode']}@{best_config['config']['filter_band']:.1%}")
        print(f"    Gap: {best_config['config']['rank_gap']}")
        
        # Test phase: evaluate best config on out-of-sample data
        print(f"\n  Testing on {window['test_start']} to {window['test_end']}...")
        
        try:
            test_metrics = run_backtest_config(
                tickers, bucket_map,
                window["test_start"], window["test_end"],
                best_config["config"]
            )
            
            if test_metrics:
                print(f"  OUT-OF-SAMPLE Results:")
                print(f"    Sharpe: {test_metrics['sharpe']:.3f}")
                print(f"    CAGR: {test_metrics['cagr']:.2%}")
                print(f"    MaxDD: {test_metrics['max_dd']:.2%}")
                print(f"    Turnover: {test_metrics['turnover']:.2f}")
                
                all_results.append({
                    "window": window_idx + 1,
                    "test_start": window["test_start"],
                    "test_end": window["test_end"],
                    "config": best_config["config"],
                    "train_sharpe": best_config["sharpe"],
                    "test_sharpe": test_metrics["sharpe"],
                    "train_cagr": best_config["cagr"],
                    "test_cagr": test_metrics["cagr"],
                    "train_max_dd": best_config["max_dd"],
                    "test_max_dd": test_metrics["max_dd"],
                    "train_turnover": best_config["turnover"],
                    "test_turnover": test_metrics["turnover"],
                })
            else:
                print("  No valid test results!")
        except Exception as e:
            print(f"  Error testing config: {e}")
    
    # Convert to DataFrame
    df_results = pd.DataFrame(all_results)
    
    # Print summary
    print(f"\n{'='*100}")
    print("WALK-FORWARD TEST SUMMARY")
    print(f"{'='*100}\n")
    
    if len(df_results) > 0:
        print(f"Completed {len(df_results)} test windows\n")
        
        print("Aggregate Out-of-Sample Performance:")
        print(f"  Average Sharpe: {df_results['test_sharpe'].mean():.3f}")
        print(f"  Median Sharpe: {df_results['test_sharpe'].median():.3f}")
        print(f"  Average CAGR: {df_results['test_cagr'].mean():.2%}")
        print(f"  Median CAGR: {df_results['test_cagr'].median():.2%}")
        print(f"  Worst MaxDD: {df_results['test_max_dd'].min():.2%}")
        print(f"  Average Turnover: {df_results['test_turnover'].mean():.2f}")
        
        print("\nDegradation (Train vs Test):")
        sharpe_degradation = df_results['train_sharpe'].mean() - df_results['test_sharpe'].mean()
        print(f"  Sharpe degradation: {sharpe_degradation:.3f}")
        print(f"  CAGR degradation: {(df_results['train_cagr'].mean() - df_results['test_cagr'].mean()):.2%}")
        
        # Most common winning configurations
        print("\nConfiguration Frequency (what won most often):")
        score_modes = [str(c['score_mode']) for c in df_results['config']]
        filter_modes = [c['filter_mode'] for c in df_results['config']]
        
        from collections import Counter
        print(f"  Score modes: {Counter(score_modes).most_common(3)}")
        print(f"  Filter modes: {Counter(filter_modes).most_common(3)}")
        
    else:
        print("No valid results!")
    
    return df_results


if __name__ == "__main__":
    # Run walk-forward test
    results_df = walk_forward_test(
        train_years=3,
        test_years=1,
        start_year=2015,
        end_year=2025,
    )
    
    # Save results
    output_file = "walk_forward_results.csv"
    results_df.to_csv(output_file, index=False)
    print(f"\nResults saved to {output_file}")

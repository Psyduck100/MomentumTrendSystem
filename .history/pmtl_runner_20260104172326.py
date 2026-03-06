"""PMTL Refactored Backtest Runner

Simple, configurable runner using composition of config, strategies, and engine.
Change pmtl_config.ACTIVE_CONFIG to switch between test scenarios.
"""

import sys
from pathlib import Path

from pmtl_config import ACTIVE_CONFIG, FallbackType
from pmtl_backtest_engine import PMTLBacktestEngine
from pmtl_fallback_strategies import get_fallback_strategy


def export_results(results_df, annual_returns_all, monthly_returns_all, config) -> None:
    """Export results to CSV files (summary, annual, monthly)."""
    # Sort by primary metric
    metric_col = config.primary_metric.value
    results_df = results_df.sort_values(metric_col, ascending=False)
    
    # Export summary results
    summary_filename = f"{config.output_prefix}_results.csv"
    results_df.to_csv(summary_filename, index=False)
    print(f"\n✓ Exported summary to {summary_filename}")
    
    # Export annual returns (compounded)
    annual_filename = f"{config.output_prefix}_annual_returns.csv"
    annual_df = pd.DataFrame(annual_returns_all).T
    annual_df = annual_df.fillna(0)
    annual_df.to_csv(annual_filename)
    print(f"✓ Exported annual returns to {annual_filename}")

    # Export monthly returns (aligned by month-end)
    monthly_filename = f"{config.output_prefix}_monthly_returns.csv"
    monthly_df = pd.DataFrame(monthly_returns_all)
    monthly_df = monthly_df.sort_index()
    monthly_df.to_csv(monthly_filename)
    print(f"✓ Exported monthly returns to {monthly_filename}")
    
    # Print top 5 by primary metric
    print(f"\n{'='*70}")
    print(f"TOP 5 STRATEGIES BY {config.primary_metric.value.upper()}")
    print(f"{'='*70}")
    
    cols_to_show = ['type', 'window', config.primary_metric.value, 'sharpe', 'max_drawdown']
    print(results_df[cols_to_show].head(5).to_string(index=False))


def main():
    """Run complete backtest with active configuration."""
    print("="*70)
    print("PMTL BACKTEST RUNNER (Refactored)")
    print("="*70)
    
    # Print active configuration
    print(f"\nActive Configuration:")
    print(f"   Main Ticker: {ACTIVE_CONFIG.main_ticker}")
    print(f"   Fallback: {ACTIVE_CONFIG.fallback_type.value}")
    print(f"   Windows: {ACTIVE_CONFIG.windows.start_window}-{ACTIVE_CONFIG.windows.end_window} "
          f"(step {ACTIVE_CONFIG.windows.step})")
    print(f"   Date Range: {ACTIVE_CONFIG.dates.start_date} to {ACTIVE_CONFIG.dates.end_date}")
    print(f"   Rebalance Frequency: {ACTIVE_CONFIG.dates.frequency}")
    print(f"   Output Prefix: {ACTIVE_CONFIG.output_prefix}")
    
    # Initialize backtest engine
    print(f"\nLoading Data...")
    engine = PMTLBacktestEngine(
        main_ticker=ACTIVE_CONFIG.main_ticker,
        start_date=ACTIVE_CONFIG.dates.start_date,
        end_date=ACTIVE_CONFIG.dates.end_date,
        frequency=ACTIVE_CONFIG.dates.frequency
    )
    
    # Create fallback strategy
    print(f"\nCreating Fallback Strategy ({ACTIVE_CONFIG.fallback_type.value})...")
    fallback = get_fallback_strategy(
        fallback_type=ACTIVE_CONFIG.fallback_type.value,
        start_date=ACTIVE_CONFIG.dates.start_date,
        end_date=ACTIVE_CONFIG.dates.end_date,
        fallback_ticker=ACTIVE_CONFIG.fallback_ticker,
        fallback_csv=ACTIVE_CONFIG.fallback_csv
    )
    
    # Run window sweep
    print(f"\nRunning Window Sweep...")
    results_df, annual_returns_all, monthly_returns_all = engine.run_window_sweep(
        fallback=fallback,
        windows=ACTIVE_CONFIG.windows.get_windows(),
        use_sma=ACTIVE_CONFIG.windows.use_sma,
        use_ema=ACTIVE_CONFIG.windows.use_ema
    )
    
    # Export results
    print(f"\nExporting Results...")
    export_results(results_df, annual_returns_all, monthly_returns_all, ACTIVE_CONFIG)
    
    print(f"\n{'='*70}")
    print("✅ Backtest Complete!")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    import pandas as pd  # Import here to avoid circular imports
    
    try:
        main()
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

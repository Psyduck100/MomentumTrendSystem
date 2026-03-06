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
    df_gap0 = result_gap0['bucket_returns']['US_equities']
    df_gap3 = result_gap3['bucket_returns']['US_equities']
    
    print("\n" + "=" * 100)
    print("POSITION CHANGES (first 36 months)")
    print("=" * 100)
    print(f"{'Month':<6} {'Gap=0':<10} {'Gap=3':<10} {'Same?':<8} {'Gap=0 Return':<15} {'Gap=3 Return':<15}")
    print("-" * 100)
    
    # Compare first 36 months
    for i in range(min(36, len(df_gap0), len(df_gap3))):
        # Get the actual symbol held from the bucket_returns 'symbols' field
        symbols0 = df_gap0.iloc[i]['symbols']
        symbols3 = df_gap3.iloc[i]['symbols']
        
        pos0 = symbols0[0] if len(symbols0) > 0 else None
        pos3 = symbols3[0] if len(symbols3) > 0 else None
        same = "Yes" if pos0 == pos3 else "No"
        
        ret0 = f"{df_gap0.iloc[i]['return']:.2%}"
        ret3 = f"{df_gap3.iloc[i]['return']:.2%}"
        
        print(f"{i:<6} {pos0 or 'None':<10} {pos3 or 'None':<10} {same:<8} {ret0:<15} {ret3:<15}")
    
    # Calculate cumulative returns and find worst drawdowns
    cum_gap0 = (1 + df_gap0['return']).cumprod()
    cum_gap3 = (1 + df_gap3['return']).cumprod()
    
    dd_gap0 = (cum_gap0 / cum_gap0.cummax() - 1)
    dd_gap3 = (cum_gap3 / cum_gap3.cummax() - 1)
    
    worst_dd0_idx = dd_gap0.idxmin()
    worst_dd3_idx = dd_gap3.idxmin()
    
    print("\n" + "=" * 100)
    print("WORST DRAWDOWN PERIODS")
    print("=" * 100)
    print(f"\nGap=0: MaxDD = {dd_gap0.min():.2%} at month {worst_dd0_idx}")
    print(f"  Period around worst DD:")
    
    # Get position in the series
    worst_loc0 = df_gap0.index.get_loc(worst_dd0_idx)
    start_idx = max(0, worst_loc0 - 5)
    end_idx = min(len(positions_gap0), worst_loc0 + 6)
    for i in range(start_idx, end_idx):
        pos = positions_gap0[i][0] if len(positions_gap0[i]) > 0 else None
        ret = df_gap0.iloc[i]['return']
        dd = dd_gap0.iloc[i]
        marker = " <-- WORST DD" if i == worst_loc0 else ""
        print(f"  Month {i}: Holding {pos}, Return {ret:.2%}, DD {dd:.2%}{marker}")
    
    print(f"\nGap=3: MaxDD = {dd_gap3.min():.2%} at month {worst_dd3_idx}")
    print(f"  Period around worst DD:")
    
    # Get position in the series
    worst_loc3 = df_gap3.index.get_loc(worst_dd3_idx)
    start_idx = max(0, worst_loc3 - 5)
    end_idx = min(len(positions_gap3), worst_loc3 + 6)
    for i in range(start_idx, end_idx):
        pos = positions_gap3[i][0] if len(positions_gap3[i]) > 0 else None
        ret = df_gap3.iloc[i]['return']
        dd = dd_gap3.iloc[i]
        marker = " <-- WORST DD" if i == worst_loc3 else ""
        print(f"  Month {i}: Holding {pos}, Return {ret:.2%}, DD {dd:.2%}{marker}")

if __name__ == "__main__":
    analyze_us_equities_trades()

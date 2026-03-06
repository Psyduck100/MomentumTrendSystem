"""
Sanity check: Investigate the suspicious pattern
- Returns increasing with SMALLER windows (counterintuitive)
- Massive collapse from weekly (37.96%) to 3-day (3.32%)
"""

import pandas as pd
import numpy as np

print("="*100)
print("INVESTIGATION: SUSPICIOUS PATTERN IN RESULTS")
print("="*100)

# Load monthly returns for all frequencies
weekly = pd.read_csv('pmtl_weekly_monthly_returns.csv')
threeday = pd.read_csv('pmtl_3day_monthly_returns.csv')

print("\n1. DATA SHAPE COMPARISON")
print("-"*100)
print(f"Weekly:   {len(weekly)} periods")
print(f"3-day:    {len(threeday)} periods")
print(f"Ratio:    {len(threeday) / len(weekly):.2f}x more periods in 3-day")

print("\n2. EXAMINING THE MONTHLY RETURNS DATA TYPES")
print("-"*100)
print("\nWeekly columns:", weekly.columns.tolist())
print("\n3-day columns:", threeday.columns.tolist())

# Check data quality
print("\n3. DATA QUALITY CHECK")
print("-"*100)
print(f"\nWeekly:")
print(f"  Total rows: {len(weekly)}")
print(f"  Rows with all NaN returns: {(weekly.iloc[:, 1:].isna().all(axis=1)).sum()}")
print(f"  Sample (first 10 rows):")
print(weekly.head(10))

print(f"\n3-day:")
print(f"  Total rows: {len(threeday)}")
print(f"  Rows with all NaN returns: {(threeday.iloc[:, 1:].isna().all(axis=1)).sum()}")
print(f"  Sample (first 10 rows):")
print(threeday.head(10))

# Look at EMA_50 specifically
print("\n4. DETAILED COMPARISON: EMA 50 (THE BEST STRATEGY)")
print("-"*100)

weekly_ema50 = weekly['EMA_50'].dropna()
threeday_ema50 = threeday['EMA_50'].dropna()

print(f"\nWeekly EMA 50:")
print(f"  Non-NaN periods: {len(weekly_ema50)}")
print(f"  Mean return per period: {weekly_ema50.mean():.6f}")
print(f"  Std dev: {weekly_ema50.std():.6f}")
print(f"  Min: {weekly_ema50.min():.6f}")
print(f"  Max: {weekly_ema50.max():.6f}")
print(f"  Periods at 0%: {(weekly_ema50 == 0).sum()}")
print(f"  Periods in CASH: {(weekly_ema50 == 0).sum() / len(weekly_ema50) * 100:.1f}%")

print(f"\n3-day EMA 50:")
print(f"  Non-NaN periods: {len(threeday_ema50)}")
print(f"  Mean return per period: {threeday_ema50.mean():.6f}")
print(f"  Std dev: {threeday_ema50.std():.6f}")
print(f"  Min: {threeday_ema50.min():.6f}")
print(f"  Max: {threeday_ema50.max():.6f}")
print(f"  Periods at 0%: {(threeday_ema50 == 0).sum()}")
print(f"  Periods in CASH: {(threeday_ema50 == 0).sum() / len(threeday_ema50) * 100:.1f}%")

# Calculate CAGR manually
cum_weekly = (1 + weekly_ema50).cumprod()
cum_threeday = (1 + threeday_ema50).cumprod()

print(f"\nManual CAGR calculation:")
print(f"  Weekly: final cumulative = {cum_weekly.iloc[-1]:.4f}")
print(f"  3-day: final cumulative = {cum_threeday.iloc[-1]:.4f}")

# Check if we're in cash too much in 3-day
print(f"\nCASH EXPOSURE ANALYSIS:")
cash_pct_weekly = (weekly_ema50 == 0).sum() / len(weekly_ema50) * 100
cash_pct_3day = (threeday_ema50 == 0).sum() / len(threeday_ema50) * 100
print(f"  Weekly in CASH: {cash_pct_weekly:.1f}%")
print(f"  3-day in CASH: {cash_pct_3day:.1f}%")

# Compare benchmark too
print(f"\n5. BENCHMARK COMPARISON")
print("-"*100)
weekly_bench = weekly['benchmark'].dropna()
threeday_bench = threeday['benchmark'].dropna()

cum_weekly_bench = (1 + weekly_bench).cumprod()
cum_threeday_bench = (1 + threeday_bench).cumprod()

years_weekly = len(weekly_bench) / 52
years_3day = len(threeday_bench) / (365/3)

cagr_weekly_bench = cum_weekly_bench.iloc[-1] ** (1/years_weekly) - 1
cagr_3day_bench = cum_threeday_bench.iloc[-1] ** (1/years_3day) - 1

print(f"Benchmark (GLD buy & hold):")
print(f"  Weekly: CAGR {cagr_weekly_bench:.2%}, periods={len(weekly_bench)}")
print(f"  3-day: CAGR {cagr_3day_bench:.2%}, periods={len(threeday_bench)}")

# Check if returns are even present in 3-day data
print(f"\n6. SIGNAL DISTRIBUTION CHECK")
print("-"*100)

for window in [50, 60, 70]:
    col = f'EMA_{window}'
    if col in threeday.columns:
        ema_col = threeday[col].dropna()
        zero_count = (ema_col == 0).sum()
        nonzero_count = (ema_col != 0).sum()
        print(f"\n{col}:")
        print(f"  In CASH (0%): {zero_count} periods ({zero_count/len(ema_col)*100:.1f}%)")
        print(f"  In GLD (≠0%): {nonzero_count} periods ({nonzero_count/len(ema_col)*100:.1f}%)")
        if nonzero_count > 0:
            print(f"  Average GLD return when in position: {ema_col[ema_col != 0].mean():.4f}")

print("\n" + "="*100)
print("POTENTIAL ISSUES:")
print("="*100)
print("""
1. TOO MANY CASH PERIODS: If 3-day is in CASH >95% of the time, that explains the collapse
2. NaN HANDLING: Are NaN values being dropped incorrectly, causing misalignment?
3. FREQUENCY ARITHMETIC: With 3-day rebalancing, do we have enough data points?
4. MA CALCULATION: Is the 50-day EMA even valid for 3-day periods?
   (50 trading days = ~2.5 months = 25 periods of 3-day)
5. REINDEX/FORWARD FILL: Is the forward fill in MA calculation creating stale values?
""")

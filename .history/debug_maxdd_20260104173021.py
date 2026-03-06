import pandas as pd
import numpy as np

# Load data
w = pd.read_csv('pmtl_weekly_monthly_returns.csv')
b = pd.read_csv('pmtl_biweekly_monthly_returns.csv')

ema100_w = w['EMA_100'].dropna()
ema100_b = b['EMA_100'].dropna()

# Calculate cumulative and drawdowns
cum_w = (1 + ema100_w).cumprod()
cum_b = (1 + ema100_b).cumprod()

dd_w = (cum_w / cum_w.cummax() - 1)
dd_b = (cum_b / cum_b.cummax() - 1)

# Find max DD locations
idx_w = np.argmin(dd_w.values)
idx_b = np.argmin(dd_b.values)

print("="*80)
print("MAX DRAWDOWN ANALYSIS")
print("="*80)

print(f'\nWeekly: Max DD = {dd_w.iloc[idx_w]:.10f}')
print(f'  Occurred at index {idx_w}, date {w.iloc[idx_w+1]["Date"]}')
print(f'  Cumulative value at max DD: {cum_w.iloc[idx_w]:.4f}')
print(f'  Previous peak: {cum_w.iloc[:idx_w+1].max():.4f}')

print(f'\nBiweekly: Max DD = {dd_b.iloc[idx_b]:.10f}')
print(f'  Occurred at index {idx_b}, date {b.iloc[idx_b+1]["Date"]}')
print(f'  Cumulative value at max DD: {cum_b.iloc[idx_b]:.4f}')
print(f'  Previous peak: {cum_b.iloc[:idx_b+1].max():.4f}')

print("\n" + "="*80)
print("WEEKLY CONTEXT AROUND MAX DD:")
print("="*80)
print(w.iloc[idx_w-2:idx_w+5][['Date', 'benchmark', 'EMA_100']])

print("\n" + "="*80)
print("BIWEEKLY CONTEXT AROUND MAX DD:")
print("="*80)
print(b.iloc[idx_b-2:idx_b+5][['Date', 'benchmark', 'EMA_100']])

# Check if returns are identical over the same time period
print("\n" + "="*80)
print("CHECKING IF RETURNS ARE SUSPICIOUSLY SIMILAR:")
print("="*80)

# Find overlapping dates
w_dates = pd.to_datetime(w['Date'])
b_dates = pd.to_datetime(b['Date'])
overlap = w_dates.isin(b_dates)

print(f"Weekly has {len(w)} periods")
print(f"Biweekly has {len(b)} periods")
print(f"Overlapping dates: {overlap.sum()}")

# Compare final cumulative returns
print(f"\nFinal cumulative (weekly): {cum_w.iloc[-1]:.4f}")
print(f"Final cumulative (biweekly): {cum_b.iloc[-1]:.4f}")
print(f"Ratio: {cum_w.iloc[-1] / cum_b.iloc[-1]:.4f}")

# Most importantly - check if we're using the SAME price data
print("\n" + "="*80)
print("CRITICAL CHECK: Are we resampling correctly?")
print("="*80)

# Read results CSV to see what the engine reported
weekly_results = pd.read_csv('pmtl_weekly_results.csv')
biweekly_results = pd.read_csv('pmtl_biweekly_results.csv')

weekly_ema100 = weekly_results[(weekly_results['type'] == 'EMA') & (weekly_results['window'] == 100)]
biweekly_ema100 = biweekly_results[(biweekly_results['type'] == 'EMA') & (biweekly_results['window'] == 100)]

print("\nFrom results CSV:")
print(f"Weekly EMA 100 MaxDD: {weekly_ema100['max_drawdown'].values[0]:.10f}")
print(f"Biweekly EMA 100 MaxDD: {biweekly_ema100['max_drawdown'].values[0]:.10f}")

print("\nFrom manual calculation:")
print(f"Weekly EMA 100 MaxDD: {dd_w.min():.10f}")
print(f"Biweekly EMA 100 MaxDD: {dd_b.min():.10f}")

# Check benchmark returns too
bench_w = w['benchmark'].dropna()
bench_b = b['benchmark'].dropna()
cum_bench_w = (1 + bench_w).cumprod()
cum_bench_b = (1 + bench_b).cumprod()
dd_bench_w = (cum_bench_w / cum_bench_w.cummax() - 1)
dd_bench_b = (cum_bench_b / cum_bench_b.cummax() - 1)

print("\n" + "="*80)
print("BENCHMARK COMPARISON:")
print("="*80)
print(f"Weekly benchmark MaxDD: {dd_bench_w.min():.10f}")
print(f"Biweekly benchmark MaxDD: {dd_bench_b.min():.10f}")
print(f"Weekly benchmark final cum: {cum_bench_w.iloc[-1]:.4f}")
print(f"Biweekly benchmark final cum: {cum_bench_b.iloc[-1]:.4f}")

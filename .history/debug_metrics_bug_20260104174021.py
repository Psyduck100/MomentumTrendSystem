"""
Debug the metrics calculation bug for 3-day frequency
"""

import pandas as pd
import numpy as np

print("="*100)
print("METRICS CALCULATION BUG INVESTIGATION")
print("="*100)

# Load 3-day returns
threeday = pd.read_csv('pmtl_3day_monthly_returns.csv')
ema50 = threeday['EMA_50'].dropna()

# Convert dates
threeday['Date'] = pd.to_datetime(threeday['Date'])
threeday_with_dates = threeday.dropna(subset=['EMA_50'])

start_date = threeday_with_dates['Date'].iloc[0]
end_date = threeday_with_dates['Date'].iloc[-1]

print(f"\nDate Range:")
print(f"  Start: {start_date}")
print(f"  End: {end_date}")
print(f"  Actual calendar days: {(end_date - start_date).days}")
print(f"  Actual calendar years: {(end_date - start_date).days / 365.25:.4f}")

print(f"\nPeriod Count:")
print(f"  Non-NaN EMA_50 returns: {len(ema50)}")
print(f"  Average days per period: {(end_date - start_date).days / len(ema50):.4f}")

# Calculate CAGR correctly
cum = (1 + ema50).cumprod()
final_value = cum.iloc[-1]

# Method 1: Using actual calendar years
years_actual = (end_date - start_date).days / 365.25
cagr_actual = final_value ** (1 / years_actual) - 1

# Method 2: Using periods/365*3 (treating each period as 3 calendar days)
years_3day = len(ema50) / (365 / 3)
cagr_3day = final_value ** (1 / years_3day) - 1

# Method 3: Using periods/52 (treating it as weeks???)
years_52 = len(ema50) / 52
cagr_52 = final_value ** (1 / years_52) - 1

print(f"\nCAGR Calculations:")
print(f"  Final cumulative value: {final_value:.4f}")
print(f"\nMethod 1 (actual calendar years = {years_actual:.4f}):")
print(f"  CAGR = {cagr_actual:.4f} = {cagr_actual:.2%}")

print(f"\nMethod 2 (periods / (365/3) = {years_3day:.4f}):")
print(f"  CAGR = {cagr_3day:.4f} = {cagr_3day:.2%}")

print(f"\nMethod 3 (periods / 52 = {years_52:.4f}):")
print(f"  CAGR = {cagr_52:.4f} = {cagr_52:.2%}")

# Now let's check what the code is actually using
print(f"\n" + "="*100)
print("WHAT THE CODE IS DOING:")
print("="*100)

# Simulate what metrics.py does
periods_per_year = 365 / 3  # This is what we set for 3-day
years = len(ema50) / periods_per_year
cagr_code = final_value ** (1/years) - 1

print(f"\nCurrent code logic:")
print(f"  periods_per_year = 365 / 3 = {periods_per_year:.4f}")
print(f"  years = {len(ema50)} / {periods_per_year:.4f} = {years:.4f}")
print(f"  cagr = {final_value:.4f} ** (1 / {years:.4f}) - 1")
print(f"  cagr = {cagr_code:.4f} = {cagr_code:.2%}")

print(f"\n" + "="*100)
print("THE PROBLEM:")
print("="*100)
print(f"""
The issue is that we're using periods_per_year = 365/3 = 121.67
But the actual data spans {(end_date - start_date).days} calendar days / {365.25} = {years_actual:.2f} years

If we have {len(ema50)} 3-day periods in {years_actual:.2f} years:
  Implied periods per year = {len(ema50) / years_actual:.2f}

The discrepancy:
  Assumed: 121.67 periods/year
  Actual: {len(ema50) / years_actual:.2f} periods/year
  Ratio: {(len(ema50) / years_actual) / periods_per_year:.4f}

This causes CAGR to be calculated as {cagr_code:.2%} instead of correct {cagr_actual:.2%}

The 3-day frequency is working correctly! It's just the metrics calculation
that's using the wrong periods_per_year denominator.

SOLUTION: Don't use periods_per_year from frequency alone.
Instead, calculate it from actual data: actual_periods / actual_years
""")

print(f"\nComparison to weekly:")
weekly = pd.read_csv('pmtl_weekly_monthly_returns.csv')
weekly_ema50 = weekly['EMA_50'].dropna()
weekly['Date'] = pd.to_datetime(weekly['Date'])
weekly_with_dates = weekly.dropna(subset=['EMA_50'])
weekly_start = weekly_with_dates['Date'].iloc[0]
weekly_end = weekly_with_dates['Date'].iloc[-1]
weekly_years = (weekly_end - weekly_start).days / 365.25
weekly_cum = (1 + weekly_ema50).cumprod()
weekly_cagr_correct = weekly_cum.iloc[-1] ** (1 / weekly_years) - 1

print(f"\nWeekly (correct calculation):")
print(f"  Periods: {len(weekly_ema50)}")
print(f"  Years: {weekly_years:.4f}")
print(f"  Final cum: {weekly_cum.iloc[-1]:.4f}")
print(f"  CAGR: {weekly_cagr_correct:.2%}")

print(f"\n3-day (should be):")
print(f"  Periods: {len(ema50)}")
print(f"  Years: {years_actual:.4f}")
print(f"  Final cum: {final_value:.4f}")
print(f"  CAGR: {cagr_actual:.2%}")

# So 3-day is actually performing WORSE than weekly (not better by 10.64%)
# But it's not 3.32% either - it should be around 26% or so

import pandas as pd
import numpy as np

# Load MONTHLY returns (what the backtest uses)
# The backtest computes monthly returns and then converts to CAGR

annual_data = pd.read_csv("pmtl_cash_annual_returns.csv", index_col=0)
ema_100_annual = annual_data.loc["EMA_100", "2005":"2025"].astype(float)

print("=" * 80)
print("INVESTIGATING THE DISCREPANCY")
print("=" * 80)

print("\n1. ANNUAL RETURNS (what we see in CSV):")
print(f"   EMA 100 annual returns: {ema_100_annual.tolist()}")

print("\n2. CAGR FROM ANNUAL RETURNS:")
cum_value = (1 + ema_100_annual).cumprod()
cagr_from_annual = cum_value.iloc[-1] ** (1 / len(ema_100_annual)) - 1
print(f"   Calculated from annual data: {cagr_from_annual:.4%}")

print("\n3. CAGR CALCULATION IN CODE (from monthly returns):")
print("   The engine works with MONTHLY returns, not annual")
print("   Years = len(monthly_returns) / 12")
print("   CAGR = final_value ** (1/years) - 1")

# Simulate: if we have 21 years of monthly data
n_months = 21 * 12
print(f"\n4. MONTHLY DATA DETAILS:")
print(f"   Number of months: {n_months}")
print(f"   Number of years (from code): {n_months} / 12 = {n_months / 12}")

# The issue might be that the first return is NaN (pct_change on first row)
# Let's calculate what CAGR we'd get if we're missing the first month

print("\n5. TESTING DIFFERENT MONTH COUNTS:")
for months in [250, 251, 252, 253]:
    years = months / 12
    cagr_calc = cum_value.iloc[-1] ** (1 / years) - 1
    print(f"   {months} months ({years:.3f} years): CAGR = {cagr_calc:.4%}")

# Actually, let's check what the code actually computed
print("\n6. LOADING RESULTS FILE TO CHECK WHAT WAS REPORTED:")
results = pd.read_csv("pmtl_cash_results.csv")
ema_100_result = results[results["type"] == "EMA_100"]
if len(ema_100_result) > 0:
    reported_cagr = ema_100_result.iloc[0]["cagr"]
    print(f"   Reported CAGR in results.csv: {reported_cagr:.6f} = {reported_cagr:.4%}")
else:
    reported_cagr = results[
        (results["type"] == "EMA") & (results["window"] == 100)
    ].iloc[0]["cagr"]
    print(f"   Reported CAGR in results.csv: {reported_cagr:.6f} = {reported_cagr:.4%}")

print("\n7. DIFFERENCE:")
print(f"   From annual data: {cagr_from_annual:.4%}")
print(f"   Reported in CSV:  {reported_cagr:.4%}")
print(f"   Difference:       {reported_cagr - cagr_from_annual:+.4%}")

# The difference might be because the backtest uses MONTH-END to MONTH-END returns
# But the annual CSV is summing MONTHLY returns (which isn't the same!)

print("\n" + "=" * 80)
print("KEY INSIGHT: Are the annual returns in the CSV computed correctly?")
print("=" * 80)

print(
    """
The annual returns CSV shows MONTHLY returns summed by year.
But that's not the same as year-over-year returns!

Example:
- If Jan returns +10% and Feb returns +10%
- Annual sum = 20% (what CSV shows)
- But actual Jan 1 to Dec 31 return ≠ 20% (it's more like 21% due to compounding)

So the CSV is showing a simplified version for checking.
The ACTUAL CAGR is computed from MONTHLY returns by the engine.

Let me check if the monthly return calculation is correct...
"""
)

import pandas as pd
import yfinance as yf
import numpy as np

# Load data
data = yf.download("GLD", start="2005-01-01", end="2025-12-31", progress=False)
print(f"Data columns: {data.columns.tolist()}")
gld = data[("Close", "GLD")] if ("Close", "GLD") in data.columns else data.iloc[:, 0]

monthly_gld = gld.resample("ME").last()
monthly_gld_ret = monthly_gld.pct_change()

# Calculate GLD CAGR
start_price = monthly_gld.iloc[0]
end_price = monthly_gld.iloc[-1]
total_ret = (end_price / start_price) - 1
years = (monthly_gld.index[-1] - monthly_gld.index[0]).days / 365.25
cagr = (1 + total_ret) ** (1 / years) - 1

print(f"GLD start price: ${start_price:.2f}")
print(f"GLD end price: ${end_price:.2f}")
print(f"Total return: {total_ret:.4f} ({total_ret*100:.2f}%)")
print(f"Period: {years:.2f} years")
print(f"GLD CAGR: {cagr:.4f} ({cagr*100:.2f}%)")

# Calculate Sharpe
annual_rets = monthly_gld_ret.groupby(monthly_gld_ret.index.year).apply(
    lambda x: (1 + x).prod() - 1
)
mean_annual = annual_rets.mean()
std_annual = annual_rets.std()
sharpe = mean_annual / std_annual

print(f"\nAnnual returns stats:")
print(f"Mean: {mean_annual:.4f} ({mean_annual*100:.2f}%)")
print(f"Std: {std_annual:.4f} ({std_annual*100:.2f}%)")
print(f"Sharpe: {sharpe:.3f}")

# Check by year
print("\nYear-by-year returns:")
print(annual_rets.to_string())

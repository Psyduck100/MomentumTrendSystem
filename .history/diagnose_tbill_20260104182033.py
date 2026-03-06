import pandas as pd
import yfinance as yf
import numpy as np

# Load data
data = yf.download("GLD", start="2005-01-01", end="2025-12-31", progress=False)
print(f"Data columns: {data.columns.tolist()}")
if isinstance(data.columns, pd.MultiIndex):
    gld = (
        data[("GLD", "Adj Close")]
        if ("GLD", "Adj Close") in data.columns
        else data.iloc[:, 0]
    )
else:
    gld = data["Adj Close"]
tbill_df = pd.read_csv("CSVs/TB3MS.csv")
tbill_df["observation_date"] = pd.to_datetime(tbill_df["observation_date"])
tbill_df = tbill_df.set_index("observation_date")
tbill_df["TB3MS_monthly_ret"] = tbill_df["TB3MS"] / 100 / 12

print(f"GLD data points: {len(gld)}")
print(f"TB3MS data points: {len(tbill_df)}")

# Get monthly GLD
monthly_gld = gld.resample("ME").last()
print(f"Monthly GLD data points: {len(monthly_gld)}")
print(
    f"GLD monthly dates range: {monthly_gld.index.min()} to {monthly_gld.index.max()}"
)
print(f"TB3MS dates range: {tbill_df.index.min()} to {tbill_df.index.max()}")

# Check first few monthly GLD returns and TB3MS alignment
monthly_gld_ret = monthly_gld.pct_change()
print("\nFirst 10 monthly GLD returns:")
print(monthly_gld_ret.head(10))

# Try to align with TB3MS
tbill_monthly = tbill_df.loc[
    tbill_df.index.to_period("M").isin(monthly_gld_ret.index.to_period("M")),
    "TB3MS_monthly_ret",
]
print(f"\nTB3MS aligned to monthly GLD: {len(tbill_monthly)} values")
print("First 10:")
print(tbill_monthly.head(10))

# Check the blend - 100% TB3MS (when signal=0)
tbill_only = tbill_df.loc[
    tbill_df.index.to_period("M").isin(monthly_gld.index.to_period("M")),
    "TB3MS_monthly_ret",
]
tbill_only = tbill_only.reindex(monthly_gld.index, method="ffill")
print(f"\nTB3MS-only annualized return:")
annual_ret = (1 + tbill_only.values).prod() - 1
print(f"Full period return: {annual_ret:.4f} ({annual_ret*100:.2f}%)")
years = (monthly_gld.index[-1] - monthly_gld.index[0]).days / 365.25
cagr = (1 + annual_ret) ** (1 / years) - 1
print(f"CAGR: {cagr:.4f} ({cagr*100:.2f}%)")

import pandas as pd
import yfinance as yf
import numpy as np
from momentum_program.backtest.metrics import compute_metrics

# Load data
data = yf.download('GLD', start='2005-01-01', end='2025-12-31', progress=False)
gld = data[('Close', 'GLD')] if ('Close', 'GLD') in data.columns else data.iloc[:, 0]

tbill_df = pd.read_csv('CSVs/TB3MS.csv')
tbill_df['observation_date'] = pd.to_datetime(tbill_df['observation_date'])
tbill_df = tbill_df.set_index('observation_date')
tbill_df['TB3MS_monthly_ret'] = tbill_df['TB3MS'] / 100 / 12

# Test SMA 100 manually
window = 100
ma = gld.rolling(window=window, min_periods=1).mean()

# Get monthly prices and dates
monthly_gld = gld.resample('ME').last()
monthly_dates = monthly_gld.index

# Get MA values at month-end (reindex to handle missing dates)
ma_at_month_end = ma.reindex(monthly_dates, method='ffill')

# Generate signal at month-end dates: 1 if price > MA (hold GLD), 0 if price <= MA (hold TB3MS)
signal_at_month_end = (monthly_gld > ma_at_month_end).astype(int)

# Calculate GLD monthly returns
monthly_gld_ret = monthly_gld.pct_change()

# Align TB3MS with monthly dates
tbill_monthly = tbill_df.loc[tbill_df.index.to_period('M').isin(
    monthly_dates.to_period('M')), 'TB3MS_monthly_ret']

# Forward-fill TB3MS for dates not in the data
tbill_monthly = tbill_monthly.reindex(monthly_dates, method='ffill')

# Blend: signal% GLD + (1-signal%) TB3MS
blended_ret = (signal_at_month_end.values * monthly_gld_ret.values +
               (1 - signal_at_month_end.values) * tbill_monthly.values)

# Remove first NaN
blended_ret = pd.Series(blended_ret, index=monthly_dates, name=f"SMA_{window}")[1:]
print(f"Blended return series length: {len(blended_ret)}")
print(f"First 10 blended returns:")
print(blended_ret.head(10))

# Manual CAGR calculation
cumulative = (1 + blended_ret).prod()
years = (blended_ret.index[-1] - blended_ret.index[0]).days / 365.25
cagr_manual = (cumulative) ** (1/years) - 1
print(f"\nManual CAGR (product of returns): {cagr_manual:.4f} ({cagr_manual*100:.2f}%)")

# Use compute_metrics
metrics = compute_metrics(blended_ret)
print(f"\ncompute_metrics CAGR: {metrics['cagr']:.4f} ({metrics['cagr']*100:.2f}%)")

# Check what signal values are
print(f"\nSignal value distribution:")
print(f"Hold GLD (signal=1): {(signal_at_month_end.values[1:] == 1).sum()} months")
print(f"Hold TB3MS (signal=0): {(signal_at_month_end.values[1:] == 0).sum()} months")

# Check some month where we held TB3MS
zero_idx = np.where(signal_at_month_end.values[1:] == 0)[0]
if len(zero_idx) > 0:
    print(f"\nExample: Month {zero_idx[0]}, date {blended_ret.index[zero_idx[0]]}")
    print(f"  GLD return: {monthly_gld_ret.values[zero_idx[0]+1]:.6f}")
    print(f"  TB3MS return: {tbill_monthly.values[zero_idx[0]+1]:.6f}")
    print(f"  Blended: {blended_ret.values[zero_idx[0]]:.6f}")

"""
Debug the actual monthly returns calculation to find where the discrepancy comes from.
"""

import pandas as pd
import yfinance as yf
import numpy as np
from pmtl_fallback_strategies import get_fallback_strategy
from momentum_program.backtest.metrics import compute_metrics

# Parameters
main_ticker = "GLD"
start_date = "2005-01-01"
end_date = "2025-12-31"
window = 100

print("="*80)
print("DETAILED MONTHLY RETURN CALCULATION - EMA 100 + CASH")
print("="*80)

# Download prices
print(f"\n1. Downloading {main_ticker} prices ({start_date} to {end_date})...")
data = yf.download(main_ticker, start=start_date, end=end_date, progress=False)
# Handle both single and multi-ticker downloads
if isinstance(data.columns, pd.MultiIndex):
    prices = data[('Adj Close', main_ticker)] if ('Adj Close', main_ticker) in data.columns else data.iloc[:, 0]
else:
    prices = data['Adj Close'] if 'Adj Close' in data.columns else data['Close']
print(f"   Total trading days: {len(prices)}")
print(f"   Price range: ${prices.min():.2f} - ${prices.max():.2f}")

# Calculate EMA
print(f"\n2. Calculating {window}-day EMA...")
ema = prices.ewm(span=window, adjust=False).mean()
print(f"   EMA range: ${ema.min():.2f} - ${ema.max():.2f}")

# Get month-end data
print(f"\n3. Extracting month-end data...")
monthly_prices = prices.resample('ME').last()
monthly_dates = monthly_prices.index
print(f"   Total months: {len(monthly_prices)}")
print(f"   Date range: {monthly_dates[0].date()} to {monthly_dates[-1].date()}")

# Get EMA at month-end
ma_at_month_end = ema.reindex(monthly_dates, method='ffill')
print(f"\n4. EMA values at month-end:")
print(f"   First: ${ma_at_month_end.iloc[0]:.2f}")
print(f"   Last: ${ma_at_month_end.iloc[-1]:.2f}")

# Generate signal
signal = (monthly_prices > ma_at_month_end).astype(int)
print(f"\n5. Signal analysis (price > EMA):")
print(f"   Months in GLD: {signal.sum()}")
print(f"   Months in CASH: {(1-signal).sum()}")
print(f"   In GLD: {signal.sum() / len(signal):.1%}")
print(f"   In CASH: {(1-signal).sum() / len(signal):.1%}")

# Calculate main asset monthly returns
main_monthly_ret = monthly_prices.pct_change()
print(f"\n6. GLD monthly returns:")
print(f"   Mean: {main_monthly_ret.mean():.4f} ({main_monthly_ret.mean()*100:.2%} per month)")
print(f"   Std:  {main_monthly_ret.std():.4f}")
print(f"   Min:  {main_monthly_ret.min():.4f}")
print(f"   Max:  {main_monthly_ret.max():.4f}")
print(f"   Months: {len(main_monthly_ret)}")

# Get fallback returns (CASH)
fallback = get_fallback_strategy('cash', end_date)
fallback_ret = fallback.get_monthly_returns(monthly_dates)
print(f"\n7. Fallback (CASH) monthly returns:")
print(f"   All zeros (as expected): {np.allclose(fallback_ret, 0)}")

# Blend returns
blended_ret = (signal.values * main_monthly_ret.values + 
               (1 - signal.values) * fallback_ret.values)
blended_ret_series = pd.Series(blended_ret, index=monthly_dates)

print(f"\n8. Blended returns (EMA 100 + CASH):")
print(f"   Mean: {blended_ret_series.mean():.4f} ({blended_ret_series.mean()*100:.2%} per month)")
print(f"   Std:  {blended_ret_series.std():.4f}")
print(f"   Min:  {blended_ret_series.min():.4f}")
print(f"   Max:  {blended_ret_series.max():.4f}")

# Compute CAGR
print(f"\n9. Computing CAGR from monthly returns:")
cumulative = (1 + blended_ret_series).cumprod()
final_value = cumulative.iloc[-1]
years = len(blended_ret_series) / 12
cagr = final_value ** (1 / years) - 1

print(f"   Final cumulative value: {final_value:.4f}")
print(f"   Number of returns: {len(blended_ret_series)}")
print(f"   Years (n/12): {years:.4f}")
print(f"   CAGR: {cagr:.6f} = {cagr:.4%}")

# Compare with metrics function
metrics = compute_metrics(blended_ret_series)
print(f"\n10. Metrics function result:")
print(f"   CAGR: {metrics['cagr']:.6f} = {metrics['cagr']:.4%}")
print(f"   Sharpe: {metrics['sharpe']:.4f}")
print(f"   Max DD: {metrics['max_drawdown']:.4%}")

# Now compute from annual data for comparison
print(f"\n11. Cross-check with annual returns from CSV:")
annual_data = pd.read_csv('pmtl_cash_annual_returns.csv', index_col=0)
ema_100_annual = annual_data.loc['EMA_100', '2005':'2025'].astype(float)
annual_cum = (1 + ema_100_annual).cumprod()
annual_cagr = annual_cum.iloc[-1] ** (1 / len(ema_100_annual)) - 1
print(f"   From CSV annual sums: {annual_cagr:.4%}")
print(f"   From code monthly: {cagr:.4%}")
print(f"   Difference: {cagr - annual_cagr:+.4%}")

# The issue: the annual CSV sums MONTHLY returns, but that's not the same as 
# year-by-year compounding!
print(f"\n12. IMPORTANT: Difference explanation")
print(f"   Annual returns in CSV are SUMS of monthly returns")
print(f"   But actual year returns should be COMPOUNDS of monthly returns")
print(f"   Example: Jan +10%, Feb +10%")
print(f"   - CSV shows: 20% (simple sum)")
print(f"   - Actual year: 1.1 * 1.1 - 1 = 21% (compounding)")

# Let's verify the numbers from the results CSV
print(f"\n13. Results CSV reports:")
results = pd.read_csv('pmtl_cash_results.csv')
ema_result = results[(results['type'] == 'EMA') & (results['window'] == 100)]
if len(ema_result) > 0:
    reported_cagr = ema_result.iloc[0]['cagr']
    print(f"   Reported CAGR: {reported_cagr:.6f} = {reported_cagr:.4%}")
    print(f"   Calculated from monthly: {cagr:.4%}")
    print(f"   Match: {abs(reported_cagr - cagr) < 0.0001}")

print("\n" + "="*80)

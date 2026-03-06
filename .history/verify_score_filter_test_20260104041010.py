"""Verify the score/filter test by examining specific years in detail."""

import pandas as pd
from pathlib import Path
from us_rotation_custom import BUCKET_MAP, BACKTEST_CACHE
from momentum_program.backtest.engine import backtest_momentum
from momentum_program.analytics.constants import SCORE_MODE_BLEND_6_12

TICKERS_WITH_BOND = ["SPTM", "SPY", "QQQ", "OEF", "IWD", "IEF"]
START_DATE = "2001-01-01"
END_DATE = "2026-01-04"

print("Checking BUCKET_MAP:")
print(BUCKET_MAP)
print()

print("Checking tickers in BUCKET_MAP:")
for ticker in TICKERS_WITH_BOND:
    bucket = BUCKET_MAP.get(ticker, "NOT IN MAP")
    print(f"  {ticker}: {bucket}")
print()

# Run one test case and examine holdings
print("Running blend_filter_12m test...")
result = backtest_momentum(
    tickers=TICKERS_WITH_BOND,
    bucket_map=BUCKET_MAP,
    start_date=START_DATE,
    end_date=END_DATE,
    top_n_per_bucket=1,
    cache_dir=BACKTEST_CACHE,
    slippage_bps=3.0,
    expense_ratio=0.001,
    rank_gap_threshold=0,
    score_mode=SCORE_MODE_BLEND_6_12,
    abs_filter_mode="ret_12m",
    abs_filter_cash_annual=0.025,
    defensive_symbol="IEF",
)

# Check 2008 and 2022 holdings
overall_positions = result['overall_positions']
overall_returns = result['overall_returns']

print("\n2008 Holdings (blend_filter_12m):")
print("-" * 60)
for item in overall_positions:
    date = item[0] if isinstance(item, (list, tuple)) else None
    if date and date.year == 2008:
        print(f"{date.strftime('%Y-%m-%d')}: {item}")

print("\n2022 Holdings (blend_filter_12m):")
print("-" * 60)
for item in overall_positions:
    date = item[0] if isinstance(item, (list, tuple)) else None
    if date and date.year == 2022:
        print(f"{date.strftime('%Y-%m-%d')}: {item}")

print("\n2008 Monthly Returns:")
print("-" * 60)
returns_2008 = overall_returns[overall_returns.index.year == 2008]
for date, row in returns_2008.iterrows():
    print(f"{date.strftime('%Y-%m')}: {row['return']:7.2%} (symbols: {row['symbols']})")

print("\n2022 Monthly Returns:")
print("-" * 60)
returns_2022 = overall_returns[overall_returns.index.year == 2022]
for date, row in returns_2022.iterrows():
    print(f"{date.strftime('%Y-%m')}: {row['return']:7.2%} (symbols: {row['symbols']})")

# Check IEF price in 2022
monthly_prices = result['monthly_prices']
ief_2022 = monthly_prices['IEF'][monthly_prices.index.year == 2022]
print("\nIEF Monthly Returns 2022:")
print("-" * 60)
ief_rets_2022 = ief_2022.pct_change()
for date, ret in ief_rets_2022.items():
    if pd.notna(ret):
        print(f"{date.strftime('%Y-%m')}: {ret:7.2%}")

# Annual return calculation
annual_2022 = (1 + returns_2022['return']).prod() - 1
print(f"\n2022 Annual Return: {annual_2022:.2%}")

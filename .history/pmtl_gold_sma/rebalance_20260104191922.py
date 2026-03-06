"""
PMTL Gold SMA Strategy

- Asset: Gold (GLD)
- Rule: If price > x-day SMA at month start, hold GLD; else, hold cash
- SMA window x is a parameter
- No lookforward bias: all signals use only data available up to the rebalance date
- Rebalance: First trading day of each month
"""

import pandas as pd
import yfinance as yf
from datetime import datetime
from pathlib import Path

# === CONFIGURATION ===
TICKER = "GLD"
CASH_SYMBOL = "CASH"
SMA_WINDOW = 100  # Change this value to test different x
START_DATE = "2005-01-01"  # GLD inception


# Download daily prices
def download_prices(ticker: str, start_date: str = "2005-01-01") -> pd.Series:
    data = yf.download(ticker, start=start_date, progress=False)
    if "Adj Close" in data:
        return data["Adj Close"]
    return data["Close"]


# Compute SMA (no lookforward bias)
def compute_sma(prices: pd.Series, window: int) -> pd.Series:
    return prices.rolling(window=window, min_periods=window).mean()


# Main rebalance logic
def run_strategy(sma_window: int = SMA_WINDOW):
    prices = download_prices(TICKER, START_DATE)
    # Get first trading day of each month
    month_starts = prices.resample("MS").first().index
    # Compute SMA up to each day (no lookforward)
    sma = compute_sma(prices, sma_window)
    # Prepare results
    signals = []
    for dt in month_starts:
        if dt not in prices.index:
            continue  # skip if no trading on first calendar day
        price = prices.loc[dt]
        sma_val = sma.loc[dt]
        if pd.isna(sma_val):
            signals.append((dt, CASH_SYMBOL, price, sma_val, "No SMA yet"))
            continue
        if price > sma_val:
            signals.append((dt, TICKER, price, sma_val, "GLD above SMA: HOLD GOLD"))
        else:
            signals.append(
                (dt, CASH_SYMBOL, price, sma_val, "GLD below SMA: HOLD CASH")
            )
    # Print summary
    print(f"PMTL Gold SMA Strategy (SMA window: {sma_window} days)")
    print("Date       | Signal | Price   | SMA     | Note")
    print("-" * 55)
    for dt, signal, price, sma_val, note in signals:
        print(f"{dt.date()} | {signal:5s}  | {price:8.2f} | {sma_val:8.2f} | {note}")
    return signals


if __name__ == "__main__":
    run_strategy()

"""Sweep MA filters with CASH ONLY (0% return) as fallback.

This tests whether the high PMTL returns come from:
1. Good MA timing (avoiding GLD downturns)
2. TB3MS contribution (should be minimal)
"""

from __future__ import annotations
import pandas as pd
import yfinance as yf
from pathlib import Path

from momentum_program.backtest.metrics import compute_metrics

TICKERS = ["GLD"]
START_DATE = "2005-01-01"
END_DATE = "2025-12-31"

def download_prices(tickers: list[str], end_date: str = END_DATE) -> pd.DataFrame:
    """Download daily prices."""
    print(f"Downloading {len(tickers)} tickers from {START_DATE} to {end_date}...")
    data = yf.download(tickers, start=START_DATE, end=end_date, progress=False)

    if isinstance(data.columns, pd.MultiIndex):
        data = data.swaplevel(0, 1, axis=1).sort_index(axis=1)
        prices = pd.DataFrame()
        for ticker in tickers:
            if ticker in data.columns.get_level_values(0):
                sub = data[ticker]
                if "Adj Close" in sub.columns:
                    prices[ticker] = sub["Adj Close"]
                elif "Close" in sub.columns:
                    prices[ticker] = sub["Close"]
    else:
        prices = pd.DataFrame({tickers[0]: data["Adj Close"]})

    return prices

def backtest_sma_window_cash(prices: pd.DataFrame, window: int) -> pd.Series:
    """Test SMA filter with given window size in trading days.
    
    If price > SMA, hold GLD. Else hold CASH (0% return).
    Returns monthly return series.
    """
    gld = prices["GLD"].copy()
    
    # Calculate SMA
    ma = gld.rolling(window=window, min_periods=1).mean()
    
    # Get monthly prices and dates
    monthly_gld = gld.resample('ME').last()
    monthly_dates = monthly_gld.index
    
    # Get MA values at month-end (reindex to handle missing dates)
    ma_at_month_end = ma.reindex(monthly_dates, method='ffill')
    
    # Generate signal at month-end dates: 1 if price > MA (hold GLD), 0 if price <= MA (hold CASH)
    signal_at_month_end = (monthly_gld > ma_at_month_end).astype(int)
    
    # Calculate GLD monthly returns
    monthly_gld_ret = monthly_gld.pct_change()
    
    # Blend: signal% GLD + (1-signal%) CASH (0%)
    blended_ret = (signal_at_month_end.values * monthly_gld_ret.values)
    
    return pd.Series(blended_ret, index=monthly_dates, name=f"SMA_{window}")

def backtest_ema_window_cash(prices: pd.DataFrame, window: int) -> pd.Series:
    """Test EMA filter with given window size in trading days.
    
    If price > EMA, hold GLD. Else hold CASH (0% return).
    Returns monthly return series.
    """
    gld = prices["GLD"].copy()
    
    # Calculate EMA
    ma = gld.ewm(span=window, adjust=False).mean()
    
    # Get monthly prices and dates
    monthly_gld = gld.resample('ME').last()
    monthly_dates = monthly_gld.index
    
    # Get MA values at month-end (reindex to handle missing dates)
    ma_at_month_end = ma.reindex(monthly_dates, method='ffill')
    
    # Generate signal at month-end dates: 1 if price > MA (hold GLD), 0 if price <= MA (hold CASH)
    signal_at_month_end = (monthly_gld > ma_at_month_end).astype(int)
    
    # Calculate GLD monthly returns
    monthly_gld_ret = monthly_gld.pct_change()
    
    # Blend: signal% GLD + (1-signal%) CASH (0%)
    blended_ret = (signal_at_month_end.values * monthly_gld_ret.values)
    
    return pd.Series(blended_ret, index=monthly_dates, name=f"EMA_{window}")

def backtest_gld_only(prices: pd.DataFrame) -> pd.Series:
    """Benchmark: Hold GLD always."""
    gld = prices["GLD"].copy()
    monthly_gld = gld.resample('ME').last()
    return monthly_gld.pct_change()

def test_window_range(prices: pd.DataFrame,
                     start_window: int = 100, end_window: int = 200, step: int = 10) -> dict:
    """Test all windows from start_window to end_window in step increments."""
    results = []
    annual_returns_all = {}
    
    # Get GLD benchmark
    gld_rets = backtest_gld_only(prices)
    gld_annual = gld_rets.groupby(gld_rets.index.year).sum()
    annual_returns_all['GLD_benchmark'] = gld_annual
    
    print(f"\nGLD Benchmark: CAGR {compute_metrics(gld_rets)['cagr']:.2%}, "
          f"Sharpe {compute_metrics(gld_rets)['sharpe']:.3f}, "
          f"MaxDD {compute_metrics(gld_rets)['max_drawdown']:.2%}")
    
    # Test SMA windows
    print("\nTesting SMA windows (CASH fallback)...")
    for window in range(start_window, end_window + 1, step):
        sma_rets = backtest_sma_window_cash(prices, window)
        metrics = compute_metrics(sma_rets)
        
        results.append({
            'type': 'SMA',
            'window': window,
            'cagr': metrics['cagr'],
            'sharpe': metrics['sharpe'],
            'max_drawdown': metrics['max_drawdown'],
        })
        
        # Annual returns
        annual = sma_rets.groupby(sma_rets.index.year).sum()
        annual_returns_all[f'SMA_{window}'] = annual
        
        print(f"  SMA {window:3d}: CAGR {metrics['cagr']:6.2%}, "
              f"Sharpe {metrics['sharpe']:6.3f}, MaxDD {metrics['max_drawdown']:7.2%}")
    
    # Test EMA windows
    print("\nTesting EMA windows (CASH fallback)...")
    for window in range(start_window, end_window + 1, step):
        ema_rets = backtest_ema_window_cash(prices, window)
        metrics = compute_metrics(ema_rets)
        
        results.append({
            'type': 'EMA',
            'window': window,
            'cagr': metrics['cagr'],
            'sharpe': metrics['sharpe'],
            'max_drawdown': metrics['max_drawdown'],
        })
        
        # Annual returns
        annual = ema_rets.groupby(ema_rets.index.year).sum()
        annual_returns_all[f'EMA_{window}'] = annual
        
        print(f"  EMA {window:3d}: CAGR {metrics['cagr']:6.2%}, "
              f"Sharpe {metrics['sharpe']:6.3f}, MaxDD {metrics['max_drawdown']:7.2%}")
    
    return pd.DataFrame(results), annual_returns_all

def export_results(results_df: pd.DataFrame, annual_returns_all: dict) -> None:
    """Export results to CSV files."""
    # Export summary results
    results_df = results_df.sort_values('cagr', ascending=False)
    results_df.to_csv('pmtl_ma_sweep_results_cash.csv', index=False)
    print(f"\nExported summary to pmtl_ma_sweep_results_cash.csv")
    
    # Export annual returns
    annual_df = pd.DataFrame(annual_returns_all).T
    annual_df = annual_df.fillna(0)
    annual_df.to_csv('pmtl_ma_sweep_annual_returns_cash.csv')
    print(f"Exported annual returns to pmtl_ma_sweep_annual_returns_cash.csv")
    
    # Print top 5 by CAGR
    print("\n" + "="*60)
    print("TOP 5 WINDOWS BY CAGR (CASH Fallback)")
    print("="*60)
    print(results_df.head(5).to_string(index=False))

if __name__ == "__main__":
    # Download data
    prices = download_prices(TICKERS)
    
    # Test windows
    results_df, annual_returns_all = test_window_range(prices,
                                                       start_window=100,
                                                       end_window=200,
                                                       step=10)
    
    # Export
    export_results(results_df, annual_returns_all)

"""Sweep MA filters from 100 to 200 trading days in 10-day bands.

Test both SMA and EMA variants, report CAGR per band.
Use TB3MS (3-month T-Bills) as defensive asset instead of IEF.
Export results to CSV and benchmark vs GLD.
"""

from __future__ import annotations
import pandas as pd
import yfinance as yf
from pathlib import Path

from momentum_program.backtest.metrics import compute_metrics

TICKERS = ["GLD"]
START_DATE = "2005-01-01"
END_DATE = "2025-12-31"
TBILL_CSV = Path("CSVs/TB3MS.csv")

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

def load_tbill_data() -> pd.DataFrame:
    """Load TB3MS monthly rates from CSV."""
    df = pd.read_csv(TBILL_CSV)
    df['observation_date'] = pd.to_datetime(df['observation_date'])
    df = df.set_index('observation_date')
    # TB3MS is annualized rate; convert to monthly return
    df['TB3MS_monthly_ret'] = df['TB3MS'] / 100 / 12
    return df

def backtest_sma_window(prices: pd.DataFrame, tbill_df: pd.DataFrame, window: int) -> pd.Series:
    """Test SMA filter with given window size in trading days.
    
    If price > SMA, hold GLD. Else hold TB3MS (T-Bills).
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
    
    return pd.Series(blended_ret, index=monthly_dates, name=f"SMA_{window}")

def backtest_ema_window(prices: pd.DataFrame, tbill_df: pd.DataFrame, window: int) -> pd.Series:
    """Test EMA filter with given window size in trading days.
    
    If price > EMA, hold GLD. Else hold TB3MS (T-Bills).
    Returns monthly return series.
    """
    gld = prices["GLD"].copy()
    
    # Calculate EMA
    ma = gld.ewm(span=window, adjust=False).mean()
    
    # Get monthly prices and dates
    monthly_gld = gld.resample('ME').last()
    monthly_dates = monthly_gld.index
    
    # Generate signal at month-end dates: 1 if price > MA (hold GLD), 0 if price <= MA (hold TB3MS)
    signal_at_month_end = (monthly_gld > ma.loc[monthly_dates]).astype(int)
    
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
    
    return pd.Series(blended_ret, index=monthly_dates, name=f"EMA_{window}")

def backtest_gld_only(prices: pd.DataFrame) -> pd.Series:
    """Benchmark: Hold GLD always."""
    gld = prices["GLD"].copy()
    monthly_gld = gld.resample('ME').last()
    return monthly_gld.pct_change()

def test_window_range(prices: pd.DataFrame, tbill_df: pd.DataFrame,
                     start_window: int = 100, end_window: int = 200, step: int = 10) -> dict:
    """Test all windows from start_window to end_window in step increments."""
    results = []
    annual_returns_all = {}
    
    # Get GLD benchmark
    gld_rets = backtest_gld_only(prices)
    gld_annual = gld_rets.groupby(gld_rets.index.year).sum()
    annual_returns_all['GLD_benchmark'] = gld_annual
    
    print(f"\nGLD Benchmark: CAGR {compute_metrics(gld_rets)['cagr']:.2f}%, "
          f"Sharpe {compute_metrics(gld_rets)['sharpe']:.3f}, "
          f"MaxDD {compute_metrics(gld_rets)['max_drawdown']:.2f}%")
    
    # Test SMA windows
    print("\nTesting SMA windows...")
    for window in range(start_window, end_window + 1, step):
        sma_rets = backtest_sma_window(prices, tbill_df, window)
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
        
        print(f"  SMA {window:3d}: CAGR {metrics['cagr']:6.2f}%, "
              f"Sharpe {metrics['sharpe']:6.3f}, MaxDD {metrics['max_drawdown']:7.2f}%")
    
    # Test EMA windows
    print("\nTesting EMA windows...")
    for window in range(start_window, end_window + 1, step):
        ema_rets = backtest_ema_window(prices, tbill_df, window)
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
        
        print(f"  EMA {window:3d}: CAGR {metrics['cagr']:6.2f}%, "
              f"Sharpe {metrics['sharpe']:6.3f}, MaxDD {metrics['max_drawdown']:7.2f}%")
    
    return pd.DataFrame(results), annual_returns_all

def export_results(results_df: pd.DataFrame, annual_returns_all: dict) -> None:
    """Export results to CSV files."""
    # Export summary results
    results_df = results_df.sort_values('cagr', ascending=False)
    results_df.to_csv('pmtl_ma_sweep_results_tb3ms.csv', index=False)
    print(f"\nExported summary to pmtl_ma_sweep_results_tb3ms.csv")
    
    # Export annual returns
    annual_df = pd.DataFrame(annual_returns_all).T
    annual_df = annual_df.fillna(0)
    annual_df.to_csv('pmtl_ma_sweep_annual_returns_tb3ms.csv')
    print(f"Exported annual returns to pmtl_ma_sweep_annual_returns_tb3ms.csv")
    
    # Print top 5 by CAGR
    print("\n" + "="*60)
    print("TOP 5 WINDOWS BY CAGR (TB3MS Fallback)")
    print("="*60)
    print(results_df.head(5).to_string(index=False))

if __name__ == "__main__":
    # Download data
    prices = download_prices(TICKERS)
    tbill_df = load_tbill_data()
    
    # Test windows
    results_df, annual_returns_all = test_window_range(prices, tbill_df,
                                                       start_window=100,
                                                       end_window=200,
                                                       step=10)
    
    # Export
    export_results(results_df, annual_returns_all)

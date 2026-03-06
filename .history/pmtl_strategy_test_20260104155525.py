"""PMTL Strategy: GLD with 100-day MA filter vs 12M return filter.

Compare three strategies:
1. GLD 100-day MA filter: hold GLD if above 100MA, else IEF
2. GLD 12M return filter: hold GLD if 12M > 0, else IEF
3. GLD benchmark: hold GLD always
"""

from __future__ import annotations
import pandas as pd
import yfinance as yf
from pathlib import Path

from momentum_program.backtest.metrics import compute_metrics

TICKERS = ["GLD", "IEF"]
START_DATE = "2005-01-01"
END_DATE = pd.Timestamp.today().strftime("%Y-%m-%d")

BACKTEST_CACHE = Path("backtest_cache")
BACKTEST_CACHE.mkdir(exist_ok=True)

def download_prices(tickers: list[str]) -> pd.DataFrame:
    """Download daily prices."""
    print(f"Downloading {len(tickers)} tickers from {START_DATE} to {END_DATE}...")
    data = yf.download(tickers, start=START_DATE, end=END_DATE, progress=False)

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

def backtest_100ma(prices: pd.DataFrame) -> dict:
    """Backtest: GLD above 100-day MA -> GLD, else IEF."""
    prices["GLD_100MA"] = prices["GLD"].rolling(window=100).mean()
    monthly = prices.resample("ME").last()
    
    positions = []
    monthly_rets = []
    
    for i in range(1, len(monthly)):
        gld_price = monthly.iloc[i]["GLD"]
        gld_100ma = monthly.iloc[i]["GLD_100MA"]
        
        if pd.notna(gld_100ma) and gld_price > gld_100ma:
            position = "GLD"
        else:
            position = "IEF"
        
        positions.append(position)
        
        if i + 1 < len(monthly):
            price_now = monthly.iloc[i][position]
            price_next = monthly.iloc[i + 1][position]
            monthly_ret = (price_next - price_now) / price_now if price_now != 0 else 0
            monthly_rets.append(monthly_ret)
    
    return_series = pd.Series(monthly_rets, index=monthly.index[2:len(monthly)])
    return {"returns": return_series}

def backtest_12m_return(prices: pd.DataFrame) -> dict:
    """Backtest: GLD 12M return > 0 -> GLD, else IEF."""
    monthly = prices.resample("ME").last()
    
    positions = []
    monthly_rets = []
    
    for i in range(1, len(monthly)):
        # Compute 12M return
        if i >= 12:
            gld_12m = (monthly.iloc[i]["GLD"] - monthly.iloc[i-12]["GLD"]) / monthly.iloc[i-12]["GLD"]
        else:
            gld_12m = 0
        
        if gld_12m > 0:
            position = "GLD"
        else:
            position = "IEF"
        
        positions.append(position)
        
        if i + 1 < len(monthly):
            price_now = monthly.iloc[i][position]
            price_next = monthly.iloc[i + 1][position]
            monthly_ret = (price_next - price_now) / price_now if price_now != 0 else 0
            monthly_rets.append(monthly_ret)
    
    return_series = pd.Series(monthly_rets, index=monthly.index[2:len(monthly)])
    return {"returns": return_series}

def backtest_gld_only(prices: pd.DataFrame) -> dict:
    """Benchmark: hold GLD always."""
    monthly = prices.resample("ME").last()
    monthly_rets = []
    
    for i in range(1, len(monthly)):
        if i + 1 < len(monthly):
            price_now = monthly.iloc[i]["GLD"]
            price_next = monthly.iloc[i + 1]["GLD"]
            monthly_ret = (price_next - price_now) / price_now if price_now != 0 else 0
            monthly_rets.append(monthly_ret)
    
    return_series = pd.Series(monthly_rets, index=monthly.index[2:len(monthly)])
    return {"returns": return_series}

def main():
    prices = download_prices(TICKERS)
    
    # Run all three strategies
    ma100_result = backtest_100ma(prices)
    ret12m_result = backtest_12m_return(prices)
    gld_result = backtest_gld_only(prices)
    
    ma100_ret = ma100_result["returns"]
    ret12m_ret = ret12m_result["returns"]
    gld_ret = gld_result["returns"]
    
    if len(ma100_ret) > 0 and len(ret12m_ret) > 0 and len(gld_ret) > 0:
        ma100_metrics = compute_metrics(ma100_ret)
        ret12m_metrics = compute_metrics(ret12m_ret)
        gld_metrics = compute_metrics(gld_ret)
        
        print("\n" + "="*110)
        print(f"GLD STRATEGY COMPARISON ({ma100_ret.index[0].strftime('%Y-%m-%d')} to {ma100_ret.index[-1].strftime('%Y-%m-%d')})")
        print("="*110)
        print(f"{'Metric':<20} {'100MA Filter':<22} {'12M Return Filter':<22} {'GLD (Benchmark)':<22}")
        print("-"*110)
        
        print(f"{'CAGR':<20} {ma100_metrics['cagr']:>20.2%} {ret12m_metrics['cagr']:>20.2%} {gld_metrics['cagr']:>20.2%}")
        print(f"{'Sharpe Ratio':<20} {ma100_metrics['sharpe']:>20.3f} {ret12m_metrics['sharpe']:>20.3f} {gld_metrics['sharpe']:>20.3f}")
        print(f"{'Max Drawdown':<20} {ma100_metrics['max_drawdown']:>20.2%} {ret12m_metrics['max_drawdown']:>20.2%} {gld_metrics['max_drawdown']:>20.2%}")
        print("="*110)
        
        print("\nSUMMARY vs GLD Benchmark:")
        print(f"  100MA Filter:    {ma100_metrics['cagr'] - gld_metrics['cagr']:>+.2%} CAGR, Sharpe {ma100_metrics['sharpe'] - gld_metrics['sharpe']:>+.3f}")
        print(f"  12M Filter:      {ret12m_metrics['cagr'] - gld_metrics['cagr']:>+.2%} CAGR, Sharpe {ret12m_metrics['sharpe'] - gld_metrics['sharpe']:>+.3f}")

if __name__ == "__main__":
    main()

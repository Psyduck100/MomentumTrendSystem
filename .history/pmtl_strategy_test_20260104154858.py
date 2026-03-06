"""PMTL Strategy: GLD with 200-day MA filter.

Logic:
- If GLD > 200-day SMA: hold GLD
- Else: hold IEF
- Monthly rebalance
- Compare against GLD benchmark
"""

from __future__ import annotations
import pandas as pd
import yfinance as yf
from pathlib import Path

from momentum_program.backtest.metrics import compute_metrics

TICKERS = ["GLD", "IEF"]
START_DATE = "2005-01-01"  # GLD inception Nov 2004
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

def backtest_pmtl(prices: pd.DataFrame) -> dict:
    """Backtest PMTL: GLD above 200MA -> GLD, else IEF."""
    # Compute 200-day SMA for GLD
    prices["GLD_200MA"] = prices["GLD"].rolling(window=200).mean()
    
    # Monthly rebalance
    monthly = prices.resample("ME").last()
    
    positions = []
    monthly_rets = []
    
    for i in range(1, len(monthly)):
        gld_price = monthly.iloc[i]["GLD"]
        gld_200ma = monthly.iloc[i]["GLD_200MA"]
        
        # Check if GLD is above 200MA
        if pd.notna(gld_200ma) and gld_price > gld_200ma:
            position = "GLD"
        else:
            position = "IEF"
        
        positions.append(position)
        
        # Return for next month (from i to i+1)
        if i + 1 < len(monthly):
            price_now = monthly.iloc[i][position]
            price_next = monthly.iloc[i + 1][position]
            monthly_ret = (price_next - price_now) / price_now if price_now != 0 else 0
            monthly_rets.append(monthly_ret)
    
    return_series = pd.Series(monthly_rets, index=monthly.index[2:len(monthly)])
    return {
        "positions": positions,
        "returns": return_series,
    }

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
    return {
        "returns": return_series,
    }

def main():
    prices = download_prices(TICKERS)
    
    # Run both strategies
    pmtl_result = backtest_pmtl(prices)
    gld_result = backtest_gld_only(prices)
    
    pmtl_ret = pmtl_result["returns"]
    gld_ret = gld_result["returns"]
    
    if len(pmtl_ret) > 0 and len(gld_ret) > 0:
        pmtl_metrics = compute_metrics(pmtl_ret)
        gld_metrics = compute_metrics(gld_ret)
        
        print("\n" + "="*90)
        print(f"PMTL vs GLD COMPARISON ({pmtl_ret.index[0].strftime('%Y-%m-%d')} to {pmtl_ret.index[-1].strftime('%Y-%m-%d')})")
        print("="*90)
        print(f"{'Metric':<20} {'PMTL (200MA)':<20} {'GLD (Benchmark)':<20} {'Advantage':<20}")
        print("-"*90)
        
        cagr_diff = pmtl_metrics['cagr'] - gld_metrics['cagr']
        sharpe_diff = pmtl_metrics['sharpe'] - gld_metrics['sharpe']
        dd_diff = pmtl_metrics['max_drawdown'] - gld_metrics['max_drawdown']
        
        print(f"{'CAGR':<20} {pmtl_metrics['cagr']:>18.2%} {gld_metrics['cagr']:>18.2%} {cagr_diff:>18.2%}")
        print(f"{'Sharpe Ratio':<20} {pmtl_metrics['sharpe']:>18.3f} {gld_metrics['sharpe']:>18.3f} {sharpe_diff:>18.3f}")
        print(f"{'Max Drawdown':<20} {pmtl_metrics['max_drawdown']:>18.2%} {gld_metrics['max_drawdown']:>18.2%} {dd_diff:>18.2%}")
        print("="*90)
        
        if cagr_diff > 0:
            print(f"[+] PMTL outperforms GLD by {cagr_diff:.2%} CAGR")
        else:
            print(f"[-] PMTL underperforms GLD by {abs(cagr_diff):.2%} CAGR")
        
        if sharpe_diff > 0:
            print(f"[+] PMTL has better risk-adjusted returns (Sharpe {sharpe_diff:+.3f})")
        else:
            print(f"[-] PMTL has worse risk-adjusted returns (Sharpe {sharpe_diff:+.3f})")
        
        if dd_diff < 0:
            print(f"[+] PMTL has lower max drawdown by {abs(dd_diff):.2%}")
        else:
            print(f"[-] PMTL has higher max drawdown by {dd_diff:.2%}")

if __name__ == "__main__":
    main()

"""Sweep MA filters from 100 to 200 trading days in 10-day bands.

Test both SMA and EMA variants, report CAGR per band.
Export results to CSV and benchmark vs GLD.
"""

from __future__ import annotations
import pandas as pd
import yfinance as yf
from pathlib import Path

from momentum_program.backtest.metrics import compute_metrics

TICKERS = ["GLD", "IEF"]
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

def backtest_sma_window(prices: pd.DataFrame, window: int) -> dict:
    """Backtest: GLD above SMA(window) -> GLD, else IEF. Returns full return series."""
    ma_column = f"GLD_SMA{window}"
    prices_copy = prices.copy()
    prices_copy[ma_column] = prices_copy["GLD"].rolling(window=window).mean()
    monthly = prices_copy.resample("ME").last()
    
    monthly_rets = []
    for i in range(1, len(monthly)):
        gld_price = monthly.iloc[i]["GLD"]
        gld_ma = monthly.iloc[i][ma_column]
        
        position = "GLD" if pd.notna(gld_ma) and gld_price > gld_ma else "IEF"
        
        if i + 1 < len(monthly):
            price_now = monthly.iloc[i][position]
            price_next = monthly.iloc[i + 1][position]
            monthly_ret = (price_next - price_now) / price_now if price_now != 0 else 0
            monthly_rets.append(monthly_ret)
    
    return_series = pd.Series(monthly_rets, index=monthly.index[2:len(monthly)])
    return {"returns": return_series, "monthly_returns": return_series}

def backtest_ema_window(prices: pd.DataFrame, window: int) -> dict:
    """Backtest: GLD above EMA(window) -> GLD, else IEF. Returns full return series."""
    ema_column = f"GLD_EMA{window}"
    prices_copy = prices.copy()
    prices_copy[ema_column] = prices_copy["GLD"].ewm(span=window, adjust=False).mean()
    monthly = prices_copy.resample("ME").last()
    
    monthly_rets = []
    for i in range(1, len(monthly)):
        gld_price = monthly.iloc[i]["GLD"]
        gld_ema = monthly.iloc[i][ema_column]
        
        position = "GLD" if pd.notna(gld_ema) and gld_price > gld_ema else "IEF"
        
        if i + 1 < len(monthly):
            price_now = monthly.iloc[i][position]
            price_next = monthly.iloc[i + 1][position]
            monthly_ret = (price_next - price_now) / price_now if price_now != 0 else 0
            monthly_rets.append(monthly_ret)
    
    return_series = pd.Series(monthly_rets, index=monthly.index[2:len(monthly)])
    return {"returns": return_series, "monthly_returns": return_series}

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
    prices = download_prices(TICKERS, end_date=END_DATE)
    
    # Benchmark GLD
    gld_result = backtest_gld_only(prices)
    gld_metrics = compute_metrics(gld_result["returns"])
    gld_cagr = gld_metrics["cagr"]
    
    results = []
    
    print("\nTesting MA windows from 100 to 200 trading days (10-day bands)...\n")
    print(f"{'Window':<8} {'SMA CAGR':>10} {'SMA vs GLD':>12} {'EMA CAGR':>10} {'EMA vs GLD':>12}")
    print("-" * 60)
    
    for window in range(100, 201, 10):
        # Test SMA
        sma_result = backtest_sma_window(prices, window)
        if len(sma_result["returns"]) > 0:
            sma_metrics = compute_metrics(sma_result["returns"])
            sma_cagr = sma_metrics["cagr"]
            sma_sharpe = sma_metrics["sharpe"]
            sma_maxdd = sma_metrics["max_drawdown"]
            sma_delta = sma_cagr - gld_cagr
        else:
            sma_cagr = None
            sma_sharpe = None
            sma_maxdd = None
            sma_delta = None
        
        # Test EMA
        ema_result = backtest_ema_window(prices, window)
        if len(ema_result["returns"]) > 0:
            ema_metrics = compute_metrics(ema_result["returns"])
            ema_cagr = ema_metrics["cagr"]
            ema_sharpe = ema_metrics["sharpe"]
            ema_maxdd = ema_metrics["max_drawdown"]
            ema_delta = ema_cagr - gld_cagr
        else:
            ema_cagr = None
            ema_sharpe = None
            ema_maxdd = None
            ema_delta = None
        
        results.append({
            "window": window,
            "sma_cagr": sma_cagr,
            "sma_sharpe": sma_sharpe,
            "sma_maxdd": sma_maxdd,
            "sma_vs_gld": sma_delta,
            "ema_cagr": ema_cagr,
            "ema_sharpe": ema_sharpe,
            "ema_maxdd": ema_maxdd,
            "ema_vs_gld": ema_delta,
        })
        
        print(f"{window:3d} days | SMA: {sma_cagr:>7.2%} ({sma_delta:>+6.2%}) | EMA: {ema_cagr:>7.2%} ({ema_delta:>+6.2%})")
    
    # Create DataFrame and export to CSV
    df = pd.DataFrame(results)
    df["gld_cagr"] = gld_cagr
    df["gld_sharpe"] = gld_metrics["sharpe"]
    df["gld_maxdd"] = gld_metrics["max_drawdown"]
    
    csv_path = Path("pmtl_ma_sweep_results.csv")
    df.to_csv(csv_path, index=False, float_format="%.4f")
    print(f"\nResults exported to {csv_path}")
    print(f"\nGLD Benchmark CAGR: {gld_cagr:.2%} | Sharpe: {gld_metrics['sharpe']:.3f} | MaxDD: {gld_metrics['max_drawdown']:.2%}")
    print(f"\nBest SMA window: {df.loc[df['sma_cagr'].idxmax(), 'window']:.0f} days ({df['sma_cagr'].max():.2%})")
    print(f"Best EMA window: {df.loc[df['ema_cagr'].idxmax(), 'window']:.0f} days ({df['ema_cagr'].max():.2%})")

if __name__ == "__main__":
    main()

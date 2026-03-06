"""
Backtest PMTL Gold SMA strategy vs Buy & Hold GLD
- Compares cumulative returns of the SMA strategy and GLD buy & hold
- Uses only data available up to each rebalance date (no lookforward bias)
"""

import pandas as pd
import yfinance as yf
from rebalance import run_strategy, TICKER, SMA_WINDOW, START_DATE, CASH_SYMBOL

import matplotlib.pyplot as plt

# Download daily prices
def download_prices(ticker: str, start_date: str = "2005-01-01") -> pd.Series:
    data = yf.download(ticker, start=start_date, progress=False)
    if "Adj Close" in data:
        return data["Adj Close"]
    return data["Close"]

def backtest_vs_gld(sma_window: int = SMA_WINDOW):
    prices = download_prices(TICKER, START_DATE)
    signals = run_strategy(sma_window)
    # Build a DataFrame of signals
    df = pd.DataFrame(signals, columns=["date", "signal", "price", "sma", "note"])
    df = df.set_index("date")
    # Forward fill signal until next rebalance
    df_full = pd.DataFrame(index=prices.index)
    df_full["signal"] = df["signal"].reindex(prices.index, method="ffill")
    df_full["price"] = prices
    # Calculate daily returns for strategy
    df_full["hold_gld"] = df_full["signal"] == TICKER
    df_full["ret"] = 0.0
    df_full.loc[df_full["hold_gld"], "ret"] = prices.pct_change().loc[df_full["hold_gld"]]
    # Buy & hold GLD returns
    df_full["gld_ret"] = prices.pct_change()
    # Cumulative returns
    df_full["strat_cum"] = (1 + df_full["ret"]).cumprod()
    df_full["gld_cum"] = (1 + df_full["gld_ret"]).cumprod()
    # Plot
    plt.figure(figsize=(10,6))
    plt.plot(df_full.index, df_full["strat_cum"], label=f"PMTL SMA {sma_window}d")
    plt.plot(df_full.index, df_full["gld_cum"], label="GLD Buy & Hold", alpha=0.7)
    plt.title(f"PMTL Gold SMA Strategy vs GLD Buy & Hold (SMA={sma_window}d)")
    plt.ylabel("Cumulative Return")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.show()

    # Calculate CAGR for both strategies
    n_years = (df_full.index[-1] - df_full.index[0]).days / 365.25
    strat_cagr = df_full['strat_cum'].iloc[-1] ** (1/n_years) - 1
    gld_cagr = df_full['gld_cum'].iloc[-1] ** (1/n_years) - 1
    print(f"Final PMTL SMA return: {df_full['strat_cum'].iloc[-1]:.2f}x")
    print(f"Final GLD return: {df_full['gld_cum'].iloc[-1]:.2f}x")
    print(f"PMTL SMA CAGR: {strat_cagr*100:.2f}%")
    print(f"GLD Buy & Hold CAGR: {gld_cagr*100:.2f}%")

if __name__ == "__main__":
    backtest_vs_gld()

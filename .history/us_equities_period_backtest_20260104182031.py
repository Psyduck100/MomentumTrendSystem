"""Backtest US equities strategy for specific period: Jan 2008 - Sep 2025."""

from __future__ import annotations
import pandas as pd
import yfinance as yf
from pathlib import Path

from momentum_program.backtest.metrics import compute_metrics

# Load universe from CSV
UNIVERSE_CSV = Path("CSVs/US_equities.csv")
DEFENSIVE_SYMBOL = "IEF"


def load_universe() -> list[str]:
    """Load the US equities universe from CSV file."""
    df = pd.read_csv(UNIVERSE_CSV, encoding="latin-1")
    return df["ticker"].tolist()


UNIVERSE = load_universe()
ALL_TICKERS = UNIVERSE + [DEFENSIVE_SYMBOL]

START_DATE = "2008-01-01"
END_DATE = "2025-09-30"


def download_prices(tickers: list[str]) -> pd.DataFrame:
    """Download daily prices for all tickers."""
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


def backtest_us_equities(prices: pd.DataFrame) -> dict:
    """Backtest US equities with blend_6_12 and ret_12m filter."""
    monthly = prices.resample("ME").last()
    positions = []
    monthly_rets = []

    # Start from month 13 (first valid 12M return)
    for i in range(12, len(monthly)):
        # Compute blend_6_12 and ret_12m for this month
        ret_6m = monthly.pct_change(6).iloc[i]
        ret_12m = monthly.pct_change(12).iloc[i]
        blend_6_12 = 0.5 * ret_6m + 0.5 * ret_12m

        # Get scores for US equities only (excluding IEF)
        us_blend = blend_6_12[UNIVERSE].dropna()
        us_12m = ret_12m[UNIVERSE].dropna()

        if len(us_blend) == 0 or len(us_12m) == 0:
            position = DEFENSIVE_SYMBOL
        else:
            # Find best US equity by blend_6_12
            best_ticker = us_blend.idxmax()
            best_12m = us_12m[best_ticker]

            # Apply ret_12m filter: if best 12M > 0, take it; else IEF
            if best_12m > 0:
                position = best_ticker
            else:
                position = DEFENSIVE_SYMBOL

        positions.append(position)

        # Return for next month (from i to i+1)
        if i + 1 < len(monthly):
            price_now = monthly.iloc[i][position]
            price_next = monthly.iloc[i + 1][position]
            monthly_ret = (price_next - price_now) / price_now if price_now != 0 else 0
            monthly_rets.append(monthly_ret)

    return_series = pd.Series(monthly_rets, index=monthly.index[13 : len(monthly)])
    return {
        "positions": positions,
        "returns": return_series,
    }


def main():
    prices = download_prices(ALL_TICKERS)
    result = backtest_us_equities(prices)
    ret_series = result["returns"]

    if len(ret_series) > 0:
        metrics = compute_metrics(ret_series)

        print("\n" + "=" * 80)
        print(f"US EQUITIES STRATEGY: Jan 2008 - Sep 2025")
        print("=" * 80)
        print(f"CAGR:          {metrics['cagr']:.2%}")
        print(f"Sharpe Ratio:  {metrics['sharpe']:.3f}")
        print(f"Max Drawdown:  {metrics['max_drawdown']:.2%}")
        print(f"Start Date:    {ret_series.index[0].strftime('%Y-%m-%d')}")
        print(f"End Date:      {ret_series.index[-1].strftime('%Y-%m-%d')}")
        print("=" * 80)


if __name__ == "__main__":
    main()

"""Monthly US Equities Momentum Rebalance Script

Strategy Configuration:
- Universe: Loaded from CSVs/US_equities.csv
- Scoring: blend_6_12 (50% 6-month + 50% 12-month returns)
- Rank Gap: 0 (allow switching between any positions)
- Absolute Filter: ret_12m (require positive 12-month return, else defensive)
- Defensive Allocation: IEF (7-10 Year Treasury ETF)
- Position: Single best-ranked ticker

Run this script at month-end to get the recommended position for next month.
"""

from __future__ import annotations
import pandas as pd
from datetime import datetime
from pathlib import Path
import yfinance as yf

# Strategy Parameters
UNIVERSE_CSV = Path("CSVs/US_equities.csv")
DEFENSIVE_SYMBOL = "IEF"

# Load universe from CSV
def load_universe() -> list[str]:
    """Load the US equities universe from CSV file."""
    df = pd.read_csv(UNIVERSE_CSV)
    return df['ticker'].tolist()

UNIVERSE = load_universe()
ALL_TICKERS = UNIVERSE + [DEFENSIVE_SYMBOL]

# Momentum weights for blend_6_12
LOOKBACK_MONTHS = [6, 12]
LOOKBACK_WEIGHTS = [0.5, 0.5]

# Cache directory
CACHE_DIR = Path("backtest_cache")
CACHE_DIR.mkdir(exist_ok=True)


def download_prices(tickers: list[str], start_date: str = "2000-01-01") -> pd.DataFrame:
    """Download daily prices for all tickers."""
    today = datetime.now().strftime("%Y-%m-%d")
    print(f"Downloading price data for {len(tickers)} tickers from {start_date} to {today}...")
    
    data = yf.download(tickers, start=start_date, end=today, progress=False)
    
    # Handle yfinance output format
    if isinstance(data.columns, pd.MultiIndex):
        # Multi-ticker: swap levels and extract Adj Close
        data = data.swaplevel(0, 1, axis=1).sort_index(axis=1)
        prices = pd.DataFrame()
        for ticker in tickers:
            if ticker in data.columns.get_level_values(0):
                sub = data[ticker]
                if 'Adj Close' in sub.columns:
                    prices[ticker] = sub['Adj Close']
                elif 'Close' in sub.columns:
                    prices[ticker] = sub['Close']
    else:
        # Single ticker
        prices = pd.DataFrame({tickers[0]: data['Adj Close']})
    
    return prices


def compute_monthly_returns(daily_prices: pd.DataFrame) -> pd.DataFrame:
    """Resample daily prices to month-end and compute returns."""
    monthly = daily_prices.resample('ME').last()
    returns = {}
    
    for months in LOOKBACK_MONTHS:
        ret_col = monthly.pct_change(months)
        # If single-column DataFrame, extract the series
        if isinstance(ret_col, pd.DataFrame) and len(ret_col.columns) == 1:
            returns[f'ret_{months}m'] = ret_col.iloc[:, 0]
        else:
            returns[f'ret_{months}m'] = ret_col
    
    return pd.DataFrame(returns, index=monthly.index)


def compute_blend_score(returns_df: pd.DataFrame) -> pd.Series:
    """Compute blend_6_12 score: 50% 6M + 50% 12M returns."""
    score = sum(w * returns_df[f'ret_{m}m'] for w, m in zip(LOOKBACK_WEIGHTS, LOOKBACK_MONTHS))
    return score


def get_current_recommendation() -> dict:
    """Get the current month's position recommendation."""
    
    # Download data
    prices = download_prices(ALL_TICKERS)
    
    # Compute monthly returns for each ticker
    all_returns = {}
    for ticker in ALL_TICKERS:
        ticker_prices = prices[[ticker]].copy()
        ticker_prices.columns = ['price']
        ticker_returns = compute_monthly_returns(ticker_prices)
        all_returns[ticker] = ticker_returns
    
    # Get latest complete month
    latest_date = prices.resample('M').last().index[-1]
    print(f"\nAnalyzing data as of {latest_date.strftime('%Y-%m-%d')}")
    print("=" * 60)
    
    # Score all universe tickers
    scores = {}
    ret_12m_values = {}
    
    for ticker in UNIVERSE:
        rets = all_returns[ticker]
        if latest_date not in rets.index:
            continue
            
        score = compute_blend_score(rets.loc[[latest_date]]).iloc[0]
        ret_12m = rets.loc[latest_date, 'ret_12m']
        
        scores[ticker] = score
        ret_12m_values[ticker] = ret_12m
    
    # Sort by score descending
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    
    # Print rankings
    print("\nUniverse Rankings (blend_6_12 score):")
    print("-" * 60)
    for i, (ticker, score) in enumerate(ranked, 1):
        ret_12m = ret_12m_values[ticker]
        print(f"{i}. {ticker:5s}  Score: {score:7.2%}  12M Return: {ret_12m:7.2%}")
    
    # Apply absolute filter (ret_12m must be positive)
    best_ticker, best_score = ranked[0]
    best_12m_return = ret_12m_values[best_ticker]
    
    if best_12m_return > 0:
        recommendation = best_ticker
        reason = f"Top-ranked with positive 12M return ({best_12m_return:.2%})"
    else:
        recommendation = DEFENSIVE_SYMBOL
        reason = f"Defensive allocation: {best_ticker} 12M return is negative ({best_12m_return:.2%})"
    
    # Get defensive ticker info
    defensive_ret_12m = all_returns[DEFENSIVE_SYMBOL].loc[latest_date, 'ret_12m']
    
    print("\n" + "=" * 60)
    print(f"\n📍 RECOMMENDED POSITION: {recommendation}")
    print(f"   Reason: {reason}")
    if recommendation == DEFENSIVE_SYMBOL:
        print(f"   {DEFENSIVE_SYMBOL} 12M Return: {defensive_ret_12m:.2%}")
    print("\n" + "=" * 60)
    
    return {
        'date': latest_date.strftime('%Y-%m-%d'),
        'recommendation': recommendation,
        'reason': reason,
        'top_ranked': best_ticker,
        'top_score': best_score,
        'top_12m_return': best_12m_return,
        'defensive_12m_return': defensive_ret_12m,
        'rankings': ranked
    }


def main():
    """Run the monthly rebalance analysis."""
    print("=" * 60)
    print("US EQUITIES MOMENTUM STRATEGY - MONTHLY REBALANCE")
    print("=" * 60)
    print(f"\nUniverse: {', '.join(UNIVERSE)}")
    print(f"Defensive: {DEFENSIVE_SYMBOL}")
    print(f"Scoring: blend_6_12 (50% 6M + 50% 12M returns)")
    print(f"Filter: Positive 12-month return required")
    print()
    
    result = get_current_recommendation()
    
    return result


if __name__ == "__main__":
    main()

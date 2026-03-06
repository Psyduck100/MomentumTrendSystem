import pandas as pd
import yfinance as yf
import numpy as np
from datetime import datetime


def backtest_momentum(
    tickers: list[str],
    bucket_map: dict[str, str],
    start_date: str,
    end_date: str,
    top_n_per_bucket: int = 1,
    lookback_long: int = 6,
    lookback_short: int = 1,
) -> dict:
    """
    Run a monthly rebalancing momentum backtest by bucket.

    Args:
        tickers: List of ETF symbols.
        bucket_map: Dict mapping symbol -> bucket name.
        start_date: Start date (YYYY-MM-DD).
        end_date: End date (YYYY-MM-DD).
        top_n_per_bucket: Number of top picks per bucket each month.
        lookback_long: Long lookback window in months (e.g., 6).
        lookback_short: Short lookback window in months (e.g., 1).

    Returns:
        Dict with per-bucket and overall results.
    """
    # Download all data at once to avoid rate limits
    print(f"Downloading {len(tickers)} tickers from {start_date} to {end_date}...")
    data = yf.download(tickers, start=start_date, end=end_date, progress=False)[
        "Adj Close"
    ]
    if isinstance(data, pd.Series):
        data = data.to_frame()
    
    # Filter out tickers with insufficient data (NaN during backtest period)
    # Keep only tickers with >80% non-null data
    valid_tickers = []
    for ticker in data.columns:
        non_null_pct = data[ticker].notna().sum() / len(data)
        if non_null_pct >= 0.8:
            valid_tickers.append(ticker)
        else:
            print(f"  Skipping {ticker} ({non_null_pct:.1%} data available)")
    
    data = data[valid_tickers]
    print(f"Using {len(valid_tickers)} tickers with sufficient history")

    # Resample to monthly (last day of month)
    print("Resampling to monthly...")
    monthly = data.resample("M").last()

    # Compute momentum: lookback_long return - lookback_short return
    print("Computing momentum...")
    ret_long = monthly.pct_change(lookback_long)
    ret_short = monthly.pct_change(lookback_short)
    momentum = ret_long - ret_short

    # Build bucket list
    buckets = set(bucket_map.values())

    # Backtest
    print("Running backtest...")
    portfolio_returns_overall = []
    bucket_returns = {b: [] for b in buckets}
    bucket_positions = {b: [] for b in buckets}
    overall_positions = []

    for i in range(max(lookback_long, lookback_short), len(momentum)):
        date = momentum.index[i]
        momentum_scores = momentum.iloc[i]

        # Group by bucket and pick top 1 per bucket (hold only top 1, display top N)
        selected_symbols = []
        for bucket in buckets:
            bucket_symbols = [s for s in valid_tickers if bucket_map.get(s) == bucket]
            if not bucket_symbols:
                continue
            bucket_momentum = momentum_scores[bucket_symbols]
            # Skip NaN scores and get top N
            valid_scores = bucket_momentum.dropna()
            if valid_scores.empty:
                continue
            top_symbols = valid_scores.nlargest(top_n_per_bucket).index.tolist()
            # Hold only top 1, but track top N for display
            selected_symbols.extend(top_symbols[:1])
            bucket_positions[bucket].append(top_symbols)

        if not selected_symbols:
            continue

        overall_positions.append(selected_symbols)

        # Get next month's returns
        if i + 1 >= len(monthly):
            break

        next_date = momentum.index[i + 1]
        weights = np.array([1 / len(selected_symbols)] * len(selected_symbols))

        # Next month returns for selected symbols
        next_returns = monthly.loc[next_date, selected_symbols]
        if isinstance(next_returns, pd.Series):
            next_returns = next_returns.values

        # Portfolio return (equal weight)
        portfolio_ret = np.mean(next_returns)  # Equal weight simplification
        portfolio_returns_overall.append(
            {
                "date": next_date,
                "return": portfolio_ret,
                "symbols": selected_symbols,
            }
        )

        # Per-bucket returns
        for bucket in buckets:
            bucket_symbols = [
                s for s in selected_symbols if bucket_map.get(s) == bucket
            ]
            if bucket_symbols:
                bucket_ret = monthly.loc[next_date, bucket_symbols]
                if isinstance(bucket_ret, pd.Series):
                    bucket_ret = np.mean(bucket_ret.values)
                else:
                    bucket_ret = np.mean(bucket_ret)
                bucket_returns[bucket].append(
                    {
                        "date": next_date,
                        "return": bucket_ret,
                        "symbols": bucket_symbols,
                    }
                )

    # Convert to DataFrames
    df_overall = pd.DataFrame(portfolio_returns_overall)
    if not df_overall.empty:
        df_overall.set_index("date", inplace=True)

    bucket_dfs = {}
    for bucket, rets in bucket_returns.items():
        if rets:
            df = pd.DataFrame(rets)
            df.set_index("date", inplace=True)
            bucket_dfs[bucket] = df
        else:
            bucket_dfs[bucket] = pd.DataFrame()

    return {
        "overall_returns": df_overall,
        "bucket_returns": bucket_dfs,
        "bucket_positions": bucket_positions,
        "overall_positions": overall_positions,
        "momentum": momentum,
        "monthly_prices": monthly,
        "tickers": tickers,
        "bucket_map": bucket_map,
    }

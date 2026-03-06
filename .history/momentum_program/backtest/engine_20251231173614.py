import pandas as pd
import yfinance as yf
import numpy as np
import time
from datetime import datetime
from pathlib import Path


def backtest_momentum(
    tickers: list[str],
    bucket_map: dict[str, str],
    start_date: str,
    end_date: str,
    top_n_per_bucket: int = 1,
    lookback_long: int = 6,
    lookback_short: int = 1,
    cache_dir: Path = Path("backtest_cache"),
    slippage_bps: float = 3.0,
    expense_ratio: float = 0.0001,
    vol_adjusted: bool = False,
    vol_lookback: int = 6,
    market_filter: bool = False,
    market_ticker: str = "SPY",
    defensive_bucket: str = "Bonds",
    market_threshold: float = 0.0,
    rank_gap_threshold: int = 0,
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
        cache_dir: Directory to cache downloaded data.
        slippage_bps: Trading slippage in basis points (default 3 bps per trade).
        expense_ratio: Annual expense ratio drag (default 0.01% annualized).
        vol_adjusted: Use volatility-adjusted momentum (score / volatility).
        vol_lookback: Months of data for rolling volatility calculation (default 6).
        market_filter: Apply absolute momentum filter (dual momentum defense).
        market_ticker: Market index to check for absolute momentum (default SPY).
        defensive_bucket: Bucket to use when market momentum is negative (default Bonds).
        market_threshold: Minimum 12M return to stay risk-on (e.g., 0.0 for >0%, -0.03 for >-3%).
        rank_gap_threshold: Only rotate if new leader beats current holding by this many ranks (reduces turnover).

    Returns:
        Dict with per-bucket and overall results.
    """
    cache_dir.mkdir(exist_ok=True)
    cache_file = cache_dir / f"price_data_{start_date}_{end_date}.csv"

    def _download_prices(symbols: list[str]) -> pd.DataFrame:
        frames: list[pd.DataFrame] = []
        chunk_size = 25
        for i in range(0, len(symbols), chunk_size):
            chunk = symbols[i : i + chunk_size]
            print(
                f"Downloading {len(chunk)} tickers ({chunk[0]}..{chunk[-1]}) from {start_date} to {end_date}..."
            )
            df_chunk = pd.DataFrame()
            for attempt in range(2):
                try:
                    df_chunk = yf.download(
                        chunk, start=start_date, end=end_date, progress=False
                    )
                except Exception as exc:  # noqa: BLE001
                    if attempt == 0:
                        print(
                            f"  Error for chunk starting {chunk[0]} ({exc}); retrying after 5s"
                        )
                        time.sleep(5)
                        continue
                    print(f"  Failed chunk starting {chunk[0]} ({exc}); skipping")
                    df_chunk = pd.DataFrame()
                if df_chunk.empty and attempt == 0:
                    print(
                        f"  Empty response for chunk starting {chunk[0]}; retrying after 5s"
                    )
                    time.sleep(5)
                    continue
                break

            if df_chunk.empty:
                continue

            if isinstance(df_chunk.columns, pd.MultiIndex):
                lvl0 = df_chunk.columns.get_level_values(0)
                if "Adj Close" in lvl0:
                    df_chunk = df_chunk["Adj Close"]
                elif "Close" in lvl0:
                    df_chunk = df_chunk["Close"]
                else:
                    df_chunk = df_chunk[lvl0[0]]
            elif "Adj Close" in df_chunk.columns:
                df_chunk = df_chunk["Adj Close"]
            elif "Close" in df_chunk.columns:
                df_chunk = df_chunk["Close"]

            if isinstance(df_chunk, pd.Series):
                df_chunk = df_chunk.to_frame()

            frames.append(df_chunk)

        if not frames:
            return pd.DataFrame()

        combined = pd.concat(frames, axis=1)
        # Drop duplicate columns that can appear across chunks
        combined = combined.loc[:, ~combined.columns.duplicated()]
        return combined

    # Try to load from cache; if empty or missing, download in chunks
    data = pd.DataFrame()
    if cache_file.exists():
        print(f"Loading cached price data from {cache_file}...")
        data = pd.read_csv(cache_file, index_col=0, parse_dates=True)

    if data.empty:
        data = _download_prices(tickers)
        if data.empty:
            print("Download returned no data; aborting backtest.")
            return {
                "overall_returns": pd.DataFrame(),
                "bucket_returns": {},
                "bucket_positions": {},
                "overall_positions": [],
                "momentum": pd.DataFrame(),
                "monthly_prices": pd.DataFrame(),
                "tickers": [],
                "bucket_map": bucket_map,
            }
        print(f"Saving price data to cache: {cache_file}")
        data.to_csv(cache_file)

    # Filter out tickers with insufficient data (NaN during backtest period)
    # Keep only tickers with >80% non-null data
    valid_tickers = []
    for ticker in data.columns:
        non_null_pct = data[ticker].notna().sum() / len(data)
        if non_null_pct >= 0.8:
            valid_tickers.append(ticker)
        else:
            print(f"  Skipping {ticker} ({non_null_pct:.1%} data available)")

    if not valid_tickers:
        print("No tickers with sufficient history after filtering.")
        return {
            "overall_returns": pd.DataFrame(),
            "bucket_returns": {},
            "bucket_positions": {},
            "overall_positions": [],
            "momentum": pd.DataFrame(),
            "monthly_prices": pd.DataFrame(),
            "tickers": [],
            "bucket_map": bucket_map,
        }

    data = data[valid_tickers]
    print(f"Using {len(valid_tickers)} tickers with sufficient history")

    # Resample to monthly (end of month)
    print("Resampling to monthly...")
    monthly = data.resample("ME").last()
    monthly_returns = monthly.pct_change()

    # Compute momentum: lookback_long return - lookback_short return
    print("Computing momentum...")
    ret_long = monthly.pct_change(lookback_long)
    ret_short = monthly.pct_change(lookback_short)
    momentum = ret_long - ret_short

    if vol_adjusted:
        print(f"Applying volatility adjustment (lookback={vol_lookback}M)...")
        # Compute rolling volatility (annualized)
        rolling_vol = monthly_returns.rolling(vol_lookback).std() * np.sqrt(12)
        # Divide momentum by volatility (replace inf/nan with 0)
        momentum_vol_adj = momentum / rolling_vol
        momentum_vol_adj = momentum_vol_adj.replace([np.inf, -np.inf], 0).fillna(0)
        momentum = momentum_vol_adj

    # Build bucket list
    buckets = set(bucket_map.values())

    # Backtest
    print("Running backtest...")
    portfolio_returns_overall = []
    bucket_returns = {b: [] for b in buckets}
    bucket_positions = {b: [] for b in buckets}
    overall_positions = []
    prev_holdings = set()
    prev_bucket_holdings = {b: set() for b in buckets}
    prev_bucket_selection: dict[str, str | None] = {b: None for b in buckets}

    for i in range(max(lookback_long, lookback_short), len(momentum)):
        date = momentum.index[i]
        momentum_scores = momentum.iloc[i]

        # Check market absolute momentum if enabled (require full 12M history)
        market_is_positive = True
        if market_filter and market_ticker in valid_tickers and i >= 12:
            price_now = monthly.iloc[i][market_ticker]
            price_12m_ago = monthly.iloc[i - 12][market_ticker]
            if pd.notna(price_now) and pd.notna(price_12m_ago) and price_12m_ago != 0:
                market_momentum = (price_now - price_12m_ago) / price_12m_ago
                market_is_positive = market_momentum > market_threshold

        # Group by bucket and pick top 1 per bucket (hold only top 1, display top N)
        selected_symbols = []
        for bucket in buckets:
            # Apply market filter: if market negative, only select from defensive bucket
            if market_filter and not market_is_positive and bucket != defensive_bucket:
                bucket_positions[bucket].append([])
                continue

            bucket_symbols = [s for s in valid_tickers if bucket_map.get(s) == bucket]
            if not bucket_symbols:
                continue
            bucket_momentum = momentum_scores[bucket_symbols]
            # Skip NaN scores and get top N
            valid_scores = bucket_momentum.dropna()
            if valid_scores.empty:
                continue
            ranked_symbols = valid_scores.sort_values(ascending=False).index.tolist()
            top_symbols = ranked_symbols[:top_n_per_bucket]
            bucket_positions[bucket].append(top_symbols)

            # Turnover control: keep current unless new leader beats by rank_gap_threshold
            leader = top_symbols[0]
            current = prev_bucket_selection.get(bucket)
            if rank_gap_threshold > 0 and current in ranked_symbols:
                leader_rank = ranked_symbols.index(leader)
                current_rank = ranked_symbols.index(current)
                # replace only if new is ahead by rank_gap_threshold ranks
                if leader_rank > current_rank - rank_gap_threshold:
                    leader = current

            selected_symbols.append(leader)
            prev_bucket_selection[bucket] = leader

        if not selected_symbols:
            continue

        overall_positions.append(selected_symbols)

        # Get next month's returns
        if i + 1 >= len(monthly):
            break

        next_date = momentum.index[i + 1]

        # Next month returns for selected symbols (already pct_change)
        next_returns = monthly_returns.loc[next_date, selected_symbols]
        if isinstance(next_returns, pd.Series):
            next_returns = next_returns.values

        # Drop NaNs and compute equal-weight return
        next_returns = next_returns[~pd.isna(next_returns)]
        if len(next_returns) == 0:
            continue
        portfolio_ret = np.mean(next_returns)

        # Apply transaction costs
        curr_holdings = set(selected_symbols)
        trades = len(curr_holdings.symmetric_difference(prev_holdings))
        slippage_cost = (
            (trades / len(selected_symbols)) * (slippage_bps / 10000)
            if len(selected_symbols) > 0
            else 0.0
        )

        # Apply monthly expense ratio drag
        expense_drag = expense_ratio / 12

        # Net return after costs
        net_return = portfolio_ret - slippage_cost - expense_drag
        prev_holdings = curr_holdings

        portfolio_returns_overall.append(
            {
                "date": next_date,
                "return": net_return,
                "gross_return": portfolio_ret,
                "slippage": slippage_cost,
                "expense": expense_drag,
                "symbols": selected_symbols,
            }
        )

        # Per-bucket returns
        for bucket in buckets:
            bucket_symbols = [
                s for s in selected_symbols if bucket_map.get(s) == bucket
            ]
            if bucket_symbols:
                bucket_ret = monthly_returns.loc[next_date, bucket_symbols]
                if isinstance(bucket_ret, pd.Series):
                    bucket_ret_gross = bucket_ret.dropna().mean()
                else:
                    bucket_ret_gross = np.nanmean(bucket_ret)

                # Apply bucket-specific transaction costs
                curr_bucket_holdings = set(bucket_symbols)
                bucket_trades = len(
                    curr_bucket_holdings.symmetric_difference(
                        prev_bucket_holdings[bucket]
                    )
                )
                bucket_slippage = (
                    (bucket_trades / len(bucket_symbols)) * (slippage_bps / 10000)
                    if len(bucket_symbols) > 0
                    else 0.0
                )
                bucket_expense = expense_ratio / 12
                bucket_ret_net = bucket_ret_gross - bucket_slippage - bucket_expense
                prev_bucket_holdings[bucket] = curr_bucket_holdings

                bucket_returns[bucket].append(
                    {
                        "date": next_date,
                        "return": bucket_ret_net,
                        "gross_return": bucket_ret_gross,
                        "slippage": bucket_slippage,
                        "expense": bucket_expense,
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
        "tickers": valid_tickers,
        "bucket_map": bucket_map,
    }

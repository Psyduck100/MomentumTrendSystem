"""
Generate detailed transaction log for a specific backtest configuration.
Shows when each stock was bought, held, and sold with dates and performance.
"""

from pathlib import Path
import pandas as pd
import numpy as np
from momentum_program.config import AppConfig
from momentum_program.analytics.constants import SCORE_MODE_RW_3_6_9_12
from momentum_program.universe.bucket_csv_provider import BucketedCsvUniverseProvider
from momentum_program.backtest.engine import backtest_momentum


def run_transaction_log() -> None:
    """Generate detailed transaction log for rw_3_6_9_12, gap=2, no filter."""
    cfg = AppConfig()
    bucket_folder = Path("CSVs")

    # Load universe
    universe = BucketedCsvUniverseProvider(bucket_folder)
    tickers = universe.get_tickers()
    bucket_map = universe.get_bucket_map()

    # Substitute QQQM with QQQ for historical data availability
    tickers = [t if t != "QQQM" else "QQQ" for t in tickers]
    bucket_map = {(k if k != "QQQM" else "QQQ"): v for k, v in bucket_map.items()}

    print("=" * 100)
    print("DETAILED TRANSACTION LOG")
    print(
        "Config: vol_adj=False, score=rw_3_6_9_12, thr=none, gap=2, abs=rw_3_6_9_12@0.00%"
    )
    print(f"Period: 2015-01-01 to 2025-12-31")
    print(f"Note: QQQM substituted with QQQ for historical data")
    print("=" * 100)
    print()

    # Run backtest with specific configuration
    backtest_data = backtest_momentum(
        tickers=tickers,
        bucket_map=bucket_map,
        start_date="2015-01-01",
        end_date="2025-12-31",
        top_n_per_bucket=cfg.strategy.top_n_per_bucket or cfg.strategy.top_n,
        lookback_long=12,
        lookback_short=1,
        vol_adjusted=False,
        vol_lookback=6,
        market_filter=False,
        market_ticker="SPY",
        defensive_bucket="Bonds",
        market_threshold=0.0,
        rank_gap_threshold=2,  # Per-bucket rank gap
        score_mode=SCORE_MODE_RW_3_6_9_12,
        abs_filter_mode=SCORE_MODE_RW_3_6_9_12,
        abs_filter_band=0.0,
    )

    # Extract position history from backtest data
    overall_positions = backtest_data.get("overall_positions", [])
    overall_returns = backtest_data.get("overall_returns", pd.DataFrame())
    bucket_returns_dict = backtest_data.get("bucket_returns", {})

    if overall_positions is None or len(overall_positions) == 0:
        print("No positions generated in backtest.")
        return

    # Build position history with dates
    dates = []
    if not overall_returns.empty:
        dates = overall_returns.index.tolist()

    if len(dates) != len(overall_positions):
        print(
            f"Warning: Date mismatch ({len(dates)} dates vs {len(overall_positions)} positions)"
        )

    # Create transaction log
    current_holdings = {}  # bucket -> symbol
    transaction_list = []

    for month_idx, (date, positions) in enumerate(zip(dates, overall_positions)):
        # positions is a list of symbols (one per bucket)
        # We need to figure out the bucket for each symbol
        month_holdings = {}

        for symbol in positions:
            bucket = bucket_map.get(symbol, "Unknown")
            month_holdings[bucket] = symbol

        # Detect rotations (changes from previous month)
        for bucket, symbol in month_holdings.items():
            prev_symbol = current_holdings.get(bucket)

            if prev_symbol is None:
                # Initial position
                action = "BUY"
                transaction_list.append(
                    {
                        "date": date,
                        "bucket": bucket,
                        "action": action,
                        "symbol": symbol,
                        "prev_symbol": None,
                        "month_idx": month_idx,
                    }
                )
            elif prev_symbol != symbol:
                # Rotation
                action = "ROTATE"
                transaction_list.append(
                    {
                        "date": date,
                        "bucket": bucket,
                        "action": action,
                        "symbol": symbol,
                        "prev_symbol": prev_symbol,
                        "month_idx": month_idx,
                    }
                )
            # else: HOLD (no transaction logged)

        # Detect holds
        for bucket, symbol in current_holdings.items():
            if bucket not in month_holdings:
                # Sold/exited
                action = "SELL"
                transaction_list.append(
                    {
                        "date": date,
                        "bucket": bucket,
                        "action": action,
                        "symbol": None,
                        "prev_symbol": symbol,
                        "month_idx": month_idx,
                    }
                )

        current_holdings = month_holdings

    # Print transaction log grouped by bucket
    print("\nTRANSACTION HISTORY BY BUCKET:")
    print("=" * 100)

    buckets = sorted(set(t["bucket"] for t in transaction_list))

    for bucket in buckets:
        bucket_txns = [t for t in transaction_list if t["bucket"] == bucket]

        print(f"\n{bucket}:")
        print("-" * 100)
        print(
            f"{'Date':<12} {'Action':<10} {'Symbol':<8} {'From':<8} {'Description':<40}"
        )
        print("-" * 100)

        for txn in bucket_txns:
            date_str = txn["date"].strftime("%Y-%m-%d")
            action = txn["action"]
            symbol = txn["symbol"] or "-"
            prev = txn["prev_symbol"] or "-"

            if action == "BUY":
                desc = f"Initial position: {symbol}"
            elif action == "ROTATE":
                desc = f"Rotated from {txn['prev_symbol']} to {txn['symbol']}"
            elif action == "SELL":
                desc = f"Exited position in {txn['prev_symbol']}"
            else:
                desc = ""

            print(f"{date_str:<12} {action:<10} {symbol:<8} {prev:<8} {desc:<40}")

    # Summary statistics
    print("\n" + "=" * 100)
    print("SUMMARY STATISTICS:")
    print("=" * 100)

    # Count rotations per bucket
    for bucket in buckets:
        bucket_txns = [
            t
            for t in transaction_list
            if t["bucket"] == bucket and t["action"] == "ROTATE"
        ]
        buys = [
            t
            for t in transaction_list
            if t["bucket"] == bucket and t["action"] == "BUY"
        ]
        print(
            f"{bucket:<20} | Initial Buys: {len(buys):<3} | Rotations: {len(bucket_txns):<3} | Total Activity: {len(buys) + len(bucket_txns)}"
        )

    total_rotations = len([t for t in transaction_list if t["action"] == "ROTATE"])
    total_buys = len([t for t in transaction_list if t["action"] == "BUY"])
    print(f"\nTotal Rotations: {total_rotations}")
    print(f"Total Initial Buys: {total_buys}")
    print(f"Backtest Months: {len(dates)}")
    print(f"Avg Rotations Per Month: {total_rotations / len(dates):.2f}")

    # Per-bucket performance summary
    print("\n" + "=" * 100)
    print("PER-BUCKET PERFORMANCE (from backtest):")
    print("=" * 100)

    for bucket, df_returns in bucket_returns_dict.items():
        if df_returns.empty:
            print(f"{bucket:<20} | No data")
            continue

        returns = df_returns["return"].dropna()
        if len(returns) == 0:
            continue

        cagr = (
            (1 + returns).prod() ** (12 / len(returns)) - 1 if len(returns) > 0 else 0
        )
        sharpe = (
            returns.mean() / returns.std() * np.sqrt(12) if returns.std() > 0 else 0
        )
        max_dd = (returns + 1).cumprod().expanding().max()
        max_dd = ((returns + 1).cumprod() / max_dd - 1).min()

        print(
            f"{bucket:<20} | CAGR: {cagr:>7.2%} | Sharpe: {sharpe:>6.2f} | MaxDD: {max_dd:>8.2%}"
        )

    print("\n" + "=" * 100)
    print("HOLDINGS SCHEDULE (Month-by-month):")
    print("=" * 100)

    # Print month-by-month holdings
    current_holdings = {}
    for month_idx, (date, positions) in enumerate(zip(dates, overall_positions)):
        month_holdings = {}
        for symbol in positions:
            bucket = bucket_map.get(symbol, "Unknown")
            month_holdings[bucket] = symbol

        # Only print if holdings changed
        if month_holdings != current_holdings:
            print(f"\n{date.strftime('%Y-%m-%d')} (Month {month_idx + 1}):")
            for bucket in sorted(month_holdings.keys()):
                symbol = month_holdings[bucket]
                prev = current_holdings.get(bucket, "N/A")
                change = (
                    " [NEW]"
                    if prev == "N/A"
                    else f" [was {prev}]" if prev != symbol else ""
                )
                print(f"  {bucket:<20}: {symbol:<8}{change}")
            current_holdings = month_holdings


if __name__ == "__main__":
    run_transaction_log()

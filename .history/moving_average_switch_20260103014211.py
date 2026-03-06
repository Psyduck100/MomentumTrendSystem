"""Moving-average switch between a stock ETF and a cash proxy."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import pandas as pd
import yfinance as yf

from momentum_program.backtest.metrics import compute_metrics


def download_prices(tickers: list[str], start: str, end: str, cache_dir: Path) -> pd.DataFrame:
    cache_dir.mkdir(exist_ok=True)
    fingerprint = hashlib.md5(",".join(tickers).encode("utf-8")).hexdigest()[:8]
    cache_file = cache_dir / f"ma_switch_prices_{start}_{end}_{fingerprint}.csv"
    if cache_file.exists():
        data = pd.read_csv(cache_file, index_col=0, parse_dates=True)
    else:
        data = pd.DataFrame()
    if data.empty:
        data = yf.download(tickers, start=start, end=end, progress=False)
        if data.empty:
            raise RuntimeError("Price download returned no data")
        if isinstance(data.columns, pd.MultiIndex):
            data = data.swaplevel(0, 1, axis=1)
            data = data.sort_index(axis=1)
            cols = []
            for ticker in data.columns.get_level_values(0).unique():
                sub = data[ticker]
                if isinstance(sub, pd.Series):
                    chosen = sub
                elif "Adj Close" in sub.columns:
                    chosen = sub["Adj Close"]
                elif "Close" in sub.columns:
                    chosen = sub["Close"]
                else:
                    chosen = sub.iloc[:, 0]
                chosen.name = ticker
                cols.append(chosen)
            data = pd.concat(cols, axis=1)
        elif "Adj Close" in data.columns:
            data = data["Adj Close"]
        elif "Close" in data.columns:
            data = data["Close"]
        if isinstance(data, pd.Series):
            data = data.to_frame()
        data.to_csv(cache_file)
    return data


def run_strategy(
    stock_ticker: str,
    start: str,
    end: str,
    ma_length: int,
    cash_rate: float,
    cache_dir: Path,
) -> pd.Series:
    data = download_prices([stock_ticker], start, end, cache_dir).sort_index()
    if stock_ticker not in data.columns:
        raise RuntimeError("Missing stock data")
    stock = data[stock_ticker].dropna()

    daily_returns = stock.pct_change()
    ma = stock.rolling(ma_length).mean()
    cash_daily = (1 + cash_rate) ** (1 / 252) - 1

    positions = stock > ma
    returns = []
    for i in range(1, len(stock)):
        date = stock.index[i]
        in_stock = positions.iloc[i - 1]
        if in_stock:
            ret = daily_returns.iloc[i]
        else:
            ret = cash_daily
        returns.append((date, ret))

    if not returns:
        raise RuntimeError("No returns calculated; check date range and MA length")

    dates, rets = zip(*returns)
    series = pd.Series(rets, index=pd.Index(dates, name="date"))
    monthly = series.resample("ME").apply(lambda s: (1 + s).prod() - 1)
    return monthly


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Moving-average stock/cash switch")
    parser.add_argument("--stock", default="SPY", help="Broad stock ETF (e.g., SPY or VT)")
    parser.add_argument("--start", default="2005-01-01")
    parser.add_argument("--end", default="2025-12-31")
    parser.add_argument("--ma-length", type=int, default=200, help="Moving average length in days")
    parser.add_argument("--cash-rate", type=float, default=0.03, help="Annualized cash yield when defensive")
    parser.add_argument("--cache-dir", type=Path, default=Path("backtest_cache"))
    parser.add_argument("--output", type=Path, help="Optional CSV for monthly returns")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    monthly_returns = run_strategy(
        stock_ticker=args.stock,
        start=args.start,
        end=args.end,
        ma_length=args.ma_length,
        cash_rate=args.cash_rate,
        cache_dir=args.cache_dir,
    )
    metrics = compute_metrics(monthly_returns)
    print("\nMoving-Average Switch Results")
    print("------------------------------")
    print(f"Period: {monthly_returns.index[0].date()} – {monthly_returns.index[-1].date()}")
    print(
        f"CAGR {metrics['cagr']*100:5.2f}% | Sharpe {metrics['sharpe']:4.2f} | "
        f"MaxDD {metrics['max_drawdown']*100:6.2f}%"
    )
    print(f"Total Return {(metrics['total_return']*100):6.2f}%")

    if args.output:
        monthly_returns.to_csv(args.output, header=["return"])
        print(f"Saved returns to {args.output}")


if __name__ == "__main__":
    main()

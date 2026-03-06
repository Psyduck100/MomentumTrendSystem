"""Absolute/relative momentum switch between SPY, ACWX, and bonds.

The portfolio holds a single asset each month:
1. Compute 12-month returns for the two stock proxies (S&P 500 vs ACWI ex-US).
2. If the strongest 12-month return beats the bond proxy, stay in stocks and allocate to the stronger symbol.
3. Otherwise move fully into bonds, modeled as a constant annual rate (default 4%).
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import pandas as pd
import yfinance as yf

from momentum_program.backtest.metrics import compute_metrics, compute_turnover


def download_prices(tickers: Iterable[str], start: str, end: str) -> pd.DataFrame:
    tickers = list(dict.fromkeys(tickers))
    data = yf.download(tickers, start=start, end=end, progress=False)
    if data.empty:
        raise RuntimeError("Price download returned no data")
    if isinstance(data.columns, pd.MultiIndex):
        data = data.swaplevel(0, 1, axis=1)
        data = data.sort_index(axis=1)
        selected = []
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
            selected.append(chosen)
        data = pd.concat(selected, axis=1)
    elif "Adj Close" in data.columns:
        data = data["Adj Close"]
    elif "Close" in data.columns:
        data = data["Close"]
    if isinstance(data, pd.Series):
        data = data.to_frame()
    return data


def run_strategy(
    start: str,
    end: str,
    stock_a: str,
    stock_b: str,
    bond_rate: float,
) -> tuple[pd.Series, list[str]]:
    prices = download_prices([stock_a, stock_b], start, end).sort_index()
    monthly = prices.resample("ME").last().dropna(axis=1, how="all")
    if monthly.shape[1] < 2:
        raise RuntimeError("Need both stock series with sufficient data")
    monthly_returns = monthly.pct_change()
    ret_12 = monthly.pct_change(12)
    bond_monthly = (1 + bond_rate) ** (1 / 12) - 1

    returns_records: list[tuple[pd.Timestamp, float]] = []
    held_assets: list[str] = []

    for idx in range(12, len(monthly) - 1):
        next_date = monthly.index[idx + 1]
        # Relative momentum comparison using 12M return
        best_symbol = None
        best_ret = None
        for symbol in [stock_a, stock_b]:
            value = ret_12.iloc[idx].get(symbol)
            if pd.isna(value):
                continue
            if best_ret is None or value > best_ret:
                best_ret = value
                best_symbol = symbol

        # Stay in equities only if their trailing 12M return beats the bond proxy
        if best_ret is not None and best_ret > bond_rate:
            period_ret = monthly_returns.iloc[idx + 1].get(best_symbol, float("nan"))
            if pd.isna(period_ret):
                continue
            returns_records.append((next_date, float(period_ret)))
            held_assets.append(best_symbol or "Stocks")
        else:
            returns_records.append((next_date, float(bond_monthly)))
            held_assets.append("Bonds")

    if not returns_records:
        raise RuntimeError("Strategy produced no observations (check date range)")

    dates, rets = zip(*returns_records)
    series = pd.Series(rets, index=pd.Index(dates, name="date"))
    return series, held_assets


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Absolute/relative SPY vs ACWX switch")
    parser.add_argument("--start", default="2010-01-01", help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", default="2025-12-31", help="End date (YYYY-MM-DD)")
    parser.add_argument("--stock-a", default="SPY", help="US equity proxy")
    parser.add_argument("--stock-b", default="ACWX", help="International equity proxy")
    parser.add_argument(
        "--bond-rate",
        type=float,
        default=0.04,
        help="Annualized bond return when defensive / absolute hurdle",
    )
    parser.add_argument(
        "--output", type=Path, help="Optional CSV path for monthly returns"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    returns, holdings = run_strategy(
        start=args.start,
        end=args.end,
        stock_a=args.stock_a,
        stock_b=args.stock_b,
        bond_rate=args.bond_rate,
    )
    metrics = compute_metrics(returns)
    turnover = compute_turnover([[h] for h in holdings])

    print("\nAbsolute/Relative Switch Results")
    print("--------------------------------")
    print(f"Period: {returns.index[0].date()} – {returns.index[-1].date()}")
    print(
        f"CAGR {metrics['cagr']*100:5.2f}% | Sharpe {metrics['sharpe']:4.2f} | MaxDD {metrics['max_drawdown']*100:6.2f}%"
    )
    print(
        f"Total Return {(metrics['total_return']*100):6.2f}% | Turnover {turnover:4.2f}"
    )

    if args.output:
        returns.to_csv(args.output, header=["return"])
        print(f"Saved monthly returns to {args.output}")


if __name__ == "__main__":
    main()

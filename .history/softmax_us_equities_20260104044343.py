"""Softmax-weighted allocation inside the US equities bucket.

Each month:
1. Compute momentum scores (default 12m-1m) for all US equity ETFs.
2. Convert those scores to allocation weights via a temperature-controlled softmax.
3. Optionally apply an absolute filter: if the weighted 12M return is below the hurdle,
   move entirely to cash/T-bills.
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import yfinance as yf

from momentum_program.analytics.constants import (
    SCORE_MODE_12M_MINUS_1M,
    SCORE_MODE_BLEND_6_12,
    SCORE_MODE_RW_3_6_9_12,
)
from momentum_program.backtest.metrics import compute_metrics
from momentum_program.universe.bucket_csv_provider import BucketedCsvUniverseProvider

SCORE_CHOICES = {
    "12m_minus_1m": SCORE_MODE_12M_MINUS_1M,
    "blend_6_12": SCORE_MODE_BLEND_6_12,
    "rw_3_6_9_12": SCORE_MODE_RW_3_6_9_12,
}


def load_bucket(bucket_dir: Path, bucket_name: str) -> tuple[list[str], dict[str, str]]:
    provider = BucketedCsvUniverseProvider(bucket_dir)
    tickers = [
        t
        for t in provider.get_tickers()
        if provider.get_bucket_map().get(t) == bucket_name
    ]
    if not tickers:
        raise RuntimeError(f"Bucket {bucket_name} not found or empty")
    bucket_map = {ticker: bucket_name for ticker in tickers}
    return tickers, bucket_map


def download_prices(
    tickers: list[str], start: str, end: str, cache_dir: Path
) -> pd.DataFrame:
    cache_dir.mkdir(exist_ok=True)
    fingerprint = hashlib.md5(",".join(sorted(tickers)).encode("utf-8")).hexdigest()[:8]
    cache_path = cache_dir / f"softmax_prices_{start}_{end}_{fingerprint}.csv"
    if cache_path.exists():
        data = pd.read_csv(cache_path, index_col=0, parse_dates=True)
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
            data = pd.concat(cols, axis=1) if cols else pd.DataFrame(index=data.index)
        elif "Adj Close" in data.columns:
            data = data["Adj Close"]
        elif "Close" in data.columns:
            data = data["Close"]
        if isinstance(data, pd.Series):
            data = data.to_frame()
        data.to_csv(cache_path)
    return data


def compute_momentum(monthly: pd.DataFrame, score_mode: str) -> pd.DataFrame:
    ret_12 = monthly.pct_change(12)
    ret_9 = monthly.pct_change(9)
    ret_6 = monthly.pct_change(6)
    ret_3 = monthly.pct_change(3)
    ret_1 = monthly.pct_change(1)

    if score_mode == SCORE_MODE_12M_MINUS_1M:
        return ret_12 - ret_1
    if score_mode == SCORE_MODE_BLEND_6_12:
        return 0.5 * ret_6 + 0.5 * ret_12
    if score_mode == SCORE_MODE_RW_3_6_9_12:
        return 0.4 * ret_3 + 0.2 * ret_6 + 0.2 * ret_9 + 0.2 * ret_12
    raise ValueError(f"Unknown score mode {score_mode}")


def softmax(
    scores: pd.Series, temperature: float, top_n: int | None = None
) -> pd.Series:
    series = scores.dropna().sort_values(ascending=False)
    if top_n is not None and top_n > 0:
        series = series.iloc[:top_n]
    if series.empty:
        return pd.Series(dtype=float)
    scaled = series / max(temperature, 1e-6)
    scaled -= scaled.max()  # stability shift
    weights = np.exp(scaled)
    weights /= weights.sum()
    return pd.Series(weights, index=series.index)


def run_strategy(
    tickers: list[str],
    start: str,
    end: str,
    score_mode: str,
    temperature: float,
    top_n: int | None,
    abs_threshold: float,
    cash_rate: float,
    cache_dir: Path,
) -> tuple[pd.Series, pd.DataFrame]:
    data = download_prices(tickers, start, end, cache_dir)
    data = data.sort_index()
    monthly = data.resample("ME").last().dropna(axis=1, how="all")
    monthly = monthly[[t for t in tickers if t in monthly.columns]]
    if monthly.shape[1] < 2:
        raise RuntimeError("Not enough tickers with data")
    ret = monthly.pct_change()
    ret12 = monthly.pct_change(12)
    momentum = compute_momentum(monthly, score_mode)
    cash_monthly = (1 + cash_rate) ** (1 / 12) - 1

    returns_list: list[tuple[pd.Timestamp, float]] = []
    weights_history: list[pd.Series] = []

    min_idx = 12
    for idx in range(min_idx, len(monthly) - 1):
        date = monthly.index[idx]
        weights = softmax(momentum.iloc[idx], temperature, top_n)
        if weights.empty:
            continue
        weighted_ret12 = float(
            (ret12.iloc[idx].reindex(weights.index).fillna(0) * weights).sum()
        )
        if weighted_ret12 < abs_threshold:
            period_ret = cash_monthly
            weights_history.append(pd.Series({"CASH": 1.0}, name=date))
        else:
            next_ret = ret.iloc[idx + 1].reindex(weights.index)
            period_ret = float((next_ret.fillna(0) * weights).sum())
            weights_history.append(weights.rename(date))
        returns_list.append((monthly.index[idx + 1], period_ret))

    if not returns_list:
        raise RuntimeError("Strategy produced no observations")

    dates, rets = zip(*returns_list)
    series = pd.Series(rets, index=pd.Index(dates, name="date"))
    weights_df = pd.DataFrame(weights_history).fillna(0)
    return series, weights_df


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Softmax-weighted US equities bucket")
    parser.add_argument("--bucket-dir", type=Path, default=Path("CSVs"))
    parser.add_argument("--bucket", default="US_equities")
    parser.add_argument("--start", default="2015-01-01")
    parser.add_argument("--end", default="2025-12-31")
    parser.add_argument("--score", choices=SCORE_CHOICES.keys(), default="12m_minus_1m")
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.5,
        help="Softmax temperature (lower=sharper)",
    )
    parser.add_argument(
        "--top-n", type=int, default=0, help="Limit softmax to top-N scores (0=all)"
    )
    parser.add_argument(
        "--abs-threshold",
        type=float,
        default=0.0,
        help="Weighted 12M hurdle to stay invested",
    )
    parser.add_argument(
        "--cash-rate",
        type=float,
        default=0.04,
        help="Annual cash return when defensive",
    )
    parser.add_argument("--cache-dir", type=Path, default=Path("backtest_cache"))
    parser.add_argument("--weights-output", type=Path)
    parser.add_argument("--returns-output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    tickers, _ = load_bucket(args.bucket_dir, args.bucket)
    top_n = args.top_n if args.top_n > 0 else None
    returns, weights = run_strategy(
        tickers=tickers,
        start=args.start,
        end=args.end,
        score_mode=SCORE_CHOICES[args.score],
        temperature=args.temperature,
        top_n=top_n,
        abs_threshold=args.abs_threshold,
        cash_rate=args.cash_rate,
        cache_dir=args.cache_dir,
    )
    metrics = compute_metrics(returns)
    print("\nSoftmax US Equities Results")
    print("----------------------------")
    print(f"Period: {returns.index[0].date()} – {returns.index[-1].date()}")
    print(
        f"CAGR {metrics['cagr']*100:5.2f}% | Sharpe {metrics['sharpe']:4.2f} | "
        f"MaxDD {metrics['max_drawdown']*100:6.2f}%"
    )
    print(f"Total Return {(metrics['total_return']*100):6.2f}%")

    if args.returns_output:
        returns.to_csv(args.returns_output, header=["return"])
        print(f"Saved returns to {args.returns_output}")
    if args.weights_output:
        weights.to_csv(args.weights_output)
        print(f"Saved weights to {args.weights_output}")


if __name__ == "__main__":
    main()

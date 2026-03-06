"""Two-stage bucket momentum strategy.

1. Within each bucket, select the top ETF by momentum (12m-1m by default).
2. Rank those bucket winners globally and hold the top-K (default 2).
3. Apply absolute filters (6m, 12m, or BOTH) per pick; failed picks go to cash/T-bills.

The script sweeps rank-gap and filter combinations so we can compare configs quickly.
"""

from __future__ import annotations

import argparse
import hashlib
import time
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import numpy as np
import pandas as pd
import yfinance as yf

from momentum_program.analytics.constants import (
    SCORE_MODE_12M_MINUS_1M,
    SCORE_MODE_BLEND_6_12,
    SCORE_MODE_RW_3_6_9_12,
)
from momentum_program.backtest.metrics import compute_metrics, compute_turnover
from momentum_program.universe.bucket_csv_provider import BucketedCsvUniverseProvider

SCORE_CHOICES = {
    "12m_minus_1m": SCORE_MODE_12M_MINUS_1M,
    "blend_6_12": SCORE_MODE_BLEND_6_12,
    "rw_3_6_9_12": SCORE_MODE_RW_3_6_9_12,
}

FILTER_CHOICES = {"ret_12m", "ret_6m", "ret_and", "none"}


@dataclass
class StrategyResult:
    rank_gap: int
    filter_mode: str
    cagr: float
    sharpe: float
    max_dd: float
    total_return: float
    turnover: float
    returns: pd.Series


def load_universe(bucket_dir: Path) -> tuple[list[str], dict[str, str]]:
    provider = BucketedCsvUniverseProvider(bucket_dir)
    tickers = provider.get_tickers()
    bucket_map = provider.get_bucket_map()
    if not tickers:
        raise RuntimeError(f"No CSV tickers found in {bucket_dir}")
    return tickers, bucket_map


def download_prices(tickers: Sequence[str], start: str, end: str) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    chunk_size = 25
    for i in range(0, len(tickers), chunk_size):
        chunk = tickers[i : i + chunk_size]
        print(f"Downloading {len(chunk)} tickers ({chunk[0]}..{chunk[-1]}) from {start} to {end}...")
        df_chunk = pd.DataFrame()
        for attempt in range(2):
            try:
                df_chunk = yf.download(chunk, start=start, end=end, progress=False)
            except Exception as exc:  # noqa: BLE001
                if attempt == 0:
                    print(f"  Error for chunk starting {chunk[0]} ({exc}); retry in 5s")
                    time.sleep(5)
                    continue
                print(f"  Failed chunk starting {chunk[0]} ({exc}); skipping")
                df_chunk = pd.DataFrame()
            if df_chunk.empty and attempt == 0:
                print(f"  Empty response for chunk starting {chunk[0]}; retry in 5s")
                time.sleep(5)
                continue
            break

        if df_chunk.empty:
            continue

        if isinstance(df_chunk.columns, pd.MultiIndex):
            df_chunk = df_chunk.swaplevel(0, 1, axis=1)
            df_chunk = df_chunk.sort_index(axis=1)
            selected_cols: list[pd.Series] = []
            for ticker in df_chunk.columns.get_level_values(0).unique():
                sub = df_chunk[ticker]
                if isinstance(sub, pd.Series):
                    chosen = sub
                else:
                    if "Adj Close" in sub.columns:
                        chosen = sub["Adj Close"]
                    elif "Close" in sub.columns:
                        chosen = sub["Close"]
                    else:
                        chosen = sub.iloc[:, 0]
                chosen.name = ticker
                selected_cols.append(chosen)
            df_chunk = pd.concat(selected_cols, axis=1) if selected_cols else pd.DataFrame(index=df_chunk.index)
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
    combined = combined.loc[:, ~combined.columns.duplicated()]
    return combined


def load_price_history(tickers: list[str], start: str, end: str, cache_dir: Path) -> pd.DataFrame:
    cache_dir.mkdir(exist_ok=True)
    fingerprint = hashlib.md5(",".join(sorted(tickers)).encode("utf-8")).hexdigest()[:10]
    cache_file = cache_dir / f"price_data_{start}_{end}_{fingerprint}.csv"

    if cache_file.exists():
        print(f"Loading cached price data from {cache_file}...")
        data = pd.read_csv(cache_file, index_col=0, parse_dates=True)
    else:
        data = pd.DataFrame()

    if data.empty:
        data = download_prices(tickers, start, end)
        if data.empty:
            raise RuntimeError("Could not download any price data")
        data.to_csv(cache_file)

    refreshed = False
    while True:
        valid = [t for t in data.columns if data[t].notna().mean() >= 0.8]
        for ticker in data.columns:
            if ticker not in valid:
                print(f"  Skipping {ticker} (insufficient data)")
        if valid:
            return data[valid]
        if refreshed:
            raise RuntimeError("No tickers left after coverage filter")
        print("Cache missing coverage; refreshing...")
        cache_file.unlink(missing_ok=True)
        data = download_prices(tickers, start, end)
        if data.empty:
            raise RuntimeError("Download empty after cache refresh")
        data.to_csv(cache_file)
        refreshed = True


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
    raise ValueError(f"Unsupported score mode: {score_mode}")


def _window_return(monthly: pd.DataFrame, idx: int, symbol: str, months: int) -> float | None:
    if idx < months:
        return None
    now = monthly.iloc[idx][symbol]
    past = monthly.iloc[idx - months][symbol]
    if pd.isna(now) or pd.isna(past) or past == 0:
        return None
    return (now - past) / past


def passes_abs_filter(
    monthly: pd.DataFrame,
    idx: int,
    symbol: str,
    mode: str,
    band: float,
) -> bool:
    if mode == "none":
        return True

    ret_12 = _window_return(monthly, idx, symbol, 12)
    ret_6 = _window_return(monthly, idx, symbol, 6)
    if ret_12 is None or ret_6 is None:
        return False

    if mode == "ret_12m":
        return ret_12 > band
    if mode == "ret_6m":
        return ret_6 > band
    if mode == "ret_and":
        return (ret_12 > band) and (ret_6 > band)
    raise ValueError(f"Unknown abs filter mode: {mode}")


def run_two_stage_strategy(
    monthly: pd.DataFrame,
    bucket_map: Dict[str, str],
    score_mode: str,
    rank_gap: int,
    filter_mode: str,
    filter_band: float,
    top_k: int,
    cash_rate: float,
    slippage_bps: float,
    expense_ratio: float,
) -> tuple[pd.Series, list[list[str]]]:
    momentum = compute_momentum(monthly, score_mode)
    monthly_returns = monthly.pct_change()
    buckets = sorted(set(bucket_map.values()))
    valid_symbols = [s for s in monthly.columns if s in bucket_map]
    prev_bucket_selection: dict[str, str | None] = {b: None for b in buckets}
    prev_positions: list[str] = []
    returns_records: list[tuple[pd.Timestamp, float]] = []
    positions_history: list[list[str]] = []
    cash_ret_monthly = (1 + cash_rate) ** (1 / 12) - 1
    min_lookback = 12

    for idx in range(min_lookback, len(momentum) - 1):
        date = momentum.index[idx]
        next_date = momentum.index[idx + 1]
        scores = momentum.iloc[idx]

        bucket_leaders: dict[str, str] = {}
        for bucket in buckets:
            bucket_symbols = [s for s in valid_symbols if bucket_map.get(s) == bucket]
            if not bucket_symbols:
                continue
            bucket_scores = scores[bucket_symbols].dropna()
            if bucket_scores.empty:
                continue
            ranked = bucket_scores.sort_values(ascending=False)
            leader = ranked.index[0]
            current = prev_bucket_selection.get(bucket)
            if rank_gap > 0 and current in ranked.index:
                leader_rank = ranked.index.get_loc(leader)
                current_rank = ranked.index.get_loc(current)
                if leader_rank >= current_rank - rank_gap:
                    leader = current
            bucket_leaders[bucket] = leader
            prev_bucket_selection[bucket] = leader

        if not bucket_leaders:
            continue

        candidates: list[tuple[str, str, float]] = []
        for bucket, symbol in bucket_leaders.items():
            score = scores.get(symbol)
            if pd.notna(score):
                candidates.append((symbol, bucket, float(score)))
        if not candidates:
            continue
        candidates.sort(key=lambda tup: tup[2], reverse=True)
        picks = candidates[:top_k]

        if not picks:
            continue

        weight_per_pick = 1.0 / top_k
        period_return = 0.0
        invested_symbols: list[str] = []

        for symbol, bucket, _score in picks:
            filter_pass = passes_abs_filter(monthly, idx, symbol, filter_mode, filter_band)
            if filter_pass:
                raw_ret = monthly_returns.loc[next_date, symbol]
                raw_ret = 0.0 if pd.isna(raw_ret) else float(raw_ret)
                net_ret = raw_ret - (expense_ratio / 12.0)
                invested_symbols.append(symbol)
            else:
                net_ret = cash_ret_monthly
            period_return += weight_per_pick * net_ret

        prev_set = set(prev_positions)
        curr_set = set(invested_symbols)
        additions = len(curr_set - prev_set)
        deletions = len(prev_set - curr_set)
        slippage_per_weight = slippage_bps / 10000.0 / max(top_k, 1)
        period_return -= (additions + deletions) * slippage_per_weight

        returns_records.append((next_date, period_return))
        positions_history.append(invested_symbols)
        prev_positions = invested_symbols

    if not returns_records:
        return pd.Series(dtype=float), []

    dates, rets = zip(*returns_records)
    series = pd.Series(rets, index=pd.Index(dates, name="date"))
    return series, positions_history


def run_grid(
    monthly: pd.DataFrame,
    bucket_map: Dict[str, str],
    score_mode: str,
    rank_gaps: Sequence[int],
    filter_modes: Sequence[str],
    filter_band: float,
    top_k: int,
    cash_rate: float,
    slippage_bps: float,
    expense_ratio: float,
) -> list[StrategyResult]:
    results: list[StrategyResult] = []
    for gap, filt in product(rank_gaps, filter_modes):
        print(f"\n=== rank_gap={gap} filter={filt} ===")
        returns, positions = run_two_stage_strategy(
            monthly=monthly,
            bucket_map=bucket_map,
            score_mode=score_mode,
            rank_gap=gap,
            filter_mode=filt,
            filter_band=filter_band,
            top_k=top_k,
            cash_rate=cash_rate,
            slippage_bps=slippage_bps,
            expense_ratio=expense_ratio,
        )
        if returns.empty:
            print("  No valid return series produced; skipping")
            continue
        metrics = compute_metrics(returns)
        turnover = compute_turnover(positions)
        print(
            f"  CAGR {metrics['cagr']*100:5.2f}% | Sharpe {metrics['sharpe']:4.2f} | "
            f"MaxDD {metrics['max_drawdown']*100:6.2f}% | Turnover {turnover:4.2f}"
        )
        results.append(
            StrategyResult(
                rank_gap=gap,
                filter_mode=filt,
                cagr=metrics["cagr"],
                sharpe=metrics["sharpe"],
                max_dd=metrics["max_drawdown"],
                total_return=metrics["total_return"],
                turnover=turnover,
                returns=returns,
            )
        )
    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Two-stage bucket momentum sweep")
    parser.add_argument("--start", default="2015-01-01", help="Backtest start date (YYYY-MM-DD)")
    parser.add_argument("--end", default="2025-12-31", help="Backtest end date (YYYY-MM-DD)")
    parser.add_argument("--bucket-dir", type=Path, default=Path("CSVs"))
    parser.add_argument(
        "--score",
        choices=SCORE_CHOICES.keys(),
        default="12m_minus_1m",
        help="Momentum score mode",
    )
    parser.add_argument(
        "--exclude-buckets",
        nargs="*",
        default=[],
        help="Bucket names to exclude from the universe (e.g., Commodities)",
    )
    parser.add_argument(
        "--rank-gaps",
        nargs="+",
        type=int,
        default=[0, 1],
        help="Rank-gap thresholds to test",
    )
    parser.add_argument(
        "--filter-modes",
        nargs="+",
        default=["ret_12m", "ret_and", "ret_6m"],
        help="Absolute filter modes (subset of ret_12m, ret_6m, ret_and, none)",
    )
    parser.add_argument("--filter-band", type=float, default=0.0, help="Filter hurdle (e.g., 0.01 for +1%)")
    parser.add_argument("--top-k", type=int, default=2, help="Number of bucket winners to hold")
    parser.add_argument("--cash-rate", type=float, default=0.04, help="Annual cash rate when sidelined")
    parser.add_argument("--slippage-bps", type=float, default=3.0, help="Per-trade slippage in bps")
    parser.add_argument("--expense-ratio", type=float, default=0.001, help="Annual expense drag applied when invested")
    parser.add_argument("--cache-dir", type=Path, default=Path("backtest_cache"))
    parser.add_argument("--output", type=Path, help="Optional CSV to store summary metrics")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    invalid_filters = set(args.filter_modes) - FILTER_CHOICES
    if invalid_filters:
        raise ValueError(f"Unknown filter mode(s): {', '.join(sorted(invalid_filters))}")

    tickers, bucket_map = load_universe(args.bucket_dir)
    if args.exclude_buckets:
        excluded = set(args.exclude_buckets)
        bucket_map = {sym: bucket for sym, bucket in bucket_map.items() if bucket not in excluded}
        tickers = [sym for sym in tickers if sym in bucket_map]
        if not tickers:
            raise RuntimeError("No tickers remain after excluding buckets")
        print(f"Excluding buckets: {', '.join(sorted(excluded))}")
    price_history = load_price_history(tickers, args.start, args.end, args.cache_dir)
    price_history = price_history.sort_index()
    monthly = price_history.resample("ME").last()

    # Drop tickers lost during resampling (all-NaN columns)
    monthly = monthly.dropna(axis=1, how="all")
    symbols = [s for s in monthly.columns if s in bucket_map]
    if not symbols:
        raise RuntimeError("No overlapping tickers between price data and buckets")
    monthly = monthly[symbols]
    filtered_bucket_map = {sym: bucket_map[sym] for sym in symbols}

    results = run_grid(
        monthly=monthly,
        bucket_map=filtered_bucket_map,
        score_mode=SCORE_CHOICES[args.score],
        rank_gaps=args.rank_gaps,
        filter_modes=args.filter_modes,
        filter_band=args.filter_band,
        top_k=max(args.top_k, 1),
        cash_rate=args.cash_rate,
        slippage_bps=args.slippage_bps,
        expense_ratio=args.expense_ratio,
    )

    if args.output and results:
        df = pd.DataFrame(
            {
                "rank_gap": [r.rank_gap for r in results],
                "filter_mode": [r.filter_mode for r in results],
                "cagr": [r.cagr for r in results],
                "sharpe": [r.sharpe for r in results],
                "max_drawdown": [r.max_dd for r in results],
                "total_return": [r.total_return for r in results],
                "turnover": [r.turnover for r in results],
            }
        )
        df.to_csv(args.output, index=False)
        print(f"Saved summary metrics to {args.output}")


if __name__ == "__main__":
    main()

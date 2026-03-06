from __future__ import annotations

from dataclasses import dataclass
import os
import sys

import pandas as pd

if __name__ == "__main__" and __package__ is None:
    project_root = os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    )
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

import strategy.SECTOR170 as SECTOR170
from backtest.common.metrics import (
    compute_return_metrics,
    compute_relative_metrics,
    format_metrics_block,
)


@dataclass(frozen=True)
class BacktestConfig:
    universe: tuple[str, ...] = ("XLK", "XLV", "XLI", "XLE", "XAR")
    lookback_days: int = SECTOR170.DEFAULT_LOOKBACK_DAYS
    start_date: str = "2012-09-29"
    benchmark: str = "SPY"
    transaction_cost_bps: float = 3.0


def month_end_trading_days(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    proxy = pd.Series(index, index=index)
    month_ends = proxy.resample("ME").last().dropna()
    return pd.DatetimeIndex(month_ends.values)


def build_monthly_recommendations(
    prices_daily: pd.DataFrame,
    universe: list[str],
    lookback_days: int,
) -> pd.Series:
    recs: dict[pd.Timestamp, str] = {}
    for asof in month_end_trading_days(prices_daily.index):
        try:
            scores = SECTOR170.compute_170d_scores(
                daily_prices=prices_daily,
                asof=asof,
                lookback_days=lookback_days,
                universe=universe,
            )
        except ValueError:
            continue

        rank_table = SECTOR170.build_rank_table(scores)
        decision = SECTOR170.pick_recommendation(rank_table)
        ticker = decision.get("recommendation")
        if isinstance(ticker, str):
            recs[asof] = ticker

    return pd.Series(recs).sort_index()


def positions_from_monthly_recs(
    recs: pd.Series,
    daily_index: pd.DatetimeIndex,
) -> pd.Series:
    pos = pd.Series(index=daily_index, dtype=object)
    for asof, ticker in recs.items():
        future = daily_index[daily_index > asof]
        if len(future) == 0:
            continue
        pos.loc[future[0]] = str(ticker)
    pos = pos.ffill()
    return pos


def returns_from_positions(
    returns_df: pd.DataFrame,
    positions: pd.Series,
    transaction_cost_bps: float,
) -> pd.Series:
    pos = positions.reindex(returns_df.index).ffill()
    strategy_returns = pd.Series(index=returns_df.index, dtype=float)

    for ticker in returns_df.columns:
        mask = pos == ticker
        if mask.any():
            strategy_returns.loc[mask] = returns_df.loc[mask, ticker]

    switched = pos.ne(pos.shift(1)).fillna(False)
    if transaction_cost_bps > 0:
        strategy_returns = strategy_returns - switched.astype(float) * (
            transaction_cost_bps / 10000.0
        )

    return strategy_returns.dropna()


def run_strategy_backtest(cfg: BacktestConfig) -> dict:
    universe = list(cfg.universe)
    all_tickers = universe + [cfg.benchmark]

    prices = SECTOR170.download_prices(
        all_tickers, start_date=cfg.start_date, auto_adjust=True
    )
    prices_universe = prices.reindex(
        columns=[t for t in universe if t in prices.columns]
    ).dropna(how="all")

    recs = build_monthly_recommendations(
        prices_daily=prices_universe,
        universe=list(prices_universe.columns),
        lookback_days=cfg.lookback_days,
    )
    positions = positions_from_monthly_recs(recs, prices_universe.index)

    daily_returns = prices_universe.pct_change().fillna(0.0)
    strategy_returns = returns_from_positions(
        daily_returns,
        positions,
        transaction_cost_bps=cfg.transaction_cost_bps,
    )

    benchmark_returns = prices[cfg.benchmark].pct_change().reindex(strategy_returns.index).dropna()

    base_metrics = compute_return_metrics(
        strategy_returns,
        positions=positions.reindex(strategy_returns.index).notna().astype(float),
    )
    rel_metrics = compute_relative_metrics(strategy_returns, benchmark_returns)
    benchmark_metrics = compute_return_metrics(benchmark_returns, positions=None)

    hold_counts = positions.value_counts(dropna=True)
    hold_weights = (
        (hold_counts / hold_counts.sum()).sort_values(ascending=False)
        if hold_counts.sum() > 0
        else hold_counts
    )

    return {
        "config": cfg,
        "prices": prices,
        "recommendations": recs,
        "positions": positions,
        "returns": strategy_returns,
        "benchmark_returns": benchmark_returns,
        "benchmark_metrics": benchmark_metrics,
        "metrics": {**base_metrics, **rel_metrics},
        "hold_weights": hold_weights,
    }


def main() -> None:
    cfg = BacktestConfig()
    result = run_strategy_backtest(cfg)

    returns = result["returns"]
    metrics = result["metrics"]
    benchmark_returns = result["benchmark_returns"]
    benchmark_metrics = result["benchmark_metrics"]

    print("=" * 72)
    print("SECTOR170 BACKTEST")
    print("=" * 72)
    print(f"Window: {returns.index.min().date()} -> {returns.index.max().date()}")
    print(f"Universe: {', '.join(cfg.universe)}")
    print(f"Lookback: {cfg.lookback_days} trading days")
    print(
        f"Benchmark: {cfg.benchmark} | Cost per switch: {cfg.transaction_cost_bps:.2f} bps"
    )
    print()
    print("Strategy Stats")
    print(format_metrics_block(metrics, include_relative=True))
    print()
    print(f"{cfg.benchmark} Buy & Hold Stats (same window)")
    print(format_metrics_block(benchmark_metrics, include_relative=False))

    hold_weights = result["hold_weights"]
    if len(hold_weights):
        print("\nHold Concentration:")
        for ticker, w in hold_weights.items():
            print(f"  {ticker}: {w:.1%}")


if __name__ == "__main__":
    main()

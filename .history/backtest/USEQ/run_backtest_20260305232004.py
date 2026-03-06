from __future__ import annotations

from dataclasses import dataclass
import os
import sys

import pandas as pd

if __name__ == "__main__" and __package__ is None:
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

from backtest.USEQ.Test import (
    AbsFilterConfig,
    StrategyConfig,
    run_backtest,
    normalize_tickers,
)
import strategy.USEQ as USEQ
from backtest.common.metrics import (
    compute_return_metrics,
    compute_relative_metrics,
    format_metrics_block,
)


@dataclass(frozen=True)
class BacktestConfig:
    universe: tuple[str, ...] = ("SCHB", "XLG", "SCHV", "QQQ", "RSP")
    defensive_symbol: str = "IEF"
    start_date: str = "2000-01-01"
    rebalance_freq: str = "M"
    benchmark: str = "SPY"
    transaction_cost_bps: float = 0.0


def run_strategy_backtest(cfg: BacktestConfig) -> dict:
    strategy_cfg = StrategyConfig(
        universe_override=list(cfg.universe),
        defensive_symbol=cfg.defensive_symbol,
        start_date=cfg.start_date,
        rebalance_freq=cfg.rebalance_freq,
    )
    abs_cfg = AbsFilterConfig(kind="ret_12m_pos", trading_days_per_month=strategy_cfg.trading_days_per_month)

    result = run_backtest(strategy_cfg, abs_cfg, transaction_cost_bps=cfg.transaction_cost_bps)

    bt = result["bt"]
    strategy_returns = bt["ret"].copy()
    in_position = (bt["position"] != cfg.defensive_symbol).astype(float)

    prices = result["prices"]
    if cfg.benchmark in prices.columns:
        benchmark_returns = prices[cfg.benchmark].pct_change().reindex(strategy_returns.index).dropna()
    else:
        bench_px = USEQ.download_prices([cfg.benchmark], start_date=cfg.start_date)
        benchmark_returns = bench_px[cfg.benchmark].pct_change().reindex(strategy_returns.index).dropna()

    base_metrics = compute_return_metrics(strategy_returns, positions=in_position)
    rel_metrics = compute_relative_metrics(strategy_returns, benchmark_returns)

    metrics = {**base_metrics, **rel_metrics}

    benchmark_metrics = compute_return_metrics(benchmark_returns, positions=None)

    return {
        "config": cfg,
        "strategy_config": strategy_cfg,
        "abs_filter": abs_cfg,
        "bt": bt,
        "returns": strategy_returns,
        "benchmark_returns": benchmark_returns,
        "benchmark_metrics": benchmark_metrics,
        "metrics": metrics,
    }


def main() -> None:
    cfg = BacktestConfig()
    result = run_strategy_backtest(cfg)

    returns = result["returns"]
    metrics = result["metrics"]
    benchmark_returns = result["benchmark_returns"]
    benchmark_metrics = result["benchmark_metrics"]

    print("=" * 72)
    print("USEQ BACKTEST")
    print("=" * 72)
    print(f"Window: {returns.index.min().date()} -> {returns.index.max().date()}")
    print(f"Universe: {', '.join(normalize_tickers(cfg.universe))}")
    print(f"Defensive: {cfg.defensive_symbol} | Rebalance: {cfg.rebalance_freq}")
    print(f"Benchmark: {cfg.benchmark}")
    print()
    print("Strategy Stats")
    print(format_metrics_block(metrics, include_relative=True))
    print()
    print(f"{cfg.benchmark} Buy & Hold Stats (same window)")
    print(format_metrics_block(benchmark_metrics, include_relative=False))


if __name__ == "__main__":
    main()

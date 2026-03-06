from __future__ import annotations

from dataclasses import dataclass
import os
import sys

import pandas as pd
import yfinance as yf

if __name__ == "__main__" and __package__ is None:
    project_root = os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    )
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

import strategy.PMTL as PMTL
import strategy.USEQ as USEQ
from strategy.runPMTL import (
    build_useq_recommendations,
    positions_from_monthly_recs,
    returns_from_positions,
)
from backtest.common.metrics import compute_return_metrics, format_metrics_block


@dataclass(frozen=True)
class BacktestConfig:
    gld_ticker: str = "GLD"
    useq_start_date: str = "2001-01-01"
    useq_universe: tuple[str, ...] = ("SPY", "QQQ", "VTI")


def _build_regime(close: pd.Series) -> pd.DataFrame:
    feature_params_up = dict(
        ma_len_entry=200,
        ma_len_exit=200,
        slope_lookback=20,
        entry_len=260,
        exit_len=90,
    )
    feature_params_down = dict(
        ma_len_entry=200,
        ma_len_exit=270,
        slope_lookback=20,
        entry_len=90,
        exit_len=90,
    )

    return PMTL.build_regime(
        close,
        feature_params_up=feature_params_up,
        rule_params_up=dict(slope_min=0.0, ma_buffer=0.005),
        feature_params_down=feature_params_down,
        rule_params_down=dict(slope_min=0.0, ma_buffer=0.005),
        gate_up="BME",
        gate_down="W-FRI",
        down_enter_gates=2,
        down_exit_gates=2,
    )


def run_strategy_backtest(cfg: BacktestConfig) -> dict:
    gld_raw = yf.download(
        cfg.gld_ticker, period="max", auto_adjust=True, progress=False
    )
    gld_close = PMTL.as_series(gld_raw["Close"], "Close").dropna()

    regime = _build_regime(gld_close)
    is_bull = regime["is_up"].reindex(gld_close.index).fillna(False)

    useq_cfg = USEQ.StrategyConfig(start_date=cfg.useq_start_date)
    universe = list(cfg.useq_universe)
    all_tickers = universe + [useq_cfg.defensive_symbol]
    useq_prices = USEQ.download_prices(all_tickers, start_date=useq_cfg.start_date)

    useq_recs = build_useq_recommendations(useq_prices, useq_cfg, universe=universe)
    useq_positions = positions_from_monthly_recs(
        useq_recs,
        useq_prices.index,
        default_ticker=useq_cfg.defensive_symbol,
    )

    gld_returns = gld_close.pct_change().fillna(0.0)
    useq_returns_df = useq_prices.pct_change().fillna(0.0)
    useq_returns = returns_from_positions(useq_returns_df, useq_positions)

    useq_returns = useq_returns.reindex(gld_returns.index).fillna(0.0)
    is_bull = is_bull.reindex(gld_returns.index).fillna(False)

    layered_returns = gld_returns.where(is_bull, useq_returns)
    metrics = compute_return_metrics(layered_returns, positions=is_bull.astype(float))

    useq_mask = ~is_bull
    metrics["cagr_non_bull_sleeve"] = compute_return_metrics(
        layered_returns[useq_mask],
        positions=None,
    )["cagr"]

    return {
        "config": cfg,
        "gld_close": gld_close,
        "regime": regime,
        "is_bull": is_bull,
        "useq_prices": useq_prices,
        "useq_recommendations": useq_recs,
        "useq_positions": useq_positions,
        "returns": layered_returns,
        "metrics": metrics,
    }


def main() -> None:
    cfg = BacktestConfig()
    result = run_strategy_backtest(cfg)

    returns = result["returns"]
    metrics = result["metrics"]

    print("=" * 72)
    print("PMTL BACKTEST (GLD + USEQ SLEEVES)")
    print("=" * 72)
    print(f"Window: {returns.index.min().date()} -> {returns.index.max().date()}")
    print(f"USEQ Universe: {', '.join(cfg.useq_universe)} + IEF")
    print()
    print(format_metrics_block(metrics, include_relative=False))
    print(f"CAGR in Non-Bull Sleeve: {metrics['cagr_non_bull_sleeve']:.2%}")


if __name__ == "__main__":
    main()

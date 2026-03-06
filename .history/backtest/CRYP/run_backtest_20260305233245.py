from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import os
import sys

import pandas as pd

if __name__ == "__main__" and __package__ is None:
    project_root = os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    )
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

import strategy.CRYP as CRYP
from backtest.CRYP.backtest import run_backtest as run_signal_backtest
from backtest.common.metrics import compute_return_metrics, format_metrics_block


@dataclass(frozen=True)
class BacktestConfig:
    start: datetime = datetime(2013, 1, 1)
    end: datetime = datetime.today()
    use_btc_calendar: bool = False
    config_name: str = "primary"  # primary | alt
    cost_bps: float = 2.0
    slippage_bps: float = 1.0


def _resolve_signal_config(config_name: str) -> dict:
    key = str(config_name).strip().lower()
    if key == "primary":
        return CRYP.CONFIG_PRIMARY
    if key == "alt":
        return CRYP.CONFIG_ALT
    raise ValueError("config_name must be 'primary' or 'alt'.")


def run_strategy_backtest(cfg: BacktestConfig) -> dict:
    price, raw_returns, btc_close = CRYP.load_proxy_data(
        cfg.start,
        cfg.end,
        use_btc_calendar=cfg.use_btc_calendar,
    )

    signal_cfg = _resolve_signal_config(cfg.config_name)
    signal = CRYP.build_signal_for_config(price, signal_cfg)

    bt = run_signal_backtest(
        returns=raw_returns,
        signal=signal,
        cost_bps=cfg.cost_bps,
        slippage_bps=cfg.slippage_bps,
    )

    metrics = compute_return_metrics(bt.returns, positions=bt.positions)

    return {
        "config": cfg,
        "signal_config": signal_cfg,
        "btc_close": btc_close,
        "proxy_price": price,
        "signal": signal,
        "returns": bt.returns,
        "equity": bt.equity,
        "positions": bt.positions,
        "metrics": metrics,
    }


def main() -> None:
    cfg = BacktestConfig()
    result = run_strategy_backtest(cfg)

    start = result["returns"].index.min().date()
    end = result["returns"].index.max().date()

    print("=" * 72)
    print(f"CRYP BACKTEST ({cfg.config_name.upper()})")
    print("=" * 72)
    print(f"Window: {start} -> {end}")
    print(f"Costs: {cfg.cost_bps + cfg.slippage_bps:.2f} bps per switch")
    print(
        f"Calendar: {'BTC (24/7)' if cfg.use_btc_calendar else 'SPY (US market sessions)'}"
    )
    print()
    print(format_metrics_block(result["metrics"], include_relative=False))

    latest_signal = int(result["signal"].iloc[-1]) if len(result["signal"]) else 0
    print(f"Latest signal: {'ON' if latest_signal == 1 else 'OFF'}")


if __name__ == "__main__":
    main()

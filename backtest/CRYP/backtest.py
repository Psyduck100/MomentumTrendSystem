from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class BacktestResult:
    returns: pd.Series
    equity: pd.Series
    positions: pd.Series
    metrics: dict


def _max_drawdown(equity: pd.Series) -> float:
    peak = equity.cummax()
    drawdown = equity / peak - 1.0
    return float(drawdown.min())


def _annualized_return(equity: pd.Series) -> float:
    if equity.empty:
        return 0.0
    if equity.iloc[0] == 0:
        return 0.0
    total_return = equity.iloc[-1] / equity.iloc[0] - 1.0
    if total_return == 0.0:
        return 0.0
    years = len(equity) / 252.0
    if years <= 0:
        return 0.0
    return (1.0 + total_return) ** (1.0 / years) - 1.0


def _annualized_return_for_mask(returns: pd.Series, mask: pd.Series) -> float:
    masked = returns[mask]
    if masked.empty:
        return 0.0
    equity = (1.0 + masked).cumprod()
    total_return = equity.iloc[-1] / equity.iloc[0] - 1.0
    if total_return == 0.0:
        return 0.0
    years = len(masked) / 252.0
    if years <= 0:
        return 0.0
    return (1.0 + total_return) ** (1.0 / years) - 1.0


def _sharpe(returns: pd.Series) -> float:
    if returns.std() < 1e-12:
        return 0.0
    return float(np.sqrt(252.0) * returns.mean() / returns.std())


def run_backtest(
    returns: pd.Series,
    signal: pd.Series,
    cost_bps: float = 0.0,
    slippage_bps: float = 0.0,
    vol_target: Optional[float] = None,
    vol_lookback: int = 20,
    cash_returns: Optional[pd.Series] = None,
) -> BacktestResult:
    aligned = returns.align(signal, join="inner")[0]
    signal = signal.reindex(aligned.index).fillna(0.0)
    returns = aligned.fillna(0.0)

    positions = signal.shift(1).fillna(0.0)
    if vol_target is not None:
        vol = returns.rolling(vol_lookback).std() * np.sqrt(252.0)
        scale = (vol_target / vol).clip(lower=0.0, upper=1.0).fillna(0.0)
        positions = positions * scale

    if cash_returns is None:
        cash_returns = pd.Series(0.0, index=returns.index)
    else:
        cash_returns = cash_returns.reindex(returns.index).fillna(0.0)

    trades = positions.diff().abs().fillna(0.0)
    cost = trades * (cost_bps + slippage_bps) / 10000.0
    strat_returns = positions * returns + (1.0 - positions) * cash_returns - cost
    equity = (1.0 + strat_returns).cumprod()

    switches = int(trades.sum())
    years = len(returns) / 252.0 if len(returns) else 0.0
    in_position = positions > 0
    metrics = {
        "cagr": _annualized_return(equity),
        "cagr_in_position": _annualized_return_for_mask(strat_returns, in_position),
        "sharpe": _sharpe(strat_returns),
        "max_drawdown": _max_drawdown(equity),
        "time_in_market": float((positions > 0).mean()),
        "switches_per_year": float(switches / years) if years > 0 else 0.0,
    }
    return BacktestResult(
        returns=strat_returns,
        equity=equity,
        positions=positions,
        metrics=metrics,
    )

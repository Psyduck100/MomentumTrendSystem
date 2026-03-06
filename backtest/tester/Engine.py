from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

try:
    from .Config import EngineConfig
    from .Data_model import compute_returns, validate_prices
    from .Defensive import DefensiveAsset
    from .Selector import Selector
except ImportError:
    from Config import EngineConfig
    from Data_model import compute_returns, validate_prices
    from Defensive import DefensiveAsset
    from Selector import Selector


def _apply_costs(turnover: float, cost_bps: float, slippage_bps: float) -> float:
    total_cost_bps = float(cost_bps) + float(slippage_bps)
    return float(turnover * (total_cost_bps / 10000.0))


def _normalize_weights(weights: np.ndarray) -> np.ndarray:
    s = float(np.sum(weights))
    if s <= 0.0:
        return np.zeros_like(weights)
    return weights / s


def _weights_dict_to_vector(
    all_tickers: list[str], weights_dict: dict[str, float]
) -> np.ndarray:
    w = np.array([float(weights_dict.get(t, 0.0)) for t in all_tickers], dtype=float)
    if np.any(w < 0.0):
        raise ValueError("Negative weights are not supported in this engine.")
    return _normalize_weights(w)


def _sanitize_universe(universe: list[str], all_tickers: list[str]) -> list[str]:
    allowed = set(all_tickers)
    return [t for t in universe if t in allowed]


def _next_exec_date(
    idx: int, dates: pd.DatetimeIndex, trade_delay: int
) -> pd.Timestamp | None:
    target_idx = idx + int(trade_delay)
    if target_idx >= len(dates):
        return None
    return pd.Timestamp(dates[target_idx])


@dataclass
class EngineResult:
    daily: pd.DataFrame
    trades: pd.DataFrame


def run_engine(
    prices: pd.DataFrame,
    config: EngineConfig,
    universe_provider: Any,
    rebalance_gate: Any,
    entry_rule: Any,
    exit_rule: Any,
    selector: Selector,
    defensive_asset: DefensiveAsset,
) -> EngineResult:
    """
    Generic modular backtest engine.

    Lifecycle:
    1) On rebalance days, evaluate entry/exit/select rules and schedule target weights.
    2) Execute scheduled trades after `trade_delay` bars.
    3) Apply portfolio or defensive return daily.
    """
    validate_prices(prices)
    returns = compute_returns(prices)

    dates = prices.index
    all_tickers = list(prices.columns)

    w_current = np.zeros(len(all_tickers), dtype=float)
    in_market = False

    pending_w: np.ndarray | None = None
    pending_exec_date: pd.Timestamp | None = None
    pending_in_market: bool | None = None
    pending_reason: str | None = None

    equity = float(config.initial_capital)

    daily_rows: list[dict[str, Any]] = []
    trade_rows: list[dict[str, Any]] = []

    for i, date in enumerate(dates):
        date = pd.Timestamp(date)
        trade_cost = 0.0
        turnover = 0.0

        # Execute pending orders at open/close proxy on this bar before return application.
        if pending_exec_date is not None and date == pending_exec_date:
            w_target = pending_w if pending_w is not None else np.zeros_like(w_current)
            w_target = _normalize_weights(w_target)
            turnover = float(np.abs(w_target - w_current).sum())
            trade_cost = _apply_costs(turnover, config.cost_bps, config.slippage_bps)

            w_current = w_target
            if pending_in_market is not None:
                in_market = bool(pending_in_market)

            trade_rows.append(
                {
                    "date": date,
                    "event": "EXECUTE",
                    "reason": pending_reason,
                    "turnover": turnover,
                    "cost": trade_cost,
                    "in_market_after": in_market,
                }
            )

            pending_w = None
            pending_exec_date = None
            pending_in_market = None
            pending_reason = None

        # Daily return from risky sleeve or defensive sleeve.
        if in_market and float(w_current.sum()) > 0.0:
            r_vec = returns.loc[date].to_numpy(dtype=float)
            daily_ret = float(np.dot(w_current, r_vec))
        else:
            daily_ret = float(defensive_asset.daily_return(date))

        daily_ret -= trade_cost
        equity *= 1.0 + daily_ret

        daily_rows.append(
            {
                "date": date,
                "ret": daily_ret,
                "equity": equity,
                "in_market": in_market,
                "turnover": turnover,
                "cost": trade_cost,
            }
        )

        # New decision can be scheduled only if there is no pending order.
        if pending_exec_date is not None:
            continue

        if not rebalance_gate.is_rebalance_day(date, dates):
            continue

        raw_universe = list(universe_provider.tickers_on(date))
        universe = _sanitize_universe(raw_universe, all_tickers)

        should_enter = bool(entry_rule.enter(date, prices, universe))
        should_exit = bool(exit_rule.exit(date, prices, universe))

        next_date = _next_exec_date(i, dates, config.trade_delay)
        if next_date is None:
            continue

        if in_market and should_exit:
            pending_w = np.zeros_like(w_current)
            pending_exec_date = next_date
            pending_in_market = False
            pending_reason = "exit_rule"
            trade_rows.append(
                {"date": date, "event": "SCHEDULE", "reason": pending_reason}
            )
            continue

        if (not in_market) and should_enter:
            selected = selector.select(date, prices, returns, universe)
            target_w = _weights_dict_to_vector(all_tickers, selected)
            if float(target_w.sum()) > 0.0:
                pending_w = target_w
                pending_exec_date = next_date
                pending_in_market = True
                pending_reason = "entry_rule"
                trade_rows.append(
                    {"date": date, "event": "SCHEDULE", "reason": pending_reason}
                )
            continue

        # In-market rebalance: refresh weights if selector output changes.
        if in_market and (not should_exit):
            selected = selector.select(date, prices, returns, universe)
            target_w = _weights_dict_to_vector(all_tickers, selected)
            if float(target_w.sum()) > 0.0:
                if float(np.abs(target_w - w_current).sum()) > 1e-12:
                    pending_w = target_w
                    pending_exec_date = next_date
                    pending_in_market = True
                    pending_reason = "rebalance"
                    trade_rows.append(
                        {"date": date, "event": "SCHEDULE", "reason": pending_reason}
                    )

    daily_df = pd.DataFrame(daily_rows).set_index("date")
    trades_df = pd.DataFrame(trade_rows)
    return EngineResult(daily=daily_df, trades=trades_df)

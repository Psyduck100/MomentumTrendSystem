from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Dict, List, Any

import numpy as np
import pandas as pd

from Config import EngineConfig
from Data_model import compute_returns, validate_prices
from Selector import Selector
from UniverseProvider import UniverseProvider
from Rules import EntryRule, ExitRule
from Defensive import DefensiveAsset


def _apply_costs(turnover: float, cost_bps: float, slippage_bps: float) -> float:
    total_cost_bps = cost_bps + slippage_bps
    cost = turnover * (total_cost_bps / 10000.0)
    return cost


def _weights_dict_to_vector(
    all_tickers: List[str], weights_dict: Dict[str, float]
) -> np.ndarray:
    return np.array([weights_dict.get(t, 0.0) for t in all_tickers])


def _normalize_weights(weights: np.ndarray) -> np.ndarray:
    s = float(np.sum(weights))
    if s > 0.0:
        return weights / s
    return weights


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
    validate_prices(prices)
    returns = compute_returns(prices)

    dates = prices.index
    all_tickers = list(prices.columns)

    w_current = np.zeros(len(all_tickers), dtype=float)

    pending_w: Optional[np.ndarray] = None
    pending_exec_date = Optional[pd.Timestamp] = None
    pending_state: Optional[str] = None

    in_market = False

    port_rets: List[float] = []
    equities: List[float] = []
    states: List[str] = []

    equity = config.initial_capital

    trade_rows = []

    for i, date in enumerate(dates):
        # Execute pending trade if we are on exec date
        cost = 0.0
        if pending_exec_date is not None and date == pending_exec_date:
            w_target = pending_w if pending_w is not None else np.zeros_like(w_current)
            w_target = _normalize_weights(w_target)

            turnover = float(np.abs(w_target - w_current).sum())
            cost = _apply_costs(turnover, config.cost_bps, config.slippage_bps)
            w_current = w_target
            if pending_state is not None:
                in_market = pending_state
            
            trade_rows.append({
                "date":date,
                "action":"EXECUTE",
                "cost":cost,
                "in_market_after": in_market,
            })

            pending_w = None
            pending_exec_date = None
            pending_state = None
        
        #compute today's return
        if float(w_current.sum()) > 0.0:
            daily_ret = defensive_asset.daily_return(date, prices, returns, w_current)

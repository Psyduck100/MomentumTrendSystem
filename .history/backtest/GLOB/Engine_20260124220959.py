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
    universe_provider: UniverseProvider,







)
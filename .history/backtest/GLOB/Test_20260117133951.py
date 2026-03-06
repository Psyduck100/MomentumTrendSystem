from dataclasses import dataclass
from typing import Optional
import pandas as pd


@dataclass(frozen=True)
class EngineConfig:
    cost_bps: float = 0.0
    slippage_bps: float = 0.0
    initial_capital: float = 1.0
    trade_delay: int = 1

def compute_returns(prices: pd.DataFrame)
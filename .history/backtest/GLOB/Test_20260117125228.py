from dataclassses import dataclass
from typing import Optional


@dataclass(frozen=True)
class EngineConfig:
    cost_bps: float = 0.0
    slippage_bps: float = 0.0
    initial_capital: float = 1.0

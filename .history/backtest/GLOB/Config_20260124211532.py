from dataclasses import dataclass


@dataclass(frozen=True)
class EngineConfig:
    cost_bps: float = 0.0
    slippage_bps: float = 0.0
    initial_capital: float = 1.0
    trade_delay: int = 1

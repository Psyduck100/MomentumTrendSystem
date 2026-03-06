from dataclasses import dataclass


@dataclass(frozen=True)
class EngineConfig:
    """Execution and cost configuration for the modular testing engine."""
    cost_bps: float = 0.0
    slippage_bps: float = 0.0
    initial_capital: float = 1.0
    trade_delay: int = 1

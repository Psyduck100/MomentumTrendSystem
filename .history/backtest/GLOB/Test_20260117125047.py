from dataclassses import dataclass
from typing import Optional


@dataclass(frozen=True)
class EngineConfig:
    cost_bps: float

from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List


@dataclass(slots=True)
class PriceBar:
    symbol: str
    as_of: datetime
    close: float


@dataclass(slots=True)
class PriceSeries:
    symbol: str
    bars: List[PriceBar]

    def closes(self) -> List[float]:
        return [bar.close for bar in self.bars]


@dataclass(slots=True)
class MomentumScore:
    symbol: str
    score: float


@dataclass(slots=True)
class PortfolioTarget:
    weights: Dict[str, float]
    as_of: datetime
    bucket_rankings: Dict[str, List[str]] | None = None
    # Track selected leader per bucket so we can persist and apply rank-gap on next run.
    bucket_selection: Dict[str, str | None] | None = None

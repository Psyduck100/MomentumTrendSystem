from dataclasses import dataclass, field
from pathlib import Path
from typing import List

from momentum_program.analytics.constants import (
    SCORE_MODE_RW_3_6_9_12,
    SCORE_MODE_12M_MINUS_1M,
    SCORE_MODE_BLEND_6_12,
)

VOLUME_THRESHOLD = 500_000
AUM_THRESHOLD = 200_000_000
PRICE_THRESHOLD = 5.0
COOKIES = {
    "sessionid": "qk7kmdr4ilzw1qjxmyshtgb89onkla9t",
    "sessionid_sign": "v3:odro6V2zj19g9RARCj7gU/KeAp9dHuJhbT55oC23FTc=",
}  # personal TradingView cookies


@dataclass(slots=True)
class DataConfig:
    tickers: List[str] = field(
        default_factory=lambda: ["SPY", "QQQ", "IWM", "EFA", "EEM"]
    )
    # Fetch enough history to cover 252 trading days (~12 months) plus cushion for holidays.
    lookback_days: int = 400
    warmup_days: int = 30
    data_dir: Path = Path("data_cache")


@dataclass(slots=True)
class StrategyConfig:
    top_n: int = 3
    top_n_per_bucket: int | None = None
    rebalance_frequency_days: int = 21
    momentum_weight: float = 1.0
    score_mode: str = (
        SCORE_MODE_12M_MINUS_1M  # default live: recency-weighted 3/6/9/12M
    )
    vol_adjusted: bool = False
    market_filter: bool = False
    market_ticker: str = "SPY"
    defensive_bucket: str = "Bonds"
    rank_gap_threshold: int = 2  # require new pick to beat current by this many ranks
    force_broad_us: bool = False  # keep false by default per user preference
    broad_us_ticker: str = "SPY"
    vol_target: float | None = None
    excluded_buckets: list[str] = field(default_factory=lambda: ["Bonds", "REITs"])


@dataclass(slots=True)
class RiskConfig:
    max_position_pct: float = 0.33
    cash_buffer_pct: float = 0.05


@dataclass(slots=True)
class AppConfig:
    data: DataConfig = field(default_factory=DataConfig)
    strategy: StrategyConfig = field(default_factory=StrategyConfig)
    risk: RiskConfig = field(default_factory=RiskConfig)

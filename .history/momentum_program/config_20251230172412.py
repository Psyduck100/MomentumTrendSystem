from dataclasses import dataclass, field
from pathlib import Path
from typing import List

VOLUME_THRESHOLD = 500_000
AUM_THRESHOLD = 20_000_000
PRICE_THRESHOLD = 5.0
COOKIES = {
    "sessionid": "qk7kmdr4ilzw1qjxmyshtgb89onkla9t",
    "sessionid_sign": "v3:odro6V2zj19g9RARCj7gU/KeAp9dHuJhbT55oC23FTc=",
}  # personal TradingView cookies (do not commit publicly)


@dataclass(slots=True)
class DataConfig:
    tickers: List[str] = field(
        default_factory=lambda: ["SPY", "QQQ", "IWM", "EFA", "EEM"]
    )
    lookback_days: int = 252
    warmup_days: int = 30
    data_dir: Path = Path("data_cache")


@dataclass(slots=True)
class StrategyConfig:
    top_n: int = 3
    rebalance_frequency_days: int = 21
    momentum_weight: float = 1.0


@dataclass(slots=True)
class RiskConfig:
    max_position_pct: float = 0.33
    cash_buffer_pct: float = 0.05


@dataclass(slots=True)
class AppConfig:
    data: DataConfig = field(default_factory=DataConfig)
    strategy: StrategyConfig = field(default_factory=StrategyConfig)
    risk: RiskConfig = field(default_factory=RiskConfig)

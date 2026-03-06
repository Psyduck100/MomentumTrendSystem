from datetime import datetime, timedelta

from momentum_program.config import AppConfig
from momentum_program.data_providers.yfinance_provider import YFinanceProvider
from momentum_program.strategy.ranking import build_targets


class MomentumPipeline:
    def __init__(self, cfg: AppConfig) -> None:
        self.cfg = cfg
        self.provider = YFinanceProvider()

    def run(self) -> None:
        end = datetime.utcnow()
        start = end - timedelta(days=self.cfg.data.lookback_days)
        price_series = self.provider.fetch(self.cfg.data.tickers, start, end)
        targets = build_targets(price_series, self.cfg.strategy, end)
        self._emit_targets(targets)

    def _emit_targets(self, targets) -> None:
        # Skeleton placeholder; wire into broker or reporting later.
        print(f"Target weights as of {targets.as_of:%Y-%m-%d} -> {targets.weights}")

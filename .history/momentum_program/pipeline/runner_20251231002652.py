from datetime import datetime, timedelta, timezone

from momentum_program.config import AppConfig
from momentum_program.strategy.ranking import build_targets
from momentum_program.universe.base import UniverseProvider
from momentum_program.data_providers.base import DataProvider


class MomentumPipeline:
    def __init__(
        self,
        cfg: AppConfig,
        universe_provider: UniverseProvider,
        price_provider: DataProvider,
    ) -> None:
        self.cfg = cfg
        self.universe_provider = universe_provider
        self.price_provider = price_provider

    def run(self) -> None:
        end = datetime.now(timezone.utc)
        start = end - timedelta(days=self.cfg.data.lookback_days)
        tickers = self.universe_provider.get_tickers()
        bucket_map = self.universe_provider.get_bucket_map()
        if not tickers:
            tickers = self.cfg.data.tickers
            bucket_map = {}
        price_series = self.price_provider.fetch(tickers, start, end)
        targets = build_targets(price_series, self.cfg.strategy, end, bucket_map)
        self._emit_targets(targets)

    def _emit_targets(self, targets) -> None:
        # Skeleton placeholder; wire into broker or reporting later.
        print(f"Target weights as of {targets.as_of:%Y-%m-%d} -> {targets.weights}")

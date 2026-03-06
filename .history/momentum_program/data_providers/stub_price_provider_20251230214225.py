from datetime import datetime, timedelta
from typing import Sequence

from momentum_program.data_models import PriceBar, PriceSeries
from momentum_program.data_providers.base import DataProvider


class StubPriceProvider(DataProvider):
    """Deterministic stub data for testing."""

    def fetch(self, symbols: Sequence[str], start: datetime, end: datetime) -> Sequence[PriceSeries]:
        series: list[PriceSeries] = []
        days = (end - start).days or 1
        for symbol in symbols:
            bars: list[PriceBar] = []
            for day in range(days):
                as_of = start + timedelta(days=day)
                price = 100 + day * 0.1  # deterministic dummy data
                bars.append(PriceBar(symbol=symbol, as_of=as_of, close=price))
            series.append(PriceSeries(symbol=symbol, bars=bars))
        return series

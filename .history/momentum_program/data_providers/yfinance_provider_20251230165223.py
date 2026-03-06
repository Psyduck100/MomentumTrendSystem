from tradingview_screener import Query, col
from momentum_program.config import AUM_THRESHOLD, VOLUME_THRESHOLD, PRICE_THRESHOLD
from pprint import pprint

q = (
    Query()
    .select(
        "ticker",
        "name",
        "volume",
        "type",
        "asset_class",
        "category",
        "focus",
        "holdings_region",
        "country",
        "inverse_flag",
        "leverage_flag",
    )
    .where(
        col("type").equals("etf"),
        col("aum").greater_than(AUM_THRESHOLD),
        col("average_volume_30d_calc").greater_than(VOLUME_THRESHOLD),
    )
)

pprint(q.query)


class YFinanceProvider(DataProvider):

    def fetch(
        self, symbols: Sequence[str], start: datetime, end: datetime
    ) -> Sequence[PriceSeries]:
        series: list[PriceSeries] = []
        for symbol in symbols:
            bars = self._build_stub_series(symbol, start, end)
            series.append(PriceSeries(symbol=symbol, bars=bars))
        return series

    def _build_stub_series(
        self, symbol: str, start: datetime, end: datetime
    ) -> list[PriceBar]:
        days = (end - start).days or 1
        bars: list[PriceBar] = []
        for day in range(days):
            as_of = start + timedelta(days=day)
            price = 100 + day * 0.1  # deterministic dummy data for now
            bars.append(PriceBar(symbol=symbol, as_of=as_of, close=price))
        return bars

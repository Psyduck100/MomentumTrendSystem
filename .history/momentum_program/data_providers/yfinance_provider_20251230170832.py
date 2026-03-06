from datetime import datetime, timedelta
from typing import Sequence

import pandas as pd
from tradingview_screener import Query, col

from momentum_program.config import AUM_THRESHOLD, VOLUME_THRESHOLD, PRICE_THRESHOLD
from momentum_program.data_models import PriceBar, PriceSeries
from momentum_program.data_providers.base import DataProvider
from pprint import pprint

q = (
    Query()
    .select(
        "name",
        "volume",
        "type",
        "typespecs",
        "asset_class",
        "category",
        "focus",
        "holdings_region",
        "country",
        "inverse_flag",
        "leveraged_flag",
    )
    .where(
        col("type")
        == "fund",  # ETFs are classified as funds; typespecs carries the ETF tag
        col("typespecs").has(["etf"]),
        col("aum") > AUM_THRESHOLD,
        col("average_volume_30d_calc") > VOLUME_THRESHOLD,
        col("close") >= PRICE_THRESHOLD,
    )
)


def fetch_all_etfs(q, chunk=500):
    all_rows = []
    offset = 0
    total = None
    while True:
        q.range(offset, offset + chunk)
        total, df = q.get_scanner_data()
        all_rows.append(df)
        offset += chunk
        if len(df) < chunk:
            break
    return pd.concat(all_rows, ignore_index=True)


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


if __name__ == "__main__":
    # Allow running this file to inspect the screener payload and first rows.
    pprint(q.query)
    _, etf_df = q.get_scanner_data()
    print(etf_df.head())

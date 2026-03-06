from datetime import datetime, timedelta
from typing import Sequence

import pandas as pd
from tradingview_screener import Query, col

from momentum_program.config import (
    AUM_THRESHOLD,
    VOLUME_THRESHOLD,
    PRICE_THRESHOLD,
    COOKIES,
)
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


def fetch_all_etfs(q, chunk=50):
    all_rows = []
    offset = 0
    total = None
    while True:
        page = q.copy().offset(offset).limit(chunk)
        try:
            total, df = page.get_scanner_data()
        except Exception as exc:
            print(f"Error fetching page at offset {offset}: {exc}")
            break
        if df is None or df.empty:
            break
        all_rows.append(df)
        offset += len(df)
        if total is not None and offset >= total:
            break
    if not all_rows:
        return total, pd.DataFrame()
    return total, pd.concat(all_rows, ignore_index=True)


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
    total, etf_df = fetch_all_etfs(q, chunk=50)
    print("total:", total)
    print(etf_df.shape)

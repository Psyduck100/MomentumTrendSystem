from datetime import datetime
from typing import Sequence

import yfinance as yf

from momentum_program.data_models import PriceBar, PriceSeries
from momentum_program.data_providers.base import DataProvider


class YahooPriceProvider(DataProvider):
    """Fetches historical prices from Yahoo Finance using yfinance."""

    def fetch(
        self, symbols: Sequence[str], start: datetime, end: datetime
    ) -> Sequence[PriceSeries]:
        if not symbols:
            return []
        data = yf.download(
            list(symbols),
            start=start,
            end=end,
            progress=False,
            auto_adjust=True,
            interval="1d",
            threads=True,
        )
        series_list: list[PriceSeries] = []

        # flatten multi-index
        close_frame = data["Close"] if "Close" in data else data
        if close_frame.empty:
            return []

        # if only one symbol
        if close_frame.ndim == 1:
            symbol = symbols[0]
            bars = [
                PriceBar(symbol=symbol, as_of=idx.to_pydatetime(), close=float(val))
                for idx, val in close_frame.dropna().items()
            ]
            series_list.append(PriceSeries(symbol=symbol, bars=bars))
        # multiple symbols
        else:
            for symbol in close_frame.columns:
                closes = close_frame[symbol].dropna()
                bars = [
                    PriceBar(symbol=symbol, as_of=idx.to_pydatetime(), close=float(val))
                    for idx, val in closes.items()
                ]
                series_list.append(PriceSeries(symbol=symbol, bars=bars))
        return series_list

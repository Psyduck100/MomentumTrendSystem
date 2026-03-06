from datetime import datetime
from typing import Sequence

import yfinance as yf

from momentum_program.data_models import PriceBar, PriceSeries
from momentum_program.data_providers.base import DataProvider


class YahooPriceProvider(DataProvider):
    """Fetches historical prices from Yahoo Finance using yfinance."""

    def fetch(self, symbols: Sequence[str], start: datetime, end: datetime) -> Sequence[PriceSeries]:
        if not symbols:
            return []
        data = yf.download(list(symbols), start=start, end=end, progress=False, auto_adjust=False)
        series_list: list[PriceSeries] = []

        # If multiple symbols, data has columns like ('Close', 'AAPL'); single symbol returns flat columns
        close_frame = data["Close"] if "Close" in data else data
        if close_frame.empty:
            return []

        if close_frame.ndim == 1:
            # Single symbol
            symbol = symbols[0]
            bars = [
                PriceBar(symbol=symbol, as_of=idx.to_pydatetime(), close=float(val))
                for idx, val in close_frame.dropna().items()
            ]
            series_list.append(PriceSeries(symbol=symbol, bars=bars))
        else:
            for symbol in close_frame.columns:
                closes = close_frame[symbol].dropna()
                bars = [
                    PriceBar(symbol=symbol, as_of=idx.to_pydatetime(), close=float(val))
                    for idx, val in closes.items()
                ]
                series_list.append(PriceSeries(symbol=symbol, bars=bars))
        return series_list

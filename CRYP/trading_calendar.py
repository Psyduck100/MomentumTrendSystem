from __future__ import annotations

from datetime import datetime
from typing import Iterable

import pandas as pd
import yfinance as yf


def trading_days_from_series(series: pd.Series) -> pd.DatetimeIndex:
    if series.empty:
        raise ValueError("series is empty")
    return pd.DatetimeIndex(series.index).tz_localize(None)


def get_trading_days(
    start: datetime,
    end: datetime,
    symbol: str = "SPY",
) -> pd.DatetimeIndex:
    data = yf.download(
        symbol,
        start=start,
        end=end,
        progress=False,
        auto_adjust=True,
        interval="1d",
        threads=True,
    )
    if data.empty:
        raise ValueError(f"No data returned for {symbol}")
    close = data["Close"] if "Close" in data else data
    return trading_days_from_series(close)


def gate_days(index: Iterable[pd.Timestamp], gate: str) -> pd.DatetimeIndex:
    idx = pd.DatetimeIndex(index).tz_localize(None)
    gate_index = idx.to_series().resample(gate).last().index
    return gate_index.intersection(idx)

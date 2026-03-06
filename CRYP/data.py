from __future__ import annotations

from datetime import datetime
from typing import Iterable

import pandas as pd
import yfinance as yf


def fetch_close(symbol: str, start: datetime, end: datetime) -> pd.Series:
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
    close = close.dropna()
    if isinstance(close, pd.Series):
        return close.rename(symbol)
    if close.shape[1] == 1:
        series = close.iloc[:, 0]
        series.name = symbol
        return series
    raise ValueError(f"Expected single series for {symbol}, got columns={list(close.columns)}")


def fetch_closes(symbols: Iterable[str], start: datetime, end: datetime) -> pd.DataFrame:
    data = yf.download(
        list(symbols),
        start=start,
        end=end,
        progress=False,
        auto_adjust=True,
        interval="1d",
        threads=True,
    )
    if data.empty:
        raise ValueError(f"No data returned for {symbols}")
    close = data["Close"] if "Close" in data else data
    return close.dropna(how="all")


def build_btc_proxy(
    btc_close: pd.Series,
    trading_days: pd.DatetimeIndex,
    fee_annual: float = 0.0025,
) -> pd.DataFrame:
    if btc_close.empty:
        raise ValueError("btc_close is empty")
    btc = btc_close.copy()
    btc.index = pd.DatetimeIndex(btc.index).tz_localize(None)
    td = pd.DatetimeIndex(trading_days).tz_localize(None)
    btc_td = btc.reindex(td).ffill()
    ret_gross = btc_td.pct_change().fillna(0.0)
    fee_daily = fee_annual / 252.0
    ret_net = ret_gross - fee_daily
    return pd.DataFrame(
        {
            "btc_td": btc_td,
            "ret_gross": ret_gross,
            "ret_net": ret_net,
        }
    )

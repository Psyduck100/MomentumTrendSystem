from __future__ import annotations

import pandas as pd


class RebalanceGate:
    def is_rebalance_day(
        self, date: pd.Timestamp, trading_calendar: pd.DatetimeIndex
    ) -> bool:
        raise NotImplementedError


class DailyRebalanceGate(RebalanceGate):
    def is_rebalance_day(
        self, date: pd.Timestamp, trading_calendar: pd.DatetimeIndex
    ) -> bool:
        return date in trading_calendar


class WeeklyRebalanceGate(RebalanceGate):
    """Rebalance on first or last trading day of each calendar week."""

    def __init__(self, when: str = "last"):
        if when not in {"first", "last"}:
            raise ValueError("when must be 'first' or 'last'.")
        self.when = when

    def is_rebalance_day(
        self, date: pd.Timestamp, trading_calendar: pd.DatetimeIndex
    ) -> bool:
        s = pd.Series(trading_calendar, index=trading_calendar)
        if self.when == "first":
            target = s.resample("W").first().dropna()
        else:
            target = s.resample("W").last().dropna()
        return date in set(pd.DatetimeIndex(target.values))


class MonthlyRebalanceGate(RebalanceGate):
    """Rebalance on first or last trading day of each calendar month."""

    def __init__(self, when: str = "last"):
        if when not in {"first", "last"}:
            raise ValueError("when must be 'first' or 'last'.")
        self.when = when

    def is_rebalance_day(
        self, date: pd.Timestamp, trading_calendar: pd.DatetimeIndex
    ) -> bool:
        s = pd.Series(trading_calendar, index=trading_calendar)
        if self.when == "first":
            target = s.resample("MS").first().dropna()
        else:
            target = s.resample("ME").last().dropna()
        return date in set(pd.DatetimeIndex(target.values))

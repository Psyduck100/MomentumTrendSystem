from __future__ import annotations

import pandas as pd


class DefensiveAsset:
    def daily_return(self, date: pd.Timestamp) -> float:
        raise NotImplementedError


class ConstantReturnDefensive(DefensiveAsset):
    def __init__(self, rate_annual: float = 0.0, trading_days: int = 252):
        self.rate_annual = float(rate_annual)
        self.trading_days = int(trading_days)

    def daily_return(self, date: pd.Timestamp) -> float:
        return float((1.0 + self.rate_annual) ** (1.0 / self.trading_days) - 1.0)


class SeriesReturnDefensive(DefensiveAsset):
    """
    Defensive returns supplied as a daily return time series.
    Missing dates are treated as 0.0 return.
    """

    def __init__(self, returns: pd.Series):
        s = returns.copy()
        s.index = pd.to_datetime(s.index)
        self.returns = s.sort_index()

    def daily_return(self, date: pd.Timestamp) -> float:
        return float(self.returns.get(date, 0.0))

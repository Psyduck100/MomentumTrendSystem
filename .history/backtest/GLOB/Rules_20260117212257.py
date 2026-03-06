import pandas as pd


class EntryRule:
    def enter(self, date: pd.Timestamp, prices, universe) -> bool:
        raise NotImplementedError


class ExitRule:
    def exit(self, date: pd.Timestamp, prices, universe) -> bool:
        raise NotImplementedError


class TwelveMonthMomentumEntryRule(EntryRule):
    def __init__(self, lookback_days: float = 252):
        self.lookback_days = lookback_days

    def enter(self, date: pd.Timestamp, prices, universe) -> bool:
        px = prices.loc[:date, universe]
        if len(px) < self.lookback_days:
            return False
        start = px.iloc[-self.lookback_days]
        end = px.iloc[-1]
        mom = (end / start) - 1.0
        return mom > 0.0


class TwelveMonthMomentumExitRule(ExitRule):
    def __init__(self, lookback_days: float = 252):
        self.lookback_days = lookback_days

    def exit(self, date: pd.Timestamp, prices, universe) -> bool:
        px = prices.loc[:date, universe]
        if len(px) < self.lookback_days:
            return False
        start = px.iloc[-self.lookback_days]
        end = px.iloc[-1]
        mom = (end / start) - 1.0
        return mom <= 0.0

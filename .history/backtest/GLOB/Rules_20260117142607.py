import pandas as pd


class EntryRule:
    def enter(self, date: pd.Timestamp, prices: pd.Series) -> bool:
        raise NotImplementedError

class ExitRule:
    def exit(self, date: pd.Timestamp, prices: pd.Series) -> bool:
        raise NotImplementedError
    

class 12MonthMomentumEntryRule(EntryRule):
    def __init__(self, lookback_days: float = 252):
        self.lookback_days = lookback_days

    def enter(self, date: pd.Timestamp, prices: pd.Series) -> bool:
        pass
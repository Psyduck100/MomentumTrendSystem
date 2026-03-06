import pandas as pd


class EntryRule:
    def enter(self, date: pd.Timestamp, prices: pd.Series) -> bool:
        raise NotImplementedError

import pandas as pd
from typing import List


class UniverseProvider:
    def tickers_on(self, date: pd.Timestamp) -> List[str]:
        raise NotImplementedError


class StaticUniverse(UniverseProvider):
    def __init__(self, tickers: list[str]):
        self.tickers = list(tickers)
    def tickers_on(self, date: pd.Timestamp) -> list[str]:
        return self.tickers

        
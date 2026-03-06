import pandas as pd
from typing import List


class UniverseProvider:
    def tickers_on(self, date: pd.Timestamp) -> List[str]:
        raise NotImplementedError


class StaticUniverse(UniverseProvider):
    def __init__(self, tickes: list[str]):
        self.tickers = list(tickers)

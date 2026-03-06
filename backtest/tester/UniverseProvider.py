from __future__ import annotations

from typing import List

import pandas as pd


class UniverseProvider:
    def tickers_on(self, date: pd.Timestamp) -> List[str]:
        raise NotImplementedError


class StaticUniverse(UniverseProvider):
    def __init__(self, tickers: list[str]):
        self.tickers = list(dict.fromkeys(str(t).strip() for t in tickers if str(t).strip()))

    def tickers_on(self, date: pd.Timestamp) -> list[str]:
        return self.tickers


class ScheduledUniverse(UniverseProvider):
    """
    Universe that can change over time.
    `schedule` keys are effective dates; values are ticker lists.
    """

    def __init__(self, schedule: dict[pd.Timestamp | str, list[str]]):
        parsed = []
        for k, v in schedule.items():
            eff = pd.Timestamp(k)
            tickers = list(dict.fromkeys(str(t).strip() for t in v if str(t).strip()))
            parsed.append((eff, tickers))
        if len(parsed) == 0:
            raise ValueError("schedule cannot be empty.")
        self.schedule = sorted(parsed, key=lambda x: x[0])

    def tickers_on(self, date: pd.Timestamp) -> list[str]:
        current = self.schedule[0][1]
        for eff, tickers in self.schedule:
            if date >= eff:
                current = tickers
            else:
                break
        return current

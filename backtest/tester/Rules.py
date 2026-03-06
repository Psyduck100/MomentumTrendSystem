from __future__ import annotations

import pandas as pd


class EntryRule:
    def enter(
        self, date: pd.Timestamp, prices: pd.DataFrame, universe: list[str]
    ) -> bool:
        raise NotImplementedError


class ExitRule:
    def exit(self, date: pd.Timestamp, prices: pd.DataFrame, universe: list[str]) -> bool:
        raise NotImplementedError


class AlwaysEnterRule(EntryRule):
    def enter(self, date: pd.Timestamp, prices: pd.DataFrame, universe: list[str]) -> bool:
        return len(universe) > 0


class NeverExitRule(ExitRule):
    def exit(self, date: pd.Timestamp, prices: pd.DataFrame, universe: list[str]) -> bool:
        return False


def _basket_momentum(
    date: pd.Timestamp, prices: pd.DataFrame, universe: list[str], lookback_days: int
) -> float | None:
    if len(universe) == 0:
        return None

    px = prices.loc[:date, universe].dropna(how="all")
    if len(px) <= lookback_days:
        return None

    start = px.iloc[-(lookback_days + 1)]
    end = px.iloc[-1]
    mom = (end / start) - 1.0
    return float(mom.mean(skipna=True))


class NDaysMomentumEntryRule(EntryRule):
    def __init__(self, lookback_days: int = 252):
        self.lookback_days = int(lookback_days)

    def enter(self, date: pd.Timestamp, prices: pd.DataFrame, universe: list[str]) -> bool:
        mom = _basket_momentum(date, prices, universe, self.lookback_days)
        return bool(mom is not None and mom > 0.0)


class NDaysMomentumExitRule(ExitRule):
    def __init__(self, lookback_days: int = 252):
        self.lookback_days = int(lookback_days)

    def exit(self, date: pd.Timestamp, prices: pd.DataFrame, universe: list[str]) -> bool:
        mom = _basket_momentum(date, prices, universe, self.lookback_days)
        return bool(mom is not None and mom <= 0.0)

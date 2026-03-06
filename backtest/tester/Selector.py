from __future__ import annotations

import pandas as pd


class Selector:
    def select(
        self,
        date: pd.Timestamp,
        prices: pd.DataFrame,
        returns: pd.DataFrame,
        universe: list[str],
    ) -> dict[str, float]:
        raise NotImplementedError


class EqualWeightSelector(Selector):
    def select(
        self,
        date: pd.Timestamp,
        prices: pd.DataFrame,
        returns: pd.DataFrame,
        universe: list[str],
    ) -> dict[str, float]:
        universe = [t for t in universe if t in prices.columns]
        if len(universe) == 0:
            return {}
        w = 1.0 / len(universe)
        return {t: w for t in universe}


class TopMomentumSelector(Selector):
    def __init__(self, lookback_days: int = 252, top_n: int = 1):
        self.lookback_days = int(lookback_days)
        self.top_n = int(top_n)

    def select(
        self,
        date: pd.Timestamp,
        prices: pd.DataFrame,
        returns: pd.DataFrame,
        universe: list[str],
    ) -> dict[str, float]:
        universe = [t for t in universe if t in prices.columns]
        if len(universe) == 0:
            return {}

        px = prices.loc[:date, universe].dropna(how="all")
        if len(px) <= self.lookback_days:
            return {}

        start = px.iloc[-(self.lookback_days + 1)]
        end = px.iloc[-1]
        mom = (end / start) - 1.0
        mom = mom.dropna().sort_values(ascending=False)
        if mom.empty:
            return {}

        winners = mom.index[: self.top_n].tolist()
        w = 1.0 / len(winners)
        return {t: w for t in winners}

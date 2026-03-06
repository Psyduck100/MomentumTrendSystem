import pandas as pd


class DefensiveAsset:
    def daily_returns(self, date: pd.Timestamp) -> float:
        raise NotImplementedError


class ConstantReturnsDefensive(DefensiveAsset):
    def __init__(self, rate_annual: float = 0.0, trading_days: int = 252):
        self.rate_annual = rate_annual
        self.trading_days = trading_days

    def daily_returns(self, date: pd.Timestamp) -> float:
        return s(1.0 + self.rate_annual) ** (1.0 / self.trading_days) - 1.0

import pandas as pd


class RebalanceGate:
    def is_rebalance_day(
        self, date: pd.Timestamp, trading_calendar: pd.DatetimeIndex
    ) -> bool:
        raise NotImplementedError

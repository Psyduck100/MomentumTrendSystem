import pandas as pd


class RebalanceGate:
    def is_rebalance_day(
        self, date: pd.Timestamp, trading_calendar: pd.DatetimeIndex
    ) -> bool:
        raise NotImplementedError


class MonthlyRebalanceGate(RebalanceGate):
    def is_rebalance_day(self, date, trading_calendar):
        """Returns True if the given date is the first trading day of the month."""
        month = date.month
        year = date.year
        month_dates = trading_calendar[trading_calendar]

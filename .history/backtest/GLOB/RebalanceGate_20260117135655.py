import pandas as pd


class RebalanceGate:
    def is_rebalance_day(self, date: pd.Timestamp) -> bool:
        raise NotImplementedError

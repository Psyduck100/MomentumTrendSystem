
    @property
    def name(self) -> str:
        return "SPY_THEN_IEF"
"""Fallback Strategy Implementations

Different defensive asset strategies that can be plugged into backtests.
"""

from abc import ABC, abstractmethod
import pandas as pd
import yfinance as yf
from pathlib import Path


class FallbackStrategy(ABC):
    """Abstract base class for fallback strategies."""

    def __init__(self, start_date: str, end_date: str):
        self.start_date = start_date
        self.end_date = end_date
        self._data = None

    @abstractmethod
    def get_monthly_returns(self, monthly_dates: pd.DatetimeIndex) -> pd.Series:
        """Get monthly returns aligned to month-end dates.

        Args:
            monthly_dates: Month-end dates from main strategy

        Returns:
            Series of monthly returns aligned to those dates
        """
        pass

    @property
    def name(self) -> str:
        """Human-readable name for this strategy."""
        return self.__class__.__name__


class CashFallback(FallbackStrategy):
    """0% return (cash equivalent)."""

    def get_monthly_returns(self, monthly_dates: pd.DatetimeIndex) -> pd.Series:
        """Returns zero for all periods."""
        return pd.Series(0.0, index=monthly_dates, name="cash_returns")

    @property
    def name(self) -> str:
        return "CASH"


class TBillsFallback(FallbackStrategy):
    """TB3MS 3-month T-Bills from CSV."""

    def __init__(self, start_date: str, end_date: str, csv_path: str):
        super().__init__(start_date, end_date)
        self.csv_path = csv_path
        self._load_tbills()

    def _load_tbills(self):
        """Load TB3MS data from CSV."""
        df = pd.read_csv(self.csv_path)
        df["observation_date"] = pd.to_datetime(df["observation_date"])
        df = df.set_index("observation_date")
        # TB3MS is annualized rate; convert to monthly return
        df["monthly_ret"] = df["TB3MS"] / 100 / 12
        self._data = df

    def get_monthly_returns(self, monthly_dates: pd.DatetimeIndex) -> pd.Series:
        """Get TB3MS monthly returns aligned to dates."""
        if self._data is None:
            self._load_tbills()

        # Extract monthly returns and align to dates
        tbill_monthly = self._data.loc[
            self._data.index.to_period("M").isin(monthly_dates.to_period("M")),
            "monthly_ret",
        ]

        # Forward-fill for dates not in data
        tbill_monthly = tbill_monthly.reindex(monthly_dates, method="ffill")

        return pd.Series(
            tbill_monthly.values, index=monthly_dates, name="tbills_returns"
        )

    @property
    def name(self) -> str:
        return "TB3MS"


class YFinanceFallback(FallbackStrategy):
    """Fallback using ticker from yfinance (IEF, SPY, TLT, AGG, etc.)."""

    def __init__(self, start_date: str, end_date: str, ticker: str):
        super().__init__(start_date, end_date)
        self.ticker = ticker
        self._load_prices()

    def _load_prices(self):
        """Download prices from yfinance."""
        print(f"  Downloading {self.ticker} prices...")
        data = yf.download(
            self.ticker, start=self.start_date, end=self.end_date, progress=False
        )

        # Handle both single and multi-ticker downloads
        if isinstance(data.columns, pd.MultiIndex):
            self._data = (
                data[("Close", self.ticker)]
                if ("Close", self.ticker) in data.columns
                else data.iloc[:, 0]
            )
        else:
            self._data = data["Close"] if "Close" in data.columns else data["Adj Close"]

    def get_monthly_returns(self, monthly_dates: pd.DatetimeIndex) -> pd.Series:
        """Get monthly price returns aligned to dates."""
        if self._data is None:
            self._load_prices()

        # Resample to month-end
        monthly_prices = self._data.resample("ME").last()

        # Calculate returns
        monthly_ret = monthly_prices.pct_change()

        # Align to requested dates
        result = monthly_ret.reindex(monthly_dates, method="ffill")

        return pd.Series(
            result.values, index=monthly_dates, name=f"{self.ticker.lower()}_returns"
        )

    @property
    def name(self) -> str:
        return self.ticker.upper()


def get_fallback_strategy(
    fallback_type: str,
    start_date: str,
    end_date: str,
    fallback_ticker: str = None,
    fallback_csv: str = None,
) -> FallbackStrategy:
    """Factory function to create appropriate fallback strategy.

    Args:
        fallback_type: Type of fallback (from FallbackType enum)
        start_date: Backtest start date
        end_date: Backtest end date
        fallback_ticker: Ticker symbol for yfinance fallbacks
        fallback_csv: CSV path for TB3MS fallback

    Returns:
        FallbackStrategy instance
    """
    fallback_type = fallback_type.lower()

    if fallback_type == "cash":
        return CashFallback(start_date, end_date)

    elif fallback_type == "tbills":
        if fallback_csv is None:
            fallback_csv = "CSVs/TB3MS.csv"
        return TBillsFallback(start_date, end_date, fallback_csv)

    elif fallback_type == "ief":
        return YFinanceFallback(start_date, end_date, "IEF")

    elif fallback_type in ["spy", "tlt", "agg", "vgit"]:
        return YFinanceFallback(start_date, end_date, fallback_type.upper())

    elif fallback_type == "spy_then_ief":
        return SpyThenIefFallback(start_date, end_date)
    else:
        raise ValueError(f"Unknown fallback type: {fallback_type}")


if __name__ == "__main__":
    # Test fallback strategies
    print("Testing Fallback Strategies\n")

    # Test Cash
    cash = CashFallback("2005-01-01", "2025-12-31")
    print(f"✓ {cash.name} strategy loaded")

    # Test TB3MS
    try:
        tbills = TBillsFallback("2005-01-01", "2025-12-31", "CSVs/TB3MS.csv")
        print(f"✓ {tbills.name} strategy loaded")
    except Exception as e:
        print(f"✗ TB3MS strategy failed: {e}")

    # Test IEF
    try:
        ief = YFinanceFallback("2005-01-01", "2025-12-31", "IEF")
        print(f"✓ {ief.name} strategy loaded")
    except Exception as e:
        print(f"✗ IEF strategy failed: {e}")

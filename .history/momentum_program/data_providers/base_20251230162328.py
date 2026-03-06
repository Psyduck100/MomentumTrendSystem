from abc import ABC, abstractmethod
from datetime import datetime
from typing import Sequence

from momentum_program.data_models import PriceSeries


class DataProvider(ABC):
    """Defines interface for fetching historical price data."""

    @abstractmethod
    def fetch(
        self, symbols: Sequence[str], start: datetime, end: datetime
    ) -> Sequence[PriceSeries]:
        """Return price history between the given dates."""

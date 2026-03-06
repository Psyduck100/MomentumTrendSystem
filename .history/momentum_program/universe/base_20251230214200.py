from abc import ABC, abstractmethod
from typing import List


class UniverseProvider(ABC):
    """Abstract universe provider that returns a list of tickers."""

    @abstractmethod
    def get_tickers(self) -> List[str]:
        raise NotImplementedError

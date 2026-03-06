from abc import ABC, abstractmethod
from typing import Dict, List


class UniverseProvider(ABC):
    """Abstract universe provider that returns a list of tickers."""

    @abstractmethod
    def get_tickers(self) -> List[str]:
        raise NotImplementedError

    def get_bucket_map(self) -> Dict[str, str]:
        """Optional bucket metadata keyed by symbol. Defaults to none."""
        return {}

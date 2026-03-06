from pathlib import Path
from typing import List

import pandas as pd

from momentum_program.universe.base import UniverseProvider


class CsvUniverseProvider(UniverseProvider):
    """Loads tickers from a CSV file. Expects a 'ticker' column or uses the first column."""

    def __init__(self, path: Path) -> None:
        self.path = path

    def get_tickers(self) -> List[str]:
        if not self.path.exists():
            return []
        df = pd.read_csv(self.path)
        if "ticker" in df.columns:
            return df["ticker"].dropna().astype(str).tolist()
        return df.iloc[:, 0].dropna().astype(str).tolist()

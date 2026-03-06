from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd

from momentum_program.universe.base import UniverseProvider


class BucketedCsvUniverseProvider(UniverseProvider):
    """Loads tickers from multiple CSV files in a folder, bucketed by filename."""

    def __init__(self, folder: Path) -> None:
        self.folder = folder

    @staticmethod
    def _clean_symbol(raw: str) -> str:
        # Keep only the ticker portion before any comma, strip quotes/whitespace.
        return raw.split(",")[0].strip().strip("\"").strip("'")

    def _load(self) -> Tuple[List[str], Dict[str, str]]:
        tickers: List[str] = []
        bucket_map: Dict[str, str] = {}
        if not self.folder.exists():
            return tickers, bucket_map

        for csv_path in self.folder.glob("*.csv"):
            bucket = csv_path.stem
            try:
                df = pd.read_csv(csv_path, encoding="utf-8")
            except UnicodeDecodeError:
                df = pd.read_csv(csv_path, encoding="latin1")
            if df.empty:
                continue
            if "ticker" in df.columns:
                series = df["ticker"]
            else:
                series = df.iloc[:, 0]
            symbols = []
            seen: set[str] = set()
            for raw in series.dropna().astype(str).tolist():
                symbol = self._clean_symbol(raw)
                if not symbol or symbol in seen:
                    continue
                seen.add(symbol)
                symbols.append(symbol)
            tickers.extend(symbols)
            bucket_map.update({symbol: bucket for symbol in symbols})

        return tickers, bucket_map

    def get_tickers(self) -> List[str]:
        tickers, _ = self._load()
        return tickers

    def get_bucket_map(self) -> Dict[str, str]:
        _, bucket_map = self._load()
        return bucket_map

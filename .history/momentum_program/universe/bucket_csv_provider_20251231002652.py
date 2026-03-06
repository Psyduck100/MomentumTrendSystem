from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd

from momentum_program.universe.base import UniverseProvider


class BucketedCsvUniverseProvider(UniverseProvider):
    """Loads tickers from multiple CSV files in a folder, bucketed by filename."""

    def __init__(self, folder: Path) -> None:
        self.folder = folder

    def _load(self) -> Tuple[List[str], Dict[str, str]]:
        tickers: List[str] = []
        bucket_map: Dict[str, str] = {}
        if not self.folder.exists():
            return tickers, bucket_map

        for csv_path in self.folder.glob("*.csv"):
            bucket = csv_path.stem
            df = pd.read_csv(csv_path)
            if df.empty:
                continue
            if "ticker" in df.columns:
                series = df["ticker"]
            else:
                series = df.iloc[:, 0]
            symbols = (
                series.dropna().astype(str).str.strip().loc[lambda s: s != ""].tolist()
            )
            tickers.extend(symbols)
            bucket_map.update({symbol: bucket for symbol in symbols})

        return tickers, bucket_map

    def get_tickers(self) -> List[str]:
        tickers, _ = self._load()
        return tickers

    def get_bucket_map(self) -> Dict[str, str]:
        _, bucket_map = self._load()
        return bucket_map

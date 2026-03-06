from itertools import chain
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

        paths = chain(self.folder.glob("*.csv"), self.folder.glob("*.xlsx"))

        for path in paths:
            bucket = path.stem
            try:
                if path.suffix.lower() == ".csv":
                    try:
                        df = pd.read_csv(path, encoding="utf-8-sig")
                    except UnicodeDecodeError:
                        df = pd.read_csv(path, encoding="latin-1")
                else:
                    df = pd.read_excel(path)
            except Exception:
                continue
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

from typing import List

import pandas as pd
from tradingview_screener import Query, col

from momentum_program.config import (
    AUM_THRESHOLD,
    VOLUME_THRESHOLD,
    PRICE_THRESHOLD,
    COOKIES,
)

from momentum_program.universe.base import UniverseProvider


class TradingViewUniverseProvider(UniverseProvider):
    """Fetches ETF tickers from TradingView screener."""

    def __init__(self, chunk: int = 50) -> None:
        self.chunk = chunk

    def _build_query(self) -> Query:
        return (
            Query()
            .select("name")
            .where(
                col("type") == "fund",
                col("typespecs").has(["etf"]),
                col("aum") > AUM_THRESHOLD,
                col("average_volume_30d_calc") > VOLUME_THRESHOLD,
                col("close") >= PRICE_THRESHOLD,
                col("expense_ratio") < 0.5,
            )
            .set_markets("america")
        )

    def get_tickers(self) -> List[str]:
        q = self._build_query()
        all_rows = []
        offset = 0
        total = None
        while True:
            page = q.copy().offset(offset).limit(self.chunk)
            try:
                total, df = page.get_scanner_data(cookies=COOKIES)
            except Exception:
                break
            if df is None or df.empty:
                break
            all_rows.append(df)
            offset += len(df)
            if total is not None and offset >= total:
                break
        if not all_rows:
            return []
        df_all = pd.concat(all_rows, ignore_index=True)
        if "ticker" in df_all.columns:
            return df_all["ticker"].dropna().astype(str).tolist()
        # TradingView returns ticker in index as 's' column if not selected explicitly
        if "s" in df_all.columns:
            return df_all["s"].dropna().astype(str).tolist()
        return []

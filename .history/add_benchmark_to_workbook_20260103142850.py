"""Append SPY/QQQ annual returns to each sheet in us_rotation_annual_returns_all.xlsx."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import yfinance as yf

WORKBOOK = Path("us_rotation_annual_returns_all.xlsx")
START_YEAR = 2000
END_YEAR = 2026
BENCH_TICKERS = ["SPY", "QQQ"]


def compute_benchmark_annual_returns() -> dict[str, dict[int, float]]:
    data = yf.download(
        BENCH_TICKERS,
        start=f"{START_YEAR}-01-01",
        end=f"{END_YEAR}-01-01",
        interval="1mo",
        auto_adjust=True,
        progress=False,
    )
    if isinstance(data.columns, pd.MultiIndex):
        adj_close = data["Adj Close"].copy()
    else:
        adj_close = data.copy()
    adj_close = adj_close.dropna(how="all")
    monthly = adj_close.pct_change().dropna(how="all")
    annual = (1.0 + monthly).groupby(monthly.index.year).prod() - 1.0
    return {ticker: annual[ticker].to_dict() for ticker in BENCH_TICKERS}


def main() -> None:
    if not WORKBOOK.exists():
        raise FileNotFoundError(WORKBOOK)

    benchmark_maps = compute_benchmark_annual_returns()

    xls = pd.ExcelFile(WORKBOOK)
    updated: dict[str, pd.DataFrame] = {}
    for sheet in xls.sheet_names:
        df = xls.parse(sheet)
        if "year" in df.columns:
            df = df.copy()
            df["spy_annual_return"] = df["year"].map(benchmark_maps["SPY"])
            df["qqq_annual_return"] = df["year"].map(benchmark_maps["QQQ"])
        updated[sheet] = df

    with pd.ExcelWriter(WORKBOOK, engine="openpyxl") as writer:
        for sheet, frame in updated.items():
            frame.to_excel(writer, sheet_name=sheet, index=False)

    print(f"Updated {len(updated)} sheets in {WORKBOOK} with benchmark annual returns.")


if __name__ == "__main__":
    main()

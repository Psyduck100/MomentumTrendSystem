"""
US EQUITIES MOMENTUM STRATEGY - Entry Point

This module implements the primary monthly momentum strategy for US equities.

Strategy Configuration:
- Universe: US equity ETFs (SCHB, XLG, SCHV, QQQ, RSP) from CSVs/US_equities.csv
- Scoring: blend_6_12 (50% 6-month + 50% 12-month returns)
- Rank Gap: 0 (allow switching between any positions)
- Absolute Filter: ret_12m > 0 (require positive 12-month return, else defensive)
- Defensive Allocation: IEF (7-10 Year Treasury ETF)
- Rebalancing: Monthly (end-of-month)

Expected Performance (2002-2026 backtest):
- CAGR: 13.28%
- Sharpe Ratio: 0.92
- Max Drawdown: -27.18%

For monthly rebalancing, run: UsEquitiesRebalance.py

For strategy backtesting and analysis, use momentum_program/ package directly.
"""

from pathlib import Path

from momentum_program.config import AppConfig
from momentum_program.pipeline.runner import MomentumPipeline
from momentum_program.universe.bucket_csv_provider import BucketedCsvUniverseProvider
from momentum_program.universe.csv_provider import CsvUniverseProvider
from momentum_program.universe.tradingview_provider import TradingViewUniverseProvider
from momentum_program.data_providers.yahoo_price_provider import YahooPriceProvider


def main() -> None:
    """Run the momentum pipeline with configured universe and price provider."""
    cfg = AppConfig()
    bucket_folder = Path("CSVs")
    bucket_csvs = list(bucket_folder.glob("*.csv")) if bucket_folder.exists() else []
    csv_path = Path("etfs.csv")

    # Choose universe source: bucketed CSV folder, single CSV, else TradingView
    if bucket_csvs:
        universe = BucketedCsvUniverseProvider(bucket_folder)
    elif csv_path.exists():
        universe = CsvUniverseProvider(csv_path)
    else:
        universe = TradingViewUniverseProvider(chunk=50)

    # Choose price source: Yahoo (real) or Stub (tests)
    price_provider = YahooPriceProvider()
    # price_provider = StubPriceProvider()

    pipeline = MomentumPipeline(
        cfg, universe_provider=universe, price_provider=price_provider
    )
    pipeline.run()


if __name__ == "__main__":
    main()

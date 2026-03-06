from pathlib import Path

from momentum_program.config import AppConfig
from momentum_program.pipeline.runner import MomentumPipeline
from momentum_program.universe.bucket_csv_provider import BucketedCsvUniverseProvider
from momentum_program.universe.csv_provider import CsvUniverseProvider
from momentum_program.universe.tradingview_provider import TradingViewUniverseProvider
from momentum_program.data_providers.yahoo_price_provider import YahooPriceProvider


def main() -> None:
    cfg = AppConfig()
    bucket_folder = Path("CSVs")
    bucket_csvs = list(bucket_folder.glob("*.csv")) if bucket_folder.exists() else []
    csv_path = Path("etfs.csv")

    # Choose universe source: bucketed CSV folder, single CSV, else TradingView
    if bucket_csvs:
        # Exclude unwanted buckets (e.g., Bonds, REITs) for the live portfolio
        exclude = {"Bonds", "REITs"}
        universe = BucketedCsvUniverseProvider(bucket_folder, exclude_buckets=exclude)
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

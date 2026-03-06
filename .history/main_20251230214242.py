from pathlib import Path

from momentum_program.config import AppConfig
from momentum_program.pipeline.runner import MomentumPipeline
from momentum_program.universe.csv_provider import CsvUniverseProvider
from momentum_program.universe.tradingview_provider import TradingViewUniverseProvider
from momentum_program.data_providers.yahoo_price_provider import YahooPriceProvider
from momentum_program.data_providers.stub_price_provider import StubPriceProvider


def main() -> None:
    cfg = AppConfig()
    csv_path = Path("etfs.csv")

    # Choose universe source: CSV if present, else TradingView
    if csv_path.exists():
        universe = CsvUniverseProvider(csv_path)
    else:
        universe = TradingViewUniverseProvider(chunk=50)

    # Choose price source: Yahoo (real) or Stub (tests)
    price_provider = YahooPriceProvider()
    # price_provider = StubPriceProvider()

    pipeline = MomentumPipeline(cfg, universe_provider=universe, price_provider=price_provider)
    pipeline.run()


if __name__ == "__main__":
    main()

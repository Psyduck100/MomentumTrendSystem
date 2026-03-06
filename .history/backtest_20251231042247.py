from pathlib import Path

from momentum_program.config import AppConfig
from momentum_program.universe.bucket_csv_provider import BucketedCsvUniverseProvider
from momentum_program.universe.csv_provider import CsvUniverseProvider
from momentum_program.universe.tradingview_provider import TradingViewUniverseProvider
from momentum_program.backtest.engine import backtest_momentum
from momentum_program.backtest.metrics import compute_metrics, compute_turnover


def run_backtest() -> None:
    """Load universe, run monthly momentum backtest, and report metrics per bucket."""
    cfg = AppConfig()
    bucket_folder = Path("CSVs")
    bucket_csvs = list(bucket_folder.glob("*.csv")) if bucket_folder.exists() else []
    csv_path = Path("etfs.csv")

    # Choose universe source
    if bucket_csvs:
        universe = BucketedCsvUniverseProvider(bucket_folder)
    elif csv_path.exists():
        universe = CsvUniverseProvider(csv_path)
    else:
        universe = TradingViewUniverseProvider(chunk=50)

    tickers = universe.get_tickers()
    bucket_map = universe.get_bucket_map()

    if not tickers:
        print("No tickers found in universe.")
        return

    print(
        f"Loaded {len(tickers)} tickers across {len(set(bucket_map.values()))} buckets."
    )

    thresholds = [0.0, -0.05]
    lookback_options = [6, 12]
    rank_gaps = [0, 2]
    vol_flags = [True, False]

    for vol_flag in vol_flags:
        for lookback in lookback_options:
            for rank_gap_setting in rank_gaps:
                for threshold in thresholds:
                    print("\n" + "=" * 80)
                    print(
                        f"BACKTEST RESULTS (vol_adj: {vol_flag}, lookback: {lookback}M/1M, threshold: {threshold:.0%}, rank_gap: {rank_gap_setting})"
                    )
                    print("=" * 80)

                    backtest_data = backtest_momentum(
                        tickers=tickers,
                        bucket_map=bucket_map,
                        start_date="2015-01-01",
                        end_date="2025-12-31",
                        top_n_per_bucket=cfg.strategy.top_n_per_bucket or cfg.strategy.top_n,
                        lookback_long=lookback,
                        lookback_short=1,
                        vol_adjusted=vol_flag,
                        vol_lookback=6,
                        market_filter=True,
                        market_ticker="SPY",
                        defensive_bucket="Bonds",
                        market_threshold=threshold,
                        rank_gap_threshold=rank_gap_setting,
                    )
                    if not backtest_data["overall_returns"].empty:
                        overall_metrics = compute_metrics(
                            backtest_data["overall_returns"]["return"]
                        )
                        overall_turnover = compute_turnover(
                            backtest_data["overall_positions"]
                        )
                        print("\nOVERALL (net):")
                        print(f"  CAGR:       {overall_metrics['cagr']:.2%}")
                        print(f"  Volatility: {overall_metrics['volatility']:.2%}")
                        print(f"  Sharpe:     {overall_metrics['sharpe']:.2f}")
                        print(f"  Max DD:     {overall_metrics['max_drawdown']:.2%}")
                        print(f"  Total Ret:  {overall_metrics['total_return']:.2%}")
                        print(f"  Turnover:   {overall_turnover:.2%}")

    print("\n" + "=" * 80)


if __name__ == "__main__":
    run_backtest()

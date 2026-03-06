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

    thresholds = [0.0, -0.03, -0.05]
    lookback_options = [6, 12]
    rank_gap_setting = 2  # try rank-gap=2 for turnover control tests

    for lookback in lookback_options:
        for threshold in thresholds:
            print("\n" + "=" * 80)
            print(
                f"BACKTEST RESULTS (lookback: {lookback}M long / 1M short, market threshold: {threshold:.0%}, rank_gap: {rank_gap_setting})"
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
                vol_adjusted=True,
                vol_lookback=6,
                market_filter=True,
                market_ticker="SPY",
                defensive_bucket="Bonds",
                market_threshold=threshold,
                rank_gap_threshold=rank_gap_setting,
            )

        if not backtest_data["overall_returns"].empty:
            overall_metrics = compute_metrics(backtest_data["overall_returns"]["return"])
            overall_turnover = compute_turnover(backtest_data["overall_positions"])

            if "gross_return" in backtest_data["overall_returns"].columns:
                gross_metrics = compute_metrics(
                    backtest_data["overall_returns"]["gross_return"]
                )
                print("\nOVERALL PORTFOLIO (NET OF COSTS):")
            else:
                print("\nOVERALL PORTFOLIO:")

            print(f"  CAGR:         {overall_metrics['cagr']:.2%}")
            print(f"  Volatility:   {overall_metrics['volatility']:.2%}")
            print(f"  Sharpe Ratio: {overall_metrics['sharpe']:.2f}")
            print(f"  Max Drawdown: {overall_metrics['max_drawdown']:.2%}")
            print(f"  Sortino:      {overall_metrics['sortino']:.2f}")
            print(f"  Total Return: {overall_metrics['total_return']:.2%}")
            print(f"  Avg Turnover: {overall_turnover:.2%}")

            if "gross_return" in backtest_data["overall_returns"].columns:
                print("\n  GROSS (before costs):")
                print(f"    CAGR:         {gross_metrics['cagr']:.2%}")
                print(f"    Total Return: {gross_metrics['total_return']:.2%}")
                avg_slippage = backtest_data["overall_returns"]["slippage"].mean() * 12
                avg_expense = backtest_data["overall_returns"]["expense"].mean() * 12
                print(f"  Annual Cost (slippage):  {avg_slippage:.2%}")
                print(f"  Annual Cost (expense):   {avg_expense:.2%}")

        print("\nPER-BUCKET METRICS:")
        for bucket, df_returns in backtest_data["bucket_returns"].items():
            if df_returns.empty:
                print(f"\n  {bucket}: No returns data")
                continue

            bucket_metrics = compute_metrics(df_returns["return"])
            bucket_turnover = compute_turnover(backtest_data["bucket_positions"][bucket])

            print(f"\n  {bucket} (NET OF COSTS):")
            print(f"    CAGR:         {bucket_metrics['cagr']:.2%}")
            print(f"    Volatility:   {bucket_metrics['volatility']:.2%}")
            print(f"    Sharpe Ratio: {bucket_metrics['sharpe']:.2f}")
            print(f"    Max Drawdown: {bucket_metrics['max_drawdown']:.2%}")
            print(f"    Sortino:      {bucket_metrics['sortino']:.2f}")
            print(f"    Total Return: {bucket_metrics['total_return']:.2%}")
            print(f"    Avg Turnover: {bucket_turnover:.2%}")

            if "gross_return" in df_returns.columns:
                bucket_gross = compute_metrics(df_returns["gross_return"])
                print(f"    GROSS (before costs):")
                print(f"      CAGR:         {bucket_gross['cagr']:.2%}")
                print(f"      Total Return: {bucket_gross['total_return']:.2%}")
                avg_bucket_slip = df_returns["slippage"].mean() * 12
                avg_bucket_exp = df_returns["expense"].mean() * 12
                print(f"      Annual Cost (slippage): {avg_bucket_slip:.2%}")
                print(f"      Annual Cost (expense):  {avg_bucket_exp:.2%}")

    print("\n" + "=" * 80)


if __name__ == "__main__":
    run_backtest()

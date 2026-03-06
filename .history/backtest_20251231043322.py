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

    thresholds = [None, 0.0, -0.05]
    lookback_options = [6, 12]
    rank_gaps = [0, 2]
    vol_flags = [True, False]
    start_date = "2005-01-01"
    end_date = "2015-12-31"

    scenarios = []

    for vol_flag in vol_flags:
        for lookback in lookback_options:
            for rank_gap_setting in rank_gaps:
                for threshold in thresholds:
                    backtest_data = backtest_momentum(
                        tickers=tickers,
                        bucket_map=bucket_map,
                        start_date=start_date,
                        end_date=end_date,
                        top_n_per_bucket=cfg.strategy.top_n_per_bucket or cfg.strategy.top_n,
                        lookback_long=lookback,
                        lookback_short=1,
                        vol_adjusted=vol_flag,
                        vol_lookback=6,
                        market_filter=threshold is not None,
                        market_ticker="SPY",
                        defensive_bucket="Bonds",
                        market_threshold=threshold or 0.0,
                        rank_gap_threshold=rank_gap_setting,
                    )
                    if backtest_data["overall_returns"].empty:
                        continue

                    overall_metrics = compute_metrics(
                        backtest_data["overall_returns"]["return"]
                    )
                    overall_turnover = compute_turnover(
                        backtest_data["overall_positions"]
                    )

                    bucket_cagrs: dict[str, float] = {}
                    for bucket, df_returns in backtest_data["bucket_returns"].items():
                        if df_returns.empty:
                            bucket_cagrs[bucket] = float("nan")
                        else:
                            bucket_cagrs[bucket] = compute_metrics(df_returns["return"])[
                                "cagr"
                            ]

                    scenarios.append(
                        {
                            "vol_adj": vol_flag,
                            "lookback": lookback,
                            "rank_gap": rank_gap_setting,
                            "threshold": threshold,
                            "overall": overall_metrics,
                            "turnover": overall_turnover,
                            "buckets": bucket_cagrs,
                        }
                    )

    # Print summary with bucket breakdowns (CAGR per bucket)
    print("\n" + "=" * 80)
    print(f"SCENARIO SUMMARY (net, {start_date} to {end_date})")
    print("Columns: vol_adj, lookbackM, threshold, rank_gap, CAGR, Sharpe, MaxDD, Turnover")
    for s in scenarios:
        o = s["overall"]
        thr_txt = "none" if s["threshold"] is None else f"{s['threshold']:.0%}"
        print(
            f"vol_adj={s['vol_adj']}, lookback={s['lookback']}M, thr={thr_txt}, gap={s['rank_gap']} | "
            f"CAGR={o['cagr']:.2%}, Sharpe={o['sharpe']:.2f}, MaxDD={o['max_drawdown']:.2%}, Turnover={s['turnover']:.2%}"
        )
        # bucket CAGRs in a compact line
        bucket_parts = [f"{b}:{c:.2%}" for b, c in s["buckets"].items()]
        print("  Buckets " + ", ".join(bucket_parts))

    print("\n" + "=" * 80)


if __name__ == "__main__":
    run_backtest()

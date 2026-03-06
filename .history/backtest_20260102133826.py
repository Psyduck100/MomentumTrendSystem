from pathlib import Path

from momentum_program.config import AppConfig
from momentum_program.analytics.constants import (
    SCORE_MODE_12M_MINUS_1M,
    SCORE_MODE_BLEND_6_12,
    SCORE_MODE_RW_3_6_9_12,
)
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

    thresholds = [None]  # market filter off per preference
    # Rank gap scenarios: uniform 0, uniform 2, and per-bucket from config
    all_buckets = set(bucket_map.values())
    rank_gaps = [
        0,  # uniform 0 (no gap)
        2,  # uniform 2 (all buckets)
        cfg.strategy.rank_gap_threshold,  # per-bucket config (US_equities:2, US_small_mid_cap:1, others:0)
    ]
    vol_flags = [False]  # vol adj off per preference
    score_modes = [
        SCORE_MODE_12M_MINUS_1M,
        SCORE_MODE_BLEND_6_12,
        SCORE_MODE_RW_3_6_9_12,
    ]

    # Absolute momentum filter scenarios
    abs_filters = [
        ("none", 0.0),  # control
        ("ret_12m", 0.00),
        ("ret_12m", 0.01),
        ("ret_6m", 0.00),
        ("ret_6m", 0.02),
        ("ret_and", 0.00),
        ("ret_and", 0.01),
        ("ret_blend", 0.00),
        ("ret_blend", 0.01),
    ]

    scenarios = []

    # Core ranges plus overlapping 5-year rolling windows to sanity-check consistency
    date_ranges = [
        ("2012-01-01", "2022-12-31"),
        ("2015-01-01", "2025-12-31"),
    ]
    for start_year in range(2012, 2022):  # 2012-2021 starts -> 5y windows through 2025
        end_year = min(start_year + 4, 2025)
        date_ranges.append((f"{start_year}-01-01", f"{end_year}-12-31"))

    for start_date, end_date in date_ranges:
        print(f"\nRunning window {start_date} -> {end_date}")
        for vol_flag in vol_flags:
            for rank_gap_setting in rank_gaps:
                for threshold in thresholds:
                    for score_mode in score_modes:
                        for abs_mode, abs_band in abs_filters:
                            backtest_data = backtest_momentum(
                                tickers=tickers,
                                bucket_map=bucket_map,
                                start_date=start_date,
                                end_date=end_date,
                                top_n_per_bucket=cfg.strategy.top_n_per_bucket
                                or cfg.strategy.top_n,
                                lookback_long=12,
                                lookback_short=1,
                                vol_adjusted=vol_flag,
                                vol_lookback=6,
                                market_filter=threshold is not None,
                                market_ticker="SPY",
                                defensive_bucket="Bonds",
                                market_threshold=threshold or 0.0,
                                rank_gap_threshold=rank_gap_setting,
                                score_mode=score_mode,
                                abs_filter_mode=abs_mode,
                                abs_filter_band=abs_band,
                                abs_filter_cash_annual=cfg.strategy.abs_filter_cash_annual,
                            )
                            if backtest_data["overall_returns"].empty:
                                continue

                            overall_metrics = compute_metrics(
                                backtest_data["overall_returns"]["return"]
                            )
                            overall_turnover = compute_turnover(
                                backtest_data["overall_positions"]
                            )

                            bucket_metrics: dict[str, dict] = {}
                            for bucket, df_returns in backtest_data[
                                "bucket_returns"
                            ].items():
                                if df_returns.empty:
                                    bucket_metrics[bucket] = {
                                        "cagr": float("nan"),
                                        "sharpe": float("nan"),
                                        "max_drawdown": float("nan"),
                                    }
                                else:
                                    bucket_metrics[bucket] = compute_metrics(
                                        df_returns["return"]
                                    )

                            scenarios.append(
                                {
                                    "start": start_date,
                                    "end": end_date,
                                    "vol_adj": vol_flag,
                                    "score_mode": score_mode,
                                    "rank_gap": rank_gap_setting,
                                    "threshold": threshold,
                                    "abs_filter": abs_mode,
                                    "abs_band": abs_band,
                                    "overall": overall_metrics,
                                    "turnover": overall_turnover,
                                    "buckets": bucket_metrics,
                                }
                            )

    # Print summary with bucket breakdowns (CAGR per bucket)
    print("\n" + "=" * 80)
    print("SCENARIO SUMMARY (net, 2012-2022)")
    print(
        "Columns: window, vol_adj, score_mode, threshold, rank_gap, abs_filter, band, CAGR, Sharpe, MaxDD, Turnover"
    )
    for s in scenarios:
        o = s["overall"]
        thr_txt = "none" if s["threshold"] is None else f"{s['threshold']:.0%}"
        # Format rank_gap: show as dict or int
        gap_val = s["rank_gap"]
        if isinstance(gap_val, dict):
            gap_txt = "per_bucket"
        else:
            gap_txt = str(gap_val)
        print(
            f"{s['start']}->{s['end']} | vol_adj={s['vol_adj']}, score={s['score_mode']}, thr={thr_txt}, gap={gap_txt}, abs={s['abs_filter']}@{s['abs_band']:.2%} | "
            f"CAGR={o['cagr']:.2%}, Sharpe={o['sharpe']:.2f}, MaxDD={o['max_drawdown']:.2%}, Turnover={s['turnover']:.2%}"
        )
        bucket_parts = [f"{b}:{m['cagr']:.2%}" for b, m in s["buckets"].items()]
        print("  Buckets " + ", ".join(bucket_parts))

    print("\n" + "=" * 80)
    
    # Write detailed bucket metrics to separate file
    output_path = Path("backtest_output_by_bucket.txt")
    with output_path.open("w") as f:
        f.write("=" * 100 + "\n")
        f.write("BUCKET-LEVEL METRICS (Sharpe, MaxDD, CAGR)\n")
        f.write("=" * 100 + "\n\n")
        
        for s in scenarios:
            o = s["overall"]
            thr_txt = "none" if s["threshold"] is None else f"{s['threshold']:.0%}"
            # Format rank_gap: show as dict or int
            gap_val = s["rank_gap"]
            if isinstance(gap_val, dict):
                gap_txt = str(gap_val)
            else:
                gap_txt = str(gap_val)
            f.write(f"Window: {s['start']} -> {s['end']}\n")
            f.write(f"Config: vol_adj={s['vol_adj']}, score={s['score_mode']}, thr={thr_txt}, gap={gap_txt}, abs={s['abs_filter']}@{s['abs_band']:.2%}\n")
            f.write(f"Overall: CAGR={o['cagr']:.2%}, Sharpe={o['sharpe']:.2f}, MaxDD={o['max_drawdown']:.2%}, Turnover={s['turnover']:.2%}\n")
            f.write("\nBucket Metrics:\n")
            
            for bucket, metrics in s["buckets"].items():
                f.write(f"  {bucket:25s} | CAGR: {metrics['cagr']:7.2%} | Sharpe: {metrics['sharpe']:5.2f} | MaxDD: {metrics['max_drawdown']:7.2%}\n")
            
            f.write("\n" + "-" * 100 + "\n\n")
    
    print(f"Detailed bucket metrics written to {output_path}")
    print("\n" + "=" * 80)


if __name__ == "__main__":
    run_backtest()

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

                                bucket_cagrs: dict[str, float] = {}
                                for bucket, df_returns in backtest_data[
                                    "bucket_returns"
                                ].items():
                                    if df_returns.empty:
                                        bucket_cagrs[bucket] = float("nan")
                                    else:
                                        bucket_cagrs[bucket] = compute_metrics(
                                            df_returns["return"]
                                        )["cagr"]

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
                                        "buckets": bucket_cagrs,
                                    }
                                )
        end_year = min(end_year, 2025)
        date_ranges.append((f"{start_year}-01-01", f"{end_year}-12-31"))

    for start_date, end_date in date_ranges:
        print(f"\nRunning window {start_date} -> {end_date}")
        for vol_flag in vol_flags:
            for rank_gap_setting in rank_gaps:
                for threshold in thresholds:
                    for score_mode in score_modes:
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
                        for bucket, df_returns in backtest_data[
                            "bucket_returns"
                        ].items():
                            if df_returns.empty:
                                bucket_cagrs[bucket] = float("nan")
                            else:
                                bucket_cagrs[bucket] = compute_metrics(
                                    df_returns["return"]
                                )["cagr"]

                        scenarios.append(
                            {
                                "start": start_date,
                                "end": end_date,
                                "vol_adj": vol_flag,
                                "score_mode": score_mode,
                                "rank_gap": rank_gap_setting,
                                "threshold": threshold,
                                "overall": overall_metrics,
                                "turnover": overall_turnover,
                                "buckets": bucket_cagrs,
                            }
                        )

    # Print summary with bucket breakdowns (CAGR per bucket)
    print("\n" + "=" * 80)
    print("SCENARIO SUMMARY (net, 2012-2022)")
    print(
        "Columns: window, vol_adj, score_mode, threshold, rank_gap, CAGR, Sharpe, MaxDD, Turnover"
    )
    for s in scenarios:
        o = s["overall"]
        thr_txt = "none" if s["threshold"] is None else f"{s['threshold']:.0%}"
        print(
            f"{s['start']}->{s['end']} | vol_adj={s['vol_adj']}, score={s['score_mode']}, thr={thr_txt}, gap={s['rank_gap']} | "
            f"CAGR={o['cagr']:.2%}, Sharpe={o['sharpe']:.2f}, MaxDD={o['max_drawdown']:.2%}, Turnover={s['turnover']:.2%}"
        )
        # bucket CAGRs in a compact line
        bucket_parts = [f"{b}:{c:.2%}" for b, c in s["buckets"].items()]
        print("  Buckets " + ", ".join(bucket_parts))

    print("\n" + "=" * 80)


if __name__ == "__main__":
    run_backtest()

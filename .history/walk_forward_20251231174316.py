"""Walk-forward validation: 5y train / 1y test, rolled."""
from pathlib import Path
import numpy as np
from momentum_program.config import AppConfig
from momentum_program.universe.bucket_csv_provider import BucketedCsvUniverseProvider
from momentum_program.universe.csv_provider import CsvUniverseProvider
from momentum_program.universe.tradingview_provider import TradingViewUniverseProvider
from momentum_program.backtest.engine import backtest_momentum
from momentum_program.backtest.metrics import compute_metrics, compute_turnover


def run_walk_forward() -> None:
    """Run walk-forward validation with 5y train / 1y test, rolled."""
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

    # Parameter grid
    thresholds = [None, 0.0, -0.05]
    lookback_options = [6, 12]
    rank_gaps = [0, 2]
    vol_flags = [True, False]

    # Walk-forward folds: 5y train, 1y test
    # For 2012-2022 (11 years): 6 folds
    folds = [
        {"train_start": "2012-01-01", "train_end": "2016-12-31", "test_start": "2017-01-01", "test_end": "2017-12-31"},
        {"train_start": "2013-01-01", "train_end": "2017-12-31", "test_start": "2018-01-01", "test_end": "2018-12-31"},
        {"train_start": "2014-01-01", "train_end": "2018-12-31", "test_start": "2019-01-01", "test_end": "2019-12-31"},
        {"train_start": "2015-01-01", "train_end": "2019-12-31", "test_start": "2020-01-01", "test_end": "2020-12-31"},
        {"train_start": "2016-01-01", "train_end": "2020-12-31", "test_start": "2021-01-01", "test_end": "2021-12-31"},
        {"train_start": "2017-01-01", "train_end": "2021-12-31", "test_start": "2022-01-01", "test_end": "2022-12-31"},
    ]

    print(f"\nRunning {len(folds)} walk-forward folds (5y train / 1y test)...")

    # Store results: combo -> list of fold metrics
    combo_results: dict[tuple, list[dict]] = {}

    for vol_flag in vol_flags:
        for lookback in lookback_options:
            for rank_gap_setting in rank_gaps:
                for threshold in thresholds:
                    combo_key = (vol_flag, lookback, rank_gap_setting, threshold)
                    combo_results[combo_key] = []

                    for fold_idx, fold in enumerate(folds, 1):
                        # Run backtest on test period only (we're not optimizing, just testing fixed params)
                        backtest_data = backtest_momentum(
                            tickers=tickers,
                            bucket_map=bucket_map,
                            start_date=fold["test_start"],
                            end_date=fold["test_end"],
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

                        combo_results[combo_key].append({
                            "fold": fold_idx,
                            "test_year": fold["test_start"][:4],
                            "cagr": overall_metrics["cagr"],
                            "sharpe": overall_metrics["sharpe"],
                            "max_drawdown": overall_metrics["max_drawdown"],
                            "turnover": overall_turnover,
                        })

    # Compute aggregate statistics across folds
    print("\n" + "=" * 100)
    print("WALK-FORWARD RESULTS (5y train / 1y test, 6 folds)")
    print("Columns: vol_adj, lookback, threshold, rank_gap | Median/Mean/Std of CAGR, Sharpe, MaxDD, Turnover")
    print("=" * 100)

    aggregated = []
    for combo_key, fold_metrics in combo_results.items():
        if not fold_metrics:
            continue

        vol_flag, lookback, rank_gap_setting, threshold = combo_key
        
        cagrs = [m["cagr"] for m in fold_metrics]
        sharpes = [m["sharpe"] for m in fold_metrics]
        maxdds = [m["max_drawdown"] for m in fold_metrics]
        turnovers = [m["turnover"] for m in fold_metrics]

        aggregated.append({
            "vol_adj": vol_flag,
            "lookback": lookback,
            "rank_gap": rank_gap_setting,
            "threshold": threshold,
            "n_folds": len(fold_metrics),
            "median_cagr": np.median(cagrs),
            "mean_cagr": np.mean(cagrs),
            "std_cagr": np.std(cagrs),
            "median_sharpe": np.median(sharpes),
            "mean_sharpe": np.mean(sharpes),
            "std_sharpe": np.std(sharpes),
            "median_maxdd": np.median(maxdds),
            "mean_maxdd": np.mean(maxdds),
            "std_maxdd": np.std(maxdds),
            "median_turnover": np.median(turnovers),
            "mean_turnover": np.mean(turnovers),
            "std_turnover": np.std(turnovers),
        })

    # Sort by median Sharpe (descending)
    aggregated.sort(key=lambda x: x["median_sharpe"], reverse=True)

    for agg in aggregated:
        thr_txt = "none" if agg["threshold"] is None else f"{agg['threshold']:.0%}"
        print(
            f"vol_adj={agg['vol_adj']}, lookback={agg['lookback']}M, thr={thr_txt}, gap={agg['rank_gap']} | "
            f"n={agg['n_folds']} | "
            f"CAGR med={agg['median_cagr']:.2%} μ={agg['mean_cagr']:.2%} σ={agg['std_cagr']:.2%} | "
            f"Sharpe med={agg['median_sharpe']:.2f} μ={agg['mean_sharpe']:.2f} σ={agg['std_sharpe']:.2f} | "
            f"MaxDD med={agg['median_maxdd']:.2%} μ={agg['mean_maxdd']:.2%} σ={agg['std_maxdd']:.2%} | "
            f"Turnover med={agg['median_turnover']:.2%} μ={agg['mean_turnover']:.2%} σ={agg['std_turnover']:.2%}"
        )

    print("\n" + "=" * 100)
    print(f"Top 10 combos by median Sharpe (across {len(folds)} folds):")
    print("=" * 100)
    for i, agg in enumerate(aggregated[:10], 1):
        thr_txt = "none" if agg["threshold"] is None else f"{agg['threshold']:.0%}"
        print(
            f"{i:2d}. vol_adj={agg['vol_adj']}, lookback={agg['lookback']}M, thr={thr_txt}, gap={agg['rank_gap']} | "
            f"Sharpe med={agg['median_sharpe']:.2f} CAGR med={agg['median_cagr']:.2%} MaxDD med={agg['median_maxdd']:.2%}"
        )


if __name__ == "__main__":
    run_walk_forward()

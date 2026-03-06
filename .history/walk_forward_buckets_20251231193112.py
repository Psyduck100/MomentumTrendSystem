"""Walk-forward validation of bucket optimization: compare REITs vs No REITs over longer period."""

from pathlib import Path
import numpy as np
from momentum_program.config import AppConfig
from momentum_program.universe.bucket_csv_provider import BucketedCsvUniverseProvider
from momentum_program.universe.csv_provider import CsvUniverseProvider
from momentum_program.universe.tradingview_provider import TradingViewUniverseProvider
from momentum_program.backtest.engine import backtest_momentum
from momentum_program.backtest.metrics import compute_metrics, compute_turnover


def run_walk_forward_bucket_comparison() -> None:
    """Walk-forward test: WITH REITs vs WITHOUT REITs over longest available period."""
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
    full_bucket_map = universe.get_bucket_map()

    if not tickers:
        print("No tickers found in universe.")
        return

    all_buckets = sorted(set(full_bucket_map.values()))
    print(f"Testing bucket optimization via walk-forward: WITH vs WITHOUT REITs")
    print(f"Universe: {len(tickers)} tickers across {len(all_buckets)} buckets\n")

    # Best params from walk-forward analysis
    params = {
        "lookback": 12,
        "vol_adjusted": False,
        "rank_gap": 2,
    }

    # Walk-forward folds: 2015-2025 (11 years) with expanding windows, 2y test
    # This gives us 5 folds across the full 11-year period
    folds = [
        {
            "train_start": "2015-01-01",
            "train_end": "2016-12-31",
            "test_start": "2017-01-01",
            "test_end": "2018-12-31",
        },
        {
            "train_start": "2015-01-01",
            "train_end": "2017-12-31",
            "test_start": "2018-01-01",
            "test_end": "2019-12-31",
        },
        {
            "train_start": "2015-01-01",
            "train_end": "2018-12-31",
            "test_start": "2019-01-01",
            "test_end": "2020-12-31",
        },
        {
            "train_start": "2015-01-01",
            "train_end": "2019-12-31",
            "test_start": "2020-01-01",
            "test_end": "2021-12-31",
        },
        {
            "train_start": "2015-01-01",
            "train_end": "2020-12-31",
            "test_start": "2021-01-01",
            "test_end": "2022-12-31",
        },
        {
            "train_start": "2015-01-01",
            "train_end": "2021-12-31",
            "test_start": "2022-01-01",
            "test_end": "2023-12-31",
        },
        {
            "train_start": "2015-01-01",
            "train_end": "2022-12-31",
            "test_start": "2023-01-01",
            "test_end": "2024-12-31",
        },
    ]

    print(f"Walk-forward period: 2015-2025 (11 years)")
    print(f"Folds: 7 (expanding window, 2y test periods)")
    print(f"Test years: 2017-2024\n")
    print("=" * 100)

    # Test configurations
    configs = {
        "WITH REITs": all_buckets,
        "WITHOUT REITs": [b for b in all_buckets if b != "REITs"],
    }

    results = {}

    for config_name, buckets_to_include in configs.items():
        print(f"\n{config_name}: {buckets_to_include}")
        print("-" * 100)

        fold_results = []

        for fold_idx, fold in enumerate(folds, 1):
            # Filter to only include tickers from specified buckets
            filtered_bucket_map = {
                ticker: bucket
                for ticker, bucket in full_bucket_map.items()
                if bucket in buckets_to_include
            }

            filtered_tickers = list(filtered_bucket_map.keys())

            # Run backtest on test period
            backtest_data = backtest_momentum(
                tickers=filtered_tickers,
                bucket_map=filtered_bucket_map,
                start_date=fold["test_start"],
                end_date=fold["test_end"],
                top_n_per_bucket=cfg.strategy.top_n_per_bucket or cfg.strategy.top_n,
                lookback_long=params["lookback"],
                lookback_short=1,
                vol_adjusted=params["vol_adjusted"],
                vol_lookback=6,
                market_filter=False,
                market_ticker="SPY",
                defensive_bucket="Bonds" if "Bonds" in buckets_to_include else None,
                market_threshold=0.0,
                rank_gap_threshold=params["rank_gap"],
            )

            if backtest_data["overall_returns"].empty:
                print(f"  Fold {fold_idx} ({fold['test_start'][:4]}): NO DATA")
                continue

            metrics = compute_metrics(backtest_data["overall_returns"]["return"])
            turnover = compute_turnover(backtest_data["overall_positions"])

            fold_results.append(
                {
                    "fold": fold_idx,
                    "year": f"{fold['test_start'][:4]}-{fold['test_end'][:4]}",
                    "cagr": metrics["cagr"],
                    "sharpe": metrics["sharpe"],
                    "max_drawdown": metrics["max_drawdown"],
                    "turnover": turnover,
                }
            )

            print(
                f"  Fold {fold_idx} ({fold['test_start'][:4]}-{fold['test_end'][:4]}): "
                f"Sharpe={metrics['sharpe']:.2f}, CAGR={metrics['cagr']:.2%}, "
                f"MaxDD={metrics['max_drawdown']:.2%}, Turnover={turnover:.2%}"
            )

        if fold_results:
            cagrs = [r["cagr"] for r in fold_results]
            sharpes = [r["sharpe"] for r in fold_results]
            maxdds = [r["max_drawdown"] for r in fold_results]
            turnovers = [r["turnover"] for r in fold_results]

            results[config_name] = {
                "n_folds": len(fold_results),
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
                "fold_results": fold_results,
            }

    # Summary comparison
    print("\n" + "=" * 100)
    print("SUMMARY: Walk-Forward Comparison (2015-2025, 7 folds)")
    print("=" * 100)

    for config_name, stats in results.items():
        print(f"\n{config_name}:")
        print(f"  Folds tested: {stats['n_folds']}")
        print(
            f"  Sharpe:  median={stats['median_sharpe']:.2f}, mean={stats['mean_sharpe']:.2f} (±{stats['std_sharpe']:.2f})"
        )
        print(
            f"  CAGR:    median={stats['median_cagr']:.2%}, mean={stats['mean_cagr']:.2%} (±{stats['std_cagr']:.2%})"
        )
        print(
            f"  MaxDD:   median={stats['median_maxdd']:.2%}, mean={stats['mean_maxdd']:.2%} (±{stats['std_maxdd']:.2%})"
        )
        print(
            f"  Turnover: median={stats['median_turnover']:.2%}, mean={stats['mean_turnover']:.2%} (±{stats['std_turnover']:.2%})"
        )

    # Comparison
    if len(results) == 2:
        with_reits = results.get("WITH REITs", {})
        without_reits = results.get("WITHOUT REITs", {})

        if with_reits and without_reits:
            print("\n" + "=" * 100)
            print("IMPACT OF REMOVING REITs (walk-forward median stats):")
            print("=" * 100)

            sharpe_gain = without_reits["median_sharpe"] - with_reits["median_sharpe"]
            cagr_gain = without_reits["median_cagr"] - with_reits["median_cagr"]
            dd_gain = (
                without_reits["median_maxdd"] - with_reits["median_maxdd"]
            )  # Positive = less negative (improvement)
            turnover_change = (
                without_reits["median_turnover"] - with_reits["median_turnover"]
            )

            print(
                f"\nSharpe ratio:   {sharpe_gain:+.2f} ({with_reits['median_sharpe']:.2f} -> {without_reits['median_sharpe']:.2f})"
            )
            print(
                f"CAGR:           {cagr_gain:+.2%} ({with_reits['median_cagr']:.2%} -> {without_reits['median_cagr']:.2%})"
            )
            print(
                f"Max Drawdown:   {dd_gain:+.2%} ({with_reits['median_maxdd']:.2%} -> {without_reits['median_maxdd']:.2%})"
            )
            print(
                f"Turnover:       {turnover_change:+.2%} ({with_reits['median_turnover']:.2%} -> {without_reits['median_turnover']:.2%})"
            )

            if sharpe_gain > 0:
                print(
                    f"\n[+] RECOMMENDATION: Remove REITs (consistent {sharpe_gain:+.2f} Sharpe improvement across 7 folds)"
                )
            else:
                print(
                    f"\n[-] RECOMMENDATION: Keep REITs (removing REITs hurts by {-sharpe_gain:.2f} Sharpe)"
                )

    # Per-fold breakdown
    print("\n" + "=" * 100)
    print("PER-FOLD BREAKDOWN:")
    print("=" * 100)

    if "WITH REITs" in results and "WITHOUT REITs" in results:
        with_reits_folds = results["WITH REITs"]["fold_results"]
        without_reits_folds = results["WITHOUT REITs"]["fold_results"]

        print(
            f"{'Test Period':<20} {'WITH REITs Sharpe':<20} {'WITHOUT REITs Sharpe':<20} {'Difference':<15}"
        )
        print("-" * 75)

        for wr, wo in zip(with_reits_folds, without_reits_folds):
            diff = wo["sharpe"] - wr["sharpe"]
            print(
                f"{wr['year']:<20} {wr['sharpe']:<20.2f} {wo['sharpe']:<20.2f} {diff:+.2f}"
            )


if __name__ == "__main__":
    run_walk_forward_bucket_comparison()

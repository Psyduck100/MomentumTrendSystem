"""Walk-forward validation of bucket optimization: compare REITs vs No REITs over longer period."""

from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd
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

    # Drop REITs entirely per decision
    full_bucket_map = {t: b for t, b in full_bucket_map.items() if b != "REITs"}
    tickers = [t for t in tickers if t in full_bucket_map]

    if not tickers:
        print("No tickers found in universe.")
        return

    all_buckets = sorted(set(full_bucket_map.values()))
    print(f"Testing bucket optimization via walk-forward: REITs dropped by design")
    print(f"Universe: {len(tickers)} tickers across {len(all_buckets)} buckets (REITs removed)\n")

    # Best params from walk-forward analysis
    params = {
        "lookback": 12,
        "vol_adjusted": False,
        "rank_gap": 2,
    }

    # Walk-forward folds: align with data availability (3-year test windows)
    folds = [
        {
            "train_start": "2015-01-01",
            "train_end": "2017-12-31",
            "test_start": "2018-01-01",
            "test_end": "2020-12-31",
        },
        {
            "train_start": "2015-01-01",
            "train_end": "2020-12-31",
            "test_start": "2021-01-01",
            "test_end": "2023-12-31",
        },
        {
            "train_start": "2015-01-01",
            "train_end": "2021-12-31",
            "test_start": "2022-01-01",
            "test_end": "2024-12-31",
        },
    ]

    print(f"Walk-forward period: 2018-2024 (7 years)")
    print(f"Folds: 3 (expanding window, 3y test periods)")
    print(f"Test years: 2018-2024\n")
    print("=" * 100)

    # Test configurations (REITs already removed, so only one config makes sense)
    configs = {
        "REITs removed": all_buckets,
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

            # Require warmup: ensure at least lookback months before test_start
            train_start_dt = datetime.fromisoformat(fold["train_start"])
            test_start_dt = datetime.fromisoformat(fold["test_start"])
            warmup_months = params["lookback"]
            if (test_start_dt.year - train_start_dt.year) * 12 + (test_start_dt.month - train_start_dt.month) < warmup_months:
                print(f"  Fold {fold_idx} ({fold['test_start'][:4]}): skipped (insufficient warmup)")
                continue

            # Run backtest on train+test window, then slice to test window for metrics
            backtest_data = backtest_momentum(
                tickers=filtered_tickers,
                bucket_map=filtered_bucket_map,
                start_date=fold["train_start"],
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

            overall_returns = backtest_data["overall_returns"]
            raw_positions = backtest_data["overall_positions"]

            # Normalize positions: list -> DataFrame aligned to returns index
            if isinstance(raw_positions, list):
                min_len = min(len(overall_returns), len(raw_positions))
                pos_index = overall_returns.index[:min_len]
                overall_positions = pd.DataFrame(
                    {"positions": raw_positions[:min_len]}, index=pos_index
                )
            else:
                overall_positions = raw_positions

            print(
                f"    Fold {fold_idx}: raw overall_returns rows={len(overall_returns)}, overall_positions len={len(raw_positions)}"
            )

            # Normalize positions to DataFrame if a list is returned
            if isinstance(overall_positions, list):
                overall_positions = pd.DataFrame()

            # Slice to test window
            if not overall_returns.empty:
                mask = (overall_returns.index >= pd.to_datetime(fold["test_start"])) & (
                    overall_returns.index <= pd.to_datetime(fold["test_end"])
                )
                test_returns = overall_returns.loc[mask]
            else:
                test_returns = overall_returns

            if not overall_positions.empty:
                mask_pos = (overall_positions.index >= pd.to_datetime(fold["test_start"])) & (
                    overall_positions.index <= pd.to_datetime(fold["test_end"])
                )
                test_positions = overall_positions.loc[mask_pos]["positions"].tolist()
            else:
                test_positions = []

            print(
                f"    Fold {fold_idx}: test window returns rows={len(test_returns)}, positions rows={len(test_positions)}"
            )

            # If we have no usable data (e.g., early folds with few tickers), skip
            if test_returns.empty:
                print(f"  Fold {fold_idx} ({fold['test_start'][:4]}): NO DATA")
                continue

            metrics = compute_metrics(test_returns["return"])
            turnover = compute_turnover(test_positions)

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

    # Summary
    print("\n" + "=" * 100)
    print("SUMMARY: Walk-Forward (REITs removed)")
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

    # Per-fold breakdown
    print("\n" + "=" * 100)
    print("PER-FOLD BREAKDOWN:")
    print("=" * 100)

    for config_name, stats in results.items():
        print(f"\n{config_name}:")
        print(f"{'Test Period':<20} {'Sharpe':<10} {'CAGR':<10} {'MaxDD':<10} {'Turnover':<10}")
        print("-" * 75)
        for row in stats.get("fold_results", []):
            print(
                f"{row['year']:<20} {row['sharpe']:<10.2f} {row['cagr']:<10.2%} {row['max_drawdown']:<10.2%} {row['turnover']:<10.2%}"
            )


if __name__ == "__main__":
    run_walk_forward_bucket_comparison()

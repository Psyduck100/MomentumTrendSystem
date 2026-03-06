"""Walk-forward bucket drop test (10y span): evaluate removing each bucket using best params.

Outputs median/mean CAGR, Sharpe, MaxDD, Turnover across folds, sorted by Sharpe.
"""

from pathlib import Path
from statistics import median, mean
from typing import Dict, List

import numpy as np

from momentum_program.config import AppConfig
from momentum_program.universe.bucket_csv_provider import BucketedCsvUniverseProvider
from momentum_program.universe.csv_provider import CsvUniverseProvider
from momentum_program.universe.tradingview_provider import TradingViewUniverseProvider
from momentum_program.backtest.engine import backtest_momentum
from momentum_program.backtest.metrics import compute_metrics, compute_turnover


FOLDS = [
    {
        "train_start": "2019-01-01",
        "train_end": "2020-12-31",
        "test_start": "2021-01-01",
        "test_end": "2022-12-31",
    },
    {
        "train_start": "2019-01-01",
        "train_end": "2021-12-31",
        "test_start": "2022-01-01",
        "test_end": "2023-12-31",
    },
    {
        "train_start": "2019-01-01",
        "train_end": "2023-12-31",
        "test_start": "2024-01-01",
        "test_end": "2024-12-31",
    },
]

# Best params validated earlier
BEST_PARAMS = {
    "lookback": 12,
    "vol_adjusted": False,
    "rank_gap": 2,
    "threshold": None,
}


def load_universe():
    bucket_folder = Path("CSVs")
    bucket_csvs = list(bucket_folder.glob("*.csv")) if bucket_folder.exists() else []
    csv_path = Path("etfs.csv")

    if bucket_csvs:
        universe = BucketedCsvUniverseProvider(bucket_folder)
    elif csv_path.exists():
        universe = CsvUniverseProvider(csv_path)
    else:
        universe = TradingViewUniverseProvider(chunk=50)

    tickers = universe.get_tickers()
    bucket_map = universe.get_bucket_map()
    return tickers, bucket_map


def evaluate_buckets(
    tickers: List[str], bucket_map: Dict[str, str], buckets_to_include: List[str], cfg: AppConfig
) -> Dict[str, float]:
    # Filter tickers/bucket map
    filtered_map = {t: b for t, b in bucket_map.items() if b in buckets_to_include}
    filtered_tickers = list(filtered_map.keys())

    if not filtered_tickers:
        return {
            "median_cagr": 0.0,
            "median_sharpe": 0.0,
            "median_maxdd": 0.0,
            "median_turnover": 0.0,
            "mean_cagr": 0.0,
            "mean_sharpe": 0.0,
            "mean_maxdd": 0.0,
            "mean_turnover": 0.0,
            "n_folds": 0,
            "n_tickers": 0,
            "fold_details": [],
        }

    fold_metrics = []
    fold_details = []

    for fold_idx, fold in enumerate(FOLDS, 1):
        backtest_data = backtest_momentum(
            tickers=filtered_tickers,
            bucket_map=filtered_map,
            start_date=fold["test_start"],
            end_date=fold["test_end"],
            top_n_per_bucket=cfg.strategy.top_n_per_bucket or cfg.strategy.top_n,
            lookback_long=BEST_PARAMS["lookback"],
            lookback_short=1,
            vol_adjusted=BEST_PARAMS["vol_adjusted"],
            vol_lookback=6,
            market_filter=BEST_PARAMS["threshold"] is not None,
            market_ticker="SPY",
            defensive_bucket="Bonds" if "Bonds" in buckets_to_include else None,
            market_threshold=BEST_PARAMS["threshold"] or 0.0,
            rank_gap_threshold=BEST_PARAMS["rank_gap"],
        )

        if backtest_data["overall_returns"].empty:
            fold_details.append({
                "fold": fold_idx,
                "year": fold["test_start"][:4],
                "cagr": None,
                "sharpe": None,
                "maxdd": None,
                "turnover": None,
                "n_tickers_used": 0,
                "status": "NO DATA"
            })
            continue

        metrics = compute_metrics(backtest_data["overall_returns"]["return"])
        turnover = compute_turnover(backtest_data["overall_positions"])

        fold_metrics.append(
            {
                "cagr": metrics["cagr"],
                "sharpe": metrics["sharpe"],
                "maxdd": metrics["max_drawdown"],
                "turnover": turnover,
            }
        )
        
        fold_details.append({
            "fold": fold_idx,
            "year": fold["test_start"][:4],
            "cagr": metrics["cagr"],
            "sharpe": metrics["sharpe"],
            "maxdd": metrics["max_drawdown"],
            "turnover": turnover,
            "n_tickers_used": len(backtest_data["overall_returns"]),
            "status": "OK"
        })

    if not fold_metrics:
        return {
            "median_cagr": 0.0,
            "median_sharpe": 0.0,
            "median_maxdd": 0.0,
            "median_turnover": 0.0,
            "mean_cagr": 0.0,
            "mean_sharpe": 0.0,
            "mean_maxdd": 0.0,
            "mean_turnover": 0.0,
            "n_folds": 0,
            "n_tickers": len(filtered_tickers),
            "fold_details": fold_details,
        }

    return {
        "median_cagr": median(m["cagr"] for m in fold_metrics),
        "median_sharpe": median(m["sharpe"] for m in fold_metrics),
        "median_maxdd": median(m["maxdd"] for m in fold_metrics),
        "median_turnover": median(m["turnover"] for m in fold_metrics),
        "mean_cagr": mean(m["cagr"] for m in fold_metrics),
        "mean_sharpe": mean(m["sharpe"] for m in fold_metrics),
        "mean_maxdd": mean(m["maxdd"] for m in fold_metrics),
        "mean_turnover": mean(m["turnover"] for m in fold_metrics),
        "n_folds": len(fold_metrics),
        "n_tickers": len(filtered_tickers),
        "fold_details": fold_details,
    }


def run_bucket_drop_walk_forward():
    cfg = AppConfig()
    tickers, bucket_map = load_universe()

    if not tickers:
        print("No tickers found in universe.")
        return

    all_buckets = sorted(set(bucket_map.values()))
    print(f"Walk-forward bucket drop test (2019-2024, targeting CAGR + diversity)")
    print(f"Universe: {len(tickers)} tickers across {len(all_buckets)} buckets: {all_buckets}")
    print(f"Params: lookback={BEST_PARAMS['lookback']}M, vol_adj={BEST_PARAMS['vol_adjusted']}, gap={BEST_PARAMS['rank_gap']}, threshold={BEST_PARAMS['threshold']}")
    print(f"Folds: {len(FOLDS)} (expanding window, 2y test), covering 2019-2024")
    print("=" * 120)

    configs = []

    # Baseline
    baseline_stats = evaluate_buckets(tickers, bucket_map, all_buckets, cfg)
    configs.append({"name": "All buckets", "buckets": all_buckets, **baseline_stats})

    # Remove each bucket individually
    for bucket in all_buckets:
        remaining = [b for b in all_buckets if b != bucket]
        stats = evaluate_buckets(tickers, bucket_map, remaining, cfg)
        configs.append({"name": f"Without {bucket}", "buckets": remaining, **stats})

    # Sort by median Sharpe (primary), then median CAGR (secondary)
    configs.sort(key=lambda x: (x["median_sharpe"], x["median_cagr"]), reverse=True)

    print("\nTop configurations (sorted by median Sharpe, tiebreak on CAGR):")
    print("-" * 120)
    for idx, cfg_res in enumerate(configs, 1):
        print(
            f"{idx:2d}. {cfg_res['name']:<25} | Sharpe med={cfg_res['median_sharpe']:.2f}, CAGR med={cfg_res['median_cagr']:.2%}, "
            f"MaxDD med={cfg_res['median_maxdd']:.2%}, Turnover med={cfg_res['median_turnover']:.2%}, n_folds={cfg_res['n_folds']}, "
            f"n_tickers={cfg_res['n_tickers']}"
        )

    # Per-fold detail
    print("\n" + "=" * 120)
    print("DETAILED FOLD BREAKDOWN:")
    print("=" * 120)
    for cfg_res in configs:
        print(f"\n{cfg_res['name']:<25} (buckets: {cfg_res['buckets']}, n_tickers={cfg_res['n_tickers']})")
        print(f"  Fold | Year | Status    | CAGR     | Sharpe | MaxDD    | Turnover")
        print(f"  -----|------|-----------|----------|--------|----------|----------")
        for detail in cfg_res["fold_details"]:
            if detail["status"] == "NO DATA":
                print(f"  {detail['fold']:4d} | {detail['year']} | NO DATA   | -        | -      | -        | -")
            else:
                print(
                    f"  {detail['fold']:4d} | {detail['year']} | OK        | {detail['cagr']:7.2%} | {detail['sharpe']:6.2f} | {detail['maxdd']:8.2%} | {detail['turnover']:8.2%}"
                )

    # Impact vs baseline
    print("\n" + "=" * 120)
    print("IMPACT VS BASELINE (median stats):")
    print("=" * 120)
    for cfg_res in configs:
        if cfg_res["name"] == "All buckets":
            baseline = cfg_res
            break
    else:
        baseline = None

    if baseline:
        for cfg_res in configs:
            if cfg_res["name"] == "All buckets":
                continue
            sharpe_gain = cfg_res["median_sharpe"] - baseline["median_sharpe"]
            cagr_gain = cfg_res["median_cagr"] - baseline["median_cagr"]
            dd_gain = cfg_res["median_maxdd"] - baseline["median_maxdd"]
            turnover_change = cfg_res["median_turnover"] - baseline["median_turnover"]
            print(
                f"{cfg_res['name']:<25} | Sharpe {sharpe_gain:+.2f}, CAGR {cagr_gain:+.2%}, MaxDD {dd_gain:+.2%}, Turnover {turnover_change:+.2%}"
            )


if __name__ == "__main__":
    run_bucket_drop_walk_forward()

"""Walk-forward per-bucket performance over 2015-2024 (3y test folds)."""

from pathlib import Path
from statistics import median, mean
from typing import Dict, List

import pandas as pd

from momentum_program.config import AppConfig
from momentum_program.universe.bucket_csv_provider import BucketedCsvUniverseProvider
from momentum_program.universe.csv_provider import CsvUniverseProvider
from momentum_program.universe.tradingview_provider import TradingViewUniverseProvider
from momentum_program.backtest.engine import backtest_momentum
from momentum_program.backtest.metrics import compute_metrics, compute_turnover


# Three expanding folds with 3-year test windows (same as long bucket-drop script)
FOLDS = [
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

    # Remove REITs per prior decision
    bucket_map = {t: b for t, b in bucket_map.items() if b != "REITs"}
    tickers = [t for t in tickers if t in bucket_map]
    return tickers, bucket_map


def evaluate_buckets(
    tickers: List[str],
    bucket_map: Dict[str, str],
    buckets_to_include: List[str],
    cfg: AppConfig,
) -> Dict[str, float]:
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
            start_date=fold["train_start"],
            end_date=fold["test_end"],
            cache_dir=Path("backtest_cache") / ("bucket_" + "_".join(sorted(buckets_to_include))),
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

        overall_returns = backtest_data["overall_returns"]
        raw_positions = backtest_data["overall_positions"]

        if isinstance(raw_positions, list):
            min_len = min(len(overall_returns), len(raw_positions))
            pos_index = overall_returns.index[:min_len]
            overall_positions = pd.DataFrame(
                {"positions": raw_positions[:min_len]}, index=pos_index
            )
        else:
            overall_positions = raw_positions

        if not overall_returns.empty:
            mask = (overall_returns.index >= pd.to_datetime(fold["test_start"])) & (
                overall_returns.index <= pd.to_datetime(fold["test_end"])
            )
            test_returns = overall_returns.loc[mask]
        else:
            test_returns = overall_returns

        if not overall_positions.empty:
            mask_pos = (
                overall_positions.index >= pd.to_datetime(fold["test_start"])
            ) & (overall_positions.index <= pd.to_datetime(fold["test_end"]))
            test_positions = overall_positions.loc[mask_pos]["positions"].tolist()
        else:
            test_positions = []

        if test_returns.empty:
            fold_details.append(
                {
                    "fold": fold_idx,
                    "period": f"{fold['test_start'][:4]}-{fold['test_end'][:4]}",
                    "cagr": None,
                    "sharpe": None,
                    "maxdd": None,
                    "turnover": None,
                    "n_tickers_used": 0,
                    "status": "NO DATA",
                }
            )
            continue

        metrics = compute_metrics(test_returns["return"])
        turnover = compute_turnover(test_positions)

        fold_metrics.append(
            {
                "cagr": metrics["cagr"],
                "sharpe": metrics["sharpe"],
                "maxdd": metrics["max_drawdown"],
                "turnover": turnover,
            }
        )

        fold_details.append(
            {
                "fold": fold_idx,
                "period": f"{fold['test_start'][:4]}-{fold['test_end'][:4]}",
                "cagr": metrics["cagr"],
                "sharpe": metrics["sharpe"],
                "maxdd": metrics["max_drawdown"],
                "turnover": turnover,
                "n_tickers_used": len(backtest_data["overall_returns"]),
                "status": "OK",
            }
        )

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


def main():
    cfg = AppConfig()
    tickers, bucket_map = load_universe()
    if not tickers:
        print("No tickers found.")
        return

    buckets = sorted(set(bucket_map.values()))
    print("Walk-forward per-bucket test (2015-2024, 3y test folds)")
    print(f"Buckets in universe: {buckets}")
    print("Params: lookback=12M, vol_adj=False, gap=2, threshold=None")
    print("Folds: 2018-2020, 2021-2023, 2022-2024 (expanding train)")
    print("=" * 118)

    results = []
    for bucket in buckets:
        stats = evaluate_buckets(tickers, bucket_map, [bucket], cfg)
        results.append((bucket, stats))

    # Sort by median Sharpe then median CAGR
    results.sort(
        key=lambda r: (r[1]["median_sharpe"], r[1]["median_cagr"]), reverse=True
    )

    print("Top-line per-bucket medians:")
    for bucket, stats in results:
        print(
            f"- {bucket:20s} | Sharpe {stats['median_sharpe']:.2f} | CAGR {stats['median_cagr']*100:5.2f}% | "
            f"MaxDD {stats['median_maxdd']*100:6.2f}% | Turn {stats['median_turnover']:5.2f}% | "
            f"folds {stats['n_folds']} | n_tickers {stats['n_tickers']}"
        )

    print("\nFold breakdowns:")
    for bucket, stats in results:
        print("-" * 118)
        print(f"{bucket}")
        for fd in stats["fold_details"]:
            cagr_txt = "-" if fd["cagr"] is None else f"{fd['cagr']*100:5.2f}%"
            sharpe_txt = "-" if fd["sharpe"] is None else f"{fd['sharpe']:5.2f}"
            maxdd_txt = "-" if fd["maxdd"] is None else f"{fd['maxdd']*100:6.2f}%"
            turn_txt = "-" if fd["turnover"] is None else f"{fd['turnover']:5.2f}%"

            print(
                f"  Fold {fd['fold']}: {fd['period']} | {fd['status']:<7} | "
                f"CAGR {cagr_txt} Sharpe {sharpe_txt} MaxDD {maxdd_txt} Turn {turn_txt}"
            )


if __name__ == "__main__":
    main()

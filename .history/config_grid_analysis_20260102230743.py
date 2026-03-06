"""Run every config combination on a custom date range and rank results."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

from momentum_program.universe.bucket_csv_provider import BucketedCsvUniverseProvider
from comprehensive_walkforward import generate_configs, run_backtest_config


def _load_universe(bucket_dir: Path) -> tuple[list[str], dict[str, str]]:
    provider = BucketedCsvUniverseProvider(bucket_dir)
    tickers = provider.get_tickers()
    bucket_map = provider.get_bucket_map()
    if not tickers:
        raise RuntimeError(f"No tickers found in {bucket_dir}")
    return tickers, bucket_map


def _summarize(df: pd.DataFrame, top_n: int) -> None:
    if df.empty:
        print("No successful configurations.")
        return

    print("\nTop configs by Sharpe")
    print(df.sort_values("sharpe", ascending=False).head(top_n)[
        ["score_mode", "filter_mode", "filter_band", "rank_gap", "sharpe", "cagr", "max_dd"]
    ])

    print("\nTop configs by CAGR")
    print(df.sort_values("cagr", ascending=False).head(top_n)[
        ["score_mode", "filter_mode", "filter_band", "rank_gap", "cagr", "sharpe", "max_dd"]
    ])

    counts = df.groupby("score_mode").size().sort_values(ascending=False)
    print("\nWinning score modes (count of successful runs):")
    print(counts)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate every config combo on a fixed date range (e.g., exclude 2024-2025)"
    )
    parser.add_argument("--start", default="2015-01-01", help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", default="2023-12-31", help="End date (YYYY-MM-DD)")
    parser.add_argument("--bucket-dir", type=Path, default=Path("CSVs"))
    parser.add_argument("--output", type=Path, default=Path("config_grid_results.csv"))
    parser.add_argument("--top", type=int, default=10, help="Rows to display in the console")
    args = parser.parse_args()

    tickers, bucket_map = _load_universe(args.bucket_dir)
    configs = generate_configs()
    rows: List[Dict[str, Any]] = []

    print(f"Running {len(configs)} configs from {args.start} to {args.end}...")
    for idx, config in enumerate(configs, start=1):
        metrics = run_backtest_config(
            tickers,
            bucket_map,
            args.start,
            args.end,
            config,
        )
        if metrics is None:
            continue
        row = {
            "score_mode": config["score_mode"],
            "filter_mode": config["filter_mode"],
            "filter_band": config["filter_band"],
            "rank_gap": config["rank_gap"],
            "sharpe": metrics["sharpe"],
            "cagr": metrics["cagr"],
            "max_dd": metrics["max_dd"],
            "turnover": metrics["turnover"],
            "n_months": metrics["n_months"],
        }
        rows.append(row)
        if idx % 25 == 0:
            print(f"Processed {idx}/{len(configs)} configs...")

    df = pd.DataFrame(rows)
    if df.empty:
        print("No valid configs produced returns; check the date range.")
        return

    df.to_csv(args.output, index=False)
    print(f"Saved grid results to {args.output}")

    _summarize(df, args.top)


if __name__ == "__main__":
    main()

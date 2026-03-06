"""Compute per-bucket performance metrics for a single momentum configuration."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

from momentum_program.analytics.constants import (
    SCORE_MODE_12M_MINUS_1M,
    SCORE_MODE_BLEND_6_12,
    SCORE_MODE_RW_3_6_9_12,
)
from momentum_program.backtest.engine import backtest_momentum
from momentum_program.backtest.metrics import compute_metrics
from momentum_program.config import AppConfig
from momentum_program.universe.bucket_csv_provider import BucketedCsvUniverseProvider

SCORE_CHOICES = {
    "rw_3_6_9_12": SCORE_MODE_RW_3_6_9_12,
    "12m_minus_1m": SCORE_MODE_12M_MINUS_1M,
    "blend_6_12": SCORE_MODE_BLEND_6_12,
}


def _parse_gap(value: str) -> int:
    try:
        return int(value)
    except ValueError as exc:  # pragma: no cover
        raise argparse.ArgumentTypeError("rank_gap must be an integer") from exc


def load_universe(bucket_dir: Path) -> tuple[list[str], dict[str, str]]:
    provider = BucketedCsvUniverseProvider(bucket_dir)
    return provider.get_tickers(), provider.get_bucket_map()


def summarize_bucket_metrics(result: dict) -> list[tuple[str, dict]]:
    ordered = []
    for bucket, df in sorted(result["bucket_returns"].items()):
        if df.empty:
            metrics = {
                "cagr": float("nan"),
                "sharpe": float("nan"),
                "max_drawdown": float("nan"),
            }
        else:
            metrics = compute_metrics(df["return"])
        ordered.append((bucket, metrics))
    return ordered


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Per-bucket CAGR breakdown for a given config"
    )
    parser.add_argument(
        "--start", default="2015-01-01", help="Backtest start date (YYYY-MM-DD)"
    )
    parser.add_argument(
        "--end", default="2025-12-31", help="Backtest end date (YYYY-MM-DD)"
    )
    parser.add_argument("--score", choices=SCORE_CHOICES.keys(), default="blend_6_12")
    parser.add_argument("--filter-mode", default="ret_and")
    parser.add_argument("--filter-band", type=float, default=0.01)
    parser.add_argument("--rank-gap", type=_parse_gap, default=1)
    parser.add_argument(
        "--bucket-dir",
        type=Path,
        default=Path("CSVs"),
        help="Directory containing bucket CSV files",
    )
    args = parser.parse_args()

    tickers, bucket_map = load_universe(args.bucket_dir)
    if not tickers:
        raise RuntimeError(f"No ticker CSVs found in {args.bucket_dir}")

    cfg = AppConfig()
    result = backtest_momentum(
        tickers=tickers,
        bucket_map=bucket_map,
        start_date=args.start,
        end_date=args.end,
        top_n_per_bucket=cfg.strategy.top_n_per_bucket or cfg.strategy.top_n,
        rank_gap_threshold=args.rank_gap,
        score_mode=SCORE_CHOICES[args.score],
        abs_filter_mode=args.filter_mode,
        abs_filter_band=args.filter_band,
        abs_filter_cash_annual=cfg.strategy.abs_filter_cash_annual,
    )

    if result["overall_returns"].empty:
        raise RuntimeError("Backtest returned no data; check inputs")

    overall = compute_metrics(result["overall_returns"]["return"])
    print("\nOverall Metrics")
    print("---------------")
    print(
        f"CAGR {overall['cagr']*100:.2f}% | Sharpe {overall['sharpe']:.2f} | MaxDD {overall['max_drawdown']*100:.2f}%"
    )

    print("\nPer-Bucket Breakdown")
    print("---------------------")
    for bucket, metrics in summarize_bucket_metrics(result):
        print(
            f"{bucket:20s}  CAGR {metrics['cagr']*100:6.2f}%  Sharpe {metrics['sharpe']:5.2f}  MaxDD {metrics['max_drawdown']*100:7.2f}%"
        )


if __name__ == "__main__":
    main()

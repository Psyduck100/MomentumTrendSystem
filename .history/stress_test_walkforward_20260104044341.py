"""Run robustness checks by excluding specified windows from a walk-forward CSV."""

import argparse
from pathlib import Path
from typing import Iterable, List

import pandas as pd

from walkforward_analysis import _compute_performance_stats, _load_return_series


def _parse_years(values: Iterable[str]) -> List[int]:
    years: List[int] = []
    for value in values:
        try:
            years.append(int(value))
        except ValueError as exc:  # pragma: no cover
            raise ValueError(f"Year must be an integer, got {value}") from exc
    return years


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Exclude top years and recompute stats"
    )
    parser.add_argument("csv", type=Path, help="Walk-forward CSV file")
    parser.add_argument(
        "--exclude-years",
        nargs="*",
        default=[],
        help="Test years to drop before stitching returns",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.05,
        help="Significance level for interpretation",
    )
    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    if args.exclude_years:
        if "test_year" not in df.columns:
            raise ValueError("CSV does not contain 'test_year'; cannot exclude years.")
        years = _parse_years(args.exclude_years)
        df = df[~df["test_year"].isin(years)]

    if df.empty:
        raise ValueError("No windows left after exclusions.")

    stitched = _load_return_series(df["returns_path"].tolist())
    stats = _compute_performance_stats(stitched)

    print("\nRobustness analysis")
    print("===================")
    if args.exclude_years:
        print(f"Excluded test years: {', '.join(map(str, years))}")
    print(f"Windows: {len(df)}  Observations: {stats['n_obs']}")
    print(
        f"CAGR: {stats['cagr']*100:.2f}% (95% CI {stats['cagr_ci_low']*100:.2f}% – {stats['cagr_ci_high']*100:.2f}%)"
    )
    print(f"Sharpe: {stats['sharpe']:.2f}  MaxDD: {stats['max_dd']*100:.2f}%")
    print(f"Total Return: {stats['total_return']*100:.2f}%")
    print(f"Z-stat: {stats['z_stat']:.2f}  One-sided p-value: {stats['p_value']:.4f}")
    if stats["p_value"] < args.alpha:
        print(f"Result: Significant at α={args.alpha:.2f}")
    else:
        print(f"Result: Not significant at α={args.alpha:.2f}")


if __name__ == "__main__":
    main()

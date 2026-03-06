"""Post-process walk-forward CSV outputs to run statistical significance checks.

This script loads one or more walk-forward CSV files created by
`comprehensive_walkforward.py`, stitches their out-of-sample returns,
computes aggregate CAGR/Sharpe/MaxDD, and evaluates whether the CAGR is
statistically greater than zero using a log-return z-test plus a 95% confidence
interval. That keeps backtest generation (step 1) separate from significance
analysis (step 2).
"""

import argparse
import math
from pathlib import Path
from typing import Dict, Iterable

import numpy as np
import pandas as pd


def _normal_cdf(x: float) -> float:
    """Standard normal cumulative distribution function."""

    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _load_return_series(paths: Iterable[str]) -> pd.Series:
    """Load and concatenate per-window return series."""

    series_list = []
    for raw_path in paths:
        if not isinstance(raw_path, str) or not raw_path:
            continue

        path = Path(raw_path)
        if not path.exists():
            raise FileNotFoundError(f"Return series not found: {path}")

        series = pd.read_csv(path, index_col=0, parse_dates=True).iloc[:, 0]
        series.name = "return"
        series_list.append(series)

    if not series_list:
        raise ValueError("No return series available for analysis.")

    combined = pd.concat(series_list).sort_index()
    combined = combined[~combined.index.duplicated(keep="first")]
    return combined


def _compute_performance_stats(returns: pd.Series) -> Dict[str, float]:
    """Compute descriptive metrics and CAGR significance statistics."""

    if returns.empty:
        raise ValueError("Return series is empty.")

    cumulative = (1 + returns).cumprod()
    total_return = cumulative.iloc[-1] - 1

    if returns.index[0] == returns.index[-1]:
        years = len(returns) / 12.0
    else:
        delta_days = (returns.index[-1] - returns.index[0]).days
        years = max(delta_days / 365.25, len(returns) / 12.0)

    cagr = (1 + total_return) ** (1 / years) - 1

    monthly_mean = returns.mean()
    monthly_std = returns.std(ddof=1)
    sharpe = 0.0
    if monthly_std > 0:
        sharpe = monthly_mean / monthly_std * np.sqrt(12)

    running_max = cumulative.cummax()
    drawdown = cumulative / running_max - 1
    max_dd = drawdown.min()

    log_returns = np.log1p(returns.to_numpy())
    mean_lr = float(np.mean(log_returns))
    std_lr = float(np.std(log_returns, ddof=1))
    n_obs = len(log_returns)

    if n_obs > 1 and std_lr > 0:
        se_lr = std_lr / math.sqrt(n_obs)
        z_stat = mean_lr / se_lr
        p_value = 1.0 - _normal_cdf(z_stat)
        ci_low_lr = mean_lr - 1.96 * se_lr
        ci_high_lr = mean_lr + 1.96 * se_lr
    else:
        se_lr = 0.0
        z_stat = math.inf if mean_lr > 0 else (-math.inf if mean_lr < 0 else 0.0)
        p_value = 0.0 if z_stat > 0 else 1.0
        ci_low_lr = ci_high_lr = mean_lr

    cagr_low = math.expm1(ci_low_lr * 12)
    cagr_high = math.expm1(ci_high_lr * 12)

    return {
        "n_obs": n_obs,
        "years": years,
        "cagr": cagr,
        "sharpe": sharpe,
        "max_dd": max_dd,
        "total_return": total_return,
        "z_stat": z_stat,
        "p_value": p_value,
        "cagr_ci_low": cagr_low,
        "cagr_ci_high": cagr_high,
    }


def analyze_csv(csv_path: Path, alpha: float) -> None:
    """Load a walk-forward CSV, run stats, and print a concise report."""

    df = pd.read_csv(csv_path)
    if "returns_path" not in df.columns:
        raise ValueError(
            f"Column 'returns_path' missing in {csv_path}. Re-run the generator to capture return series."
        )

    stitched_returns = _load_return_series(df["returns_path"].tolist())
    stats = _compute_performance_stats(stitched_returns)

    print("\n" + "=" * 100)
    print(f"Analysis for {csv_path}")
    print("=" * 100)
    print(
        f"Windows: {len(df)}  Observations: {stats['n_obs']}  Years: {stats['years']:.2f}"
    )
    print(
        f"CAGR: {stats['cagr']*100:.2f}% (95% CI: {stats['cagr_ci_low']*100:.2f}% – {stats['cagr_ci_high']*100:.2f}%)"
    )
    print(
        f"Sharpe: {stats['sharpe']:.2f}  MaxDD: {stats['max_dd']*100:.2f}%  Total Return: {stats['total_return']*100:.2f}%"
    )
    print(
        f"Z-stat (H0: CAGR ≤ 0): {stats['z_stat']:.2f}  One-sided p-value: {stats['p_value']:.4f}"
    )

    if stats["p_value"] < alpha:
        print(f"Result: Significant at α={alpha:.2f} (reject H0, CAGR>0)")
    else:
        print(f"Result: Not significant at α={alpha:.2f} (cannot reject H0)")

    if {"test_cagr", "test_sharpe", "config"}.issubset(df.columns):
        top_windows = df.sort_values("test_cagr", ascending=False).head(3)
        print("\nTop windows by out-of-sample CAGR:")
        for _, row in top_windows.iterrows():
            label = row.get("test_year") or row.get("regime") or f"{row['test_start']}→{row['test_end']}"
            print(
                f"  {label}: test CAGR {row['test_cagr']*100:.2f}%, Sharpe {row['test_sharpe']:.2f}, config {row['config']}"
            )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run statistical analysis on walk-forward CSV outputs."
    )
    parser.add_argument(
        "files",
        nargs="+",
        help="One or more CSV files produced by comprehensive_walkforward.py",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.05,
        help="Significance level for one-sided CAGR test (default: 0.05)",
    )
    args = parser.parse_args()

    for file_arg in args.files:
        analyze_csv(Path(file_arg), args.alpha)


if __name__ == "__main__":
    main()

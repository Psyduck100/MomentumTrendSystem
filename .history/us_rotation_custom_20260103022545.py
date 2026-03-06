"""Run the US-equities rotation strategy on an early-2000s-friendly universe.

Universe: SPTM, SPY, QQQ, OEF, IWD
Evaluation windows:
  1. 2001-01-01 → 2025-12-31
  2. 2001-01-01 → 2013-12-31
  3. 2013-01-01 → 2025-12-31
  4. 2006-01-01 → 2020-12-31

For each window we sweep momentum score modes, rank-gap thresholds, and absolute filters,
then compare the strategy versus SPY buy-and-hold (monthly, same dates).
"""

from __future__ import annotations

from itertools import product
from pathlib import Path

import pandas as pd

from momentum_program.analytics.constants import (
    SCORE_MODE_12M_MINUS_1M,
    SCORE_MODE_BLEND_6_12,
    SCORE_MODE_RW_3_6_9_12,
)
from momentum_program.backtest.engine import backtest_momentum
from momentum_program.backtest.metrics import compute_metrics

UNIVERSE = ["SPTM", "SPY", "QQQ", "OEF", "IWD"]
BUCKET_MAP = {ticker: "US_equities" for ticker in UNIVERSE}
BACKTEST_CACHE = Path("backtest_cache")

SCORE_MODES = {
    "12m_minus_1m": SCORE_MODE_12M_MINUS_1M,
    "blend_6_12": SCORE_MODE_BLEND_6_12,
    "rw_3_6_9_12": SCORE_MODE_RW_3_6_9_12,
}

FILTER_MODES = ["none", "ret_6m", "ret_12m", "ret_and"]
RANK_GAPS = [0, 1, 2]
WINDOWS = [
    ("2001_full", "2001-01-01", "2025-12-31"),
    ("2001_2013", "2001-01-01", "2013-12-31"),
    ("2013_2025", "2013-01-01", "2025-12-31"),
    ("2006_2020", "2006-01-01", "2020-12-31"),
]


def describe_metrics(label: str, metrics: dict[str, float]) -> str:
    return (
        f"{label}: CAGR {metrics['cagr']*100:5.2f}% | Sharpe {metrics['sharpe']:5.2f} | "
        f"MaxDD {metrics['max_drawdown']*100:6.2f}% | Total {metrics['total_return']*100:6.2f}%"
    )


def main() -> None:
    summary_rows: list[dict] = []
    spy_by_window: dict[str, dict[str, float]] = {}

    for window_label, start, end in WINDOWS:
        print(f"\n===== Window {window_label}: {start} → {end} =====")
        combos = list(product(SCORE_MODES.items(), RANK_GAPS, FILTER_MODES))
        for (score_label, score_mode), gap, filter_mode in combos:
            result = backtest_momentum(
                tickers=UNIVERSE,
                bucket_map=BUCKET_MAP,
                start_date=start,
                end_date=end,
                top_n_per_bucket=1,
                rank_gap_threshold=gap,
                score_mode=score_mode,
                abs_filter_mode=filter_mode,
                abs_filter_band=0.0,
                abs_filter_cash_annual=0.04,
                cache_dir=BACKTEST_CACHE,
                slippage_bps=3.0,
                expense_ratio=0.001,
            )

            overall = result["overall_returns"]
            if overall.empty:
                print(
                    f"  Skipping score={score_label} gap={gap} filter={filter_mode} (no data)"
                )
                continue

            strat_metrics = compute_metrics(overall["return"])

            spy_metrics = spy_by_window.get(window_label)
            if spy_metrics is None:
                monthly_prices = result["monthly_prices"]
                if "SPY" not in monthly_prices:
                    raise RuntimeError("SPY missing from monthly price history")
                spy_returns = monthly_prices["SPY"].pct_change().dropna()
                spy_returns = spy_returns.loc[
                    (spy_returns.index >= overall.index[0])
                    & (spy_returns.index <= overall.index[-1])
                ]
                spy_metrics = compute_metrics(spy_returns)
                spy_by_window[window_label] = spy_metrics
                print("  " + describe_metrics("SPY", spy_metrics))

            summary_rows.append(
                {
                    "window": window_label,
                    "start": overall.index[0].date(),
                    "end": overall.index[-1].date(),
                    "score": score_label,
                    "rank_gap": gap,
                    "filter": filter_mode,
                    "cagr": strat_metrics["cagr"],
                    "sharpe": strat_metrics["sharpe"],
                    "max_dd": strat_metrics["max_drawdown"],
                    "total_return": strat_metrics["total_return"],
                    "spy_cagr": spy_metrics["cagr"],
                    "spy_sharpe": spy_metrics["sharpe"],
                    "spy_max_dd": spy_metrics["max_drawdown"],
                }
            )

            print(
                "  "
                + describe_metrics(
                    f"score={score_label:12s} gap={gap} filter={filter_mode:7s}",
                    strat_metrics,
                )
            )

    df = pd.DataFrame(summary_rows)
    out_path = Path("us_rotation_summary.csv")
    df.to_csv(out_path, index=False)
    print(f"\nSaved summary for {len(df)} configs to {out_path}")

    for window_label in df["window"].unique():
        window_df = df[df["window"] == window_label].copy()
        window_df.sort_values("sharpe", ascending=False, inplace=True)
        top = window_df.head(3)
        print(f"\nTop configs by Sharpe – {window_label}")
        for _, row in top.iterrows():
            print(
                f"  score={row['score']:12s} gap={row['rank_gap']} filter={row['filter']:7s} | "
                f"CAGR {row['cagr']*100:5.2f}% | Sharpe {row['sharpe']:5.2f} | MaxDD {row['max_dd']*100:6.2f}%"
            )
        spy = spy_by_window[window_label]
        print(
            f"  SPY reference: CAGR {spy['cagr']*100:5.2f}% | Sharpe {spy['sharpe']:5.2f} | MaxDD {spy['max_drawdown']*100:6.2f}%"
        )


if __name__ == "__main__":
    main()

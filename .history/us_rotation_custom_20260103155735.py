"""Run the US-equities rotation strategy from 2001 through the present.

Universe: SPTM, SPY, QQQ, OEF, IWD
Evaluation window: Single 2001→present span to match the latest request.

For that window sweep momentum score modes, rank-gap thresholds, and absolute filters,
compare the strategy versus SPY/QQQ buy-and-hold, and export per-config monthly and
annual return series (including per-month SPY/QQQ returns) for downstream testing.
"""

from __future__ import annotations

import json
from itertools import product
from pathlib import Path
from typing import Any

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

WINDOW_LABEL = "full_run_2001_present"
START_DATE = "2001-01-01"
END_DATE = pd.Timestamp.today().strftime("%Y-%m-%d")
WINDOWS = [(WINDOW_LABEL, START_DATE, END_DATE)]


def compute_annual_returns(monthly_returns: pd.Series) -> list[dict[str, float]]:
    annual: list[dict[str, float]] = []
    grouped = monthly_returns.groupby(monthly_returns.index.year)
    for year, values in grouped:
        compounded = float((1.0 + values).prod() - 1.0)
        annual.append({"year": int(year), "return": compounded})
    return annual


def describe_metrics(label: str, metrics: dict[str, float]) -> str:
    return (
        f"{label}: CAGR {metrics['cagr']*100:5.2f}% | Sharpe {metrics['sharpe']:5.2f} | "
        f"MaxDD {metrics['max_drawdown']*100:6.2f}% | Total {metrics['total_return']*100:6.2f}%"
    )


def main() -> None:
    summary_rows: list[dict] = []
    monthly_rows: list[dict] = []
    annual_rows: list[dict] = []
    benchmarks_by_window: dict[str, dict[str, Any]] = {}

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

            benchmark_bundle = benchmarks_by_window.get(window_label)
            if benchmark_bundle is None:
                monthly_prices = result["monthly_prices"]
                missing = [t for t in ("SPY", "QQQ") if t not in monthly_prices]
                if missing:
                    raise RuntimeError(f"Missing benchmark data for {missing}")
                spy_monthly = monthly_prices["SPY"].pct_change()
                qqq_monthly = monthly_prices["QQQ"].pct_change()
                spy_metrics = compute_metrics(spy_monthly.loc[overall.index].dropna())
                qqq_metrics = compute_metrics(qqq_monthly.loc[overall.index].dropna())
                benchmark_bundle = {
                    "metrics": {"SPY": spy_metrics, "QQQ": qqq_metrics},
                    "returns": {"SPY": spy_monthly, "QQQ": qqq_monthly},
                }
                benchmarks_by_window[window_label] = benchmark_bundle
                print("  " + describe_metrics("SPY", spy_metrics))
                print("  " + describe_metrics("QQQ", qqq_metrics))
            else:
                spy_metrics = benchmark_bundle["metrics"]["SPY"]
                qqq_metrics = benchmark_bundle["metrics"]["QQQ"]

            spy_series = benchmark_bundle["returns"]["SPY"].reindex(overall.index)
            qqq_series = benchmark_bundle["returns"]["QQQ"].reindex(overall.index)

            monthly_returns = overall["return"].copy()
            annual_returns = compute_annual_returns(monthly_returns)

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
                    "qqq_cagr": qqq_metrics["cagr"],
                    "qqq_sharpe": qqq_metrics["sharpe"],
                    "qqq_max_dd": qqq_metrics["max_drawdown"],
                    "annual_returns_json": json.dumps(annual_returns),
                }
            )

            for timestamp, monthly_return in monthly_returns.items():
                monthly_rows.append(
                    {
                        "window": window_label,
                        "score": score_label,
                        "rank_gap": gap,
                        "filter": filter_mode,
                        "date": timestamp.date(),
                        "return": float(monthly_return),
                        "spy_return": float(spy_series.loc[timestamp])
                        if pd.notna(spy_series.loc[timestamp])
                        else None,
                        "qqq_return": float(qqq_series.loc[timestamp])
                        if pd.notna(qqq_series.loc[timestamp])
                        else None,
                    }
                )

            for entry in annual_returns:
                annual_rows.append(
                    {
                        "window": window_label,
                        "score": score_label,
                        "rank_gap": gap,
                        "filter": filter_mode,
                        "year": entry["year"],
                        "annual_return": entry["return"],
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

    monthly_df = pd.DataFrame(monthly_rows)
    monthly_path = Path("us_rotation_monthly_returns.csv")
    monthly_df.to_csv(monthly_path, index=False)
    print(f"Saved monthly return series ({len(monthly_df)} rows) to {monthly_path}")

    annual_df = pd.DataFrame(annual_rows)
    annual_path = Path("us_rotation_annual_returns.csv")
    annual_df.to_csv(annual_path, index=False)
    print(f"Saved annual return aggregates ({len(annual_df)} rows) to {annual_path}")

    for window_label in df["window"].unique():
        window_df = df[df["window"] == window_label].copy()
        window_df.sort_values("cagr", ascending=False, inplace=True)
        top = window_df.head(3)
        print(f"\nTop configs by CAGR – {window_label}")
        for _, row in top.iterrows():
            print(
                f"  score={row['score']:12s} gap={row['rank_gap']} filter={row['filter']:7s} | "
                f"CAGR {row['cagr']*100:5.2f}% | Sharpe {row['sharpe']:5.2f} | MaxDD {row['max_dd']*100:6.2f}%"
            )
        benchmarks = benchmarks_by_window[window_label]["metrics"]
        spy = benchmarks["SPY"]
        qqq = benchmarks["QQQ"]
        print(
            f"  SPY reference: CAGR {spy['cagr']*100:5.2f}% | Sharpe {spy['sharpe']:5.2f} | MaxDD {spy['max_drawdown']*100:6.2f}%"
        )
        print(
            f"  QQQ reference: CAGR {qqq['cagr']*100:5.2f}% | Sharpe {qqq['sharpe']:5.2f} | MaxDD {qqq['max_drawdown']*100:6.2f}%"
        )


if __name__ == "__main__":
    main()

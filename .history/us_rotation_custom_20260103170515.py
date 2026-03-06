"""Run the US-equities rotation strategy from 2001 through the present.

Universe: SPTM, SPY, QQQ, OEF, IWD
Evaluation window: Single 2001→present span to match the latest request.

For that window sweep momentum score modes, rank-gap thresholds, and absolute filters,
compare the strategy versus SPY/QQQ buy-and-hold, and export per-config monthly and
annual return series (including per-month SPY/QQQ returns) for downstream testing.
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Any

import pandas as pd

from momentum_program.analytics.constants import (
    SCORE_MODE_12M_MINUS_1M,
    SCORE_MODE_BLEND_6_12,
    SCORE_MODE_BLEND_LAMBDA,
    SCORE_MODE_RW_3_6_9_12,
    SCORE_MODE_RW_ALPHA,
)
from momentum_program.backtest.engine import backtest_momentum
from momentum_program.backtest.metrics import compute_metrics

UNIVERSE = ["SPTM", "SPY", "QQQ", "OEF", "IWD"]
BUCKET_MAP = {ticker: "US_equities" for ticker in UNIVERSE}
BACKTEST_CACHE = Path("backtest_cache")

OBJECTIVE_BENCHMARK = "SPY"
OBJECTIVE_GAMMA = 0.5
RECENCY_ALPHA_GRID = [-0.5, 0.0, 0.5, 1.0, 1.5, 2.0]
BLEND_LAMBDA_GRID = [0.0, 0.25, 0.5, 0.75, 1.0]


@dataclass(frozen=True)
class ScoreConfig:
    label: str
    mode: str
    param: float | None = None
    param_type: str | None = None


BASE_SCORE_CONFIGS = [
    ScoreConfig("12m_minus_1m", SCORE_MODE_12M_MINUS_1M),
    ScoreConfig("blend_6_12", SCORE_MODE_BLEND_6_12),
    ScoreConfig("rw_3_6_9_12", SCORE_MODE_RW_3_6_9_12),
]

RECENCY_SCORE_CONFIGS = [
    ScoreConfig(
        label=f"rw_alpha_{alpha:.2f}",
        mode=SCORE_MODE_RW_ALPHA,
        param=alpha,
        param_type="alpha",
    )
    for alpha in RECENCY_ALPHA_GRID
]

BLEND_SCORE_CONFIGS = [
    ScoreConfig(
        label=f"blend_lambda_{lam:.2f}",
        mode=SCORE_MODE_BLEND_LAMBDA,
        param=lam,
        param_type="lambda",
    )
    for lam in BLEND_LAMBDA_GRID
]

SCORE_CONFIGS = BASE_SCORE_CONFIGS + RECENCY_SCORE_CONFIGS + BLEND_SCORE_CONFIGS

FILTER_MODES = ["none", "ret_6m", "ret_12m", "ret_and"]
RANK_GAPS = [0, 1, 2]
COMBO_DEFS = list(product(SCORE_CONFIGS, RANK_GAPS, FILTER_MODES))

WINDOW_LABEL = "full_run_2001_present"
START_DATE = "2001-01-01"
END_DATE = pd.Timestamp.today().strftime("%Y-%m-%d")
WINDOWS = [(WINDOW_LABEL, START_DATE, END_DATE)]

SELECTED_CONFIG = {
    "score": "blend_6_12",
    "score_param": None,
    "rank_gap": 0,
    "filter": "ret_12m",
}
TRAIN_START_DATE = "2002-01-01"
TRAIN_END_DATE = "2014-12-31"
TEST_START_DATE = "2015-01-01"
TEST_END_DATE = END_DATE


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


def metrics_to_floats(metrics: dict[str, float]) -> dict[str, float]:
    return {k: float(v) for k, v in metrics.items()}


def compute_benchmark_metrics(
    monthly_prices: pd.DataFrame, index: pd.Index, tickers: tuple[str, ...] = ("SPY", "QQQ")
) -> dict[str, dict[str, float]]:
    bundle: dict[str, dict[str, float]] = {}
    for ticker in tickers:
        if ticker not in monthly_prices:
            continue
        series = monthly_prices[ticker].pct_change().reindex(index)
        aligned = series.dropna()
        if aligned.empty:
            continue
        bundle[ticker] = metrics_to_floats(compute_metrics(aligned))
    return bundle


def compute_excess_and_objective(
    strat_metrics: dict[str, float], benchmark_metrics: dict[str, float] | None
) -> tuple[float | None, float | None]:
    if benchmark_metrics is None:
        return None, None
    excess = strat_metrics["cagr"] - benchmark_metrics["cagr"]
    objective = excess - OBJECTIVE_GAMMA * abs(strat_metrics["max_drawdown"])
    return float(excess), float(objective)


def compute_turnover_series(overall_returns: pd.DataFrame) -> pd.Series:
    if "symbols" not in overall_returns.columns:
        raise ValueError("overall_returns is missing the 'symbols' column")

    turnovers: list[float] = []
    prev_holdings: set[str] | None = None

    for _, holdings in overall_returns["symbols"].items():
        current = set(holdings) if isinstance(holdings, (list, tuple, set)) else set()
        if prev_holdings is None:
            turnovers.append(0.0)
        else:
            prev_size = max(len(prev_holdings), 1)
            sell_fraction = len(prev_holdings - current) / prev_size
            buy_fraction = len(current - prev_holdings) / max(len(current), 1)
            turnovers.append(max(sell_fraction, buy_fraction))
        prev_holdings = current

    return pd.Series(turnovers, index=overall_returns.index)


def summarize_turnover(turnover_series: pd.Series) -> dict[str, float]:
    if turnover_series.empty:
        return {
            "monthly_median": 0.0,
            "monthly_p90": 0.0,
            "annual_median": 0.0,
            "annual_p90": 0.0,
        }

    trimmed = turnover_series.iloc[1:] if len(turnover_series) > 1 else turnover_series
    annual = trimmed.groupby(trimmed.index.year).sum() if not trimmed.empty else pd.Series(dtype=float)

    return {
        "monthly_median": float(trimmed.median()) if not trimmed.empty else 0.0,
        "monthly_p90": float(trimmed.quantile(0.9)) if not trimmed.empty else 0.0,
        "annual_median": float(annual.median()) if not annual.empty else 0.0,
        "annual_p90": float(annual.quantile(0.9)) if not annual.empty else 0.0,
    }


def evaluate_config_window(
    score_config: ScoreConfig,
    gap: int,
    filter_mode: str,
    start_date: str,
    end_date: str,
) -> dict[str, Any] | None:
    result = backtest_momentum(
        tickers=UNIVERSE,
        bucket_map=BUCKET_MAP,
        start_date=start_date,
        end_date=end_date,
        top_n_per_bucket=1,
        rank_gap_threshold=gap,
        score_mode=score_config.mode,
        score_param=score_config.param,
        abs_filter_mode=filter_mode,
        abs_filter_band=0.0,
        abs_filter_cash_annual=0.04,
        cache_dir=BACKTEST_CACHE,
        slippage_bps=3.0,
        expense_ratio=0.001,
    )

    overall = result["overall_returns"]
    if overall.empty:
        return None

    returns = overall["return"]
    metrics = metrics_to_floats(compute_metrics(returns))
    turnover_stats = summarize_turnover(compute_turnover_series(overall))
    benchmarks = compute_benchmark_metrics(result["monthly_prices"], returns.index)
    spy_metrics = benchmarks.get(OBJECTIVE_BENCHMARK)
    excess_cagr, objective = compute_excess_and_objective(metrics, spy_metrics)

    return {
        "metrics": metrics,
        "turnover": turnover_stats,
        "benchmarks": benchmarks,
        "excess_cagr": excess_cagr,
        "objective": objective,
        "start": str(returns.index[0].date()),
        "end": str(returns.index[-1].date()),
    }


def run_walk_forward_analysis() -> dict[str, Any]:
    print(
        "\nWalk-forward config selection (train 2002-01-01 → 2014-12-31; test 2015-01-01 → present)"
    )

    wf_payload: dict[str, Any] = {
        "candidates": [],
        "config": None,
        "train": None,
        "test": None,
    }

    candidates: list[dict[str, Any]] = []

    for (score_label, score_mode), gap, filter_mode in COMBO_DEFS:
        result = backtest_momentum(
            tickers=UNIVERSE,
            bucket_map=BUCKET_MAP,
            start_date=TRAIN_START_DATE,
            end_date=TRAIN_END_DATE,
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
            continue

        metrics = {k: float(v) for k, v in compute_metrics(overall["return"]).items()}
        candidate = {
            "score": score_label,
            "rank_gap": gap,
            "filter": filter_mode,
            "train": {
                "start": TRAIN_START_DATE,
                "end": TRAIN_END_DATE,
                **metrics,
            },
            "_score_mode": score_mode,
        }

        test_result = backtest_momentum(
            tickers=UNIVERSE,
            bucket_map=BUCKET_MAP,
            start_date=TEST_START_DATE,
            end_date=TEST_END_DATE,
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

        test_overall = test_result["overall_returns"]
        if not test_overall.empty:
            candidate["test"] = {
                "start": TEST_START_DATE,
                "end": TEST_END_DATE,
                **{k: float(v) for k, v in compute_metrics(test_overall["return"]).items()},
            }
        else:
            candidate["test"] = None

        candidates.append(candidate)

    if not candidates:
        print("  No configurations produced sufficient training data; skipping walk-forward.")
        return wf_payload

    candidates.sort(key=lambda c: c["train"]["cagr"], reverse=True)
    best_config = candidates[0]

    wf_payload["candidates"] = [
        {k: v for k, v in candidate.items() if not k.startswith("_")}
        for candidate in candidates
    ]
    wf_payload["config"] = {
        "score": best_config["score"],
        "rank_gap": best_config["rank_gap"],
        "filter": best_config["filter"],
    }
    wf_payload["train"] = best_config["train"]

    train_metrics = best_config["train"]
    print(
        "  Selected config (train): "
        f"score={best_config['score']} gap={best_config['rank_gap']} filter={best_config['filter']} | "
        f"CAGR {train_metrics['cagr']*100:5.2f}% | Sharpe {train_metrics['sharpe']:5.2f} | MaxDD {train_metrics['max_drawdown']*100:6.2f}%"
    )

    best_test = best_config.get("test")
    if best_test is None:
        print("  Selected config has no data during test window; skipping test stats.")
        return wf_payload

    wf_payload["test"] = best_test
    print(
        "  Out-of-sample (test): "
        f"CAGR {best_test['cagr']*100:5.2f}% | Sharpe {best_test['sharpe']:5.2f} | MaxDD {best_test['max_drawdown']*100:6.2f}%"
    )

    return wf_payload

def main() -> None:
    summary_rows: list[dict] = []
    monthly_rows: list[dict] = []
    annual_rows: list[dict] = []
    benchmarks_by_window: dict[str, dict[str, Any]] = {}
    selected_full_result: dict[str, Any] | None = None

    for window_label, start, end in WINDOWS:
        print(f"\n===== Window {window_label}: {start} → {end} =====")
        for (score_label, score_mode), gap, filter_mode in COMBO_DEFS:
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

            if (
                selected_full_result is None
                and window_label == WINDOW_LABEL
                and score_label == SELECTED_CONFIG["score"]
                and gap == SELECTED_CONFIG["rank_gap"]
                and filter_mode == SELECTED_CONFIG["filter"]
            ):
                selected_full_result = result

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

    if selected_full_result is not None:
        turnover_series = compute_turnover_series(selected_full_result["overall_returns"])
        turnover_stats = summarize_turnover(turnover_series)
        print(
            "\nTurnover diagnostics for selected config "
            f"(score={SELECTED_CONFIG['score']} gap={SELECTED_CONFIG['rank_gap']} filter={SELECTED_CONFIG['filter']}):"
        )
        print(
            f"  Per-rebalance turnover – median {turnover_stats['monthly_median']:.2%} | worst 10% {turnover_stats['monthly_p90']:.2%}"
        )
        print(
            f"  Annual turnover – median {turnover_stats['annual_median']:.2f}x | worst 10% {turnover_stats['annual_p90']:.2f}x"
        )
    else:
        print(
            "\nSelected configuration not found in the sweep; turnover diagnostics skipped."
        )

    walk_forward_data = run_walk_forward_analysis()
    if walk_forward_data["config"] is not None:
        wf_path = Path("us_rotation_walk_forward.json")
        wf_path.write_text(json.dumps(walk_forward_data, indent=2))
        print(f"\nSaved walk-forward summary to {wf_path}")


if __name__ == "__main__":
    main()

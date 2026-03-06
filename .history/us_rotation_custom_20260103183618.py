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
from typing import Any, Iterable

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


def build_combo_defs(
    score_configs: Iterable[ScoreConfig],
    rank_gaps: Iterable[int] | None = None,
    filter_modes: Iterable[str] | None = None,
) -> list[tuple[ScoreConfig, int, str]]:
    gaps = list(rank_gaps) if rank_gaps is not None else RANK_GAPS
    filters = list(filter_modes) if filter_modes is not None else FILTER_MODES
    return list(product(score_configs, gaps, filters))


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
COARSE_COMBO_DEFS = build_combo_defs(SCORE_CONFIGS)

REFINED_ALPHA_GRID = [round(1.5 + 0.1 * i, 2) for i in range(int((3.0 - 1.5) / 0.1) + 1)]
REFINED_ALPHA_CONFIGS = [
    ScoreConfig(
        label=f"rw_alpha_refined_{alpha:.2f}",
        mode=SCORE_MODE_RW_ALPHA,
        param=alpha,
        param_type="alpha",
    )
    for alpha in REFINED_ALPHA_GRID
]
REFINED_ALPHA_RANK_GAPS = [1]
REFINED_ALPHA_FILTERS = ["ret_6m"]

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

WALK_FORWARD_SPLITS = [
    {
        "name": "train_2003_2006",
        "train": ("2003-01-01", "2006-12-31"),
        "test": ("2007-01-01", "2008-12-31"),
    },
    {
        "name": "train_2003_2008",
        "train": ("2003-01-01", "2008-12-31"),
        "test": ("2009-01-01", "2010-12-31"),
    },
    {
        "name": "train_2003_2010",
        "train": ("2003-01-01", "2010-12-31"),
        "test": ("2011-01-01", "2012-12-31"),
    },
    {
        "name": "train_2003_2012",
        "train": ("2003-01-01", "2012-12-31"),
        "test": ("2013-01-01", "2014-12-31"),
    },
    {
        "name": "train_2003_2014",
        "train": ("2003-01-01", "2014-12-31"),
        "test": ("2015-01-01", "2016-12-31"),
    },
    {
        "name": "train_2003_2016",
        "train": ("2003-01-01", "2016-12-31"),
        "test": ("2017-01-01", "2018-12-31"),
    },
    {
        "name": "train_2003_2018",
        "train": ("2003-01-01", "2018-12-31"),
        "test": ("2019-01-01", "2020-12-31"),
    },
    {
        "name": "train_2003_2020",
        "train": ("2003-01-01", "2020-12-31"),
        "test": ("2021-01-01", "2022-12-31"),
    },
    {
        "name": "train_2003_2022",
        "train": ("2003-01-01", "2022-12-31"),
        "test": ("2023-01-01", END_DATE),
    },
]


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


def strip_private_keys(payload: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in payload.items() if not k.startswith("_")}


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


EVAL_CACHE: dict[tuple[Any, ...], dict[str, Any] | None] = {}


def _candidate_signature(
    score_label: str,
    score_param: float | None,
    score_param_type: str | None,
    gap: int,
    filter_mode: str,
) -> tuple[Any, ...]:
    return (score_label, score_param, score_param_type, gap, filter_mode)


def _selection_value(snapshot: dict[str, Any] | None) -> float:
    if not snapshot:
        return float("-inf")
    if snapshot.get("excess_cagr") is not None:
        return float(snapshot["excess_cagr"])
    return float(snapshot["metrics"]["cagr"])


def _avg(values: list[float]) -> float | None:
    if not values:
        return None
    return float(sum(values) / len(values))


def _frange(
    center: float,
    half_width: float,
    step: float,
    bounds: tuple[float, float],
) -> list[float]:
    lo = max(bounds[0], center - half_width)
    hi = min(bounds[1], center + half_width)
    if hi < lo:
        return []
    values: list[float] = []
    current = lo
    while current <= hi + 1e-9:
        values.append(round(current, 2))
        current += step
    return values


def build_refined_score_configs(
    aggregate_rows: list[dict[str, Any]], existing_configs: Iterable[ScoreConfig], step: float = 0.1, half_width: float = 0.3
) -> list[ScoreConfig]:
    existing_values: dict[str, set[float]] = {
        "alpha": {
            cfg.param
            for cfg in existing_configs
            if cfg.param_type == "alpha" and cfg.param is not None
        },
        "lambda": {
            cfg.param
            for cfg in existing_configs
            if cfg.param_type == "lambda" and cfg.param is not None
        },
    }

    refined: list[ScoreConfig] = []
    param_specs = {
        "alpha": (SCORE_MODE_RW_ALPHA, (-1.0, 3.0)),
        "lambda": (SCORE_MODE_BLEND_LAMBDA, (0.0, 1.0)),
    }

    for param_type, (mode, bounds) in param_specs.items():
        candidates = [
            row
            for row in aggregate_rows
            if row.get("score_param_type") == param_type and row.get("avg_test_cagr") is not None
        ]
        if not candidates:
            continue
        best_row = max(candidates, key=lambda row: row["avg_test_cagr"])
        center = best_row.get("score_param")
        if center is None:
            continue
        values = _frange(center, half_width, step, bounds)
        pool = existing_values.get(param_type, set())
        for value in values:
            if value in pool:
                continue
            refined.append(
                ScoreConfig(
                    label=f"{mode}_refined_{value:.2f}",
                    mode=mode,
                    param=value,
                    param_type=param_type,
                )
            )
            pool.add(value)

    return refined


def evaluate_config_window(
    score_config: ScoreConfig,
    gap: int,
    filter_mode: str,
    start_date: str,
    end_date: str,
) -> dict[str, Any] | None:
    cache_key = (
        score_config.label,
        score_config.mode,
        score_config.param,
        gap,
        filter_mode,
        start_date,
        end_date,
    )
    if cache_key in EVAL_CACHE:
        return EVAL_CACHE[cache_key]

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
        EVAL_CACHE[cache_key] = None
        return None

    returns = overall["return"]
    metrics = metrics_to_floats(compute_metrics(returns))
    turnover_stats = summarize_turnover(compute_turnover_series(overall))
    benchmarks = compute_benchmark_metrics(result["monthly_prices"], returns.index)
    spy_metrics = benchmarks.get(OBJECTIVE_BENCHMARK)
    excess_cagr, objective = compute_excess_and_objective(metrics, spy_metrics)

    payload = {
        "metrics": metrics,
        "turnover": turnover_stats,
        "benchmarks": benchmarks,
        "excess_cagr": excess_cagr,
        "objective": objective,
        "start": str(returns.index[0].date()),
        "end": str(returns.index[-1].date()),
    }
    EVAL_CACHE[cache_key] = payload
    return payload


def run_walk_forward_splits(
    score_configs: Iterable[ScoreConfig], splits: list[dict[str, Any]], label: str
) -> dict[str, Any]:
    combo_defs = build_combo_defs(score_configs)
    print(
        f"\nWalk-forward set '{label}': {len(splits)} splits | {len(combo_defs)} combos"
    )
    split_entries: list[dict[str, Any]] = []
    aggregate: dict[tuple[Any, ...], dict[str, Any]] = {}
    selection_counter: Counter[str] = Counter()

    for idx, block in enumerate(splits, start=1):
        train_start, train_end = block["train"]
        test_start, test_end = block["test"]
        block_name = block.get("name", f"split_{idx}")
        block_candidates: list[dict[str, Any]] = []

        for score_config, gap, filter_mode in combo_defs:
            train_snapshot = evaluate_config_window(
                score_config, gap, filter_mode, train_start, train_end
            )
            if train_snapshot is None:
                continue

            candidate = {
                "score": score_config.label,
                "score_param": score_config.param,
                "score_param_type": score_config.param_type,
                "rank_gap": gap,
                "filter": filter_mode,
                "train": train_snapshot,
                "_score_config": score_config,
            }

            test_snapshot = evaluate_config_window(
                score_config, gap, filter_mode, test_start, test_end
            )
            candidate["test"] = test_snapshot
            block_candidates.append(candidate)

            key = _candidate_signature(
                candidate["score"],
                candidate["score_param"],
                candidate["score_param_type"],
                gap,
                filter_mode,
            )
            entry = aggregate.setdefault(
                key,
                {
                    "score": candidate["score"],
                    "score_param": candidate["score_param"],
                    "score_param_type": candidate["score_param_type"],
                    "rank_gap": gap,
                    "filter": filter_mode,
                    "train_cagrs": [],
                    "train_excess": [],
                    "train_objectives": [],
                    "test_cagrs": [],
                    "test_excess": [],
                    "test_objectives": [],
                    "train_count": 0,
                    "test_count": 0,
                    "test_top_hits": 0,
                },
            )

            entry["train_count"] += 1
            entry["train_cagrs"].append(train_snapshot["metrics"]["cagr"])
            if train_snapshot.get("excess_cagr") is not None:
                entry["train_excess"].append(train_snapshot["excess_cagr"])
            if train_snapshot.get("objective") is not None:
                entry["train_objectives"].append(train_snapshot["objective"])

            if test_snapshot is not None:
                entry["test_count"] += 1
                entry["test_cagrs"].append(test_snapshot["metrics"]["cagr"])
                if test_snapshot.get("excess_cagr") is not None:
                    entry["test_excess"].append(test_snapshot["excess_cagr"])
                if test_snapshot.get("objective") is not None:
                    entry["test_objectives"].append(test_snapshot["objective"])

        if not block_candidates:
            print(f"  Split {block_name}: insufficient data; skipped")
            continue

        block_candidates.sort(
            key=lambda c: _selection_value(c["train"]),
            reverse=True,
        )
        winner = block_candidates[0]
        selection_counter[winner["score"]] += 1
        winner_train = winner["train"]
        winner_train_desc = (
            f"excess CAGR {winner_train['excess_cagr']*100:5.2f}%"
            if winner_train.get("excess_cagr") is not None
            else f"CAGR {winner_train['metrics']['cagr']*100:5.2f}%"
        )

        test_ranked = [c for c in block_candidates if c["test"]]
        test_ranked.sort(
            key=lambda c: c["test"]["metrics"]["cagr"],
            reverse=True,
        )
        for cand in test_ranked[:3]:
            agg_entry = aggregate.get(
                _candidate_signature(
                    cand["score"],
                    cand["score_param"],
                    cand["score_param_type"],
                    cand["rank_gap"],
                    cand["filter"],
                )
            )
            if agg_entry is not None:
                agg_entry["test_top_hits"] += 1

        best_test = winner.get("test")
        if best_test is None:
            print(
                f"  Split {block_name}: score={winner['score']} gap={winner['rank_gap']} filter={winner['filter']} | "
                f"train {winner_train_desc} | test N/A"
            )
        else:
            test_display = (
                best_test["excess_cagr"]
                if best_test.get("excess_cagr") is not None
                else best_test["metrics"]["cagr"]
            )
            print(
                f"  Split {block_name}: score={winner['score']} gap={winner['rank_gap']} filter={winner['filter']} | "
                f"train {winner_train_desc} | test excess CAGR {test_display*100:5.2f}%"
            )

        split_entries.append(
            {
                "name": block_name,
                "train_window": {"start": train_start, "end": train_end},
                "test_window": {"start": test_start, "end": test_end},
                "selected_config": {
                    "score": winner["score"],
                    "score_param": winner["score_param"],
                    "score_param_type": winner["score_param_type"],
                    "rank_gap": winner["rank_gap"],
                    "filter": winner["filter"],
                },
                "candidates": [strip_private_keys(c) for c in block_candidates],
            }
        )

    aggregate_rows: list[dict[str, Any]] = []
    for entry in aggregate.values():
        test_count = entry["test_count"]
        aggregate_rows.append(
            {
                "score": entry["score"],
                "score_param": entry["score_param"],
                "score_param_type": entry["score_param_type"],
                "rank_gap": entry["rank_gap"],
                "filter": entry["filter"],
                "avg_train_cagr": _avg(entry["train_cagrs"]),
                "avg_train_excess": _avg(entry["train_excess"]),
                "avg_test_cagr": _avg(entry["test_cagrs"]),
                "avg_test_excess": _avg(entry["test_excess"]),
                "avg_test_objective": _avg(entry["test_objectives"]),
                "train_count": entry["train_count"],
                "test_count": test_count,
                "test_top_hits": entry["test_top_hits"],
                "test_top_fraction": float(entry["test_top_hits"] / test_count) if test_count else 0.0,
            }
        )

    aggregate_rows.sort(
        key=lambda row: row["avg_test_cagr"] if row["avg_test_cagr"] is not None else float("-inf"),
        reverse=True,
    )

    return {
        "splits": split_entries,
        "aggregate": {
            "split_count": len(splits),
            "selection_counts": dict(selection_counter),
            "per_config": aggregate_rows,
            "top_configs": aggregate_rows[:10],
        },
    }


def run_walk_forward_analysis() -> dict[str, Any]:
    coarse_output = run_walk_forward_splits(SCORE_CONFIGS, WALK_FORWARD_SPLITS, label="coarse")
    refined_configs = build_refined_score_configs(
        coarse_output["aggregate"].get("per_config", []),
        SCORE_CONFIGS,
    )

    refined_output: dict[str, Any] | None = None
    if refined_configs:
        refined_output = run_walk_forward_splits(
            refined_configs,
            WALK_FORWARD_SPLITS,
            label="refined",
        )

    payload: dict[str, Any] = {
        "objective_gamma": OBJECTIVE_GAMMA,
        "splits": coarse_output["splits"],
        "aggregate": coarse_output["aggregate"],
        "refined": None,
    }

    if refined_output:
        payload["refined"] = {
            "score_configs": [
                {
                    "label": cfg.label,
                    "mode": cfg.mode,
                    "param": cfg.param,
                    "param_type": cfg.param_type,
                }
                for cfg in refined_configs
            ],
            "splits": refined_output["splits"],
            "aggregate": refined_output["aggregate"],
        }

    return payload

def main() -> None:
    summary_rows: list[dict] = []
    monthly_rows: list[dict] = []
    annual_rows: list[dict] = []
    benchmarks_by_window: dict[str, dict[str, Any]] = {}
    selected_full_result: dict[str, Any] | None = None

    for window_label, start, end in WINDOWS:
        print(f"\n===== Window {window_label}: {start} → {end} =====")
        for score_config, gap, filter_mode in COARSE_COMBO_DEFS:
            score_label = score_config.label
            result = backtest_momentum(
                tickers=UNIVERSE,
                bucket_map=BUCKET_MAP,
                start_date=start,
                end_date=end,
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
                and score_config.param == SELECTED_CONFIG.get("score_param")
            ):
                selected_full_result = result

            strat_metrics = metrics_to_floats(compute_metrics(overall["return"]))
            turnover_stats = summarize_turnover(
                compute_turnover_series(result["overall_returns"])
            )

            benchmark_bundle = benchmarks_by_window.get(window_label)
            if benchmark_bundle is None:
                monthly_prices = result["monthly_prices"]
                missing = [t for t in ("SPY", "QQQ") if t not in monthly_prices]
                if missing:
                    raise RuntimeError(f"Missing benchmark data for {missing}")
                spy_monthly = monthly_prices["SPY"].pct_change()
                qqq_monthly = monthly_prices["QQQ"].pct_change()
                spy_metrics = metrics_to_floats(
                    compute_metrics(spy_monthly.loc[overall.index].dropna())
                )
                qqq_metrics = metrics_to_floats(
                    compute_metrics(qqq_monthly.loc[overall.index].dropna())
                )
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

            excess_cagr = strat_metrics["cagr"] - spy_metrics["cagr"]
            objective_value = excess_cagr - OBJECTIVE_GAMMA * abs(
                strat_metrics["max_drawdown"]
            )

            monthly_returns = overall["return"].copy()
            annual_returns = compute_annual_returns(monthly_returns)

            summary_rows.append(
                {
                    "window": window_label,
                    "start": overall.index[0].date(),
                    "end": overall.index[-1].date(),
                    "score": score_label,
                    "score_param": score_config.param,
                    "score_param_type": score_config.param_type,
                    "rank_gap": gap,
                    "filter": filter_mode,
                    "cagr": strat_metrics["cagr"],
                    "sharpe": strat_metrics["sharpe"],
                    "max_dd": strat_metrics["max_drawdown"],
                    "total_return": strat_metrics["total_return"],
                    "excess_cagr_spy": excess_cagr,
                    "objective": objective_value,
                    "turnover_monthly_median": turnover_stats["monthly_median"],
                    "turnover_monthly_p90": turnover_stats["monthly_p90"],
                    "turnover_annual_median": turnover_stats["annual_median"],
                    "turnover_annual_p90": turnover_stats["annual_p90"],
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
                        "score_param": score_config.param,
                        "score_param_type": score_config.param_type,
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
                        "score_param": score_config.param,
                        "score_param_type": score_config.param_type,
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
            param_desc = ""
            if pd.notna(row.get("score_param")) and row.get("score_param_type"):
                param_desc = f" {row['score_param_type']}={row['score_param']:.2f}"
            print(
                f"  score={row['score']:12s}{param_desc:>12s} gap={row['rank_gap']} filter={row['filter']:7s} | "
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
    wf_path = Path("us_rotation_walk_forward.json")
    wf_path.write_text(json.dumps(walk_forward_data, indent=2))
    print(f"\nSaved walk-forward summary to {wf_path}")


if __name__ == "__main__":
    main()

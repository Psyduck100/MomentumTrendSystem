from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pandas as pd

from momentum_program.analytics.constants import (
    SCORE_MODE_BLEND_6_12,
    SCORE_MODE_RW_ALPHA,
)
from momentum_program.backtest.engine import backtest_momentum
from us_rotation_custom import BUCKET_MAP, BACKTEST_CACHE, UNIVERSE

START_DATE = "2001-01-01"
END_DATE = pd.Timestamp.today().strftime("%Y-%m-%d")
OUTPUT_PATH = Path("strategy_annual_returns_two_configs.csv")


@dataclass(frozen=True)
class StrategySpec:
    name: str
    score_mode: str
    score_param: Optional[float]
    rank_gap: int
    filter_mode: str


def compound_by_year(monthly_returns: pd.Series) -> pd.Series:
    monthly_returns = monthly_returns.dropna()
    if monthly_returns.empty:
        return pd.Series(dtype=float)
    frame = monthly_returns.to_frame("ret")
    frame["year"] = frame.index.year
    annual = frame.groupby("year")["ret"].apply(
        lambda sr: float((1.0 + sr).prod() - 1.0)
    )
    annual.index = annual.index.astype(int)
    return annual


def run_strategy(spec: StrategySpec) -> tuple[pd.Series, pd.Series, pd.Series]:
    print(
        f"Running strategy {spec.name}: mode={spec.score_mode} param={spec.score_param} "
        f"gap={spec.rank_gap} filter={spec.filter_mode}"
    )
    result = backtest_momentum(
        tickers=UNIVERSE,
        bucket_map=BUCKET_MAP,
        start_date=START_DATE,
        end_date=END_DATE,
        top_n_per_bucket=1,
        rank_gap_threshold=spec.rank_gap,
        score_mode=spec.score_mode,
        score_param=spec.score_param,
        abs_filter_mode=spec.filter_mode,
        abs_filter_band=0.0,
        abs_filter_cash_annual=0.04,
        cache_dir=BACKTEST_CACHE,
        slippage_bps=3.0,
        expense_ratio=0.001,
    )

    overall = result["overall_returns"]
    if overall.empty:
        raise RuntimeError(f"No data returned for {spec.name}")

    monthly = overall["return"].copy()
    monthly.index = pd.to_datetime(monthly.index)

    monthly_prices = result["monthly_prices"]
    spy_monthly = monthly_prices["SPY"].pct_change().reindex(monthly.index)
    qqq_monthly = monthly_prices["QQQ"].pct_change().reindex(monthly.index)

    return (
        compound_by_year(monthly),
        compound_by_year(spy_monthly),
        compound_by_year(qqq_monthly),
    )


def main() -> None:
    strategies = [
        StrategySpec(
            name="rw_alpha_refined_alpha1.70_gap1_ret6m",
            score_mode=SCORE_MODE_RW_ALPHA,
            score_param=1.70,
            rank_gap=1,
            filter_mode="ret_6m",
        ),
        StrategySpec(
            name="blend_6_12_gap0_ret12m",
            score_mode=SCORE_MODE_BLEND_6_12,
            score_param=None,
            rank_gap=0,
            filter_mode="ret_12m",
        ),
    ]

    annual_data: dict[str, pd.Series] = {}
    spy_annual: pd.Series | None = None
    qqq_annual: pd.Series | None = None

    for spec in strategies:
        strat_series, spy_series, qqq_series = run_strategy(spec)
        annual_data[spec.name] = strat_series
        spy_annual = (
            spy_series if spy_annual is None else spy_annual.combine_first(spy_series)
        )
        qqq_annual = (
            qqq_series if qqq_annual is None else qqq_annual.combine_first(qqq_series)
        )

    if spy_annual is None or qqq_annual is None:
        raise RuntimeError("Failed to compute benchmark annual returns")

    all_years = sorted(
        set().union(
            *(series.index.tolist() for series in annual_data.values()),
            spy_annual.index.tolist(),
        )
    )
    table = pd.DataFrame({"year": all_years})
    for name, series in annual_data.items():
        table[name] = table["year"].map(series)
    table["spy_annual_return"] = table["year"].map(spy_annual)
    table["qqq_annual_return"] = table["year"].map(qqq_annual)

    table.sort_values("year", inplace=True)
    table.to_csv(OUTPUT_PATH, index=False)
    print(f"Saved annual return comparison to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()

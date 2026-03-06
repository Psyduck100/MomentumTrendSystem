from __future__ import annotations

from pathlib import Path

import pandas as pd

from momentum_program.analytics.constants import SCORE_MODE_BLEND_6_12
from momentum_program.backtest.engine import backtest_momentum
from momentum_program.backtest.metrics import compute_metrics
from us_rotation_custom import BUCKET_MAP, BACKTEST_CACHE, UNIVERSE

START_DATE = "2002-08-01"
END_DATE = pd.Timestamp.today().strftime("%Y-%m-%d")
TICKERS_WITH_BOND = list(dict.fromkeys(UNIVERSE + ["IEF"]))
OUTPUT_PATH = Path("defensive_allocation_cagr.csv")


def run_strategy(label: str, defensive_symbol: str | None) -> tuple[dict, dict, dict, pd.Series]:
    """Run the blend_6_12 strategy with/without a defensive ETF fallback."""
    print(
        f"Running {label}: defensive_symbol={defensive_symbol or 'cash'} | "
        f"start={START_DATE} end={END_DATE}"
    )
    result = backtest_momentum(
        tickers=TICKERS_WITH_BOND,
        bucket_map=BUCKET_MAP,
        start_date=START_DATE,
        end_date=END_DATE,
        top_n_per_bucket=1,
        rank_gap_threshold=0,
        score_mode=SCORE_MODE_BLEND_6_12,
        score_param=None,
        abs_filter_mode="ret_12m",
        abs_filter_band=0.0,
        abs_filter_cash_annual=0.04,
        cache_dir=BACKTEST_CACHE,
        slippage_bps=3.0,
        expense_ratio=0.001,
        defensive_symbol=defensive_symbol,
    )

    returns = result["overall_returns"]["return"].dropna()
    metrics = compute_metrics(returns)

    monthly_prices = result["monthly_prices"].reindex(returns.index)
    spy_returns = monthly_prices["SPY"].pct_change().dropna()
    qqq_returns = monthly_prices["QQQ"].pct_change().dropna()

    spy_metrics = compute_metrics(spy_returns)
    qqq_metrics = compute_metrics(qqq_returns)

    row = {
        "strategy": label,
        "start": returns.index[0].date(),
        "end": returns.index[-1].date(),
        "cagr": metrics["cagr"],
        "sharpe": metrics["sharpe"],
        "max_drawdown": metrics["max_drawdown"],
    }
    return row, spy_metrics, qqq_metrics, returns.index


def main() -> None:
    rows: list[dict] = []
    benchmark_rows: dict[str, dict] | None = None
    shared_index: pd.Series | None = None

    for label, defensive_symbol in (
        ("blend_6_12_cash", None),
        ("blend_6_12_ief", "IEF"),
    ):
        row, spy_metrics, qqq_metrics, index = run_strategy(label, defensive_symbol)
        rows.append(row)
        shared_index = index
        benchmark_rows = {
            "SPY": {
                "strategy": "SPY",
                "start": index[0].date(),
                "end": index[-1].date(),
                "cagr": spy_metrics["cagr"],
                "sharpe": spy_metrics["sharpe"],
                "max_drawdown": spy_metrics["max_drawdown"],
            },
            "QQQ": {
                "strategy": "QQQ",
                "start": index[0].date(),
                "end": index[-1].date(),
                "cagr": qqq_metrics["cagr"],
                "sharpe": qqq_metrics["sharpe"],
                "max_drawdown": qqq_metrics["max_drawdown"],
            },
        }

    if benchmark_rows is None or shared_index is None:
        raise RuntimeError("Benchmarks were not computed.")

    rows.extend(benchmark_rows.values())
    df = pd.DataFrame(rows)
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"Saved defensive allocation CAGR comparison to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()

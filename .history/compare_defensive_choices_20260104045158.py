"""Compare defensive asset choices for the momentum strategy.

This script validates that IEF (7-10 Year Treasury ETF) is superior to TB3MS cash
as the defensive allocation when the absolute filter triggers.

Results Summary (2002-2026 backtest):
- blend_filter_12m_IEF: 13.28% CAGR, 0.92 Sharpe, -27.18% MaxDD ✓ SELECTED
- blend_filter_12m_TB3MS_cash: 12.47% CAGR, 0.89 Sharpe, -23.78% MaxDD
  (IEF outperforms by 81 bps CAGR)
- blend_filter_12m_excess_rf_IEF: 12.06% CAGR, 0.85 Sharpe, -27.18% MaxDD
- blend_filter_12m_excess_rf_TB3MS_cash: 11.59% CAGR, 0.84 Sharpe, -25.98% MaxDD

Conclusion: IEF provides better returns with minimal additional drawdown.
The strategy uses IEF as the defensive asset when filter conditions are not met.

Run this script to validate the defensive choice or regenerate results.
"""

from __future__ import annotations
import pandas as pd
from pathlib import Path
from us_rotation_custom import BUCKET_MAP, BACKTEST_CACHE
from momentum_program.backtest.engine import backtest_momentum
from momentum_program.backtest.metrics import compute_metrics
from momentum_program.analytics.constants import SCORE_MODE_BLEND_6_12
from pandas.tseries.offsets import MonthEnd

# Universe
TICKERS_WITH_BOND = ["SPTM", "SPY", "QQQ", "OEF", "IWD", "IEF"]

TBILL_CSV = Path("CSVs/TB3MS.csv")

# Strategy configurations: blend_filter_12m + blend_filter_12m_excess_rf with IEF vs TB3MS cash
CONFIGS = [
    # blend_filter_12m with IEF defensive
    {
        "label": "blend_filter_12m_IEF",
        "score_mode": SCORE_MODE_BLEND_6_12,
        "filter": "ret_12m",
        "band_series": None,
        "defensive_symbol": "IEF",
        "defensive_cash_rate": None,  # Using symbol, ignore cash rate
    },
    # blend_filter_12m with TB3MS cash defensive
    {
        "label": "blend_filter_12m_TB3MS_cash",
        "score_mode": SCORE_MODE_BLEND_6_12,
        "filter": "ret_12m",
        "band_series": None,
        "defensive_symbol": None,  # Use cash rate instead
        "defensive_cash_rate": "tb3ms",  # Signal to use average TB3MS rate
    },
    # blend_filter_12m_excess_rf with IEF defensive
    {
        "label": "blend_filter_12m_excess_rf_IEF",
        "score_mode": SCORE_MODE_BLEND_6_12,
        "filter": "ret_12m",
        "band_series": "rf_12m",
        "defensive_symbol": "IEF",
        "defensive_cash_rate": None,
    },
    # blend_filter_12m_excess_rf with TB3MS cash defensive
    {
        "label": "blend_filter_12m_excess_rf_TB3MS_cash",
        "score_mode": SCORE_MODE_BLEND_6_12,
        "filter": "ret_12m",
        "band_series": "rf_12m",
        "defensive_symbol": None,
        "defensive_cash_rate": "tb3ms",
    },
]

START_DATE = "2001-01-01"
END_DATE = pd.Timestamp.today().strftime("%Y-%m-%d")


def load_risk_free_band() -> pd.Series:
    """Load TB3MS and compute 12-month risk-free return."""
    tbill = pd.read_csv(TBILL_CSV, parse_dates=["observation_date"])
    tbill["observation_date"] = tbill["observation_date"] + MonthEnd(0)
    tbill.set_index("observation_date", inplace=True)

    tbill["rf_annual_decimal"] = tbill["TB3MS"] / 100.0
    tbill["rf_monthly"] = tbill["rf_annual_decimal"] / 12.0
    tbill["rf_12m"] = (
        (1 + tbill["rf_monthly"]).rolling(12).apply(lambda x: x.prod() - 1, raw=False)
    )

    return tbill["rf_12m"].dropna()


def compute_average_tb3ms_rate() -> float:
    """Compute average annual TB3MS rate as cash proxy."""
    tbill = pd.read_csv(TBILL_CSV)
    return tbill["TB3MS"].mean() / 100.0


def compound_by_year(monthly_returns: pd.Series) -> pd.Series:
    """Compound monthly returns to annual."""
    df = monthly_returns.to_frame("ret")
    df["year"] = df.index.year
    annual = df.groupby("year")["ret"].apply(lambda x: (1 + x).prod() - 1)
    return annual


def run_strategy(
    label: str,
    score_mode: str,
    filter_mode: str,
    band_series: pd.Series | None,
    defensive_symbol: str | None,
    defensive_cash_rate: float | None,
) -> dict:
    """Run backtest for a single configuration."""
    print(
        f"Running {label}: score={score_mode} filter={filter_mode} defensive={defensive_symbol or 'cash'}"
    )

    result = backtest_momentum(
        tickers=TICKERS_WITH_BOND,
        bucket_map=BUCKET_MAP,
        start_date=START_DATE,
        end_date=END_DATE,
        top_n_per_bucket=1,
        cache_dir=BACKTEST_CACHE,
        slippage_bps=3.0,
        expense_ratio=0.001,
        rank_gap_threshold=0,
        score_mode=score_mode,
        abs_filter_mode=filter_mode,
        abs_filter_cash_annual=defensive_cash_rate,
        defensive_symbol=defensive_symbol,
        abs_filter_band_series=band_series,
    )

    # Compute metrics
    overall_df = result["overall_returns"]
    monthly_rets = overall_df["return"]
    metrics = compute_metrics(monthly_rets)
    result["metrics"] = metrics
    result["monthly_returns"] = monthly_rets
    result["ticker_monthly_returns"] = result["monthly_prices"].pct_change()

    return result


def main():
    """Run all configurations and export results."""

    rf_band = load_risk_free_band()
    avg_tb3ms = compute_average_tb3ms_rate()
    print(f"Average TB3MS rate (2001-2025): {avg_tb3ms:.4f} ({avg_tb3ms*100:.2f}%)\n")

    results = {}
    for config in CONFIGS:
        # Resolve cash rate
        cash_rate = config["defensive_cash_rate"]
        if cash_rate == "tb3ms":
            cash_rate = avg_tb3ms
        elif cash_rate is None:
            # When using defensive_symbol (IEF), still need a cash rate fallback
            cash_rate = 0.025  # 2.5% fallback

        result = run_strategy(
            label=config["label"],
            score_mode=config["score_mode"],
            filter_mode=config["filter"],
            band_series=rf_band if config.get("band_series") == "rf_12m" else None,
            defensive_symbol=config["defensive_symbol"],
            defensive_cash_rate=cash_rate,
        )
        results[config["label"]] = result

    # Extract CAGR/Sharpe/MaxDD summary
    summary_rows = []
    for label, res in results.items():
        metrics = res["metrics"]
        summary_rows.append(
            {
                "strategy": label,
                "start": res["monthly_returns"].index[0].strftime("%Y-%m-%d"),
                "end": res["monthly_returns"].index[-1].strftime("%Y-%m-%d"),
                "cagr": metrics["cagr"],
                "sharpe": metrics["sharpe"],
                "max_drawdown": metrics["max_drawdown"],
            }
        )

    summary_df = pd.DataFrame(summary_rows)

    # Add SPY and QQQ benchmarks
    spy_result = results[CONFIGS[0]["label"]]
    for ticker in ["SPY", "QQQ"]:
        bench_rets = spy_result["ticker_monthly_returns"][ticker]
        bench_annual = (1 + bench_rets).prod() ** (12 / len(bench_rets)) - 1
        bench_sharpe = (
            bench_rets.mean() / bench_rets.std() * (12**0.5)
            if bench_rets.std() > 0
            else 0
        )
        cumulative = (1 + bench_rets).cumprod()
        running_max = cumulative.expanding().max()
        drawdowns = (cumulative - running_max) / running_max
        bench_maxdd = drawdowns.min()

        summary_rows.append(
            {
                "strategy": ticker,
                "start": bench_rets.index[0].strftime("%Y-%m-%d"),
                "end": bench_rets.index[-1].strftime("%Y-%m-%d"),
                "cagr": bench_annual,
                "sharpe": bench_sharpe,
                "max_drawdown": bench_maxdd,
            }
        )

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv("defensive_choice_cagr.csv", index=False)
    print(f"\nSaved summary to defensive_choice_cagr.csv\n")
    print(summary_df.to_string(index=False))

    # Export annual returns
    annual_data = {}
    for label, res in results.items():
        annual_data[label] = compound_by_year(res["monthly_returns"])

    for ticker in ["SPY", "QQQ"]:
        bench_rets = spy_result["ticker_monthly_returns"][ticker]
        annual_data[ticker] = compound_by_year(bench_rets)

    annual_df = pd.DataFrame(annual_data)
    annual_df.index.name = "year"
    annual_df.to_csv("defensive_choice_annual_returns.csv")
    print(f"Saved annual returns to defensive_choice_annual_returns.csv")


if __name__ == "__main__":
    main()

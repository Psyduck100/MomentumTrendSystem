"""Dual momentum test: US (SPY) vs Intl ex-US (ACWX), defensive IEF.

Implements the classic dual momentum rules:
- Relative momentum: choose winner between SPY and ACWX by trailing 12M or blend_6_12 score
- Absolute momentum: require winner's 12M return > 12M T-bill return (from TB3MS)
- If absolute filter fails: hold IEF (defensive)

Outputs CAGR/Sharpe/MaxDD and annual returns for two score modes: 12M and blend_6_12.
"""

from __future__ import annotations
import pandas as pd
from pathlib import Path
from pandas.tseries.offsets import MonthEnd

from momentum_program.backtest.engine import backtest_momentum
from momentum_program.backtest.metrics import compute_metrics
from momentum_program.analytics.constants import SCORE_MODE_BLEND_6_12, SCORE_MODE_12M

TICKERS = ["SPY", "ACWX", "IEF"]
BUCKET_MAP = {
    "SPY": "Equity",
    "ACWX": "Equity",
    "IEF": "Bonds",
}

TBILL_CSV = Path("CSVs/TB3MS.csv")
BACKTEST_CACHE = Path("backtest_cache")

START_DATE = "2008-01-01"  # ACWX inception ~2008
END_DATE = pd.Timestamp.today().strftime("%Y-%m-%d")
CASH_RATE = 0.025  # fallback if defensive_symbol missing

CONFIGS = [
    {"label": "gem_12m", "score_mode": SCORE_MODE_12M},
    {"label": "gem_blend_6_12", "score_mode": SCORE_MODE_BLEND_6_12},
]


def load_risk_free_band() -> pd.Series:
    tbill = pd.read_csv(TBILL_CSV, parse_dates=["observation_date"])
    tbill["observation_date"] = tbill["observation_date"] + MonthEnd(0)
    tbill.set_index("observation_date", inplace=True)
    tbill["rf_annual_decimal"] = tbill["TB3MS"] / 100.0
    tbill["rf_monthly"] = tbill["rf_annual_decimal"] / 12.0
    tbill["rf_12m"] = (
        (1 + tbill["rf_monthly"]).rolling(12).apply(lambda x: x.prod() - 1, raw=False)
    )
    return tbill["rf_12m"].dropna()


def compound_by_year(monthly_returns: pd.Series) -> pd.Series:
    df = monthly_returns.to_frame("ret")
    df["year"] = df.index.year
    return df.groupby("year")["ret"].apply(lambda x: (1 + x).prod() - 1)


def run_strategy(label: str, score_mode: str, rf_band: pd.Series) -> dict:
    result = backtest_momentum(
        tickers=TICKERS,
        bucket_map=BUCKET_MAP,
        start_date=START_DATE,
        end_date=END_DATE,
        top_n_per_bucket=1,
        cache_dir=BACKTEST_CACHE,
        slippage_bps=3.0,
        expense_ratio=0.001,
        rank_gap_threshold=0,
        score_mode=score_mode,
        abs_filter_mode="ret_12m",
        abs_filter_cash_annual=CASH_RATE,
        defensive_symbol="IEF",
        abs_filter_band_series=rf_band,
    )

    overall_df = result["overall_returns"]
    monthly_rets = overall_df["return"]
    metrics = compute_metrics(monthly_rets)
    result["metrics"] = metrics
    result["monthly_returns"] = monthly_rets
    result["ticker_monthly_returns"] = result["monthly_prices"].pct_change()
    return result


def main() -> None:
    rf_band = load_risk_free_band()

    results: dict[str, dict] = {}
    for cfg in CONFIGS:
        print(f"Running {cfg['label']} (score={cfg['score_mode']})")
        res = run_strategy(cfg["label"], cfg["score_mode"], rf_band)
        results[cfg["label"]] = res

    rows = []
    for label, res in results.items():
        metrics = res["metrics"]
        rows.append(
            {
                "strategy": label,
                "start": res["monthly_returns"].index[0].strftime("%Y-%m-%d"),
                "end": res["monthly_returns"].index[-1].strftime("%Y-%m-%d"),
                "cagr": metrics["cagr"],
                "sharpe": metrics["sharpe"],
                "max_drawdown": metrics["max_drawdown"],
            }
        )

    summary_df = pd.DataFrame(rows)
    summary_df.to_csv("gem_dual_momentum_cagr.csv", index=False)
    print("\nSaved gem_dual_momentum_cagr.csv")
    print(summary_df.to_string(index=False))

    annual_data = {}
    for label, res in results.items():
        annual_data[label] = compound_by_year(res["monthly_returns"])

    annual_df = pd.DataFrame(annual_data)
    annual_df.index.name = "year"
    annual_df.to_csv("gem_dual_momentum_annual_returns.csv")
    print("Saved gem_dual_momentum_annual_returns.csv")


if __name__ == "__main__":
    main()

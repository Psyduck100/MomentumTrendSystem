"""Dual momentum test: US (SPY) vs Intl ex-US using EFA pre-ACWX, then ACWX.

Rules:
- Relative momentum: winner between SPY and composite INTL (EFA until ACWX exists, then ACWX)
- No absolute momentum filter: always hold the winner (no defensive switch)

Outputs CAGR/Sharpe/MaxDD and annual returns for two score modes: 12M and blend_6_12.
"""

from __future__ import annotations
import hashlib
import pandas as pd
import yfinance as yf
from pathlib import Path

from momentum_program.backtest.engine import backtest_momentum
from momentum_program.backtest.metrics import compute_metrics
from momentum_program.analytics.constants import SCORE_MODE_BLEND_6_12, SCORE_MODE_12M

# Composite ticker setup
EQUITY_TICKERS = ["SPY", "EFA", "ACWX", "IEF"]
TICKERS = ["SPY", "INTL", "IEF"]  # INTL is EFA until ACWX data exists, then ACWX
BUCKET_MAP = {
    "SPY": "Equity",
    "INTL": "Equity",
    "IEF": "Bonds",
}

BACKTEST_CACHE = Path("backtest_cache")

START_DATE = "2001-01-01"  # EFA inception ~2001
END_DATE = pd.Timestamp.today().strftime("%Y-%m-%d")
CASH_RATE = 0.025  # fallback if defensive_symbol missing

CONFIGS = [
    {"label": "gem_12m", "score_mode": SCORE_MODE_12M},
    {"label": "gem_blend_6_12", "score_mode": SCORE_MODE_BLEND_6_12},
]


def build_composite_price_cache() -> Path:
    """Create composite INTL series (EFA until ACWX available) and save to cache."""
    BACKTEST_CACHE.mkdir(exist_ok=True)
    fingerprint = hashlib.md5(",".join(sorted(TICKERS)).encode("utf-8")).hexdigest()[
        :10
    ]
    cache_file = (
        BACKTEST_CACHE / f"price_data_{START_DATE}_{END_DATE}_{fingerprint}.csv"
    )

    if cache_file.exists():
        return cache_file

    print("Building composite INTL series (EFA until ACWX available)...")
    data = yf.download(EQUITY_TICKERS, start=START_DATE, end=END_DATE, progress=False)
    if isinstance(data.columns, pd.MultiIndex):
        data = data.swaplevel(0, 1, axis=1).sort_index(axis=1)
        adj_close = pd.DataFrame()
        for tkr in EQUITY_TICKERS:
            sub = data[tkr]
            if "Adj Close" in sub.columns:
                adj_close[tkr] = sub["Adj Close"]
            elif "Close" in sub.columns:
                adj_close[tkr] = sub["Close"]
    else:
        adj_close = data[["Adj Close"]].rename(columns={"Adj Close": EQUITY_TICKERS[0]})

    # Build composite INTL: use ACWX where available, otherwise EFA
    intl = adj_close["ACWX"].combine_first(adj_close["EFA"])
    composite = pd.DataFrame(
        {
            "SPY": adj_close["SPY"],
            "INTL": intl,
            "IEF": adj_close["IEF"],
        }
    )

    composite.to_csv(cache_file)
    print(f"Saved composite price data to cache: {cache_file}")
    return cache_file


def compound_by_year(monthly_returns: pd.Series) -> pd.Series:
    df = monthly_returns.to_frame("ret")
    df["year"] = df.index.year
    return df.groupby("year")["ret"].apply(lambda x: (1 + x).prod() - 1)


def run_strategy(label: str, score_mode: str) -> dict:
    build_composite_price_cache()
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
        abs_filter_mode="none",  # no absolute filter
        abs_filter_cash_annual=CASH_RATE,
        defensive_symbol=None,  # no defensive switch
        abs_filter_band_series=None,
        abs_filter_band=0.0,
    )

    overall_df = result["overall_returns"]
    monthly_rets = overall_df["return"]
    metrics = compute_metrics(monthly_rets)
    result["metrics"] = metrics
    result["monthly_returns"] = monthly_rets
    result["ticker_monthly_returns"] = result["monthly_prices"].pct_change()
    return result


def main() -> None:
    results: dict[str, dict] = {}
    for cfg in CONFIGS:
        print(f"Running {cfg['label']} (score={cfg['score_mode']})")
        res = run_strategy(cfg["label"], cfg["score_mode"])
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

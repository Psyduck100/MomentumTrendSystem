"""Layered dual-momentum strategy: US bucket rotation → INTL → Bonds.

Layer 1: US bucket rotation (blend_6_12 + ret_12m filter)
- Ranks: SCHB (VTI pre-2003), XLG (OEF pre-2006), SCHV (IWD pre-2004), QQQ, RSP (SPY pre-2003)
- Winner: top-ranked ETF by blend_6_12 score
- Filter: winner's 12M return > 0

Layer 2: If US filter fails
- Check INTL (EFA pre-ACWX, then ACWX)
- If INTL 12M > 0: hold INTL
- Else: hold IEF

Outputs CAGR/Sharpe/MaxDD and annual returns.
"""

from __future__ import annotations
import hashlib
import pandas as pd
import yfinance as yf
from pathlib import Path

from momentum_program.backtest.metrics import compute_metrics

PROXIES = {
    "SCHB": "VTI",  # before SCHB
    "XLG": "OEF",   # before XLG
    "SCHV": "IWD",  # before SCHV
    "QQQ": "QQQ",   # full history
    "RSP": "SPY",   # before RSP
}

US_TICKERS = list(PROXIES.keys())
PROXY_TICKERS = list(set(PROXIES.values()))
INTL_TICKERS = ["EFA", "ACWX"]
BONDS = ["IEF"]

ALL_TICKERS = US_TICKERS + PROXY_TICKERS + INTL_TICKERS + BONDS + ["SPY"]
BACKTEST_CACHE = Path("backtest_cache")

START_DATE = "2001-01-01"
END_DATE = pd.Timestamp.today().strftime("%Y-%m-%d")


def build_composite_cache() -> str:
    """Download all tickers and build composites, return cache path."""
    BACKTEST_CACHE.mkdir(exist_ok=True)
    fingerprint = hashlib.md5(
        ",".join(sorted(ALL_TICKERS)).encode("utf-8")
    ).hexdigest()[:10]
    cache_file = BACKTEST_CACHE / f"layered_prices_{START_DATE}_{END_DATE}_{fingerprint}.csv"

    if cache_file.exists():
        print(f"Using cached prices: {cache_file}")
        return str(cache_file)

    print(f"Downloading {len(set(ALL_TICKERS))} tickers...")
    unique_tickers = list(set(ALL_TICKERS))
    data = yf.download(unique_tickers, start=START_DATE, end=END_DATE, progress=False)

    if isinstance(data.columns, pd.MultiIndex):
        data = data.swaplevel(0, 1, axis=1).sort_index(axis=1)
        prices = pd.DataFrame()
        for tkr in unique_tickers:
            sub = data[tkr]
            if "Adj Close" in sub.columns:
                prices[tkr] = sub["Adj Close"]
            elif "Close" in sub.columns:
                prices[tkr] = sub["Close"]
    else:
        prices = data[["Adj Close"]].rename(columns={"Adj Close": unique_tickers[0]})

    # Build composites for each US ticker (actual where available, proxy otherwise)
    composites = pd.DataFrame()
    for ticker, proxy in PROXIES.items():
        composites[ticker] = prices[ticker].combine_first(prices[proxy])

    # INTL composite: ACWX where available, else EFA
    composites["INTL"] = prices["ACWX"].combine_first(prices["EFA"])

    # Bonds
    composites["IEF"] = prices["IEF"]

    # SPY (for gating logic)
    composites["SPY"] = prices["SPY"]

    composites.to_csv(cache_file)
    print(f"Saved composite prices to cache: {cache_file}")
    return str(cache_file)


def compute_monthly_returns(prices: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Resample to month-end and compute 6M, 12M returns."""
    monthly = prices.resample("ME").last()

    ret_6m = monthly.pct_change(6)
    ret_12m = monthly.pct_change(12)
    blend_6_12 = 0.5 * ret_6m + 0.5 * ret_12m

    return monthly, {"ret_6m": ret_6m, "ret_12m": ret_12m, "blend_6_12": blend_6_12}


def backtest_layered(prices: pd.DataFrame) -> dict:
    """
    Gated layering: SPY momentum gates access to US bucket rotation.
    
    Layer 1: SPY 12M > 0?
      YES → Run US bucket rotation (blend_6_12 score on SCHB/XLG/SCHV/QQQ/RSP)
      NO  → Check Layer 2
    
    Layer 2: INTL 12M > 0?
      YES → Hold INTL (EFA/ACWX composite)
      NO  → Check Layer 3
    
    Layer 3: Hold IEF
    """
    monthly, returns = compute_monthly_returns(prices)

    positions = []
    monthly_rets = []

    us_tickers = list(PROXIES.keys())

    # Start from month 13 (first valid 12M return)
    for i in range(12, len(returns["blend_6_12"])):
        spy_12m = returns["ret_12m"].iloc[i].get("SPY", 0) if "SPY" in returns["ret_12m"].columns else 0
        
        # Layer 1: SPY 12M > 0? If yes, run US bucket rotation
        if spy_12m > 0:
            # Run US bucket selection by blend_6_12 score
            blend_scores = returns["blend_6_12"].iloc[i][us_tickers].dropna()
            
            if len(blend_scores) > 0:
                position = blend_scores.idxmax()  # Winner by blend_6_12
            else:
                # Fallback if no valid scores: check INTL
                intl_12m = returns["ret_12m"].iloc[i].get("INTL", pd.NA)
                if not pd.isna(intl_12m) and intl_12m > 0:
                    position = "INTL"
                else:
                    position = "IEF"
        else:
            # SPY failed: Layer 2 - Check INTL
            intl_12m = returns["ret_12m"].iloc[i].get("INTL", pd.NA)
            if not pd.isna(intl_12m) and intl_12m > 0:
                position = "INTL"
            else:
                # Layer 3: Bonds
                position = "IEF"

        positions.append(position)

        # Return for next month (from i to i+1)
        if i + 1 < len(monthly):
            price_now = monthly.iloc[i][position]
            price_next = monthly.iloc[i + 1][position]
            monthly_ret = (price_next - price_now) / price_now if price_now != 0 else 0
            monthly_rets.append(monthly_ret)

    return_series = pd.Series(
        monthly_rets, index=monthly.index[13:len(monthly)]
    )
    return {
        "label": "spy_gated_us_bucket_intl_ief",
        "positions": positions,
        "returns": return_series,
    }


def compound_by_year(monthly_returns: pd.Series) -> pd.Series:
    df = monthly_returns.to_frame("ret")
    df["year"] = df.index.year
    return df.groupby("year")["ret"].apply(lambda x: (1 + x).prod() - 1)


def main() -> None:
    cache_file = build_composite_cache()
    prices = pd.read_csv(cache_file, index_col=0, parse_dates=True)

    result = backtest_layered(prices)
    ret_series = result["returns"]

    if len(ret_series) > 0:
        metrics = compute_metrics(ret_series)
        summary = pd.DataFrame(
            [
                {
                    "strategy": result["label"],
                    "start": ret_series.index[0].strftime("%Y-%m-%d"),
                    "end": ret_series.index[-1].strftime("%Y-%m-%d"),
                    "cagr": metrics["cagr"],
                    "sharpe": metrics["sharpe"],
                    "max_drawdown": metrics["max_drawdown"],
                }
            ]
        )

        summary.to_csv("layered_us_intl_bonds_cagr.csv", index=False)
        print("\nSaved layered_us_intl_bonds_cagr.csv")
        print(summary.to_string(index=False))

        annual_data = {"layered_us_intl_bonds": compound_by_year(ret_series)}
        annual_df = pd.DataFrame(annual_data)
        annual_df.index.name = "year"
        annual_df.to_csv("layered_us_intl_bonds_annual_returns.csv")
        print("\nSaved layered_us_intl_bonds_annual_returns.csv")


if __name__ == "__main__":
    main()

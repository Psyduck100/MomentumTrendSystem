"""Dual momentum with custom switching rules: SPY-default with fallback chain, and sticky SPY.

Strategy 1: "SPY default"
- If SPY 12M > 0: hold SPY
- Else if INTL 12M > 0: hold INTL
- Else: hold IEF

Strategy 2: "Sticky SPY" with hysteresis
- Maintain position state (on_spy True initially)
- If on_spy and SPY > 0: stay SPY
- If on_spy and SPY <= 0: switch to INTL (on_spy = False)
- If not on_spy and INTL <= 0: go to IEF
- If not on_spy and SPY > INTL: switch back to SPY (on_spy = True)
- If not on_spy and SPY <= INTL: stay INTL

Outputs CAGR/Sharpe/MaxDD and annual returns for both strategies.
"""

from __future__ import annotations
import hashlib
import pandas as pd
import yfinance as yf
from pathlib import Path

from momentum_program.backtest.metrics import compute_metrics
from momentum_program.analytics.constants import SCORE_MODE_BLEND_6_12, SCORE_MODE_12M

EQUITY_TICKERS = ["SPY", "EFA", "ACWX", "IEF"]
BACKTEST_CACHE = Path("backtest_cache")
RISK_FREE_ANNUAL = 0.02  # annual hurdle for INTL before allocation

START_DATE = "2001-01-01"
END_DATE = pd.Timestamp.today().strftime("%Y-%m-%d")


def build_composite_price_cache() -> pd.DataFrame:
    """Create composite INTL series (EFA until ACWX available) and return prices."""
    fingerprint = hashlib.md5(
        ",".join(sorted(["SPY", "EFA", "ACWX", "IEF"])).encode("utf-8")
    ).hexdigest()[:10]
    cache_file = BACKTEST_CACHE / f"price_data_{START_DATE}_{END_DATE}_{fingerprint}.csv"

    if cache_file.exists():
        prices = pd.read_csv(cache_file, index_col=0, parse_dates=True)
        return prices

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
    prices = pd.DataFrame({
        "SPY": adj_close["SPY"],
        "INTL": intl,
        "IEF": adj_close["IEF"],
    })

    prices.to_csv(cache_file)
    print(f"Saved composite price data to cache: {cache_file}")
    return prices


def compute_monthly_momentum(prices: pd.DataFrame, lookback_months: int = 12) -> pd.DataFrame:
    """Compute 12M momentum for each ticker."""
    monthly = prices.resample("ME").last()
    momentum = monthly.pct_change(lookback_months)
    return monthly, momentum


def backtest_spy_default(monthly_prices: pd.DataFrame, momentum: pd.DataFrame) -> dict:
    """
    Strategy 1: SPY default with fallback chain.
    - If SPY > 0: SPY
    - Else if INTL > 0: INTL
    - Else: IEF
    """
    positions = []
    returns = []

    for i in range(1, len(momentum)):
        spy_mom = momentum.iloc[i]["SPY"]
        intl_mom = momentum.iloc[i]["INTL"]

        if spy_mom > 0:
            position = "SPY"
        elif intl_mom > RISK_FREE_ANNUAL:
            # Only take INTL if it clears the risk-free hurdle
            position = "INTL"
        else:
            position = "IEF"

        positions.append(position)
        
        # Return for next month (from i to i+1)
        if i + 1 < len(monthly_prices):
            price_now = monthly_prices.iloc[i][position]
            price_next = monthly_prices.iloc[i + 1][position]
            monthly_ret = (price_next - price_now) / price_now if price_now != 0 else 0
            returns.append(monthly_ret)

    return_series = pd.Series(returns, index=monthly_prices.index[2:len(monthly_prices)])
    return {
        "label": "spy_default",
        "positions": positions,
        "returns": return_series,
    }


def backtest_spy_abs(monthly_prices: pd.DataFrame, momentum: pd.DataFrame) -> dict:
    """
    Absolute momentum only: If SPY 12M > 0 → SPY, else IEF (no INTL leg).
    """
    positions = []
    returns = []

    for i in range(1, len(momentum)):
        spy_mom = momentum.iloc[i]["SPY"]

        if spy_mom > 0:
            position = "SPY"
        else:
            position = "IEF"

        positions.append(position)

        if i + 1 < len(monthly_prices):
            price_now = monthly_prices.iloc[i][position]
            price_next = monthly_prices.iloc[i + 1][position]
            monthly_ret = (price_next - price_now) / price_now if price_now != 0 else 0
            returns.append(monthly_ret)

    return_series = pd.Series(returns, index=monthly_prices.index[2:len(monthly_prices)])
    return {
        "label": "spy_abs",
        "positions": positions,
        "returns": return_series,
    }


def backtest_sticky_spy(monthly_prices: pd.DataFrame, momentum: pd.DataFrame) -> dict:
    """
    Strategy 2: Sticky SPY with hysteresis.
    - Maintain position state (on_spy)
    - If on_spy and SPY > 0: stay SPY
    - If on_spy and SPY <= 0: switch to INTL (on_spy = False)
    - If not on_spy and INTL <= 0: go to IEF
    - If not on_spy and SPY > INTL: switch back to SPY (on_spy = True)
    - If not on_spy and SPY <= INTL: stay INTL
    """
    positions = []
    returns = []
    on_spy = True  # start on SPY

    for i in range(1, len(momentum)):
        spy_mom = momentum.iloc[i]["SPY"]
        intl_mom = momentum.iloc[i]["INTL"]

        if on_spy:
            if spy_mom > 0:
                position = "SPY"
            else:
                position = "INTL"
                on_spy = False
        else:  # on INTL
            if intl_mom <= 0:
                position = "IEF"
            elif spy_mom > intl_mom:
                position = "SPY"
                on_spy = True
            else:
                position = "INTL"

        positions.append(position)

        # Return for next month (from i to i+1)
        if i + 1 < len(monthly_prices):
            price_now = monthly_prices.iloc[i][position]
            price_next = monthly_prices.iloc[i + 1][position]
            monthly_ret = (price_next - price_now) / price_now if price_now != 0 else 0
            returns.append(monthly_ret)

    return_series = pd.Series(returns, index=monthly_prices.index[2:len(monthly_prices)])
    return {
        "label": "sticky_spy",
        "positions": positions,
        "returns": return_series,
    }


def compound_by_year(monthly_returns: pd.Series) -> pd.Series:
    df = monthly_returns.to_frame("ret")
    df["year"] = df.index.year
    return df.groupby("year")["ret"].apply(lambda x: (1 + x).prod() - 1)


def main() -> None:
    build_composite_price_cache()
    
    prices = pd.read_csv(
        BACKTEST_CACHE / f"price_data_{START_DATE}_{END_DATE}_{hashlib.md5(','.join(sorted(['SPY', 'EFA', 'ACWX', 'IEF'])).encode('utf-8')).hexdigest()[:10]}.csv",
        index_col=0,
        parse_dates=True,
    )

    monthly_prices, momentum = compute_monthly_momentum(prices)

    results = {}
    results["spy_default"] = backtest_spy_default(monthly_prices, momentum)
    results["sticky_spy"] = backtest_sticky_spy(monthly_prices, momentum)
    results["spy_abs"] = backtest_spy_abs(monthly_prices, momentum)

    rows = []
    for label, res in results.items():
        ret_series = res["returns"]
        if len(ret_series) > 0:
            metrics = compute_metrics(ret_series)
            rows.append(
                {
                    "strategy": label,
                    "start": ret_series.index[0].strftime("%Y-%m-%d"),
                    "end": ret_series.index[-1].strftime("%Y-%m-%d"),
                    "cagr": metrics["cagr"],
                    "sharpe": metrics["sharpe"],
                    "max_drawdown": metrics["max_drawdown"],
                }
            )

    summary_df = pd.DataFrame(rows)
    summary_df.to_csv("gem_switching_rules_cagr.csv", index=False)
    print("\nSaved gem_switching_rules_cagr.csv")
    print(summary_df.to_string(index=False))

    annual_data = {}
    for label, res in results.items():
        annual_data[label] = compound_by_year(res["returns"])

    annual_df = pd.DataFrame(annual_data)
    annual_df.index.name = "year"
    annual_df.to_csv("gem_switching_rules_annual_returns.csv")
    print("\nSaved gem_switching_rules_annual_returns.csv")


if __name__ == "__main__":
    main()

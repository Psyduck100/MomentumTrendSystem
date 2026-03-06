# GLD-SPY-IEF ladder (monthly, no-lookahead)
# Rule (evaluated at month-end t, held during month t+1):
# 1) If GLD > GLD_SMA150  -> hold GLD
# 2) Else if SPY 12M momentum > 0 -> hold SPY
# 3) Else -> hold IEF
#
# Uses Adj Close from yfinance (dividends included). Monthly resample = month-end.
# Outputs equity curve + summary stats + allocation percentages.

import numpy as np
import pandas as pd
import yfinance as yf


def max_drawdown(equity: pd.Series) -> float:
    peak = equity.cummax()
    dd = equity / peak - 1.0
    return float(dd.min())


def cagr_from_monthly(equity: pd.Series) -> float:
    equity = equity.dropna()
    if len(equity) < 2:
        return np.nan
    years = (equity.index[-1] - equity.index[0]).days / 365.25
    return float((equity.iloc[-1] / equity.iloc[0]) ** (1 / years) - 1)


def sharpe_monthly(rets: pd.Series) -> float:
    r = rets.dropna()
    if r.std(ddof=1) == 0 or len(r) < 12:
        return np.nan
    # Annualized Sharpe using monthly returns, rf assumed 0
    return float((r.mean() / r.std(ddof=1)) * np.sqrt(12))


def build_ladder(
    start="2004-01-01",
    end=None,
    gld_sma_window=150,  # this is a "months" window after resampling monthly
    spy_lookback_months=12,
):
    tickers = ["GLD", "SPY", "IEF"]
    px = yf.download(tickers, start=start, end=end, auto_adjust=True, progress=False)[
        "Close"
    ]

    # Month-end prices
    mpx = px.resample("M").last().dropna(how="all")

    # Monthly returns (held during the month)
    mret = mpx.pct_change()

    # --- Signals computed at month-end ---
    # 1) GLD trend: price > SMA(window) on month-end prices
    # IMPORTANT: gld_sma_window is in MONTHS (since we work monthly now).
    # If you truly meant 150 *TRADING DAYS*, use ~7 months instead.
    gld_sma = (
        mpx["GLD"].rolling(window=gld_sma_window, min_periods=gld_sma_window).mean()
    )
    gld_on = (mpx["GLD"] > gld_sma).astype(int)

    # 2) SPY 12M momentum: (P(t)/P(t-12) - 1) > 0
    spy_mom12 = mpx["SPY"] / mpx["SPY"].shift(spy_lookback_months) - 1.0
    spy_on = (spy_mom12 > 0).astype(int)

    # Ladder decision at month-end t (0=IEF, 1=SPY, 2=GLD)
    # Compute at t, then SHIFT by 1 to apply for month t+1 returns (no-lookahead).
    choice = pd.Series(index=mpx.index, dtype="int64")
    choice[:] = 0
    choice[(spy_on == 1)] = 1
    choice[(gld_on == 1)] = 2

    hold = choice.shift(1).fillna(0).astype(int)

    # Strategy returns by held asset
    strat_ret = pd.Series(index=mpx.index, dtype="float64")
    strat_ret[hold == 2] = mret["GLD"][hold == 2]
    strat_ret[hold == 1] = mret["SPY"][hold == 1]
    strat_ret[hold == 0] = mret["IEF"][hold == 0]
    strat_ret = strat_ret.fillna(0.0)

    equity = (1 + strat_ret).cumprod().rename("Ladder")

    # Benchmarks
    gld_eq = (1 + mret["GLD"].fillna(0)).cumprod().rename("GLD_BuyHold")
    spy_eq = (1 + mret["SPY"].fillna(0)).cumprod().rename("SPY_BuyHold")
    ief_eq = (1 + mret["IEF"].fillna(0)).cumprod().rename("IEF_BuyHold")

    out = pd.concat([equity, gld_eq, spy_eq, ief_eq], axis=1).dropna()

    # Allocation stats
    alloc = hold.value_counts(normalize=True).rename({0: "IEF", 1: "SPY", 2: "GLD"})
    alloc = alloc.reindex(["GLD", "SPY", "IEF"]).fillna(0.0)

    summary = pd.DataFrame(
        {
            "CAGR": [
                cagr_from_monthly(out["Ladder"]),
                cagr_from_monthly(out["GLD_BuyHold"]),
                cagr_from_monthly(out["SPY_BuyHold"]),
                cagr_from_monthly(out["IEF_BuyHold"]),
            ],
            "Sharpe(0rf)": [
                sharpe_monthly(strat_ret.loc[out.index]),
                sharpe_monthly(mret["GLD"].loc[out.index]),
                sharpe_monthly(mret["SPY"].loc[out.index]),
                sharpe_monthly(mret["IEF"].loc[out.index]),
            ],
            "MaxDD": [
                max_drawdown(out["Ladder"]),
                max_drawdown(out["GLD_BuyHold"]),
                max_drawdown(out["SPY_BuyHold"]),
                max_drawdown(out["IEF_BuyHold"]),
            ],
        },
        index=["Ladder", "GLD", "SPY", "IEF"],
    )

    return out, strat_ret, alloc, summary


if __name__ == "__main__":
    # NOTE: If you literally meant "150 trading-day MA" for GLD,
    # change gld_sma_window to ~7 (months), not 150.
    out, strat_ret, alloc, summary = build_ladder(
        start="2004-01-01",
        end=None,
        gld_sma_window=7,  # ~150 trading days ≈ ~7 months (recommended for monthly system)
        spy_lookback_months=12,
    )

    print("Allocation (% of months):")
    print((alloc * 100).round(1).astype(str) + "%")
    print("\nSummary:")
    print(summary.applymap(lambda x: f"{x:.2%}" if isinstance(x, float) else x))

    # Optional: save outputs
    out.to_csv("ladder_equity_curves.csv")
    strat_ret.to_csv("ladder_monthly_returns.csv")

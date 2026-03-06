# Export ladder backtest returns to CSV (for OOS tests)
# Ladder:
#   If GLD > SMA(gld_ma_months) at month-end t -> hold GLD in month t+1
#   Else -> run US-equities layer on proxies [SPY,VTI,QQQ,OEF,IWM]:
#           pick winner by blend_6_12 = (ret_6m + ret_12m)/2 at month-end t
#           if winner ret_12m <= 0 -> hold IEF in month t+1
#           else hold winner in month t+1
#
# Outputs:
#   1) monthly_returns_wide.csv  (date, gld_ret, spy_ret, ief_ret, ladder_ret_ma5..ma10, hold_name_ma5..ma10)
#   2) annual_returns_wide.csv   (year, gld_ret, spy_ret, ief_ret, ladder_ret_ma5..ma10)
#
# Self-contained and no-lookahead (decisions shifted by 1 month).

import numpy as np
import pandas as pd
import yfinance as yf

US_EQUITY_PROXIES = ["SPY", "VTI", "QQQ", "OEF", "IWM"]
DEFENSIVE = "IEF"


def build_monthly_inputs(start="2004-01-01", end=None):
    tickers = ["GLD", DEFENSIVE] + US_EQUITY_PROXIES
    tickers = list(dict.fromkeys(tickers))  # unique, preserve order

    px = yf.download(tickers, start=start, end=end, auto_adjust=True, progress=False)[
        "Close"
    ]

    # Month-end prices
    mpx = px.resample("M").last()
    # Month-end to month-end returns
    mret = mpx.pct_change()

    return mpx, mret


def _safe_ret(price: pd.Series, months: int) -> pd.Series:
    """Return series: P(t)/P(t-months)-1 with proper NaNs when insufficient history."""
    return price / price.shift(months) - 1.0


def us_equity_layer_recommendation(
    mpx: pd.DataFrame,
    asof: pd.Timestamp,
    universe=US_EQUITY_PROXIES,
    defensive_symbol=DEFENSIVE,
):
    """
    At month-end `asof`, score universe by blend_6_12 = (ret_6m + ret_12m)/2.
    Winner must also pass absolute filter: ret_12m > 0.
    Returns recommended symbol: winner or defensive_symbol.
    """
    # Need asof in index
    if asof not in mpx.index:
        return defensive_symbol

    # Compute 6m and 12m returns for each ticker at asof
    scores = {}
    ret12 = {}

    for tkr in universe:
        if tkr not in mpx.columns:
            continue
        series = mpx[tkr]
        r6 = _safe_ret(series, 6).loc[asof]
        r12 = _safe_ret(series, 12).loc[asof]

        if pd.isna(r6) or pd.isna(r12):
            continue

        scores[tkr] = 0.5 * float(r6) + 0.5 * float(r12)
        ret12[tkr] = float(r12)

    # If nothing is scoreable yet, go defensive
    if not scores:
        return defensive_symbol

    winner = max(scores, key=scores.get)

    # Absolute filter: winner must have 12m return > 0
    if ret12[winner] <= 0:
        return defensive_symbol

    return winner


def ladder_monthly_returns(
    mpx: pd.DataFrame,
    mret: pd.DataFrame,
    gld_ma_months: int,
):
    """
    Returns:
      strat_ret (monthly), hold_name (symbol held during that month)
    """
    # GLD SMA gate computed at month-end
    gld_sma = mpx["GLD"].rolling(window=gld_ma_months, min_periods=gld_ma_months).mean()
    gld_on = mpx["GLD"] > gld_sma

    # Choice at month-end t (symbol to hold next month after shift)
    choice = pd.Series(index=mpx.index, dtype=object)
    for asof in mpx.index:
        if pd.notna(gld_on.loc[asof]) and bool(gld_on.loc[asof]):
            choice.loc[asof] = "GLD"
        else:
            choice.loc[asof] = us_equity_layer_recommendation(
                mpx=mpx,
                asof=asof,
                universe=US_EQUITY_PROXIES,
                defensive_symbol=DEFENSIVE,
            )

    # --- THIS SHIFT PREVENTS LOOKAHEAD ---
    # Decision made at end of month t is applied to returns during month t+1.
    hold_name = choice.shift(1)

    # Compute strategy returns by held symbol
    strat_ret = pd.Series(index=mpx.index, dtype="float64")
    for sym in ["GLD", DEFENSIVE] + US_EQUITY_PROXIES:
        if sym in mret.columns:
            mask = hold_name == sym
            strat_ret.loc[mask] = mret[sym].loc[mask]

    # Keep only months where we actually have a held symbol AND returns exist
    df = pd.concat(
        [
            strat_ret.rename("strat_ret"),
            hold_name.rename("hold_name"),
            mret["GLD"].rename("gld_ret"),
            mret.get("SPY", pd.Series(index=mret.index, dtype=float)).rename("spy_ret"),
            mret[DEFENSIVE].rename("ief_ret"),
        ],
        axis=1,
    ).dropna(subset=["strat_ret", "hold_name", "gld_ret", "ief_ret"])

    return df["strat_ret"].astype(float), df["hold_name"].astype(str)


def export_ladder_sweep_csv(
    start="2004-01-01",
    end=None,
    gld_ma_windows=range(5, 11),  # 5..10 months
    out_monthly_csv="monthly_returns_wide.csv",
    out_annual_csv="annual_returns_wide.csv",
):
    mpx, mret = build_monthly_inputs(start=start, end=end)

    monthly = pd.DataFrame(index=mret.index)
    monthly["gld_ret"] = mret["GLD"]
    monthly["spy_ret"] = mret["SPY"] if "SPY" in mret.columns else np.nan
    monthly["ief_ret"] = mret[DEFENSIVE]

    # Add ladder returns + hold names for each MA window
    for w in gld_ma_windows:
        strat_ret, hold_name = ladder_monthly_returns(mpx, mret, gld_ma_months=int(w))
        monthly[f"ladder_ret_ma{w}"] = strat_ret
        monthly[f"hold_name_ma{w}"] = hold_name

    # Keep only months where at least one ladder series exists (after warmups)
    ladder_cols = [c for c in monthly.columns if c.startswith("ladder_ret_")]
    monthly = monthly.dropna(subset=ladder_cols, how="all").copy()

    # Export monthly
    monthly_out = monthly.reset_index()
    monthly_out = monthly_out.rename(columns={monthly_out.columns[0]: "date"})
    monthly_out["date"] = pd.to_datetime(monthly_out["date"]).dt.strftime("%Y-%m-%d")
    monthly_out.to_csv(out_monthly_csv, index=False)

    # Annual: compound ONLY return columns
    def compound_to_annual(x: pd.Series) -> float:
        x = x.dropna()
        if len(x) == 0:
            return np.nan
        return float((1 + x).prod() - 1)

    annual = monthly.copy()
    annual["year"] = annual.index.year

    ret_cols = [
        c for c in annual.columns if c.endswith("_ret") or c.startswith("ladder_ret_")
    ]
    annual_out = annual.groupby("year")[ret_cols].agg(compound_to_annual).reset_index()
    annual_out.to_csv(out_annual_csv, index=False)

    print(f"Wrote: {out_monthly_csv}")
    print(f"Wrote: {out_annual_csv}")


if __name__ == "__main__":
    export_ladder_sweep_csv(
        start="2004-01-01",
        end=None,
        gld_ma_windows=range(5, 11),
        out_monthly_csv="monthly_returns_wide.csv",
        out_annual_csv="annual_returns_wide.csv",
    )

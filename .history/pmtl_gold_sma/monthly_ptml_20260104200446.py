# Export ladder backtest returns to CSV (for OOS tests)
# - Sweeps GLD MA windows (5..10 months by default)
# - Outputs:
#   1) monthly_returns_wide.csv  (date, gld_ret, spy_ret, ief_ret, ladder_ret_ma5..ma10, hold_ma5..ma10)
#   2) annual_returns_wide.csv   (year, gld_ret, spy_ret, ief_ret, ladder_ret_ma5..ma10)
#
# Assumes you already have a price download method (yfinance) and monthly ladder logic.
# This version is self-contained.

import numpy as np
import pandas as pd
import yfinance as yf


def build_monthly_inputs(start="2004-01-01", end=None):
    tickers = ["GLD", "SPY", "IEF"]
    px = yf.download(tickers, start=start, end=end, auto_adjust=True, progress=False)[
        "Close"
    ]
    mpx = px.resample("M").last().dropna()
    mret = mpx.pct_change()
    return mpx, mret


def ladder_monthly_returns(
    mpx: pd.DataFrame,
    mret: pd.DataFrame,
    gld_ma_months: int,
    spy_lookback_months: int = 12,
):
    """
    Monthly no-lookahead ladder:
      if GLD > SMA(gld_ma_months) at month-end t => hold GLD in t+1
      else if SPY 12m momentum > 0 at month-end t => hold SPY in t+1
      else hold IEF in t+1
    Returns:
      strat_ret (monthly), hold (0=IEF,1=SPY,2=GLD), signals (optional)
    """
    # Signals at month-end
    gld_sma = mpx["GLD"].rolling(window=gld_ma_months, min_periods=gld_ma_months).mean()
    gld_on = mpx["GLD"] > gld_sma

    spy_mom = mpx["SPY"] / mpx["SPY"].shift(spy_lookback_months) - 1.0
    spy_on = spy_mom > 0

    # Choice at month-end t (0=IEF,1=SPY,2=GLD)
    choice = pd.Series(0, index=mpx.index, dtype="int64")
    choice.loc[spy_on] = 1
    choice.loc[gld_on] = 2

    # Hold during month t (decided at t-1)
    hold = choice.shift(1)

    # Strategy returns aligned by index (no .values)
    strat_ret = pd.Series(index=mpx.index, dtype="float64")
    strat_ret.loc[hold == 2] = mret["GLD"].loc[hold == 2]
    strat_ret.loc[hold == 1] = mret["SPY"].loc[hold == 1]
    strat_ret.loc[hold == 0] = mret["IEF"].loc[hold == 0]

    # Drop warmup months where SMA or momentum are not available
    # (This also removes the initial NaN return month.)
    valid = (
        (~strat_ret.isna())
        & (~mret["GLD"].isna())
        & (~mret["SPY"].isna())
        & (~mret["IEF"].isna())
    )
    strat_ret = strat_ret.loc[valid].astype(float)
    hold = hold.loc[valid].astype("int64")

    return strat_ret, hold


def export_ladder_sweep_csv(
    start="2004-01-01",
    end=None,
    gld_ma_windows=range(5, 11),  # 5..10 months
    spy_lookback_months=12,
    out_monthly_csv="monthly_returns_wide.csv",
    out_annual_csv="annual_returns_wide.csv",
):
    mpx, mret = build_monthly_inputs(start=start, end=end)

    # Base monthly returns for benchmarks
    monthly = pd.DataFrame(index=mret.index)
    monthly["gld_ret"] = mret["GLD"]
    monthly["spy_ret"] = mret["SPY"]
    monthly["ief_ret"] = mret["IEF"]

    # Add ladder returns + holds for each MA window
    for w in gld_ma_windows:
        strat_ret, hold = ladder_monthly_returns(
            mpx, mret, gld_ma_months=int(w), spy_lookback_months=spy_lookback_months
        )
        # Align into the same frame
        monthly[f"ladder_ret_ma{w}"] = strat_ret
        monthly[f"hold_ma{w}"] = hold  # 0=IEF,1=SPY,2=GLD

    # Keep only months where at least one ladder series exists (after warmups)
    ladder_cols = [c for c in monthly.columns if c.startswith("ladder_ret_")]
    monthly = monthly.dropna(subset=ladder_cols, how="all").copy()

    # Export monthly
    monthly_out = monthly.reset_index()
    monthly_out = monthly_out.rename(columns={monthly_out.columns[0]: "date"})
    monthly_out["date"] = pd.to_datetime(monthly_out["date"]).dt.strftime("%Y-%m-%d")
    monthly_out.to_csv(out_monthly_csv, index=False)

    # Annual (compound monthly to yearly)
    def compound_to_annual(x: pd.Series) -> float:
        x = x.dropna()
        if len(x) == 0:
            return np.nan
        return float((1 + x).prod() - 1)

    annual = monthly.copy()
    annual["year"] = annual.index.year

    annual_out = (
        annual.groupby("year")
        .agg({col: compound_to_annual for col in annual.columns if col != "year"})
        .reset_index()
    )
    annual_out.to_csv(out_annual_csv, index=False)

    print(f"Wrote: {out_monthly_csv}")
    print(f"Wrote: {out_annual_csv}")


if __name__ == "__main__":
    export_ladder_sweep_csv(
        start="2004-01-01",
        end=None,
        gld_ma_windows=range(5, 11),
        spy_lookback_months=12,
        out_monthly_csv="monthly_returns_wide.csv",
        out_annual_csv="annual_returns_wide.csv",
    )

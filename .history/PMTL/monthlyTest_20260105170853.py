import numpy as np
import pandas as pd
import yfinance as yf
from typing import Callable, Dict, Any, Optional, Tuple


# ----------------------------
# Metrics
# ----------------------------
def max_drawdown(equity: pd.Series) -> float:
    peak = equity.cummax()
    dd = equity / peak - 1.0
    return float(dd.min())


def cagr_from_equity(equity: pd.Series) -> float:
    equity = equity.dropna()
    if len(equity) < 2:
        return np.nan
    years = (equity.index[-1] - equity.index[0]).days / 365.25
    return float((equity.iloc[-1] / equity.iloc[0]) ** (1 / years) - 1)


def sharpe_annualized_daily(rets: pd.Series) -> float:
    r = rets.dropna()
    if len(r) < 252:
        return np.nan
    sd = r.std(ddof=1)
    if sd == 0:
        return np.nan
    return float((r.mean() / sd) * np.sqrt(252))


# ----------------------------
# Data (daily)
# ----------------------------
def load_daily_close(symbol: str, start="2004-01-01", end=None) -> pd.Series:
    px = yf.download(symbol, start=start, end=end, auto_adjust=True, progress=False)[
        "Close"
    ]
    close = px.dropna()
    close.name = symbol
    return close


def load_daily_flat_returns(
    flat_symbol: str, index_like: pd.DatetimeIndex, start="2004-01-01", end=None
) -> pd.Series:
    px = yf.download(
        flat_symbol, start=start, end=end, auto_adjust=True, progress=False
    )["Close"]
    close = px.dropna().reindex(index_like).ffill()
    return close.pct_change().fillna(0.0)


# ----------------------------
# Modular engine: plug in entry/exit functions
# ----------------------------
# entry_fn / exit_fn return boolean Series indexed like close:
# - evaluated using info up to bar t
# - engine applies position on bar t+1 (no lookahead)
EntryExitFn = Callable[[pd.Series, Dict[str, Any]], pd.Series]


def run_long_only_engine(
    close,
    entry_fn,
    exit_fn,
    params,
    *,
    name="STRAT",
    flat_ret=None,  # None => cash 0% when flat
):
    # ---- force close into a 1D Series ----
    if isinstance(close, pd.DataFrame):
        if close.shape[1] != 1:
            raise ValueError(
                "close must be a Series (one symbol). You passed a DataFrame with multiple columns."
            )
        close = close.iloc[:, 0]
    close = close.dropna()

    # remove duplicate timestamps (can happen with some data sources)
    close = close[~close.index.duplicated(keep="last")]

    ret = close.pct_change().fillna(0.0)

    if flat_ret is None:
        flat_ret = pd.Series(0.0, index=close.index)
    else:
        if isinstance(flat_ret, pd.DataFrame):
            if flat_ret.shape[1] != 1:
                raise ValueError("flat_ret must be a Series or 1-column DataFrame.")
            flat_ret = flat_ret.iloc[:, 0]
        flat_ret = flat_ret.reindex(close.index).fillna(0.0)

    entry = entry_fn(close, params)
    exit_ = exit_fn(close, params)

    # force entry/exit to 1D boolean Series
    if isinstance(entry, pd.DataFrame):
        if entry.shape[1] != 1:
            raise ValueError(
                "entry_fn returned a multi-column DataFrame; return a Series instead."
            )
        entry = entry.iloc[:, 0]
    if isinstance(exit_, pd.DataFrame):
        if exit_.shape[1] != 1:
            raise ValueError(
                "exit_fn returned a multi-column DataFrame; return a Series instead."
            )
        exit_ = exit_.iloc[:, 0]

    entry = entry.reindex(close.index).fillna(False).astype(bool)
    exit_ = exit_.reindex(close.index).fillna(False).astype(bool)

    # ---- use numpy arrays so values are always scalars ----
    entry_np = entry.to_numpy(dtype=bool)
    exit_np = exit_.to_numpy(dtype=bool)

    pos = np.zeros(len(close), dtype=np.int8)
    in_pos = 0
    for i in range(len(close)):
        if in_pos == 0:
            if entry_np[i]:
                in_pos = 1
        else:
            if exit_np[i]:
                in_pos = 0
        pos[i] = in_pos

    pos = pd.Series(pos, index=close.index, dtype="int64")
    hold = pos.shift(1).fillna(0).astype(int)

    strat_ret = (hold * ret) + ((1 - hold) * flat_ret)
    strat_ret = strat_ret.fillna(0.0).rename(name + "_ret")
    equity = (1 + strat_ret).cumprod().rename(name)

    summary = pd.DataFrame(
        {
            "CAGR": [cagr_from_equity(equity)],
            "Sharpe(0rf)": [sharpe_annualized_daily(strat_ret)],
            "MaxDD": [max_drawdown(equity)],
            "Long_alloc": [float(hold.mean())],
        },
        index=[name],
    )

    out = pd.concat([equity, close.rename("Close"), hold.rename("Hold")], axis=1)
    return out, strat_ret, summary


# ----------------------------
# Plug-ins (entry/exit examples)
# ----------------------------
# 1) Donchian asymmetric breakout (your main one)
def donchian_entry(close: pd.Series, p: Dict[str, Any]) -> pd.Series:
    L_entry = int(p["L_entry"])
    high = (
        close.shift(1).rolling(L_entry, min_periods=L_entry).max()
    )  # yesterday's channel high
    return close > high


def donchian_exit(close: pd.Series, p: Dict[str, Any]) -> pd.Series:
    L_exit = int(p["L_exit"])
    low = (
        close.shift(1).rolling(L_exit, min_periods=L_exit).min()
    )  # yesterday's channel low
    return close < low


# 2) Example swap: simple trend filter (price > SMA)
def sma_entry(close: pd.Series, p: Dict[str, Any]) -> pd.Series:
    w = int(p["w"])
    sma = close.rolling(w, min_periods=w).mean()
    return close > sma


def sma_exit(close: pd.Series, p: Dict[str, Any]) -> pd.Series:
    w = int(p["w"])
    sma = close.rolling(w, min_periods=w).mean()
    return close < sma


# ----------------------------
# Sweep helper
# ----------------------------
def sweep_params(
    close: pd.Series,
    entry_fn: EntryExitFn,
    exit_fn: EntryExitFn,
    grid: Dict[str, list],
    *,
    name_prefix="STRAT",
    flat_ret: Optional[pd.Series] = None,
):
    keys = list(grid.keys())
    rows = []
    details = {}

    def rec(i, cur):
        if i == len(keys):
            params = dict(cur)
            tag = name_prefix + "_" + "_".join(f"{k}{params[k]}" for k in keys)
            out, r, s = run_long_only_engine(
                close, entry_fn, exit_fn, params, name=tag, flat_ret=flat_ret
            )
            row = s.iloc[0].copy()
            for k in keys:
                row[k] = params[k]
            rows.append(row)
            details[tuple(params[k] for k in keys)] = {
                "out": out,
                "rets": r,
                "summary": s,
            }
            return

        k = keys[i]
        for v in grid[k]:
            cur[k] = v
            rec(i + 1, cur)

    rec(0, {})
    sweep_df = pd.DataFrame(rows).set_index(keys).sort_index()
    return sweep_df, details


# ----------------------------
# Example run (DAILY GLD)
# ----------------------------
if __name__ == "__main__":
    close = load_daily_close("GLD", start="2004-01-01")

    # optional: when flat, earn bond proxy instead of cash:
    # flat_ret = load_daily_flat_returns("IEF", close.index, start="2004-01-01")
    flat_ret = None

    # Single test (recommended first)
    out, rets, summary = run_long_only_engine(
        close,
        donchian_entry,
        donchian_exit,
        params={"L_entry": 200, "L_exit": 100},
        name="GLD_DONCH_200_100",
        flat_ret=flat_ret,
    )
    print(summary)
    gld_ret = close.pct_change().fillna(0.0)
    gld_eq = (1 + gld_ret).cumprod().rename("GLD_BH")

    bh_summary = pd.DataFrame(
        {
            "CAGR": [cagr_from_equity(gld_eq)],
            "Sharpe(0rf)": [sharpe_annualized_daily(gld_ret)],
            "MaxDD": [max_drawdown(gld_eq)],
            "Long_alloc": [1.0],
        },
        index=["GLD_BH"],
    )

    print(pd.concat([summary, bh_summary], axis=0))

    # Sweep your candidate pairs
    grid = {"L_entry": [200, 252, 180, 120], "L_exit": [100, 126, 90, 60]}
    sweep, details = sweep_params(
        close,
        donchian_entry,
        donchian_exit,
        grid,
        name_prefix="GLD_DONCH",
        flat_ret=flat_ret,
    )

    sp = sweep.copy()
    sp["CAGR"] = sp["CAGR"].map(lambda x: f"{x:.2%}")
    sp["MaxDD"] = sp["MaxDD"].map(lambda x: f"{x:.2%}")
    sp["Sharpe(0rf)"] = sp["Sharpe(0rf)"].map(lambda x: f"{x:.2f}")
    sp["Long_alloc"] = sp["Long_alloc"].map(lambda x: f"{x:.1%}")
    print(sp)

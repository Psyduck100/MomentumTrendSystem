import numpy as np
import pandas as pd
import yfinance as yf
from typing import Callable, Dict, Any, Optional


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


def cagr_in_position(underlying_ret: pd.Series, hold: pd.Series) -> float:
    """Annualized CAGR on ONLY the days you're actually long (hold==1)."""
    r = underlying_ret.reindex(hold.index).fillna(0.0)
    h = hold.reindex(r.index).fillna(0).astype(int)

    rin = r[h == 1]
    if len(rin) < 2:
        return np.nan

    growth = float((1.0 + rin).prod())
    return float(growth ** (252.0 / len(rin)) - 1.0)


def sharpe_in_position(underlying_ret: pd.Series, hold: pd.Series) -> float:
    """Annualized Sharpe on ONLY the days you're actually long (hold==1)."""
    r = underlying_ret.reindex(hold.index).fillna(0.0)
    h = hold.reindex(r.index).fillna(0).astype(int)

    rin = r[h == 1]
    if len(rin) < 60:
        return np.nan
    sd = rin.std(ddof=1)
    if sd == 0:
        return np.nan
    return float((rin.mean() / sd) * np.sqrt(252))


def _ensure_series(x, preferred_name: str | None = None) -> pd.Series:
    """Force x into a 1D Series. If DataFrame with multiple cols, try preferred_name else take first col."""
    if isinstance(x, pd.Series):
        s = x
    elif isinstance(x, pd.DataFrame):
        if x.shape[1] == 1:
            s = x.iloc[:, 0]
        else:
            if preferred_name is not None and preferred_name in x.columns:
                s = x[preferred_name]
            else:
                s = x.iloc[:, 0]
    else:
        raise TypeError("Expected Series or DataFrame.")
    s = s.dropna()
    s = s[~s.index.duplicated(keep="last")]
    return s


# ----------------------------
# Data (daily)
# ----------------------------
def load_daily_close(symbol: str, start="2004-01-01", end=None) -> pd.Series:
    px = yf.download(symbol, start=start, end=end, auto_adjust=True, progress=False)["Close"]
    close = _ensure_series(px, preferred_name=symbol)
    close.name = symbol
    return close


def load_daily_flat_returns(
    flat_symbol: str, index_like: pd.DatetimeIndex, start="2004-01-01", end=None
) -> pd.Series:
    px = yf.download(flat_symbol, start=start, end=end, auto_adjust=True, progress=False)["Close"]
    close = _ensure_series(px, preferred_name=flat_symbol).reindex(index_like).ffill()
    return close.pct_change().fillna(0.0)


# ----------------------------
# Modular engine: plug in entry/exit functions
# ----------------------------
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
    close = _ensure_series(close)

    # underlying daily returns (GLD returns)
    ret = close.pct_change().fillna(0.0)

    if flat_ret is None:
        flat_ret = pd.Series(0.0, index=close.index)
    else:
        flat_ret = _ensure_series(flat_ret).reindex(close.index).fillna(0.0)

    entry = entry_fn(close, params)
    exit_ = exit_fn(close, params)

    entry = _ensure_series(entry).reindex(close.index).fillna(False).astype(bool)
    exit_ = _ensure_series(exit_).reindex(close.index).fillna(False).astype(bool)

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
    hold = pos.shift(1).fillna(0).astype(int)  # <-- no lookahead

    # full-period strategy returns (includes flat_ret when not in position)
    strat_ret = (hold * ret) + ((1 - hold) * flat_ret)
    strat_ret = strat_ret.fillna(0.0).rename(name + "_ret")
    equity = (1 + strat_ret).cumprod()
    equity.name = name

    summary = pd.DataFrame(
        {
            # full-period metrics (depends on flat_ret)
            "CAGR": [cagr_from_equity(equity)],
            "Sharpe(0rf)": [sharpe_annualized_daily(strat_ret)],
            "MaxDD": [max_drawdown(equity)],
            "Long_alloc": [float(hold.mean())],
            # ONLY-when-in-gold metrics (independent of flat_ret)
            "CAGR_in_pos": [cagr_in_position(ret, hold)],
            "Sharpe_in_pos": [sharpe_in_position(ret, hold)],
        },
        index=[name],
    )

    out = pd.concat([equity, close.rename("Close"), hold.rename("Hold")], axis=1)
    return out, strat_ret, hold, summary


# ----------------------------
# Plug-ins (entry/exit examples)
# ----------------------------
def donchian_entry(close: pd.Series, p: Dict[str, Any]) -> pd.Series:
    L_entry = int(p["L_entry"])
    high = close.shift(1).rolling(L_entry, min_periods=L_entry).max()  # yesterday's high channel
    return (close > high).astype(bool)


def donchian_exit(close: pd.Series, p: Dict[str, Any]) -> pd.Series:
    L_exit = int(p["L_exit"])
    low = close.shift(1).rolling(L_exit, min_periods=L_exit).min()  # yesterday's low channel
    return (close < low).astype(bool)


def sma_entry(close: pd.Series, p: Dict[str, Any]) -> pd.Series:
    w = int(p["w"])
    sma = close.rolling(w, min_periods=w).mean()
    return (close > sma).astype(bool)


def sma_exit(close: pd.Series, p: Dict[str, Any]) -> pd.Series:
    w = int(p["w"])
    sma = close.rolling(w, min_periods=w).mean()
    return (close < sma).astype(bool)


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
    close = _ensure_series(close)

    keys = list(grid.keys())
    rows = []
    details = {}

    def rec(i, cur):
        if i == len(keys):
            params = dict(cur)
            tag = name_prefix + "_" + "_".join(f"{k}{params[k]}" for k in keys)
            out, r, hold, s = run_long_only_engine(
                close, entry_fn, exit_fn, params, name=tag, flat_ret=flat_ret
            )
            row = s.iloc[0].copy()
            for k in keys:
                row[k] = params[k]
            rows.append(row)
            details[tuple(params[k] for k in keys)] = {
                "out": out,
                "rets": r,
                "hold": hold,
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
    # flat_ret = load_daily_flat_returns("IEF", close.index, start="2004-01-01")
    flat_ret = None

    # Strategy (recommended first)
    out, rets, hold, summary = run_long_only_engine(
        close,
        donchian_entry,
        donchian_exit,
        params={"L_entry": 200, "L_exit": 100},
        name="GLD_DONCH_200_100",
        flat_ret=flat_ret,
    )

    # GLD buy & hold (plus in-pos columns for comparison)
    gld_ret = close.pct_change().fillna(0.0)
    gld_eq = (1.0 + gld_ret).cumprod()
    gld_eq.name = "GLD_BH"

    bh_summary = pd.DataFrame(
        {
            "CAGR": [cagr_from_equity(gld_eq)],
            "Sharpe(0rf)": [sharpe_annualized_daily(gld_ret)],
            "MaxDD": [max_drawdown(gld_eq)],
            "Long_alloc": [1.0],
            "CAGR_in_pos": [cagr_in_position(gld_ret, pd.Series(1, index=gld_ret.index))],
            "Sharpe_in_pos": [sharpe_in_position(gld_ret, pd.Series(1, index=gld_ret.index))],
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

    # Print both full-period and "in-position" rankings
    sp = sweep.copy()
    sp["CAGR"] = sp["CAGR"].map(lambda x: f"{x:.2%}")
    sp["MaxDD"] = sp["MaxDD"].map(lambda x: f"{x:.2%}")
    sp["Sharpe(0rf)"] = sp["Sharpe(0rf)"].map(lambda x: f"{x:.2f}")
    sp["Long_alloc"] = sp["Long_alloc"].map(lambda x: f"{x:.1%}")
    sp["CAGR_in_pos"] = sp["CAGR_in_pos"].map(lambda x: f"{x:.2%}")
    sp["Sharpe_in_pos"] = sp["Sharpe_in_pos"].map(lambda x: f"{x:.2f}")
    print(sp)

    # If you want: show best by "when I'm in gold" CAGR
    best = sweep.sort_values("CAGR_in_pos", ascending=False).head(10)
    print("\nTop 10 by CAGR_in_pos (when long GLD):")
    print(best[["CAGR_in_pos", "Sharpe_in_pos", "Long_alloc", "CAGR", "MaxDD"]])

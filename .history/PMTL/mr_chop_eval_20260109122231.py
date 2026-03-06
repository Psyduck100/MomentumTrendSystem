from __future__ import annotations

from typing import Dict, Any, Iterable, Tuple, List
import numpy as np
import pandas as pd


# ============================================================
# Helpers
# ============================================================


def as_series(x: pd.Series | pd.DataFrame, name: str = "x") -> pd.Series:
    if isinstance(x, pd.Series):
        return x.sort_index()
    if isinstance(x, pd.DataFrame):
        if x.shape[1] != 1:
            raise ValueError(f"Expected 1-column DataFrame for {name}, got {x.shape}")
        s = x.iloc[:, 0]
        s.name = s.name or name
        return s.sort_index()
    raise TypeError(f"Expected Series/DataFrame for {name}, got {type(x)}")


def years_in_index(idx: pd.Index, periods_per_year: int = 252) -> float:
    return max(1e-9, len(idx) / periods_per_year)


def max_drawdown(returns: pd.Series) -> float:
    equity = (1.0 + returns).cumprod()
    peak = equity.cummax()
    dd = equity / peak - 1.0
    return float(dd.min()) if len(dd) else np.nan


def perf_metrics(returns: pd.Series, *, periods_per_year: int = 252) -> Dict[str, float]:
    returns = returns.fillna(0.0)
    total = float((1.0 + returns).prod() - 1.0) if len(returns) else np.nan
    yrs = years_in_index(returns.index, periods_per_year=periods_per_year)
    ann = float((1.0 + total) ** (1.0 / yrs) - 1.0) if yrs > 0 else np.nan
    vol = float(returns.std(ddof=0) * np.sqrt(periods_per_year))
    sharpe = float(returns.mean() / returns.std(ddof=0) * np.sqrt(periods_per_year)) if returns.std(ddof=0) > 0 else np.nan
    mdd = max_drawdown(returns)
    return {
        "total_return": total,
        "ann_return": ann,
        "max_dd": mdd,
        "vol": vol,
        "sharpe": sharpe,
    }


def ann_return_over_mask(
    returns: pd.Series, mask: pd.Series, *, periods_per_year: int = 252
) -> float:
    returns = returns.fillna(0.0)
    mask = mask.reindex(returns.index).fillna(False).astype(bool)
    if not mask.any():
        return np.nan
    total = float((1.0 + returns[mask]).prod() - 1.0)
    years = float(mask.sum() / periods_per_year)
    return float((1.0 + total) ** (1.0 / years) - 1.0) if years > 0 else np.nan


# ============================================================
# MR Strategy (Chop-only)
# ============================================================


def mr_signals_and_position(
    close: pd.Series,
    regime_chop: pd.Series,
    *,
    lookback: int,
    z_enter: float,
    z_exit: float,
    max_hold_days: int,
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    close = as_series(close, "Close")
    chop = regime_chop.reindex(close.index).ffill().fillna(False).astype(bool)

    ma = close.rolling(lookback, min_periods=lookback).mean()
    sd = close.rolling(lookback, min_periods=lookback).std()
    z = (close - ma) / sd
    valid = (~ma.isna()) & (~sd.isna())

    pos = pd.Series(0, index=close.index, dtype=int)
    in_pos = False
    hold = 0

    for i in range(len(close)):
        pos.iat[i] = 1 if in_pos else 0

        if not bool(valid.iat[i]):
            in_pos = False
            hold = 0
            continue

        if not bool(chop.iat[i]):
            in_pos = False
            hold = 0
            continue

        if in_pos:
            hold += 1
            exit_cond = (z.iat[i] > -z_exit) or (hold >= max_hold_days)
            if exit_cond:
                in_pos = False
                hold = 0
        else:
            if z.iat[i] < -z_enter:
                in_pos = True
                hold = 0

    return pos, z, valid


def apply_costs(
    pos: pd.Series,
    returns: pd.Series,
    *,
    cost_bps_roundtrip: float,
) -> pd.Series:
    pos = pos.astype(float)
    returns = returns.fillna(0.0)
    strategy_ret = pos.shift(1).fillna(0.0) * returns

    if cost_bps_roundtrip and cost_bps_roundtrip > 0:
        cost = float(cost_bps_roundtrip) / 10000.0 / 2.0
        pos_diff = pos.diff().fillna(0.0)
        strategy_ret = strategy_ret - cost * (pos_diff == 1.0) - cost * (pos_diff == -1.0)

    return strategy_ret


def trade_list(
    pos: pd.Series,
    strategy_ret: pd.Series,
) -> List[Dict[str, Any]]:
    pos = pos.astype(int)
    pos_diff = pos.diff().fillna(0)
    entries = list(pos.index[pos_diff == 1])
    exits = list(pos.index[pos_diff == -1])

    trades: List[Dict[str, Any]] = []
    exit_idx = 0

    for entry in entries:
        while exit_idx < len(exits) and exits[exit_idx] <= entry:
            exit_idx += 1
        if exit_idx < len(exits):
            exit_day = exits[exit_idx]
            exit_idx += 1
        else:
            exit_day = pos.index[-1]

        trade_slice = strategy_ret.loc[entry:exit_day]
        trade_ret = float((1.0 + trade_slice).prod() - 1.0) if len(trade_slice) else np.nan
        hold_days = int(pos.loc[entry:exit_day].sum())

        trades.append(
            {
                "entry": entry,
                "exit": exit_day,
                "return": trade_ret,
                "hold_days": hold_days,
            }
        )

    return trades


def trade_stats(trades: List[Dict[str, Any]]) -> Dict[str, float]:
    if not trades:
        return {
            "trades": 0.0,
            "win_rate": np.nan,
            "avg_trade": np.nan,
            "median_trade": np.nan,
            "worst_trade": np.nan,
            "trade_p05": np.nan,
            "avg_hold": np.nan,
            "median_hold": np.nan,
        }

    rets = np.array([t["return"] for t in trades], dtype=float)
    holds = np.array([t["hold_days"] for t in trades], dtype=float)

    return {
        "trades": float(len(trades)),
        "win_rate": float(np.mean(rets > 0.0)),
        "avg_trade": float(np.mean(rets)),
        "median_trade": float(np.median(rets)),
        "worst_trade": float(np.min(rets)),
        "trade_p05": float(np.percentile(rets, 5)),
        "avg_hold": float(np.mean(holds)),
        "median_hold": float(np.median(holds)),
    }


def evaluate_mr_params(
    close: pd.Series,
    regime_chop: pd.Series,
    *,
    lookback: int,
    z_enter: float,
    z_exit: float,
    max_hold_days: int,
    cost_bps_roundtrip: float,
    periods_per_year: int = 252,
) -> Dict[str, Any]:
    close = as_series(close, "Close")
    chop = regime_chop.reindex(close.index).ffill().fillna(False).astype(bool)

    pos, z, valid = mr_signals_and_position(
        close,
        chop,
        lookback=lookback,
        z_enter=z_enter,
        z_exit=z_exit,
        max_hold_days=max_hold_days,
    )

    returns = close.pct_change().fillna(0.0)
    strategy_ret = apply_costs(pos, returns, cost_bps_roundtrip=cost_bps_roundtrip)

    pos_diff = pos.diff().fillna(0)
    trade_active = (pos == 1) | (pos_diff != 0)
    eval_mask = chop | trade_active

    chop_ret = strategy_ret.where(eval_mask, 0.0)
    full_metrics = perf_metrics(strategy_ret, periods_per_year=periods_per_year)
    chop_metrics = perf_metrics(chop_ret, periods_per_year=periods_per_year)

    trades = trade_list(pos, strategy_ret)
    tstats = trade_stats(trades)

    occupancy_in_chop = float(pos[chop].mean()) if chop.any() else np.nan
    occupancy_total = float(pos.mean())
    entries = int((pos_diff == 1).sum())
    entries_per_year = float(entries / years_in_index(close.index, periods_per_year=periods_per_year))
    chop_days = int(chop.sum())
    entries_per_chop_year = float(entries / (chop_days / periods_per_year)) if chop_days > 0 else np.nan

    inpos_mask = pos.shift(1).fillna(0).astype(bool)
    ann_over_chop_time = ann_return_over_mask(strategy_ret, chop, periods_per_year=periods_per_year)
    ann_over_inpos_time = ann_return_over_mask(strategy_ret, inpos_mask, periods_per_year=periods_per_year)

    return {
        "lookback": int(lookback),
        "z_enter": float(z_enter),
        "z_exit": float(z_exit),
        "max_hold_days": int(max_hold_days),
        "cost_bps_roundtrip": float(cost_bps_roundtrip),
        "occupancy_in_chop": occupancy_in_chop,
        "occupancy_total": occupancy_total,
        "entries_per_year": entries_per_year,
        "entries_per_chop_year": entries_per_chop_year,
        "chop_total_return": chop_metrics["total_return"],
        "chop_ann_return": chop_metrics["ann_return"],
        "chop_max_dd": chop_metrics["max_dd"],
        "chop_vol": chop_metrics["vol"],
        "chop_sharpe": chop_metrics["sharpe"],
        "ann_over_chop_time": ann_over_chop_time,
        "ann_over_inpos_time": ann_over_inpos_time,
        "full_total_return": full_metrics["total_return"],
        "full_ann_return": full_metrics["ann_return"],
        "full_max_dd": full_metrics["max_dd"],
        "full_vol": full_metrics["vol"],
        "full_sharpe": full_metrics["sharpe"],
        **tstats,
    }


def baseline_metrics(
    close: pd.Series,
    regime_chop: pd.Series,
    *,
    periods_per_year: int = 252,
) -> Dict[str, Dict[str, float]]:
    close = as_series(close, "Close")
    chop = regime_chop.reindex(close.index).ffill().fillna(False).astype(bool)
    returns = close.pct_change().fillna(0.0)
    chop_ret = returns.where(chop, 0.0)

    return {
        "cash": perf_metrics(chop_ret * 0.0, periods_per_year=periods_per_year),
        "hold_gld_in_chop": perf_metrics(chop_ret, periods_per_year=periods_per_year),
    }


def sweep_params(
    close: pd.Series,
    regime_chop: pd.Series,
    *,
    lookbacks: Iterable[int],
    z_enters: Iterable[float],
    z_exits: Iterable[float],
    max_hold_days_list: Iterable[int],
    cost_bps_roundtrip: float,
    periods_per_year: int = 252,
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for L in lookbacks:
        for ze in z_enters:
            for zx in z_exits:
                for mh in max_hold_days_list:
                    rows.append(
                        evaluate_mr_params(
                            close,
                            regime_chop,
                            lookback=L,
                            z_enter=ze,
                            z_exit=zx,
                            max_hold_days=mh,
                            cost_bps_roundtrip=cost_bps_roundtrip,
                            periods_per_year=periods_per_year,
                        )
                    )
    return pd.DataFrame(rows)


def add_neighbor_score(
    df: pd.DataFrame,
    *,
    metric: str,
    L_step: int = 20,
    z_enter_step: float = 0.5,
    z_exit_step: float = 0.5,
    max_hold_step: int = 5,
) -> pd.DataFrame:
    df = df.copy()
    scores = []
    for _, row in df.iterrows():
        mask = (
            (df["lookback"].sub(row["lookback"]).abs() <= L_step)
            & (df["z_enter"].sub(row["z_enter"]).abs() <= z_enter_step)
            & (df["z_exit"].sub(row["z_exit"]).abs() <= z_exit_step)
            & (df["max_hold_days"].sub(row["max_hold_days"]).abs() <= max_hold_step)
        )
        scores.append(float(df.loc[mask, metric].mean()))
    df["neighbor_mean"] = scores
    return df


def pick_stable_candidates(
    df: pd.DataFrame,
    *,
    metric: str,
    min_trades: int = 20,
    top_frac: float = 0.95,
) -> pd.DataFrame:
    df = df.copy()
    df = df[df["trades"] >= min_trades]
    df = df[np.isfinite(df[metric]) & np.isfinite(df["neighbor_mean"])]
    if df.empty:
        return df
    best = df["neighbor_mean"].max()
    return df[df["neighbor_mean"] >= best * top_frac].sort_values(
        ["neighbor_mean", metric], ascending=[False, False]
    )


def split_series(
    s: pd.Series,
    start: str,
    end: str | None,
) -> pd.Series:
    if end is None:
        return s.loc[start:]
    return s.loc[start:end]


def run_train_test(
    close: pd.Series,
    regime_chop: pd.Series,
    *,
    train_range: Tuple[str, str],
    test_range: Tuple[str, str | None],
    lookbacks: Iterable[int],
    z_enters: Iterable[float],
    z_exits: Iterable[float],
    max_hold_days_list: Iterable[int],
    cost_bps_roundtrip: float,
    periods_per_year: int = 252,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    train_close = split_series(close, train_range[0], train_range[1])
    train_chop = split_series(regime_chop, train_range[0], train_range[1])

    test_close = split_series(close, test_range[0], test_range[1])
    test_chop = split_series(regime_chop, test_range[0], test_range[1])

    train_df = sweep_params(
        train_close,
        train_chop,
        lookbacks=lookbacks,
        z_enters=z_enters,
        z_exits=z_exits,
        max_hold_days_list=max_hold_days_list,
        cost_bps_roundtrip=cost_bps_roundtrip,
        periods_per_year=periods_per_year,
    )
    test_df = sweep_params(
        test_close,
        test_chop,
        lookbacks=lookbacks,
        z_enters=z_enters,
        z_exits=z_exits,
        max_hold_days_list=max_hold_days_list,
        cost_bps_roundtrip=cost_bps_roundtrip,
        periods_per_year=periods_per_year,
    )

    return train_df, test_df, {}


def combine_train_test(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> pd.DataFrame:
    key_cols = ["lookback", "z_enter", "z_exit", "max_hold_days"]
    train_cols = {
        "chop_ann_return": "train_chop_ann_return",
        "occupancy_in_chop": "train_occupancy_in_chop",
        "ann_over_inpos_time": "train_ann_over_inpos_time",
        "trades": "train_trades",
    }
    test_cols = {
        "chop_ann_return": "test_chop_ann_return",
        "occupancy_in_chop": "test_occupancy_in_chop",
        "ann_over_inpos_time": "test_ann_over_inpos_time",
        "trades": "test_trades",
    }

    train_out = train_df[key_cols + list(train_cols.keys())].rename(columns=train_cols)
    test_out = test_df[key_cols + list(test_cols.keys())].rename(columns=test_cols)

    merged = train_out.merge(test_out, on=key_cols, how="inner")
    merged["robust_min"] = merged[["train_chop_ann_return", "test_chop_ann_return"]].min(axis=1)
    merged["robust_mean"] = (
        merged["train_chop_ann_return"] + merged["test_chop_ann_return"]
    ) / 2.0

    return merged


def print_summary(
    title: str,
    df: pd.DataFrame,
    baselines: Dict[str, Dict[str, float]],
    *,
    top_n: int = 5,
) -> None:
    print(f"\n{title}")
    if df.empty:
        print("No results.")
        return

    cols = [
        "lookback",
        "z_enter",
        "z_exit",
        "max_hold_days",
        "occupancy_in_chop",
        "occupancy_total",
        "entries_per_year",
        "entries_per_chop_year",
        "chop_ann_return",
        "ann_over_chop_time",
        "ann_over_inpos_time",
        "chop_max_dd",
        "trades",
        "win_rate",
        "avg_trade",
    ]
    print(df[cols].head(top_n).to_string(index=False))

    print("\nBaselines (Chop-only):")
    for name, metrics in baselines.items():
        print(
            f"{name}: ann={metrics['ann_return']:.4f}, mdd={metrics['max_dd']:.4f}, sharpe={metrics['sharpe']:.3f}"
        )


# ============================================================
# Main (demo harness)
# ============================================================


def main():
    import yfinance as yf
    from PMTL.Test2 import build_regime

    df = yf.download("GLD", period="max", auto_adjust=True, progress=False)
    close = as_series(df["Close"], "Close")

    # Regime from Model A (weekly-gated). Chop is when neither up nor down.
    feature_params_up = dict(
        ma_len_entry=200,
        ma_len_exit=200,
        slope_lookback=20,
        entry_len=260,
        exit_len=90,
    )
    rule_params_up = dict(slope_min=0.0, ma_buffer=0.005)

    feature_params_down = dict(
        ma_len_entry=200,
        ma_len_exit=270,
        slope_lookback=20,
        entry_len=90,
        exit_len=90,
    )
    rule_params_down = dict(slope_min=0.0, ma_buffer=0.005)

    reg = build_regime(
        close,
        feature_params_up=feature_params_up,
        rule_params_up=rule_params_up,
        feature_params_down=feature_params_down,
        rule_params_down=rule_params_down,
        gate="BME",
    )
    regime_chop = reg["is_chop"]

    train_range = ("2004-01-01", "2015-12-31")
    test_range = ("2015-01-01", "2025-12-31")

    train_df, test_df, _meta = run_train_test(
        close,
        regime_chop,
        train_range=train_range,
        test_range=test_range,
        lookbacks=range(20, 101, 10),
        z_enters=[1.5, 2.0, 2.5],
        z_exits=[0.0, 0.5, 1.0],
        max_hold_days_list=[5, 10, 15],
        cost_bps_roundtrip=10.0,
    )

    combined = combine_train_test(train_df, test_df)
    combined.to_csv("mr_chop_train_test_results.csv", index=False)

    train_baselines = baseline_metrics(
        split_series(close, train_range[0], train_range[1]),
        split_series(regime_chop, train_range[0], train_range[1]),
    )
    test_baselines = baseline_metrics(
        split_series(close, test_range[0], test_range[1]),
        split_series(regime_chop, test_range[0], test_range[1]),
    )

    print_summary("Train (all configs)", train_df, train_baselines)
    print_summary("Test (all configs)", test_df, test_baselines)
    print("Saved: mr_chop_train_test_results.csv")


if __name__ == "__main__":
    main()

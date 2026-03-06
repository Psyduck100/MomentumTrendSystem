from __future__ import annotations

from typing import Callable, Dict, Any
import numpy as np
import pandas as pd
import yfinance as yf

# ============================================================
# Helpers
# ============================================================


def as_series(x: pd.Series | pd.DataFrame, name: str = "x") -> pd.Series:
    """Coerce Series or 1-col DataFrame into a Series."""
    if isinstance(x, pd.Series):
        return x.sort_index()
    if isinstance(x, pd.DataFrame):
        if x.shape[1] != 1:
            raise ValueError(f"Expected 1-column DataFrame for {name}, got {x.shape}")
        s = x.iloc[:, 0]
        if s.name is None:
            s.name = name
        return s.sort_index()
    raise TypeError(f"Expected Series/DataFrame for {name}, got {type(x)}")


# Modular enter/exit rules
EnterRule = Callable[[pd.Series, Dict[str, pd.Series], Dict[str, Any]], pd.Series]
ExitRule = Callable[[pd.Series, Dict[str, pd.Series], Dict[str, Any]], pd.Series]


# ============================================================
# Features (ALL lengths are in TRADING DAYS)
# ============================================================


def compute_features(
    close: pd.Series | pd.DataFrame,
    *,
    ma_len: int = 200,
    slope_lookback: int = 20,
    entry_len: int = 90,
    exit_len: int = 90,
) -> Dict[str, pd.Series]:
    close = as_series(close, "Close")

    ma = close.rolling(ma_len, min_periods=ma_len).mean()
    ma_slope = (ma / ma.shift(slope_lookback)) - 1.0

    # Donchian channels using yesterday's channel values
    ch_high_entry = close.shift(1).rolling(entry_len, min_periods=entry_len).max()
    ch_low_exit = close.shift(1).rolling(exit_len, min_periods=exit_len).min()

    return {
        "ma": ma,
        "ma_slope": ma_slope,
        "ch_high_entry": ch_high_entry,
        "ch_low_exit": ch_low_exit,
    }


# ============================================================
# Default rules
# ============================================================


def enter_breakout_ma_slope(
    close: pd.Series,
    feat: Dict[str, pd.Series],
    params: Dict[str, Any],
) -> pd.Series:
    slope_min = float(params.get("slope_min", 0.0))
    enter = (
        (close > feat["ma"])
        & (feat["ma_slope"] > slope_min)
        & (close > feat["ch_high_entry"])
    )
    return enter.fillna(False).astype(bool)


def exit_donchian_or_ma_buffer(
    close: pd.Series,
    feat: Dict[str, pd.Series],
    params: Dict[str, Any],
) -> pd.Series:
    ma_buffer = float(params.get("ma_buffer", 0.005))
    exit_ = (close < feat["ch_low_exit"]) | (close < feat["ma"] * (1.0 - ma_buffer))
    return exit_.fillna(False).astype(bool)


# ============================================================
# Decision-gated (ANY-signal) state machine
# ============================================================


def decision_gated_is_up_anysignal(
    close: pd.Series | pd.DataFrame,
    *,
    enter_rule: EnterRule,
    exit_rule: ExitRule,
    feature_params: Dict[str, Any],
    rule_params: Dict[str, Any] | None = None,
    gate: str = "BM",
) -> pd.Series:
    """
    Signals computed daily (lengths in DAYS), but position changes only on gate days.

    ANY-signal behavior:
      - If an enter signal occurs at any time since last gate, enter at gate.
      - If an exit signal occurs at any time since last gate, exit at gate.
    """
    close_s = as_series(close, "Close")
    feat = compute_features(close_s, **feature_params)

    rp = dict(rule_params or {})
    enter = enter_rule(close_s, feat, rp).fillna(False).astype(bool)
    exit_ = exit_rule(close_s, feat, rp).fillna(False).astype(bool)

    ready = (
        (~feat["ma"].isna())
        & (~feat["ma_slope"].isna())
        & (~feat["ch_high_entry"].isna())
        & (~feat["ch_low_exit"].isna())
    )

    gate_days = close_s.resample(gate).last().index
    is_gate_day = close_s.index.isin(gate_days)

    is_up = pd.Series(False, index=close_s.index, dtype=bool)
    in_up = False
    pending_enter = False
    pending_exit = False

    for i in range(len(close_s)):
        if not bool(ready.iat[i]):
            is_up.iat[i] = False
            in_up = False
            pending_enter = False
            pending_exit = False
            continue

        if not in_up:
            if bool(enter.iat[i]):
                pending_enter = True
        else:
            if bool(exit_.iat[i]):
                pending_exit = True

        if is_gate_day[i]:
            if (not in_up) and pending_enter:
                in_up = True
            elif in_up and pending_exit:
                in_up = False
            pending_enter = False
            pending_exit = False

        is_up.iat[i] = in_up

    return is_up


# ============================================================
# Backtest + metrics
# ============================================================


def backtest_long_only(
    close: pd.Series, is_up: pd.Series, fee_bps: float = 0.0
) -> pd.Series:
    """
    Long-only when is_up is True, executed next day via shift(1).
    fee_bps applied per unit turnover.
    """
    close = as_series(close, "Close")
    rets = close.pct_change().fillna(0.0)

    pos = is_up.shift(1).fillna(False).astype(float)  # execute next day
    gross = pos * rets

    if fee_bps > 0:
        turnover = pos.diff().abs().fillna(0.0)
        net = gross - (fee_bps / 1e4) * turnover
    else:
        net = gross

    return (1.0 + net).cumprod()


def cagr_from_equity(equity: pd.Series, periods_per_year: int = 252) -> float:
    equity = equity.dropna()
    if len(equity) < 2:
        return float("nan")
    years = len(equity) / periods_per_year
    return float(equity.iloc[-1] ** (1.0 / years) - 1.0)


def max_drawdown(equity: pd.Series) -> float:
    equity = equity.dropna()
    if len(equity) == 0:
        return float("nan")
    peak = equity.cummax()
    dd = equity / peak - 1.0
    return float(dd.min())


def sharpe_from_equity(equity: pd.Series, periods_per_year: int = 252) -> float:
    """
    Simple Sharpe (rf=0) using daily returns of the strategy equity.
    """
    equity = equity.dropna()
    if len(equity) < 3:
        return float("nan")
    rets = equity.pct_change().dropna()
    vol = rets.std(ddof=0)
    if vol == 0 or np.isnan(vol):
        return float("nan")
    return float((rets.mean() / vol) * np.sqrt(periods_per_year))


# ============================================================
# Plot
# ============================================================


def plot_equity_and_regime(
    close: pd.Series, equity: pd.Series, is_up: pd.Series, ma_len: int
):
    import matplotlib.pyplot as plt

    close = as_series(close, "Close")
    equity = equity.reindex(close.index).dropna()
    is_up = as_series(is_up, "is_up").reindex(close.index).fillna(False).astype(bool)

    ma = close.rolling(ma_len, min_periods=ma_len).mean()

    plt.figure(figsize=(12, 6))
    plt.yscale("log")
    plt.plot(close.index, close.values, label="GLD Close")
    plt.plot(ma.index, ma.values, label=f"SMA{ma_len}")

    up = is_up.values
    idx = close.index
    in_run = False
    run_start = None
    for i in range(len(up)):
        if up[i] and not in_run:
            in_run = True
            run_start = idx[i]
        if in_run and ((not up[i]) or i == len(up) - 1):
            run_end = idx[i] if up[i] else idx[i - 1]
            plt.axvspan(run_start, run_end, alpha=0.18)
            in_run = False

    plt.title("GLD regime (shaded = in position)")
    plt.legend()
    plt.show()

    plt.figure(figsize=(12, 5))
    plt.yscale("log")
    plt.plot(equity.index, equity.values, label="Strategy Equity")
    plt.title("Strategy equity curve (log scale)")
    plt.legend()
    plt.show()


# ============================================================
# Main (LOCKED: gated_any_BM, entry=90, exit=90)
# ============================================================


def main():
    df = yf.download("GLD", period="max", auto_adjust=True, progress=False)
    close = as_series(df["Close"], "Close")

    # ---- LOCKED CONFIG ----
    gate = "BM"
    entry_len = 90
    exit_len = 90
    ma_len = 200
    slope_lookback = 20

    rule_params = dict(
        slope_min=0.0,
        ma_buffer=0.005,
    )

    feature_params = dict(
        ma_len=ma_len,
        slope_lookback=slope_lookback,
        entry_len=entry_len,
        exit_len=exit_len,
    )

    # ---- SIGNAL + BACKTEST ----
    is_up = decision_gated_is_up_anysignal(
        close,
        enter_rule=enter_breakout_ma_slope,
        exit_rule=exit_donchian_or_ma_buffer,
        feature_params=feature_params,
        rule_params=rule_params,
        gate=gate,
    )

    equity = backtest_long_only(close, is_up, fee_bps=2.0)

    # ---- METRICS ----
    cagr = cagr_from_equity(equity)
    mdd = max_drawdown(equity)
    shp = sharpe_from_equity(equity)

    print(f"Variant: gated_any_{gate} | entry={entry_len}d exit={exit_len}d")
    print(f"CAGR:   {cagr:.6f}")
    print(f"Sharpe: {shp:.6f}")
    print(f"MaxDD:  {mdd:.6f}")

    # ---- PLOTS ----
    plot_equity_and_regime(close, equity, is_up, ma_len=ma_len)


if __name__ == "__main__":
    main()

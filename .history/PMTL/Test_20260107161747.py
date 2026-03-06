from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Any

import numpy as np
import pandas as pd
import yfinance as yf


def as_series(x: pd.Series | pd.DataFrame, name: str = "x") -> pd.Series:
    """Coerce Series or 1-column DataFrame into a Series."""
    if isinstance(x, pd.Series):
        return x
    if isinstance(x, pd.DataFrame):
        if x.shape[1] != 1:
            raise ValueError(f"Expected 1-column DataFrame for {name}, got {x.shape}")
        s = x.iloc[:, 0]
        if s.name is None:
            s.name = name
        return s
    raise TypeError(f"Expected Series/DataFrame for {name}, got {type(x)}")


# ----------------------------
# Modular enter/exit rules
# ----------------------------

EnterRule = Callable[[pd.Series, Dict[str, pd.Series], Dict[str, Any]], pd.Series]
ExitRule = Callable[[pd.Series, Dict[str, pd.Series], Dict[str, Any]], pd.Series]


def compute_features(
    close: pd.Series | pd.DataFrame,
    *,
    ma_len: int = 200,
    slope_lookback: int = 20,
    entry_len: int = 200,
    exit_len: int = 60,
) -> Dict[str, pd.Series]:
    """
    Precompute reusable features once. Rules can pick what they need.
    All features are aligned to close.index.
    """
    close = as_series(close, "Close").sort_index()

    ma = close.rolling(ma_len, min_periods=ma_len).mean()
    ma_slope = (ma / ma.shift(slope_lookback)) - 1.0

    # Donchian-style channels based on Close (yesterday's channel)
    ch_high_entry = close.shift(1).rolling(entry_len, min_periods=entry_len).max()
    ch_low_exit = close.shift(1).rolling(exit_len, min_periods=exit_len).min()

    return {
        "ma": ma,
        "ma_slope": ma_slope,
        "ch_high_entry": ch_high_entry,
        "ch_low_exit": ch_low_exit,
    }


def enter_breakout_ma_slope(
    close: pd.Series,
    feat: Dict[str, pd.Series],
    params: Dict[str, Any],
) -> pd.Series:
    """
    Default ENTER rule:
      close > MA AND MA slope > slope_min AND close > prior entry channel high
    """
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
    """
    Default EXIT rule:
      close < prior exit channel low  OR  close < MA*(1 - ma_buffer)
    """
    ma_buffer = float(params.get("ma_buffer", 0.005))  # 0.5% default
    exit_ = (close < feat["ch_low_exit"]) | (close < feat["ma"] * (1.0 - ma_buffer))
    return exit_.fillna(False).astype(bool)


def uptrend_state_machine(
    close: pd.Series | pd.DataFrame,
    *,
    enter_rule: EnterRule,
    exit_rule: ExitRule,
    feature_params: Dict[str, Any],
    rule_params: Dict[str, Any] | None = None,
) -> pd.Series:
    """
    Produces a continuous UP state:
      - Enter when enter_rule is True
      - Stay UP until exit_rule is True
    Evaluated on day t using info up to day t (channels use shift(1) in features).
    """
    close_s = as_series(close, "Close").sort_index()
    feat = compute_features(close_s, **feature_params)

    rp = dict(rule_params or {})
    enter = enter_rule(close_s, feat, rp)
    exit_ = exit_rule(close_s, feat, rp)

    is_up = pd.Series(False, index=close_s.index, dtype=bool)
    in_up = False

    # Determine where features are "ready" (no NaNs needed by our default rules)
    # If you create custom rules that use other features, update this readiness check accordingly.
    ready = (
        (~feat["ma"].isna())
        & (~feat["ma_slope"].isna())
        & (~feat["ch_high_entry"].isna())
        & (~feat["ch_low_exit"].isna())
    )

    for i in range(len(close_s)):
        if not bool(ready.iat[i]):
            is_up.iat[i] = False
            in_up = False
            continue

        if (not in_up) and bool(enter.iat[i]):
            in_up = True
        elif in_up and bool(exit_.iat[i]):
            in_up = False

        is_up.iat[i] = in_up

    return is_up


# ----------------------------
# Utilities
# ----------------------------


def uptrend_periods(is_up: pd.Series | pd.DataFrame) -> pd.DataFrame:
    is_up = as_series(is_up, "is_up").astype(bool)
    x = is_up.astype(np.int8)
    changes = x.diff().fillna(0)

    start_mask = (changes == 1).to_numpy()
    end_mask = (changes == -1).to_numpy()

    starts = is_up.index[start_mask]
    ends = is_up.index[end_mask]

    if is_up.iloc[0]:
        starts = pd.Index([is_up.index[0]]).append(starts)
    if is_up.iloc[-1]:
        ends = ends.append(pd.Index([is_up.index[-1]]))

    n = min(len(starts), len(ends))
    return pd.DataFrame({"start": starts[:n], "end": ends[:n]})


def plot_uptrend(
    close: pd.Series | pd.DataFrame, is_up: pd.Series | pd.DataFrame, ma_len: int = 200
):
    import matplotlib.pyplot as plt

    close = as_series(close, "Close").sort_index()
    is_up = (
        as_series(is_up, "is_up").reindex(close.index, fill_value=False).astype(bool)
    )

    ma = close.rolling(ma_len, min_periods=ma_len).mean()

    plt.figure(figsize=(12, 6))
    plt.plot(close.index, close.values, label="Close")
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
            plt.axvspan(run_start, run_end, alpha=0.2)
            in_run = False

    plt.legend()
    plt.title("Uptrend state machine (shaded)")
    plt.show()


# ----------------------------
# Main
# ----------------------------


def main():
    df = yf.download("GLD", period="max", auto_adjust=True, progress=False)
    close = as_series(df["Close"], "Close")

    # Feature computation params (controls the MA/channel lengths)
    feature_params = dict(
        ma_len=200,
        slope_lookback=20,
        entry_len=200,
        exit_len=60,  # exit channel length
    )

    # Rule params (controls thresholds)
    rule_params = dict(
        slope_min=0.0,
        ma_buffer=0.005,
    )

    # Plug any enter/exit rule functions here
    is_up = uptrend_state_machine(
        close,
        enter_rule=enter_breakout_ma_slope,
        exit_rule=exit_donchian_or_ma_buffer,
        feature_params=feature_params,
        rule_params=rule_params,
    )

    periods = uptrend_periods(is_up)

    out_csv = "gld_uptrend_periods.csv"
    periods.to_csv(out_csv, index=False)

    out_txt = "gld_uptrend_periods.txt"
    with open(out_txt, "w", encoding="utf-8") as f:
        f.write(periods.to_string(index=False))

    print(f"Saved: {out_csv}")
    print(f"Saved: {out_txt}")
    print(f"UP days: {int(is_up.sum())} / {len(is_up)}")

    plot_uptrend(close, is_up, ma_len=feature_params["ma_len"])


if __name__ == "__main__":
    main()

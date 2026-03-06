from __future__ import annotations

from typing import Callable, Dict, Any, Iterable
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt


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
    # rough (works fine for trading-day indexed series)
    return max(1e-9, len(idx) / periods_per_year)


def fwd_return(close: pd.Series, horizon_days: int) -> pd.Series:
    # forward simple return at each t: close[t+h]/close[t] - 1
    return close.shift(-horizon_days) / close - 1.0


# ============================================================
# Feature computation (ALL lengths are TRADING DAYS)
# ============================================================

EnterRule = Callable[[pd.Series, Dict[str, pd.Series], Dict[str, Any]], pd.Series]
ExitRule = Callable[[pd.Series, Dict[str, pd.Series], Dict[str, Any]], pd.Series]


def compute_features(
    close: pd.Series | pd.DataFrame,
    *,
    ma_len_entry: int = 200,
    ma_len_exit: int = 270,
    slope_lookback: int = 20,
    entry_len: int = 90,
    exit_len: int = 90,
) -> Dict[str, pd.Series]:
    close = as_series(close, "Close")

    ma_entry = close.rolling(ma_len_entry, min_periods=ma_len_entry).mean()
    ma_exit = close.rolling(ma_len_exit, min_periods=ma_len_exit).mean()
    ma_slope_entry = (ma_entry / ma_entry.shift(slope_lookback)) - 1.0

    # Donchian channels (yesterday’s channel)
    ch_high_entry = close.shift(1).rolling(entry_len, min_periods=entry_len).max()
    ch_low_exit = close.shift(1).rolling(exit_len, min_periods=exit_len).min()

    return {
        "ma_entry": ma_entry,
        "ma_exit": ma_exit,
        "ma_slope_entry": ma_slope_entry,
        "ch_high_entry": ch_high_entry,
        "ch_low_exit": ch_low_exit,
    }


# ============================================================
# UP rules
# ============================================================


def enter_breakout_ma_slope(
    close: pd.Series, feat: Dict[str, pd.Series], params: Dict[str, Any]
) -> pd.Series:
    """
    ENTER UP:
      close > MA AND MA slope > slope_min AND close > prior channel high
    """
    slope_min = float(params.get("slope_min", 0.0))
    enter = (
        (close > feat["ma_entry"])
        & (feat["ma_slope_entry"] > slope_min)
        & (close > feat["ch_high_entry"])
    )
    return enter.fillna(False).astype(bool)


def exit_up_on_donchian_or_buffer(
    close: pd.Series, feat: Dict[str, pd.Series], params: Dict[str, Any]
) -> pd.Series:
    """
    EXIT UP:
      close < prior channel low OR close < MA*(1 - ma_buffer)
    """
    ma_buffer = float(params.get("ma_buffer", 0.005))
    exit_ = (close < feat["ch_low_exit"]) | (close < feat["ma_exit"] * (1.0 - ma_buffer))
    return exit_.fillna(False).astype(bool)


# ============================================================
# DOWN rules (mirrored)
# ============================================================


def enter_breakdown_ma_slope(
    close: pd.Series, feat: Dict[str, pd.Series], params: Dict[str, Any]
) -> pd.Series:
    """
    ENTER DOWN:
      close < MA AND MA slope < -slope_min AND close < prior channel low
    """
    slope_min = float(params.get("slope_min", 0.0))
    enter = (
        (close < feat["ma_entry"])
        & (feat["ma_slope_entry"] < -slope_min)
        & (close < feat["ch_low_exit"])
    )
    return enter.fillna(False).astype(bool)


def exit_down_on_ma_buffer(
    close: pd.Series, feat: Dict[str, pd.Series], params: Dict[str, Any]
) -> pd.Series:
    """
    EXIT DOWN (MA only):
      close > MA*(1 + ma_buffer)
    """
    ma_buffer = float(params.get("ma_buffer", 0.005))
    exit_ = close > feat["ma_exit"] * (1.0 + ma_buffer)
    return exit_.fillna(False).astype(bool)


# ============================================================
# Decision-gated ANY-signal state machine
# ============================================================


def decision_gated_state_anysignal(
    close: pd.Series | pd.DataFrame,
    *,
    enter_rule: EnterRule,
    exit_rule: ExitRule,
    feature_params: Dict[str, Any],
    rule_params: Dict[str, Any] | None = None,
    gate: str = "BM",
) -> pd.Series:
    """
    Compute enter/exit on DAILY data, but only transition on gate dates.
    ANY-signal:
      - If enter happened any time since last gate -> enter at gate
      - If exit happened any time since last gate  -> exit at gate
    """
    close_s = as_series(close, "Close")
    feat = compute_features(close_s, **feature_params)

    rp = dict(rule_params or {})
    enter = enter_rule(close_s, feat, rp).fillna(False).astype(bool)
    exit_ = exit_rule(close_s, feat, rp).fillna(False).astype(bool)

    ready = (
        (~feat["ma_entry"].isna())
        & (~feat["ma_exit"].isna())
        & (~feat["ma_slope_entry"].isna())
        & (~feat["ch_high_entry"].isna())
        & (~feat["ch_low_exit"].isna())
    )

    gate_days = close_s.resample(gate).last().index
    is_gate_day = close_s.index.isin(gate_days)

    state = pd.Series(False, index=close_s.index, dtype=bool)
    in_state = False
    pending_enter = False
    pending_exit = False

    for i in range(len(close_s)):
        if not bool(ready.iat[i]):
            state.iat[i] = False
            in_state = False
            pending_enter = False
            pending_exit = False
            continue

        if not in_state:
            if bool(enter.iat[i]):
                pending_enter = True
        else:
            if bool(exit_.iat[i]):
                pending_exit = True

        if is_gate_day[i]:
            if (not in_state) and pending_enter:
                in_state = True
            elif in_state and pending_exit:
                in_state = False
            pending_enter = False
            pending_exit = False

        state.iat[i] = in_state

    return state


# ============================================================
# Weekly 2-gate state machine
# ============================================================


def decision_gated_state_two_gate_weekly(
    close: pd.Series | pd.DataFrame,
    *,
    enter_rule: EnterRule,
    exit_rule: ExitRule,
    feature_params: Dict[str, Any],
    rule_params: Dict[str, Any] | None = None,
    gate: str = "W-FRI",
    enter_gates: int = 2,
    exit_gates: int = 2,
) -> pd.Series:
    """
    Compute enter/exit on DAILY data, but transition on weekly gate dates.
    Enter/exit require the signal to be present in N consecutive gates.
    """
    close_s = as_series(close, "Close")
    feat = compute_features(close_s, **feature_params)

    rp = dict(rule_params or {})
    enter = enter_rule(close_s, feat, rp).fillna(False).astype(bool)
    exit_ = exit_rule(close_s, feat, rp).fillna(False).astype(bool)

    ready = (
        (~feat["ma_entry"].isna())
        & (~feat["ma_exit"].isna())
        & (~feat["ma_slope_entry"].isna())
        & (~feat["ch_high_entry"].isna())
        & (~feat["ch_low_exit"].isna())
    )

    gate_close = close_s.resample(gate).last()
    gate_idx = gate_close.index

    enter_any = enter.resample(gate).max()
    exit_any = exit_.resample(gate).max()
    ready_any = ready.resample(gate).min()

    state_gate = pd.Series(False, index=gate_idx, dtype=bool)
    in_state = False
    enter_streak = 0
    exit_streak = 0

    for i in range(len(gate_idx)):
        if not bool(ready_any.iat[i]):
            state_gate.iat[i] = False
            in_state = False
            enter_streak = 0
            exit_streak = 0
            continue

        if not in_state:
            if bool(enter_any.iat[i]):
                enter_streak += 1
            else:
                enter_streak = 0
            if enter_streak >= enter_gates:
                in_state = True
                enter_streak = 0
        else:
            if bool(exit_any.iat[i]):
                exit_streak += 1
            else:
                exit_streak = 0
            if exit_streak >= exit_gates:
                in_state = False
                exit_streak = 0

        state_gate.iat[i] = in_state

    state_daily = state_gate.reindex(close_s.index).ffill().fillna(False).astype(bool)
    return state_daily


# ============================================================
# Regime builder: +1 up, -1 down, 0 chop
# ============================================================


def build_regime(
    close: pd.Series,
    *,
    feature_params_up: Dict[str, Any],
    rule_params_up: Dict[str, Any],
    feature_params_down: Dict[str, Any],
    rule_params_down: Dict[str, Any],
    gate: str = "BM",
) -> pd.DataFrame:
    close = as_series(close, "Close")

    is_up = decision_gated_state_anysignal(
        close,
        enter_rule=enter_breakout_ma_slope,
        exit_rule=exit_up_on_donchian_or_buffer,
        feature_params=feature_params_up,
        rule_params=rule_params_up,
        gate=gate,
    )

    is_down = decision_gated_state_two_gate_weekly(
        close,
        enter_rule=enter_breakdown_ma_slope,
        exit_rule=exit_down_on_ma_buffer,
        feature_params=feature_params_down,
        rule_params=rule_params_down,
        gate="W-FRI",
        enter_gates=2,
        exit_gates=2,
    )

    overlap = is_up & is_down
    is_up = is_up & ~overlap
    is_down = is_down & ~overlap

    regime = pd.Series(0, index=close.index, dtype=np.int8)
    regime[is_up] = 1
    regime[is_down] = -1

    chop = regime == 0

    return pd.DataFrame(
        {
            "is_up": is_up.astype(bool),
            "is_down": is_down.astype(bool),
            "is_chop": chop.astype(bool),
            "regime": regime,
        },
        index=close.index,
    )


# ============================================================
# Optimization objective for DOWN regime
# ============================================================


def down_score_and_stats(
    close: pd.Series,
    is_down: pd.Series,
    *,
    fwd_horizon_days: int = 63,
    periods_per_year: int = 252,
    lambda_time: float = 0.05,
    lambda_trades_per_year: float = 0.01,
) -> Dict[str, float]:
    """
    Score = -mean_fwd_ret(h) - lambda_time * time_in_down - lambda_trades * trades_per_year

    Notes on "reasonable" lambdas:
      - mean_fwd_ret(h) is a decimal (e.g. -0.05 = -5% over ~3 months)
      - time_in_down is 0..1
      - trades_per_year is usually 0..10ish
    Defaults make penalties comparable:
      - time penalty: 0.05 * 0.20 = 0.01 (1% score hit if down is on 20% of days)
      - trades penalty: 0.01 * 3 = 0.03 (3% score hit if 3 down-entries/year)
    """
    close = as_series(close, "Close")
    is_down = (
        as_series(is_down, "is_down").reindex(close.index).fillna(False).astype(bool)
    )

    # Down entries = first day state becomes True (based on state series itself)
    entries = is_down.astype(int).diff().fillna(0) == 1
    entry_idx = close.index[entries.to_numpy()]

    # Forward returns at entry dates
    fwd = fwd_return(close, fwd_horizon_days)
    fwd_at_entries = fwd.loc[entry_idx].dropna()

    mean_fwd = float(fwd_at_entries.mean()) if len(fwd_at_entries) else np.nan
    med_fwd = float(fwd_at_entries.median()) if len(fwd_at_entries) else np.nan

    time_in_down = float(is_down.mean())

    yrs = years_in_index(close.index, periods_per_year=periods_per_year)
    trades = int(entries.sum())
    trades_per_year = float(trades / yrs) if yrs > 0 else np.nan

    score = (
        (-mean_fwd)
        - (lambda_time * time_in_down)
        - (lambda_trades_per_year * trades_per_year)
        if np.isfinite(mean_fwd) and np.isfinite(trades_per_year)
        else np.nan
    )

    return {
        "score": float(score) if np.isfinite(score) else np.nan,
        "mean_fwd_ret": float(mean_fwd) if np.isfinite(mean_fwd) else np.nan,
        "median_fwd_ret": float(med_fwd) if np.isfinite(med_fwd) else np.nan,
        "time_in_down": time_in_down,
        "trades": float(trades),
        "trades_per_year": trades_per_year,
        "n_entries_used": float(len(fwd_at_entries)),
    }


def sweep_down_regime(
    close: pd.Series,
    *,
    gate: str = "W-FRI",
    # feature base (days)
    ma_len_entry: int = 200,
    ma_len_exit: int = 270,
    slope_lookback: int = 20,
    # sweep grids
    entry_exit_len: int = 90,
    slope_mins: Iterable[float] = (0.0, 0.0025, 0.005),  # optional (try a few)
    ma_buffers: Iterable[float] = (0.005, 0.01),  # optional (try a few)
    # objective settings
    fwd_horizon_days: int = 63,
    lambda_time: float = 0.05,
    lambda_trades_per_year: float = 0.01,
    periods_per_year: int = 252,
) -> pd.DataFrame:
    close = as_series(close, "Close")

    rows = []
    for slope_min in slope_mins:
        for ma_buffer in ma_buffers:
            feature_params = dict(
                ma_len_entry=int(ma_len_entry),
                ma_len_exit=int(ma_len_exit),
                slope_lookback=int(slope_lookback),
                entry_len=int(entry_exit_len),
                exit_len=int(entry_exit_len),
            )
            rule_params = dict(slope_min=float(slope_min), ma_buffer=float(ma_buffer))

            is_down = decision_gated_state_two_gate_weekly(
                close,
                enter_rule=enter_breakdown_ma_slope,
                exit_rule=exit_down_on_ma_buffer,
                feature_params=feature_params,
                rule_params=rule_params,
                gate=gate,
                enter_gates=2,
                exit_gates=2,
            )

            stats = down_score_and_stats(
                close,
                is_down,
                fwd_horizon_days=fwd_horizon_days,
                periods_per_year=periods_per_year,
                lambda_time=lambda_time,
                lambda_trades_per_year=lambda_trades_per_year,
            )

            rows.append(
                {
                    "gate": gate,
                    "ma_len_entry": int(ma_len_entry),
                    "ma_len_exit": int(ma_len_exit),
                    "slope_lookback": slope_lookback,
                    "slope_min": float(slope_min),
                    "ma_buffer": float(ma_buffer),
                    **stats,
                }
            )

    df = pd.DataFrame(rows)
    # higher score is better
    return df.sort_values(
        ["score", "mean_fwd_ret"], ascending=[False, True]
    ).reset_index(drop=True)


# ============================================================
# Main
# ============================================================


def _shade_runs(
    ax,
    idx: pd.Index,
    mask: pd.Series,
    *,
    color: str,
    alpha: float = 0.18,
    label: str | None = None,
):
    """Shade contiguous True runs in mask on axis ax using axvspan."""
    mask = mask.reindex(idx).fillna(False).astype(bool).to_numpy()
    in_run = False
    start = None

    for i in range(len(mask)):
        if mask[i] and not in_run:
            in_run = True
            start = idx[i]
        if in_run and ((not mask[i]) or i == len(mask) - 1):
            end = idx[i] if mask[i] else idx[i - 1]
            ax.axvspan(start, end, alpha=alpha, color=color, label=label)
            label = None
            in_run = False


def plot_up_down_regimes(
    close: pd.Series,
    reg: pd.DataFrame,
    *,
    ma_len_entry: int = 200,
    ma_len_exit: int = 270,
    logy: bool = True,
):
    close = close.sort_index()
    reg = reg.reindex(close.index).ffill()

    ma_entry = close.rolling(ma_len_entry, min_periods=ma_len_entry).mean()
    ma_exit = close.rolling(ma_len_exit, min_periods=ma_len_exit).mean()

    fig, ax = plt.subplots(figsize=(13, 6))
    if logy:
        ax.set_yscale("log")

    ax.plot(close.index, close.values, label="Close")
    ax.plot(ma_entry.index, ma_entry.values, label=f"SMA{ma_len_entry}")
    ax.plot(ma_exit.index, ma_exit.values, label=f"SMA{ma_len_exit}")

    _shade_runs(ax, close.index, reg["is_up"], color="green", alpha=0.16, label="UP")
    _shade_runs(ax, close.index, reg["is_down"], color="red", alpha=0.16, label="DOWN")

    ax.set_title("GLD regimes (shaded)")
    ax.legend()
    ax.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.4)
    plt.show()


def main():
    df = yf.download("GLD", period="max", auto_adjust=True, progress=False)
    close = as_series(df["Close"], "Close")

    # ---- Objective knobs (tweak here) ----
    fwd_h = 63  # ~3 months
    lambda_time = 0.05
    lambda_trades = 0.01

    ma_len_entry = 200
    ma_len_exit = 270
    entry_exit_len = 90

    cols = [
        "score",
        "mean_fwd_ret",
        "median_fwd_ret",
        "time_in_down",
        "trades_per_year",
        "n_entries_used",
        "slope_min",
        "ma_buffer",
        "ma_len_entry",
        "ma_len_exit",
    ]

    res = sweep_down_regime(
        close,
        gate="W-FRI",
        ma_len_entry=ma_len_entry,
        ma_len_exit=ma_len_exit,
        slope_lookback=20,
        entry_exit_len=entry_exit_len,
        slope_mins=(0.0, 0.0025, 0.005),
        ma_buffers=(0.0, 0.005, 0.01),
        fwd_horizon_days=fwd_h,
        lambda_time=lambda_time,
        lambda_trades_per_year=lambda_trades,
    )

    print("\nTop 25:")
    print(res[cols].head(25).to_string(index=False))

    out_csv = "gld_down_regime_optimize.csv"
    res.to_csv(out_csv, index=False)
    print(f"Saved: {out_csv}")

    best = res.iloc[0].to_dict()
    print("\nBEST PARAMS:")
    for k in [
        "ma_len_entry",
        "ma_len_exit",
        "slope_min",
        "ma_buffer",
        "score",
        "mean_fwd_ret",
        "time_in_down",
        "trades_per_year",
        "n_entries_used",
    ]:
        print(f"{k}: {best[k]}")

    feature_params_down = dict(
        ma_len_entry=ma_len_entry,
        ma_len_exit=ma_len_exit,
        slope_lookback=20,
        entry_len=entry_exit_len,
        exit_len=entry_exit_len,
    )
    rule_params_down = dict(slope_min=best["slope_min"], ma_buffer=best["ma_buffer"])

    feature_params_up = dict(
        ma_len_entry=200,
        ma_len_exit=200,
        slope_lookback=20,
        entry_len=260,
        exit_len=90,
    )
    rule_params_up = dict(slope_min=0.0, ma_buffer=0.005)

    reg = build_regime(
        close,
        feature_params_up=feature_params_up,
        rule_params_up=rule_params_up,
        feature_params_down=feature_params_down,
        rule_params_down=rule_params_down,
        gate="BME",
    )
    plot_up_down_regimes(
        close,
        reg,
        ma_len_entry=feature_params_up["ma_len_entry"],
        ma_len_exit=feature_params_down["ma_len_exit"],
        logy=True,
    )


if __name__ == "__main__":
    main()

from __future__ import annotations

from typing import Callable, Dict, Any
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
    ma_buffer = float(params.get("ma_buffer", 0.005))
    exit_ = (close < feat["ch_low_exit"]) | (
        close < feat["ma_exit"] * (1.0 - ma_buffer)
    )
    return exit_.fillna(False).astype(bool)


# ============================================================
# DOWN rules
# ============================================================


def enter_breakdown_ma_slope(
    close: pd.Series, feat: Dict[str, pd.Series], params: Dict[str, Any]
) -> pd.Series:
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
    ma_buffer = float(params.get("ma_buffer", 0.005))
    exit_ = close > feat["ma_exit"] * (1.0 + ma_buffer)
    return exit_.fillna(False).astype(bool)


# ============================================================
# Decision-gated state machines
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

    state_daily = state_gate.reindex(close_s.index).ffill().astype("boolean")
    state_daily = state_daily.fillna(False).astype(bool)
    return state_daily


# ============================================================
# Regime builder
# ============================================================


def build_regime(
    close: pd.Series,
    *,
    feature_params_up: Dict[str, Any],
    rule_params_up: Dict[str, Any],
    feature_params_down: Dict[str, Any],
    rule_params_down: Dict[str, Any],
    gate_up: str = "BME",
    gate_down: str = "W-FRI",
    down_enter_gates: int = 2,
    down_exit_gates: int = 2,
) -> pd.DataFrame:
    close = as_series(close, "Close")

    is_up = decision_gated_state_anysignal(
        close,
        enter_rule=enter_breakout_ma_slope,
        exit_rule=exit_up_on_donchian_or_buffer,
        feature_params=feature_params_up,
        rule_params=rule_params_up,
        gate=gate_up,
    )

    is_down = decision_gated_state_two_gate_weekly(
        close,
        enter_rule=enter_breakdown_ma_slope,
        exit_rule=exit_down_on_ma_buffer,
        feature_params=feature_params_down,
        rule_params=rule_params_down,
        gate=gate_down,
        enter_gates=down_enter_gates,
        exit_gates=down_exit_gates,
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


def _fmt_level(value: float | np.floating | None) -> str:
    if value is None or pd.isna(value):
        return "n/a"
    return f"{float(value):.2f}"


def _cmp_flag(value: float, threshold: float, op: str) -> str:
    if pd.isna(threshold):
        return "unknown"
    if op == ">":
        return "above" if value > threshold else "below"
    if op == "<":
        return "below" if value < threshold else "above"
    return "n/a"


def describe_regime_snapshot(
    close: pd.Series,
    *,
    feature_params_up: Dict[str, Any],
    rule_params_up: Dict[str, Any],
    feature_params_down: Dict[str, Any],
    rule_params_down: Dict[str, Any],
) -> Dict[str, Any]:
    close = as_series(close, "Close")
    ts = close.index[-1]
    price = float(close.loc[ts])

    feat_up = compute_features(close, **feature_params_up)
    feat_down = compute_features(close, **feature_params_down)

    slope_min_up = float(rule_params_up.get("slope_min", 0.0))
    ma_buf_up = float(rule_params_up.get("ma_buffer", 0.005))
    slope_min_down = float(rule_params_down.get("slope_min", 0.0))
    ma_buf_down = float(rule_params_down.get("ma_buffer", 0.005))

    ma_entry_up = feat_up["ma_entry"].loc[ts]
    ma_exit_up = feat_up["ma_exit"].loc[ts]
    slope_up = feat_up["ma_slope_entry"].loc[ts]
    ch_high_up = feat_up["ch_high_entry"].loc[ts]
    ch_low_up = feat_up["ch_low_exit"].loc[ts]

    ma_entry_down = feat_down["ma_entry"].loc[ts]
    ma_exit_down = feat_down["ma_exit"].loc[ts]
    slope_down = feat_down["ma_slope_entry"].loc[ts]
    ch_low_down = feat_down["ch_low_exit"].loc[ts]

    up_entry = (
        (price > ma_entry_up) and (slope_up > slope_min_up) and (price > ch_high_up)
    )
    up_exit = (price < ch_low_up) or (price < ma_exit_up * (1.0 - ma_buf_up))

    down_entry = (
        (price < ma_entry_down)
        and (slope_down < -slope_min_down)
        and (price < ch_low_down)
    )
    down_exit = price > ma_exit_down * (1.0 + ma_buf_down)

    up_reason = (
        f"Up entry: price {_cmp_flag(price, float(ma_entry_up), '>')} "
        f"MA{feature_params_up['ma_len_entry']} ({_fmt_level(ma_entry_up)}), "
        f"slope {_fmt_level(slope_up)} vs min {slope_min_up:.3f}, "
        f"price {_cmp_flag(price, float(ch_high_up), '>')} "
        f"{feature_params_up['entry_len']}d high ({_fmt_level(ch_high_up)}). "
        f"Up exit: price {_cmp_flag(price, float(ch_low_up), '<')} "
        f"{feature_params_up['exit_len']}d low ({_fmt_level(ch_low_up)}) "
        f"or below MA{feature_params_up['ma_len_exit']} - buf "
        f"({_fmt_level(float(ma_exit_up) * (1.0 - ma_buf_up))})."
    )

    down_reason = (
        f"Down entry: price {_cmp_flag(price, float(ma_entry_down), '<')} "
        f"MA{feature_params_down['ma_len_entry']} ({_fmt_level(ma_entry_down)}), "
        f"slope {_fmt_level(slope_down)} vs min -{slope_min_down:.3f}, "
        f"price {_cmp_flag(price, float(ch_low_down), '<')} "
        f"{feature_params_down['exit_len']}d low ({_fmt_level(ch_low_down)}). "
        f"Down exit: price {_cmp_flag(price, float(ma_exit_down) * (1.0 + ma_buf_down), '>')} "
        f"MA{feature_params_down['ma_len_exit']} + buf "
        f"({_fmt_level(float(ma_exit_down) * (1.0 + ma_buf_down))})."
    )

    return {
        "asof_date": ts,
        "price": price,
        "up_entry": bool(up_entry),
        "up_exit": bool(up_exit),
        "down_entry": bool(down_entry),
        "down_exit": bool(down_exit),
        "up_reason": up_reason,
        "down_reason": down_reason,
    }

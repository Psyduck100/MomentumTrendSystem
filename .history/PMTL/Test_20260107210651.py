from __future__ import annotations

from typing import Callable, Dict, Any
import numpy as np
import pandas as pd
import yfinance as yf


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


EnterRule = Callable[[pd.Series, Dict[str, pd.Series], Dict[str, Any]], pd.Series]
ExitRule = Callable[[pd.Series, Dict[str, pd.Series], Dict[str, Any]], pd.Series]


# ============================================================
# Features (ALL lengths are TRADING DAYS)
# ============================================================


def compute_features(
    close: pd.Series | pd.DataFrame,
    *,
    ma_len: int = 200,
    slope_lookback: int = 20,
    entry_len: int = 260,
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
# UP rules (what you already have)
# ============================================================


def enter_breakout_ma_slope(
    close: pd.Series, feat: Dict[str, pd.Series], params: Dict[str, Any]
) -> pd.Series:
    slope_min = float(params.get("slope_min", 0.0))
    enter = (
        (close > feat["ma"])
        & (feat["ma_slope"] > slope_min)
        & (close > feat["ch_high_entry"])
    )
    return enter.fillna(False).astype(bool)


def exit_donchian_or_ma_buffer(
    close: pd.Series, feat: Dict[str, pd.Series], params: Dict[str, Any]
) -> pd.Series:
    ma_buffer = float(params.get("ma_buffer", 0.005))
    exit_ = (close < feat["ch_low_exit"]) | (close < feat["ma"] * (1.0 - ma_buffer))
    return exit_.fillna(False).astype(bool)


# ============================================================
# DOWN rules (mirrored, to isolate true downtrends)
# ============================================================


def enter_breakdown_ma_slope(
    close: pd.Series, feat: Dict[str, pd.Series], params: Dict[str, Any]
) -> pd.Series:
    """
    ENTER DOWN:
      close < MA AND MA slope < -slope_min AND close < prior exit-channel low
    Note: uses feat["ch_low_exit"] as the breakdown level.
    """
    slope_min = float(params.get("slope_min", 0.0))
    enter = (
        (close < feat["ma"])
        & (feat["ma_slope"] < -slope_min)
        & (close < feat["ch_low_exit"])
    )
    return enter.fillna(False).astype(bool)


def exit_down_on_reclaim_or_buffer(
    close: pd.Series, feat: Dict[str, pd.Series], params: Dict[str, Any]
) -> pd.Series:
    """
    EXIT DOWN:
      close > prior entry-channel high  OR  close > MA*(1 + ma_buffer)
    Using entry-channel high as "reclaim highs" threshold.
    """
    ma_buffer = float(params.get("ma_buffer", 0.005))
    exit_ = (close > feat["ch_high_entry"]) | (close > feat["ma"] * (1.0 + ma_buffer))
    return exit_.fillna(False).astype(bool)


# ============================================================
# Decision-gated ANY-signal state machine (generic)
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
      - If enter happened at any point since last gate -> enter at gate
      - If exit happened at any point since last gate -> exit at gate
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
# Regime builder: +1 up, -1 down, 0 chop
# ============================================================


def build_regime(
    close: pd.Series,
    *,
    feature_params: Dict[str, Any],
    rule_params: Dict[str, Any],
    gate: str = "BM",
) -> pd.DataFrame:
    close = as_series(close, "Close")

    is_up = decision_gated_state_anysignal(
        close,
        enter_rule=enter_breakout_ma_slope,
        exit_rule=exit_donchian_or_ma_buffer,
        feature_params=feature_params,
        rule_params=rule_params,
        gate=gate,
    )

    is_down = decision_gated_state_anysignal(
        close,
        enter_rule=enter_breakdown_ma_slope,
        exit_rule=exit_down_on_reclaim_or_buffer,
        feature_params=feature_params,
        rule_params=rule_params,
        gate=gate,
    )

    # Make them mutually exclusive (safety): if both true, force CHOP
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
# Main (demo)
# ============================================================


def main():
    df = yf.download("GLD", period="max", auto_adjust=True, progress=False)
    close = as_series(df["Close"], "Close")

    feature_params = dict(ma_len=200, slope_lookback=20, entry_len=260, exit_len=90)
    rule_params = dict(slope_min=0.0, ma_buffer=0.005)

    reg = build_regime(
        close, feature_params=feature_params, rule_params=rule_params, gate="BM"
    )

    print("Regime counts:")
    print(reg["regime"].value_counts().sort_index())  # -1,0,1
    print("Overlap days (should be 0):", int((reg["is_up"] & reg["is_down"]).sum()))

    # Optional: save
    reg.to_csv("gld_regimes_up_down_chop.csv")
    print("Saved: gld_regimes_up_down_chop.csv")


if __name__ == "__main__":
    main()

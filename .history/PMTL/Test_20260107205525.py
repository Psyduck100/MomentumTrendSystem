from __future__ import annotations

from typing import Callable, Dict, Any, Tuple
import numpy as np
import pandas as pd
import yfinance as yf

# ============================================================
# Helpers
# ============================================================


def as_series(x: pd.Series | pd.DataFrame, name: str = "x") -> pd.Series:
    """Coerce Series or 1-column DataFrame into a Series."""
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
# Feature computation (ALL lengths are in TRADING DAYS here)
# ============================================================


def compute_features(
    close: pd.Series | pd.DataFrame,
    *,
    ma_len: int = 200,
    slope_lookback: int = 20,
    entry_len: int = 260,
    exit_len: int = 90,
) -> Dict[str, pd.Series]:
    """
    Precompute reusable features once on DAILY data.
    Lengths are in DAYS (trading days).
    """
    close = as_series(close, "Close")

    ma = close.rolling(ma_len, min_periods=ma_len).mean()
    ma_slope = (ma / ma.shift(slope_lookback)) - 1.0

    # Donchian-style channels based on Close using yesterday's channel values
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
    """
    ENTER:
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
    EXIT:
      close < prior exit channel low  OR  close < MA*(1 - ma_buffer)
    """
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
    gate: str = "BM",  # business month-end gating
) -> pd.Series:
    """
    Compute signals on DAILY data (so entry/exit lengths remain in DAYS),
    but only allow state transitions on gate dates.

    ANY-signal behavior:
      - If an enter condition happened at ANY point since last gate, enter at gate.
      - If an exit condition happened at ANY point since last gate, exit at gate.
    """
    close_s = as_series(close, "Close")
    feat = compute_features(close_s, **feature_params)

    rp = dict(rule_params or {})
    enter = enter_rule(close_s, feat, rp).fillna(False).astype(bool)
    exit_ = exit_rule(close_s, feat, rp).fillna(False).astype(bool)

    # readiness (features required by default rules)
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


def in_position_cagr(
    close: pd.Series, is_up: pd.Series, periods_per_year: int = 252
) -> float:
    """
    Annualized geometric return conditional on being in position.
    Uses only days where we were held (is_up.shift(1) == True).
    """
    close = as_series(close, "Close")
    rets = close.pct_change().fillna(0.0)
    held = is_up.shift(1).fillna(False)

    held_rets = rets[held]
    if len(held_rets) < 2:
        return float("nan")

    eq = (1.0 + held_rets).cumprod()
    years = len(held_rets) / periods_per_year
    return float(eq.iloc[-1] ** (1.0 / years) - 1.0)


def trades_from_is_up(is_up: pd.Series) -> int:
    # Trades counted on executed position (shifted) transitions into 1
    pos = is_up.shift(1).fillna(False).astype(int)
    return int((pos.diff().fillna(0) == 1).sum())


# ============================================================
# 2D sweep: entry (50..300) x exit (50..150) for gated_any_BM
# ============================================================


def sweep_gated_any_2d(
    close: pd.Series,
    *,
    enter_rule: EnterRule,
    exit_rule: ExitRule,
    rule_params: dict,
    entry_lens: list[int],
    exit_lens: list[int],
    base_feature_params: dict,
    fee_bps: float = 2.0,
    gate: str = "BM",
) -> pd.DataFrame:
    rows = []
    close = as_series(close, "Close")

    for en in entry_lens:
        for ex in exit_lens:
            feature_params = dict(base_feature_params)
            feature_params["entry_len"] = int(en)
            feature_params["exit_len"] = int(ex)

            is_up = decision_gated_is_up_anysignal(
                close,
                enter_rule=enter_rule,
                exit_rule=exit_rule,
                feature_params=feature_params,
                rule_params=rule_params,
                gate=gate,
            )

            equity = backtest_long_only(close, is_up, fee_bps=fee_bps)

            rows.append(
                {
                    "variant": f"gated_any_{gate}",
                    "entry_len_days": int(en),
                    "exit_len_days": int(ex),
                    "CAGR": cagr_from_equity(equity),
                    "MaxDD": max_drawdown(equity),
                    "Time_in_pos": float(is_up.shift(1).fillna(False).mean()),
                    "CAGR_in_pos": in_position_cagr(close, is_up),
                    "Trades": trades_from_is_up(is_up),
                    "Equity_final": (
                        float(equity.iloc[-1]) if len(equity) else float("nan")
                    ),
                }
            )

    out = pd.DataFrame(rows).sort_values("CAGR", ascending=False)
    return out


# ============================================================
# Main
# ============================================================


def main():
    df = yf.download("GLD", period="max", auto_adjust=True, progress=False)
    close = as_series(df["Close"], "Close")

    # Rules
    rule_params = dict(
        slope_min=0.0,
        ma_buffer=0.005,
    )

    # Fixed daily feature params (DAYS) except entry/exit which we sweep
    base_feature_params = dict(
        ma_len=200,
        slope_lookback=20,
        entry_len=260,  # overwritten
        exit_len=90,  # overwritten
    )

    # Sweep ranges (inclusive)
    entry_lens = list(range(50, 301, 10))  # 50..300 step 10
    exit_lens = list(range(50, 151, 10))  # 50..150 step 10

    res = sweep_gated_any_2d(
        close,
        enter_rule=enter_breakout_ma_slope,
        exit_rule=exit_donchian_or_ma_buffer,
        rule_params=rule_params,
        entry_lens=entry_lens,
        exit_lens=exit_lens,
        base_feature_params=base_feature_params,
        fee_bps=2.0,
        gate="BM",
    )

    # Show top 25
    print(res.head(25).to_string(index=False))

    # Save full grid
    out_csv = "gld_gated_any_BM_entry50_300_exit50_150.csv"
    res.to_csv(out_csv, index=False)
    print(f"Saved: {out_csv}")

    # Print best row clearly
    best = res.iloc[0].to_dict()
    print("\nBEST:")
    for k, v in best.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()

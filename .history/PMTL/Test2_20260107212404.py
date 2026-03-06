from __future__ import annotations

from typing import Callable, Dict, Any
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

    # Donchian channels using yesterday's values
    ch_high_entry = close.shift(1).rolling(entry_len, min_periods=entry_len).max()
    ch_low_exit = close.shift(1).rolling(exit_len, min_periods=exit_len).min()

    return {
        "ma": ma,
        "ma_slope": ma_slope,
        "ch_high_entry": ch_high_entry,
        "ch_low_exit": ch_low_exit,
    }


# ============================================================
# DOWN rules (mirrored)
# ============================================================


def enter_breakdown_ma_slope(
    close: pd.Series, feat: Dict[str, pd.Series], params: Dict[str, Any]
) -> pd.Series:
    """
    ENTER DOWN:
      close < MA
      MA slope < -slope_min
      close < prior ch_low_exit (breakdown)
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
      close > prior ch_high_entry  OR  close > MA*(1 + ma_buffer)
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
# Metrics for "true down"
# ============================================================


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


def strategy_equity_from_mask(close: pd.Series, mask: pd.Series) -> pd.Series:
    """
    "Down-only long" equity curve:
      - hold when mask.shift(1) is True (execute next day)
      - else flat
    """
    close = as_series(close, "Close")
    rets = close.pct_change().fillna(0.0)
    held = mask.shift(1).fillna(False)
    strat_rets = rets.where(held, 0.0)
    return (1.0 + strat_rets).cumprod()


def cagr_in_mask(
    close: pd.Series, mask: pd.Series, periods_per_year: int = 252
) -> float:
    """
    CAGR conditional on mask days ONLY (geometric, annualized to trading days).
    This answers: "when we call it DOWN, how bad is it (if long) on average?"
    """
    close = as_series(close, "Close")
    rets = close.pct_change().fillna(0.0)
    held = mask.shift(1).fillna(False)

    held_rets = rets[held]
    if len(held_rets) < 2:
        return float("nan")

    eq = (1.0 + held_rets).cumprod()
    years = len(held_rets) / periods_per_year
    return float(eq.iloc[-1] ** (1.0 / years) - 1.0)


def trades_from_mask(mask: pd.Series) -> int:
    pos = mask.shift(1).fillna(False).astype(int)
    return int((pos.diff().fillna(0) == 1).sum())


# ============================================================
# Sweep (optimize down regime)
# ============================================================


def sweep_down_regime_2d(
    close: pd.Series,
    *,
    base_feature_params: dict,
    rule_params: dict,
    entry_lens: list[int],
    exit_lens: list[int],
    gate: str = "BM",
    w_down_only: float = 0.5,
    w_inpos: float = 0.5,
) -> pd.DataFrame:
    close = as_series(close, "Close")
    rows = []

    for en in entry_lens:
        for ex in exit_lens:
            fp = dict(base_feature_params)
            fp["entry_len"] = int(en)
            fp["exit_len"] = int(ex)

            is_down = decision_gated_state_anysignal(
                close,
                enter_rule=enter_breakdown_ma_slope,
                exit_rule=exit_down_on_reclaim_or_buffer,
                feature_params=fp,
                rule_params=rule_params,
                gate=gate,
            )

            eq_down_only = strategy_equity_from_mask(close, is_down)

            cagr_down_only = cagr_from_equity(eq_down_only)
            cagr_in_down = cagr_in_mask(close, is_down)

            blend = (w_down_only * cagr_down_only) + (w_inpos * cagr_in_down)
            score = -blend  # maximize this => make blend as negative as possible

            rows.append(
                {
                    "entry_len": int(en),
                    "exit_len": int(ex),
                    "score": float(score),
                    "blend": float(blend),
                    "CAGR_down_only": float(cagr_down_only),
                    "CAGR_in_down": float(cagr_in_down),
                    "Time_in_down": float(is_down.shift(1).fillna(False).mean()),
                    "Trades": trades_from_mask(is_down),
                    "Eq_final_down_only": (
                        float(eq_down_only.iloc[-1])
                        if len(eq_down_only)
                        else float("nan")
                    ),
                    "MaxDD_down_only": max_drawdown(eq_down_only),
                }
            )

    out = pd.DataFrame(rows).sort_values("score", ascending=False)
    return out


# ============================================================
# Plotting (shade DOWN)
# ============================================================


def shade_runs(
    ax,
    idx: pd.Index,
    mask: pd.Series,
    *,
    color: str,
    alpha: float = 0.18,
    label: str | None = None,
):
    m = mask.reindex(idx).fillna(False).astype(bool).to_numpy()
    in_run = False
    start = None
    for i in range(len(m)):
        if m[i] and not in_run:
            in_run = True
            start = idx[i]
        if in_run and ((not m[i]) or i == len(m) - 1):
            end = idx[i] if m[i] else idx[i - 1]
            ax.axvspan(start, end, alpha=alpha, color=color, label=label)
            label = None
            in_run = False


def plot_down_regime(close: pd.Series, is_down: pd.Series, *, ma_len: int = 200):
    close = as_series(close, "Close")
    ma = close.rolling(ma_len, min_periods=ma_len).mean()

    fig, ax = plt.subplots(figsize=(13, 6))
    ax.set_yscale("log")
    ax.plot(close.index, close.values, label="Close")
    ax.plot(ma.index, ma.values, label=f"SMA{ma_len}")

    shade_runs(ax, close.index, is_down, color="red", alpha=0.16, label="DOWN regime")

    ax.set_title("DOWN regime (shaded)")
    ax.legend()
    ax.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.4)
    plt.show()


# ============================================================
# Main
# ============================================================


def main():
    df = yf.download("GLD", period="max", auto_adjust=True, progress=False)
    close = as_series(df["Close"], "Close")

    rule_params = dict(
        slope_min=0.0,
        ma_buffer=0.005,
    )

    base_feature_params = dict(
        ma_len=200,
        slope_lookback=20,
        entry_len=200,  # overwritten
        exit_len=90,  # overwritten
    )

    entry_lens = list(range(50, 301, 10))  # 50..300
    exit_lens = list(range(50, 151, 10))  # 50..150

    res = sweep_down_regime_2d(
        close,
        base_feature_params=base_feature_params,
        rule_params=rule_params,
        entry_lens=entry_lens,
        exit_lens=exit_lens,
        gate="BM",
        w_down_only=0.5,
        w_inpos=0.5,
    )

    print(res.head(20).to_string(index=False))

    best = res.iloc[0].to_dict()
    print("\nBEST (maximize true-down negativity):")
    for k in [
        "entry_len",
        "exit_len",
        "score",
        "blend",
        "CAGR_down_only",
        "CAGR_in_down",
        "Time_in_down",
        "Trades",
        "MaxDD_down_only",
    ]:
        print(f"{k}: {best[k]}")

    # Rebuild best and plot it
    fp = dict(base_feature_params)
    fp["entry_len"] = int(best["entry_len"])
    fp["exit_len"] = int(best["exit_len"])

    is_down_best = decision_gated_state_anysignal(
        close,
        enter_rule=enter_breakdown_ma_slope,
        exit_rule=exit_down_on_reclaim_or_buffer,
        feature_params=fp,
        rule_params=rule_params,
        gate="BM",
    )

    plot_down_regime(close, is_down_best, ma_len=base_feature_params["ma_len"])

    res.to_csv("gld_down_regime_sweep.csv", index=False)
    print("Saved: gld_down_regime_sweep.csv")


if __name__ == "__main__":
    main()

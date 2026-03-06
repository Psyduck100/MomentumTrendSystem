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
    plt.yscale("log")  # <-- add this line

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
# Exit & entry lengths
# ----------------------------
import numpy as np
import pandas as pd


def cagr_from_equity(equity: pd.Series, periods_per_year: int = 252) -> float:
    equity = equity.dropna()
    if len(equity) < 2:
        return np.nan
    years = len(equity) / periods_per_year
    return float(equity.iloc[-1] ** (1 / years) - 1)


def max_drawdown(equity: pd.Series) -> float:
    equity = equity.dropna()
    peak = equity.cummax()
    dd = equity / peak - 1.0
    return float(dd.min()) if len(dd) else np.nan


def in_position_cagr(
    close: pd.Series, is_up: pd.Series, periods_per_year: int = 252
) -> float:
    """
    Annualized geometric return conditional on being in position.
    Uses only the days where is_up.shift(1) == True (executed next day).
    """
    rets = close.pct_change().fillna(0.0)
    held = is_up.shift(1).fillna(False)
    held_rets = rets[held]
    if len(held_rets) < 2:
        return np.nan
    equity = (1 + held_rets).cumprod()
    years = len(held_rets) / periods_per_year
    return float(equity.iloc[-1] ** (1 / years) - 1)


def backtest_long_only(
    close: pd.Series, is_up: pd.Series, fee_bps: float = 0.0
) -> pd.Series:
    """
    Long-only GLD when is_up is True (executed next day).
    fee_bps is applied per unit turnover.
    """
    close = close.sort_index()
    rets = close.pct_change().fillna(0.0)

    pos = is_up.shift(1).fillna(False).astype(float)  # execute next day
    gross = pos * rets

    if fee_bps > 0:
        turnover = pos.diff().abs().fillna(0.0)
        net = gross - (fee_bps / 1e4) * turnover
    else:
        net = gross

    return (1 + net).cumprod()


def sweep_exit_lengths(
    close: pd.Series,
    *,
    enter_rule,
    exit_rule,
    base_feature_params: dict,
    rule_params: dict,
    exit_lens=(40, 60, 80, 100, 120),
    fee_bps: float = 0.0,
) -> pd.DataFrame:
    rows = []
    for ex in exit_lens:
        feature_params = dict(base_feature_params)
        feature_params["exit_len"] = ex

        is_up = uptrend_state_machine(
            close,
            enter_rule=enter_rule,
            exit_rule=exit_rule,
            feature_params=feature_params,
            rule_params=rule_params,
        )

        equity = backtest_long_only(close, is_up, fee_bps=fee_bps)

        rows.append(
            {
                "exit_len": ex,
                "CAGR": cagr_from_equity(equity),
                "MaxDD": max_drawdown(equity),
                "Time_in_pos": float(is_up.mean()),
                "CAGR_in_pos": in_position_cagr(close, is_up),
                "Trades": int((is_up.astype(int).diff().fillna(0) == 1).sum()),
            }
        )

    out = pd.DataFrame(rows).sort_values("CAGR", ascending=False)
    return out


def sweep_entry_lengths(
    close: pd.Series,
    *,
    enter_rule,
    exit_rule,
    base_feature_params: dict,
    rule_params: dict,
    entry_lens=(60, 90, 120, 150, 200, 252),
    fee_bps: float = 0.0,
) -> pd.DataFrame:
    rows = []
    for en in entry_lens:
        feature_params = dict(base_feature_params)
        feature_params["entry_len"] = en

        is_up = uptrend_state_machine(
            close,
            enter_rule=enter_rule,
            exit_rule=exit_rule,
            feature_params=feature_params,
            rule_params=rule_params,
        )

        equity = backtest_long_only(close, is_up, fee_bps=fee_bps)

        rows.append(
            {
                "entry_len": en,
                "CAGR": cagr_from_equity(equity),
                "MaxDD": max_drawdown(equity),
                "Time_in_pos": float(is_up.mean()),
                "CAGR_in_pos": in_position_cagr(close, is_up),
                "Trades": int((is_up.astype(int).diff().fillna(0) == 1).sum()),
            }
        )

    return pd.DataFrame(rows).sort_values("CAGR", ascending=False)


###----------------------------MONTHLY EVAL------###
def pure_monthly_is_up(
    close: pd.Series,
    *,
    enter_rule: EnterRule,
    exit_rule: ExitRule,
    feature_params_months: Dict[str, Any],
    rule_params: Dict[str, Any] | None = None,
    gate: str = "BM",
) -> pd.Series:
    close = close.sort_index()
    close_m = close.resample(gate).last().dropna()

    is_up_m = uptrend_state_machine(
        close_m,
        enter_rule=enter_rule,
        exit_rule=exit_rule,
        feature_params=feature_params_months,  # lengths are MONTHS
        rule_params=rule_params,
    )

    return is_up_m.reindex(close.index).ffill().fillna(False).astype(bool)


def decision_gated_is_up_strict(
    close: pd.Series | pd.DataFrame,
    *,
    enter_rule: EnterRule,
    exit_rule: ExitRule,
    feature_params: Dict[str, Any],
    rule_params: Dict[str, Any] | None = None,
    gate: str = "BM",
) -> pd.Series:
    close_s = as_series(close, "Close").sort_index()
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

    for i in range(len(close_s)):
        if not bool(ready.iat[i]):
            is_up.iat[i] = False
            in_up = False
            continue

        if is_gate_day[i]:
            if (not in_up) and bool(enter.iat[i]):
                in_up = True
            elif in_up and bool(exit_.iat[i]):
                in_up = False

        is_up.iat[i] = in_up

    return is_up


def decision_gated_is_up_anysignal(
    close: pd.Series | pd.DataFrame,
    *,
    enter_rule: EnterRule,
    exit_rule: ExitRule,
    feature_params: Dict[str, Any],
    rule_params: Dict[str, Any] | None = None,
    gate: str = "BM",
) -> pd.Series:
    close_s = as_series(close, "Close").sort_index()
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


###----SWEEP---###
def sweep_entry_all_variants(
    close: pd.Series,
    *,
    enter_rule: EnterRule,
    exit_rule: ExitRule,
    rule_params: dict,
    entry_lens_days: list[int],
    fee_bps: float = 2.0,
    gate: str = "BM",
    # Daily params that stay constant while sweeping entry_len (in DAYS)
    base_days_params: dict,
    # Monthly params baseline (in MONTHS) — we'll overwrite entry_len only
    base_months_params: dict,
    trading_days_per_month: int = 21,
    months_rounding: str = "round",  # "round" | "floor" | "ceil"
) -> pd.DataFrame:
    def days_to_months(d: int) -> int:
        x = d / trading_days_per_month
        if months_rounding == "floor":
            m = int(np.floor(x))
        elif months_rounding == "ceil":
            m = int(np.ceil(x))
        else:
            m = int(np.round(x))
        return max(1, m)

    rows = []

    for en_days in entry_lens_days:
        # ---- 1) Daily (no gating) ----
        p_days = dict(base_days_params)
        p_days["entry_len"] = en_days

        is_up_daily = uptrend_state_machine(
            close,
            enter_rule=enter_rule,
            exit_rule=exit_rule,
            feature_params=p_days,
            rule_params=rule_params,
        )
        eq = backtest_long_only(close, is_up_daily, fee_bps=fee_bps)
        rows.append(
            {
                "variant": "daily",
                "entry_len_days": en_days,
                "entry_len_months": np.nan,
                "CAGR": cagr_from_equity(eq),
                "MaxDD": max_drawdown(eq),
                "Time_in_pos": float(is_up_daily.mean()),
                "CAGR_in_pos": in_position_cagr(close, is_up_daily),
                "Trades": int((is_up_daily.astype(int).diff().fillna(0) == 1).sum()),
            }
        )

        # ---- 2) Strict gated ----
        is_up_strict = decision_gated_is_up_strict(
            close,
            enter_rule=enter_rule,
            exit_rule=exit_rule,
            feature_params=p_days,
            rule_params=rule_params,
            gate=gate,
        )
        eq = backtest_long_only(close, is_up_strict, fee_bps=fee_bps)
        rows.append(
            {
                "variant": f"gated_strict_{gate}",
                "entry_len_days": en_days,
                "entry_len_months": np.nan,
                "CAGR": cagr_from_equity(eq),
                "MaxDD": max_drawdown(eq),
                "Time_in_pos": float(is_up_strict.mean()),
                "CAGR_in_pos": in_position_cagr(close, is_up_strict),
                "Trades": int((is_up_strict.astype(int).diff().fillna(0) == 1).sum()),
            }
        )

        # ---- 3) Any-signal gated ----
        is_up_any = decision_gated_is_up_anysignal(
            close,
            enter_rule=enter_rule,
            exit_rule=exit_rule,
            feature_params=p_days,
            rule_params=rule_params,
            gate=gate,
        )
        eq = backtest_long_only(close, is_up_any, fee_bps=fee_bps)
        rows.append(
            {
                "variant": f"gated_any_{gate}",
                "entry_len_days": en_days,
                "entry_len_months": np.nan,
                "CAGR": cagr_from_equity(eq),
                "MaxDD": max_drawdown(eq),
                "Time_in_pos": float(is_up_any.mean()),
                "CAGR_in_pos": in_position_cagr(close, is_up_any),
                "Trades": int((is_up_any.astype(int).diff().fillna(0) == 1).sum()),
            }
        )

        # ---- 4) Pure monthly bars (convert days -> months) ----
        en_months = days_to_months(en_days)
        p_m = dict(base_months_params)
        p_m["entry_len"] = en_months

        is_up_pure = pure_monthly_is_up(
            close,
            enter_rule=enter_rule,
            exit_rule=exit_rule,
            feature_params_months=p_m,
            rule_params=rule_params,
            gate=gate,
        )
        eq = backtest_long_only(close, is_up_pure, fee_bps=fee_bps)
        rows.append(
            {
                "variant": f"pure_monthly_{gate}",
                "entry_len_days": en_days,
                "entry_len_months": en_months,
                "CAGR": cagr_from_equity(eq),
                "MaxDD": max_drawdown(eq),
                "Time_in_pos": float(is_up_pure.mean()),
                "CAGR_in_pos": in_position_cagr(close, is_up_pure),
                "Trades": int((is_up_pure.astype(int).diff().fillna(0) == 1).sum()),
            }
        )

    out = pd.DataFrame(rows)
    return out.sort_values(["variant", "CAGR"], ascending=[True, False])


def build_is_up_variant(
    variant: str,
    close: pd.Series,
    *,
    enter_rule,
    exit_rule,
    rule_params: dict,
    # daily params (DAYS)
    ma_len_days: int,
    slope_lookback_days: int,
    entry_len_days: int,
    exit_len_days: int,
    gate: str = "BM",
    # monthly params (MONTHS) for pure monthly variant
    ma_len_months: int = 10,
    slope_lookback_months: int = 1,
    entry_len_months: int | None = None,
    exit_len_months: int | None = None,
) -> pd.Series:
    if variant == "daily":
        feature_params = dict(
            ma_len=ma_len_days,
            slope_lookback=slope_lookback_days,
            entry_len=entry_len_days,
            exit_len=exit_len_days,
        )
        return uptrend_state_machine(
            close,
            enter_rule=enter_rule,
            exit_rule=exit_rule,
            feature_params=feature_params,
            rule_params=rule_params,
        )

    if variant == "gated_any_BM":
        feature_params = dict(
            ma_len=ma_len_days,
            slope_lookback=slope_lookback_days,
            entry_len=entry_len_days,
            exit_len=exit_len_days,
        )
        return decision_gated_is_up_anysignal(
            close,
            enter_rule=enter_rule,
            exit_rule=exit_rule,
            feature_params=feature_params,
            rule_params=rule_params,
            gate=gate,
        )
    if variant == "pure_monthly_BM":
        if entry_len_months is None or exit_len_months is None:
            raise ValueError(
                "For pure_monthly_BM, provide entry_len_months and exit_len_months."
            )

        feature_params_m = dict(
            ma_len=ma_len_months,
            slope_lookback=slope_lookback_months,
            entry_len=entry_len_months,
            exit_len=exit_len_months,
        )

        return pure_monthly_is_up(
            close,
            enter_rule=enter_rule,
            exit_rule=exit_rule,
            feature_params_months=feature_params_m,  # <-- correct kwarg
            rule_params=rule_params,
            gate=gate,  # <-- pass gate too, since your function supports it
        )

    raise ValueError(f"Unknown variant: {variant}")


def sweep_exit_for_top_entries(
    close: pd.Series,
    top_by_variant: dict,
    *,
    enter_rule,
    exit_rule,
    rule_params: dict,
    fee_bps: float = 2.0,
    gate: str = "BM",
    # daily baseline (DAYS)
    ma_len_days: int = 200,
    slope_lookback_days: int = 20,
    # monthly baseline (MONTHS) for pure monthly
    ma_len_months: int = 10,
    slope_lookback_months: int = 1,
    trading_days_per_month: int = 21,
    months_rounding: str = "round",  # "round"|"floor"|"ceil"
    exit_lens_days=(40, 60, 80, 90, 100, 120, 140, 160),
) -> pd.DataFrame:
    def days_to_months(d: int) -> int:
        x = d / trading_days_per_month
        if months_rounding == "floor":
            m = int(np.floor(x))
        elif months_rounding == "ceil":
            m = int(np.ceil(x))
        else:
            m = int(np.round(x))
        return max(1, m)

    rows = []

    for variant, best in top_by_variant.items():
        best_entry_days = int(best["entry_len_days"])

        # For pure monthly, best entry should also have a month value if available
        best_entry_months = best.get("entry_len_months", None)
        if variant == "pure_monthly_BM":
            # If your top row didn’t store months, derive it:
            if best_entry_months is None or (
                isinstance(best_entry_months, float) and np.isnan(best_entry_months)
            ):
                best_entry_months = days_to_months(best_entry_days)
            else:
                best_entry_months = int(best_entry_months)

        for ex_days in exit_lens_days:
            if variant == "pure_monthly_BM":
                ex_months = days_to_months(ex_days)
                is_up = build_is_up_variant(
                    variant,
                    close,
                    enter_rule=enter_rule,
                    exit_rule=exit_rule,
                    rule_params=rule_params,
                    ma_len_days=ma_len_days,
                    slope_lookback_days=slope_lookback_days,
                    entry_len_days=best_entry_days,
                    exit_len_days=ex_days,
                    gate=gate,
                    ma_len_months=ma_len_months,
                    slope_lookback_months=slope_lookback_months,
                    entry_len_months=best_entry_months,
                    exit_len_months=ex_months,
                )
                entry_months = best_entry_months
            else:
                is_up = build_is_up_variant(
                    variant,
                    close,
                    enter_rule=enter_rule,
                    exit_rule=exit_rule,
                    rule_params=rule_params,
                    ma_len_days=ma_len_days,
                    slope_lookback_days=slope_lookback_days,
                    entry_len_days=best_entry_days,
                    exit_len_days=ex_days,
                    gate=gate,
                )
                ex_months = np.nan
                entry_months = np.nan

            equity = backtest_long_only(close, is_up, fee_bps=fee_bps)

            rows.append(
                {
                    "variant": variant,
                    "best_entry_len_days": best_entry_days,
                    "best_entry_len_months": entry_months,
                    "exit_len_days": ex_days,
                    "exit_len_months": ex_months,
                    "CAGR": cagr_from_equity(equity),
                    "MaxDD": max_drawdown(equity),
                    "Time_in_pos": float(is_up.shift(1).fillna(False).mean()),
                    "CAGR_in_pos": in_position_cagr(close, is_up),
                    "Trades": int(
                        (
                            is_up.shift(1).fillna(False).astype(int).diff().fillna(0)
                            == 1
                        ).sum()
                    ),
                }
            )

    out = pd.DataFrame(rows).sort_values(["variant", "CAGR"], ascending=[True, False])
    return out


def get_top_by_variant(
    res: pd.DataFrame, variants=("daily", "gated_any_BM", "pure_monthly_BM")
) -> dict:
    top = {}
    for v in variants:
        dfv = res[res["variant"] == v].copy()
        best = dfv.sort_values("CAGR", ascending=False).iloc[0].to_dict()
        top[v] = best
    return top


###--------DOWN TERRITORY--------###


# ----------------------------
# Main
# ----------------------------


def main():
    df = yf.download("GLD", period="max", auto_adjust=True, progress=False)
    close = as_series(df["Close"], "Close")

    ###------------------EXIT SWEEP---------------------#

    # base_feature_params = dict(
    #     ma_len=200, slope_lookback=20, entry_len=260, exit_len=60
    # )
    # rule_params = dict(slope_min=0.0, ma_buffer=0.005)

    # results = sweep_exit_lengths(
    #     close,
    #     enter_rule=enter_breakout_ma_slope,
    #     exit_rule=exit_donchian_or_ma_buffer,
    #     base_feature_params=base_feature_params,
    #     rule_params=rule_params,
    #     exit_lens=[50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150],
    #     fee_bps=2.0,  # add a realistic cost
    # )
    # print(results)

    ###------------------ENTRY SWEEP---------------------#

    # base_feature_params = dict(
    #     ma_len=200,
    #     slope_lookback=20,
    #     entry_len=200,  # will be overwritten
    #     exit_len=90,  # lock this as baseline
    # )

    # rule_params = dict(
    #     slope_min=0.0,
    #     ma_buffer=0.005,
    # )

    # results_en = sweep_entry_lengths(
    #     close,
    #     enter_rule=enter_breakout_ma_slope,
    #     exit_rule=exit_donchian_or_ma_buffer,
    #     base_feature_params=base_feature_params,
    #     rule_params=rule_params,
    #     entry_lens=[i * 10 + 50 for i in range(30)],  # 50, 60, ..., 150
    #     fee_bps=2.0,
    # )

    # print(results_en)

    ###----------------------------------ENTER TEST------------------------------------#

    # TRUE monthly evaluation => lengths are in MONTHS, not days
    rule_params = dict(slope_min=0.0, ma_buffer=0.005)

    base_days_params = dict(
        ma_len=200,
        slope_lookback=20,
        entry_len=260,  # overwritten in sweep
        exit_len=90,
    )

    # Pure monthly baseline (MONTHS)
    base_months_params = dict(
        ma_len=10,  # ~200d
        slope_lookback=1,  # ~20d (you can set 2 if you want smoother)
        entry_len=12,  # overwritten in sweep (converted)
        exit_len=4,  # ~90d
    )

    entry_lens_days = [50 + 10 * i for i in range(30)]  # 50..340

    res = sweep_entry_all_variants(
        close,
        enter_rule=enter_breakout_ma_slope,
        exit_rule=exit_donchian_or_ma_buffer,
        rule_params=rule_params,
        entry_lens_days=entry_lens_days,
        fee_bps=2.0,
        gate="BM",
        base_days_params=base_days_params,
        base_months_params=base_months_params,
        trading_days_per_month=21,
        months_rounding="round",
    )

    print(res)

    # Optional: save
    res.to_csv("gld_entry_sweep_all_variants.csv", index=False)
    print("Saved: gld_entry_sweep_all_variants.csv")
    ###----------------------------------EXIT TEST------------------------------------#
    top = get_top_by_variant(res, variants=("daily", "gated_any_BM", "pure_monthly_BM"))
    exit_sweep = sweep_exit_for_top_entries(
        close,
        top,
        enter_rule=enter_breakout_ma_slope,
        exit_rule=exit_donchian_or_ma_buffer,
        rule_params=rule_params,
        fee_bps=2.0,
        gate="BM",
        exit_lens_days=[50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150],
    )
    print(exit_sweep.to_string(index=False))
    exit_sweep.to_csv("gld_exit_sweep_top_entries.csv", index=False)
    print("Saved: gld_exit_sweep_top_entries.csv")


if __name__ == "__main__":
    main()

from __future__ import annotations

from typing import Callable, Dict, Any, Iterable
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
    ma_len: int = 200,
    slope_lookback: int = 20,
    entry_len: int = 90,
    exit_len: int = 90,
) -> Dict[str, pd.Series]:
    close = as_series(close, "Close")

    ma = close.rolling(ma_len, min_periods=ma_len).mean()
    ma_slope = (ma / ma.shift(slope_lookback)) - 1.0

    # Donchian channels (yesterday’s channel)
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
      close < MA AND MA slope < -slope_min AND close < prior channel low
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
      close > prior channel high OR close > MA*(1 + ma_buffer)
    """
    ma_buffer = float(params.get("ma_buffer", 0.005))
    exit_ = (close > feat["ch_high_entry"]) | (close > feat["ma"] * (1.0 + ma_buffer))
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
    gate: str = "BME",
    # feature base (days)
    ma_lens: Iterable[int] = (200,),
    slope_lookback: int = 20,
    # sweep grids
    exit_len: int = 90,
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
    for ma_len in ma_lens:
        for slope_min in slope_mins:
            for ma_buffer in ma_buffers:
                feature_params = dict(
                    ma_len=int(ma_len),
                    slope_lookback=int(slope_lookback),
                    entry_len=int(exit_len),
                    exit_len=int(exit_len),
                )
                rule_params = dict(
                    slope_min=float(slope_min), ma_buffer=float(ma_buffer)
                )

                is_down = decision_gated_state_anysignal(
                    close,
                    enter_rule=enter_breakdown_ma_slope,
                    exit_rule=exit_down_on_reclaim_or_buffer,
                    feature_params=feature_params,
                    rule_params=rule_params,
                    gate=gate,
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
                        "ma_len": int(ma_len),
                        "slope_lookback": slope_lookback,
                        "entry_len": int(exit_len),
                        "exit_len": int(exit_len),
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


def main():
    df = yf.download("GLD", period="max", auto_adjust=True, progress=False)
    close = as_series(df["Close"], "Close")

    # ---- Objective knobs (tweak here) ----
    fwd_h = 63  # ~3 months
    lambda_time = 0.05
    lambda_trades = 0.01

    res = sweep_down_regime(
        close,
        gate="BME",
        ma_lens=(80, 100, 120, 150, 180, 200, 220, 250),
        slope_lookback=20,
        exit_len=90,
        slope_mins=(0.0, 0.0025, 0.005),
        ma_buffers=(0.0, 0.005, 0.01),
        fwd_horizon_days=fwd_h,
        lambda_time=lambda_time,
        lambda_trades_per_year=lambda_trades,
    )

    # Top 25
    cols = [
        "score",
        "mean_fwd_ret",
        "median_fwd_ret",
        "time_in_down",
        "trades_per_year",
        "n_entries_used",
        "entry_len",
        "exit_len",
        "slope_min",
        "ma_buffer",
    ]
    print(res[cols].head(25).to_string(index=False))

    out_csv = "gld_down_regime_optimize.csv"
    res.to_csv(out_csv, index=False)
    print(f"\nSaved: {out_csv}")

    best = res.iloc[0].to_dict()
    print("\nBEST PARAMS:")
    for k in [
        "entry_len",
        "exit_len",
        "slope_min",
        "ma_buffer",
        "score",
        "mean_fwd_ret",
        "time_in_down",
        "trades_per_year",
        "n_entries_used",
    ]:
        print(f"{k}: {best[k]}")


if __name__ == "__main__":
    main()

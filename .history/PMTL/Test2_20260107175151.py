from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Literal, Dict, Any

import numpy as np
import pandas as pd
import yfinance as yf


# ----------------------------
# Helpers
# ----------------------------


def as_series(x: pd.Series | pd.DataFrame, name: str = "x") -> pd.Series:
    """Coerce Series or 1-column DataFrame into a Series."""
    if isinstance(x, pd.Series):
        return x
    if isinstance(x, pd.DataFrame):
        if x.shape[1] != 1:
            raise ValueError(f"Expected 1-column DataFrame for {name}, got {x.shape}")
        s = x.iloc[:, 0]
        s.name = s.name if s.name is not None else name
        return s
    raise TypeError(f"Expected Series/DataFrame for {name}, got {type(x)}")


def get_close_series(raw: pd.DataFrame) -> pd.Series:
    """
    Robustly extract Close as a Series from yfinance output.
    Handles both normal columns and MultiIndex columns.
    """
    if isinstance(raw.columns, pd.MultiIndex):
        # Layout A: level0 = OHLCV, level1 = ticker(s) -> raw["Close"] is DataFrame of tickers
        if "Close" in raw.columns.get_level_values(0):
            close = raw["Close"]
        # Layout B: level0 = ticker(s), level1 = OHLCV -> need xs on level 1
        elif "Close" in raw.columns.get_level_values(1):
            close = raw.xs("Close", level=1, axis=1)
        else:
            raise KeyError("Couldn't find 'Close' in MultiIndex columns.")
        return as_series(close, "Close").sort_index()

    # Normal single-index columns
    return as_series(raw["Close"], "Close").sort_index()


def _slice_dates(
    df: pd.DataFrame, start: Optional[str], end: Optional[str]
) -> pd.DataFrame:
    if start is not None:
        df = df.loc[df.index >= pd.to_datetime(start)]
    if end is not None:
        df = df.loc[df.index <= pd.to_datetime(end)]
    return df


# ----------------------------
# Config
# ----------------------------

PriceSource = Literal["close"]


@dataclass(frozen=True)
class PaperTrendConfig:
    n_up: int = 20
    n_dn: int = 40
    k: float = 2.0
    atr_proxy_scale: float = 1.4

    ema_span_up: Optional[int] = None
    ema_span_dn: Optional[int] = None

    use_no_lookahead: bool = True

    start: Optional[str] = None
    end: Optional[str] = None

    fee_bps: float = 0.0
    price_source: PriceSource = "close"


# ----------------------------
# Indicators
# ----------------------------


def donchian_up(price: pd.Series, n: int) -> pd.Series:
    return price.rolling(n, min_periods=n).max()


def donchian_down(price: pd.Series, n: int) -> pd.Series:
    return price.rolling(n, min_periods=n).min()


def keltner_mid(price: pd.Series, span: int) -> pd.Series:
    return price.ewm(span=span, adjust=False, min_periods=span).mean()


def atr_proxy_close_only(price: pd.Series, n: int, scale: float) -> pd.Series:
    abs_dp = price.diff().abs()
    avg_abs_dp = abs_dp.rolling(n, min_periods=n).mean()
    return scale * avg_abs_dp


def keltner_up(
    price: pd.Series, n: int, k: float, scale: float, ema_span: int
) -> pd.Series:
    mid = keltner_mid(price, ema_span)
    atrp = atr_proxy_close_only(price, n, scale)
    return mid + k * atrp


def keltner_down(
    price: pd.Series, n: int, k: float, scale: float, ema_span: int
) -> pd.Series:
    mid = keltner_mid(price, ema_span)
    atrp = atr_proxy_close_only(price, n, scale)
    return mid - k * atrp


# ----------------------------
# Core: bands + trailing stop + backtest
# ----------------------------


def compute_paper_bands(
    price: pd.Series | pd.DataFrame, cfg: PaperTrendConfig
) -> pd.DataFrame:
    # >>> FIX: ensure price is 1D Series <<<
    price = as_series(price, "price").astype(float).dropna().sort_index()

    n_up, n_dn = cfg.n_up, cfg.n_dn
    ema_up = cfg.ema_span_up or n_up
    ema_dn = cfg.ema_span_dn or n_dn

    d_up = donchian_up(price, n_up)
    d_dn = donchian_down(price, n_dn)

    k_up = keltner_up(price, n_up, cfg.k, cfg.atr_proxy_scale, ema_up)
    k_dn = keltner_down(price, n_dn, cfg.k, cfg.atr_proxy_scale, ema_dn)

    upper = pd.concat([d_up, k_up], axis=1).min(axis=1)
    lower = pd.concat([d_dn, k_dn], axis=1).max(axis=1)

    # All inputs below are Series => 1D, so DataFrame constructor is safe
    out = pd.DataFrame(
        {
            "price": price,
            "donchian_up": d_up,
            "donchian_dn": d_dn,
            "keltner_up": k_up,
            "keltner_dn": k_dn,
            "upper_band": upper,
            "lower_band": lower,
        },
        index=price.index,
    )
    return out


def backtest_paper_trend(bands: pd.DataFrame, cfg: PaperTrendConfig) -> pd.DataFrame:
    price = bands["price"]
    upper = bands["upper_band"]
    lower = bands["lower_band"]

    entry_level = upper.shift(1) if cfg.use_no_lookahead else upper
    entry_signal = price >= entry_level

    pos = pd.Series(0, index=price.index, dtype=int)
    stop = pd.Series(np.nan, index=price.index, dtype=float)
    exit_signal = pd.Series(False, index=price.index, dtype=bool)

    in_pos = False
    current_stop = np.nan

    for i in range(len(price)):
        if pd.isna(entry_level.iat[i]) or pd.isna(lower.iat[i]):
            pos.iat[i] = 0
            stop.iat[i] = np.nan
            in_pos = False
            current_stop = np.nan
            continue

        px = float(price.iat[i])
        lb = float(lower.iat[i])

        if not in_pos:
            if bool(entry_signal.iat[i]):
                in_pos = True
                current_stop = lb
            pos.iat[i] = 1 if in_pos else 0
            stop.iat[i] = current_stop if in_pos else np.nan
            continue

        current_stop = max(current_stop, lb)
        stop.iat[i] = current_stop

        if px <= current_stop:
            exit_signal.iat[i] = True
            in_pos = False
            pos.iat[i] = 1  # hold through today; off next day
        else:
            pos.iat[i] = 1

        if not in_pos:
            current_stop = np.nan

    rets = price.pct_change().fillna(0.0)
    pos_exec = pos.shift(1).fillna(0.0)
    gross = pos_exec * rets

    if cfg.fee_bps > 0:
        turnover = pos_exec.diff().abs().fillna(0.0)
        net = gross - (cfg.fee_bps / 1e4) * turnover
    else:
        net = gross

    equity = (1.0 + net).cumprod()

    out = bands.copy()
    out["entry_signal"] = entry_signal.fillna(False)
    out["exit_signal"] = exit_signal
    out["position"] = pos
    out["trailing_stop"] = stop
    out["equity"] = equity
    return out


# ----------------------------
# Plot
# ----------------------------


def plot_paper_trend(
    df: pd.DataFrame,
    title: str = "Paper-style UpperBand + TrailingStop (shaded in-position)",
):
    import matplotlib.pyplot as plt

    price = df["price"]
    upper = df["upper_band"]
    lower = df["lower_band"]
    stop = df["trailing_stop"]
    pos = df["position"].fillna(0).astype(int).to_numpy()

    plt.figure(figsize=(14, 6))
    plt.plot(price.index, price.values, label="Price")
    plt.plot(upper.index, upper.values, label="UpperBand")
    plt.plot(lower.index, lower.values, label="LowerBand")
    plt.plot(stop.index, stop.values, label="TrailingStop")

    idx = price.index
    in_run = False
    run_start = None
    for i in range(len(pos)):
        if pos[i] == 1 and not in_run:
            in_run = True
            run_start = idx[i]
        if in_run and ((pos[i] == 0) or i == len(pos) - 1):
            run_end = idx[i] if pos[i] == 1 else idx[i - 1]
            plt.axvspan(run_start, run_end, alpha=0.15)
            in_run = False

    plt.legend()
    plt.title(title)
    plt.show()


# ----------------------------
# Main
# ----------------------------


def main():
    ticker = "GLD"
    cfg = PaperTrendConfig(
        n_up=20,
        n_dn=40,
        k=2.0,
        atr_proxy_scale=1.4,
        fee_bps=2.0,
        start=None,
        end=None,
        use_no_lookahead=True,
    )

    raw = yf.download(ticker, period="max", auto_adjust=True, progress=False)
    raw = _slice_dates(raw, cfg.start, cfg.end)

    # >>> FIX: always extract Close as 1D Series <<<
    price = get_close_series(raw).dropna().astype(float)

    bands = compute_paper_bands(price, cfg)
    bt = backtest_paper_trend(bands, cfg)

    bt.to_csv("gld_paper_trend_debug.csv", index=True)

    plot_paper_trend(bt)

    time_in_pos = float((bt["position"] == 1).mean())
    n_entries = int((bt["position"].diff().fillna(0) == 1).sum())

    print("Time in position:", round(time_in_pos, 4))
    print("Entries:", n_entries)
    print("Equity final:", float(bt["equity"].iloc[-1]))


if __name__ == "__main__":
    main()

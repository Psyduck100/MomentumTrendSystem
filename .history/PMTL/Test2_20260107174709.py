from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Literal, Dict, Any, Tuple

import numpy as np
import pandas as pd
import yfinance as yf


# ----------------------------
# Config
# ----------------------------

PriceSource = Literal["close"]  # paper uses close-only; keep extensible


@dataclass(frozen=True)
class PaperTrendConfig:
    # Donchian/Keltner lookbacks (paper example: up=20, down=40)
    n_up: int = 20
    n_dn: int = 40

    # Keltner parameters (paper example: k=2 and uses ~1.4 scale factor for close-only ATR proxy)
    k: float = 2.0
    atr_proxy_scale: float = 1.4  # paper's long-run ratio approx

    # EMA type for "Keltner midline"
    ema_span_up: Optional[int] = None  # if None, uses n_up
    ema_span_dn: Optional[int] = None  # if None, uses n_dn

    # Execution convention: evaluate signals at day t, trade on day t+1
    # Entry rule in paper: Price(t) >= UpperBand(t-1)
    # Exit rule: Price(t) <= TrailingStop(t) (we apply to decide next day's position)
    use_no_lookahead: bool = True

    # Optional date window (None = full)
    start: Optional[str] = None
    end: Optional[str] = None

    # Fees (simple turnover cost, optional)
    fee_bps: float = 0.0

    # Price source (paper close-only)
    price_source: PriceSource = "close"


# ----------------------------
# Helpers
# ----------------------------


def _slice_dates(
    df: pd.DataFrame, start: Optional[str], end: Optional[str]
) -> pd.DataFrame:
    if start is not None:
        df = df.loc[df.index >= pd.to_datetime(start)]
    if end is not None:
        df = df.loc[df.index <= pd.to_datetime(end)]
    return df


def donchian_up(price: pd.Series, n: int) -> pd.Series:
    """Highest close over last n days (includes today unless you shift outside)."""
    return price.rolling(n, min_periods=n).max()


def donchian_down(price: pd.Series, n: int) -> pd.Series:
    """Lowest close over last n days (includes today unless you shift outside)."""
    return price.rolling(n, min_periods=n).min()


def keltner_mid(price: pd.Series, span: int) -> pd.Series:
    """EMA midline."""
    return price.ewm(span=span, adjust=False, min_periods=span).mean()


def atr_proxy_close_only(price: pd.Series, n: int, scale: float) -> pd.Series:
    """
    Close-only ATR proxy used in the paper:
      approx_ATR_n(t) = scale * (1/n) * sum_{i=t-n+1..t} |ΔPrice(i)|
    where ΔPrice(i) = Price(i) - Price(i-1)
    """
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
# Core: build bands + trailing stop + position
# ----------------------------


def compute_paper_bands(price: pd.Series, cfg: PaperTrendConfig) -> pd.DataFrame:
    """
    Paper formulas:
      UpperBand(t) = min(DonchianUp(n_up), KeltnerUp(n_up,k))
      LowerBand(t) = max(DonchianDown(n_dn), KeltnerDown(n_dn,k))
    """
    n_up, n_dn = cfg.n_up, cfg.n_dn
    ema_up = cfg.ema_span_up or n_up
    ema_dn = cfg.ema_span_dn or n_dn

    d_up = donchian_up(price, n_up)
    d_dn = donchian_down(price, n_dn)

    k_up = keltner_up(price, n_up, cfg.k, cfg.atr_proxy_scale, ema_up)
    k_dn = keltner_down(price, n_dn, cfg.k, cfg.atr_proxy_scale, ema_dn)

    upper = pd.concat([d_up, k_up], axis=1).min(axis=1)
    lower = pd.concat([d_dn, k_dn], axis=1).max(axis=1)

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
    """
    Implements:
      Entry: Price(t) >= UpperBand(t-1)  (no lookahead)
      Trailing stop ratchet: Stop(t+1) = max(Stop(t), LowerBand(t))
      Exit: Price(t) <= Stop(t)  (decide to be out next day)

    Returns DataFrame with columns:
      position (0/1), trailing_stop, entry_signal, exit_signal, equity
    """
    price = bands["price"]
    upper = bands["upper_band"]
    lower = bands["lower_band"]

    if cfg.use_no_lookahead:
        entry_level = upper.shift(1)
    else:
        entry_level = upper

    entry_signal = price >= entry_level

    # State machine
    pos = pd.Series(0, index=price.index, dtype=int)
    stop = pd.Series(np.nan, index=price.index, dtype=float)
    exit_signal = pd.Series(False, index=price.index, dtype=bool)

    in_pos = False
    current_stop = np.nan

    for i in range(len(price)):
        t = price.index[i]

        # need bands available to operate
        if pd.isna(entry_level.iat[i]) or pd.isna(lower.iat[i]):
            pos.iat[i] = 0
            stop.iat[i] = np.nan
            in_pos = False
            current_stop = np.nan
            continue

        px = float(price.iat[i])
        lb = float(lower.iat[i])

        if not in_pos:
            # enter?
            if bool(entry_signal.iat[i]):
                in_pos = True
                # initialize trailing stop at LowerBand(t) (conservative)
                current_stop = lb
            pos.iat[i] = 1 if in_pos else 0
            stop.iat[i] = current_stop if in_pos else np.nan
            continue

        # in position: update stop (ratchet up; never loosens)
        # Paper: Stop(t+1) = max(Stop(t), LowerBand(t))
        current_stop = max(current_stop, lb)
        stop.iat[i] = current_stop

        # exit check using today's close vs today's stop (decide for next day)
        if px <= current_stop:
            exit_signal.iat[i] = True
            in_pos = False
            # position is considered OFF for next day; but for today we're still holding (close-to-close convention)
            pos.iat[i] = 1
        else:
            pos.iat[i] = 1

        if not in_pos:
            # reset stop after exit
            current_stop = np.nan

    # Trading returns (execute next day): position decided at t is held during t+1
    rets = price.pct_change().fillna(0.0)
    pos_exec = pos.shift(1).fillna(0.0)  # next-day execution
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

    # shade in-position
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
# Main demo
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

    price = raw["Close"].dropna().astype(float)

    bands = compute_paper_bands(price, cfg)
    bt = backtest_paper_trend(bands, cfg)

    # Export if you want
    bt.to_csv("gld_paper_trend_debug.csv")

    plot_paper_trend(bt)

    # Quick summary
    time_in_pos = float((bt["position"] == 1).mean())
    n_entries = int(((bt["position"].diff().fillna(0) == 1).sum()))
    print("Time in position:", round(time_in_pos, 4))
    print("Entries:", n_entries)
    print("Equity final:", float(bt["equity"].iloc[-1]))


if __name__ == "__main__":
    main()

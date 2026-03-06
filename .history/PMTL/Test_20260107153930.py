from __future__ import annotations
import numpy as np
import pandas as pd


def compute_uptrend(
    close: pd.Series,
    ma_len: int = 200,
    slope_lookback: int = 20,
    slope_min: float = 0.0,  # 0.0 means MA must be non-decreasing over lookback
    entry_len: int = 200,
) -> pd.Series:
    """
    Returns a boolean Series 'is_uptrend' evaluated on day t using close[t] only.
    No-lookahead (Donchian uses yesterday's channel high).
    """

    ma = close.rolling(ma_len, min_periods=ma_len).mean()

    # MA slope over slope_lookback days (percentage change)
    ma_slope = (ma / ma.shift(slope_lookback)) - 1.0

    # Donchian high (yesterday's channel high)
    ch_high = close.shift(1).rolling(entry_len, min_periods=entry_len).max()

    is_up = (close > ma) & (ma_slope > slope_min) & (close > ch_high)

    # optional: avoid NaN/early periods
    is_up = is_up.fillna(False)
    return is_up


def uptrend_periods(is_up: pd.Series) -> pd.DataFrame:
    """
    Convert boolean uptrend series into start/end date ranges for easy checking.
    """
    x = is_up.astype(int)
    changes = x.diff().fillna(0)

    starts = is_up.index[changes == 1]
    ends = is_up.index[changes == -1]

    # handle if series starts in uptrend
    if len(starts) == 0 and is_up.iloc[0]:
        starts = starts.insert(0, is_up.index[0])
    elif is_up.iloc[0] and (len(starts) == 0 or starts[0] != is_up.index[0]):
        starts = starts.insert(0, is_up.index[0])

    # handle if series ends in uptrend
    if is_up.iloc[-1]:
        ends = ends.append(pd.Index([is_up.index[-1]]))

    periods = pd.DataFrame({"start": starts, "end": ends})
    return periods


# --- OPTIONAL quick plot helper (matplotlib only) ---
def plot_uptrend(close: pd.Series, is_up: pd.Series, ma_len: int = 200):
    import matplotlib.pyplot as plt

    ma = close.rolling(ma_len, min_periods=ma_len).mean()

    plt.figure(figsize=(12, 6))
    plt.plot(close.index, close.values, label="Close")
    plt.plot(ma.index, ma.values, label=f"SMA{ma_len}")

    # shade uptrend regions
    up = is_up.fillna(False).values
    idx = close.index

    # find contiguous True runs
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
    plt.title("Uptrend detector (shaded)")
    plt.show()












































def main():
    # Example usage
    import matplotlib.pyplot as plt

    # Load example data (replace with your own data source)
    dates = pd.date_range(start="2020-01-01",
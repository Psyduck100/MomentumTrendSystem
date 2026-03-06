from __future__ import annotations

import numpy as np
import pandas as pd
import yfinance as yf


def _to_series(x: pd.Series | pd.DataFrame, name: str = "series") -> pd.Series:
    """Coerce a 1D Series out of either a Series or a 1-column DataFrame."""
    if isinstance(x, pd.Series):
        return x
    if isinstance(x, pd.DataFrame):
        if x.shape[1] == 1:
            s = x.iloc[:, 0]
            s.name = s.name if s.name is not None else name
            return s
        raise ValueError(f"Expected 1-column DataFrame for {name}, got shape {x.shape}")
    raise TypeError(f"Expected Series/DataFrame for {name}, got {type(x)}")


def _get_close_series(df: pd.DataFrame, ticker: str) -> pd.Series:
    """Extract close as a Series from yfinance output (handles MultiIndex columns)."""
    if isinstance(df.columns, pd.MultiIndex):
        # Possible layouts:
        # 1) level0 = OHLCV, level1 = ticker(s)  -> df["Close"] gives DataFrame of tickers
        # 2) level0 = ticker(s), level1 = OHLCV  -> need xs("Close", level=1)
        if "Close" in df.columns.get_level_values(0):
            close = df["Close"]  # Series if single ticker, else DataFrame
        elif "Close" in df.columns.get_level_values(1):
            close = df.xs("Close", level=1, axis=1)
        else:
            raise KeyError("Couldn't find 'Close' in MultiIndex columns.")
        close = _to_series(close, name="Close")
        return close

    # Normal (single-index columns)
    close = df["Close"]
    close = _to_series(close, name="Close")
    return close


def compute_uptrend(
    close: pd.Series | pd.DataFrame,
    ma_len: int = 200,
    slope_lookback: int = 20,
    slope_min: float = 0.0,  # 0.0 => MA must be non-decreasing over lookback
    entry_len: int = 200,
) -> pd.Series:
    """
    Returns boolean Series 'is_uptrend' evaluated on day t using close[t] only.
    No-lookahead (Donchian uses yesterday's channel high).
    """
    close = _to_series(close, name="Close").sort_index()

    ma = close.rolling(ma_len, min_periods=ma_len).mean()

    # MA slope over slope_lookback days (percentage change)
    ma_slope = (ma / ma.shift(slope_lookback)) - 1.0

    # Donchian high (yesterday's channel high)
    ch_high = close.shift(1).rolling(entry_len, min_periods=entry_len).max()

    is_up = (close > ma) & (ma_slope > slope_min) & (close > ch_high)
    return is_up.fillna(False)


def uptrend_periods(is_up: pd.Series | pd.DataFrame) -> pd.DataFrame:
    """Convert boolean uptrend series into start/end date ranges."""
    is_up = _to_series(is_up, name="is_up").astype(bool)

    x = is_up.astype(np.int8)
    changes = x.diff().fillna(0)

    starts = is_up.index[changes == 1]
    ends = is_up.index[changes == -1]

    # If series starts in uptrend, prepend start
    if is_up.iloc[0]:
        starts = pd.Index([is_up.index[0]]).append(starts)

    # If series ends in uptrend, append end
    if is_up.iloc[-1]:
        ends = ends.append(pd.Index([is_up.index[-1]]))

    # Safety: align lengths if anything weird happens
    n = min(len(starts), len(ends))
    starts, ends = starts[:n], ends[:n]

    return pd.DataFrame({"start": starts, "end": ends})


def plot_uptrend(close: pd.Series, is_up: pd.Series, ma_len: int = 200):
    import matplotlib.pyplot as plt

    close = _to_series(close, name="Close").sort_index()
    is_up = (
        _to_series(is_up, name="is_up")
        .reindex(close.index, fill_value=False)
        .astype(bool)
    )

    ma = close.rolling(ma_len, min_periods=ma_len).mean()

    plt.figure(figsize=(12, 6))
    plt.plot(close.index, close.values, label="Close")
    plt.plot(ma.index, ma.values, label=f"SMA{ma_len}")

    # shade uptrend regions
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
    plt.title("Uptrend detector (shaded)")
    plt.show()


def main():
    ticker = "GLD"
    df = yf.download(ticker, auto_adjust=True, progress=False)
    close = _get_close_series(df, ticker)

    is_up = compute_uptrend(
        close, ma_len=200, slope_lookback=20, slope_min=0.0, entry_len=200
    )
    periods = uptrend_periods(is_up)

    print(periods.tail(20))
    plot_uptrend(close, is_up, ma_len=200)


if __name__ == "__main__":
    main()

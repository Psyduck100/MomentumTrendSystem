from __future__ import annotations

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


def compute_uptrend(
    close: pd.Series | pd.DataFrame,
    ma_len: int = 200,
    slope_lookback: int = 20,
    slope_min: float = 0.0,
    entry_len: int = 200,
) -> pd.Series:
    close = as_series(close, "Close").sort_index()

    ma = close.rolling(ma_len, min_periods=ma_len).mean()
    ma_slope = (ma / ma.shift(slope_lookback)) - 1.0
    ch_high = close.shift(1).rolling(entry_len, min_periods=entry_len).max()

    is_up = (close > ma) & (ma_slope > slope_min) & (close > ch_high)
    return is_up.fillna(False).astype(bool)


def uptrend_periods(is_up: pd.Series | pd.DataFrame) -> pd.DataFrame:
    is_up = as_series(is_up, "is_up").astype(bool)
    x = is_up.astype(np.int8)
    changes = x.diff().fillna(0)

    # IMPORTANT: build masks as numpy arrays / Series, not DataFrames
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
    plt.title("Uptrend detector (shaded)")
    plt.show()


def main():
    df = yf.download("GLD", period="max", auto_adjust=True, progress=False)

    close = as_series(df["Close"], "Close")

    is_up = compute_uptrend(
        close, ma_len=200, slope_lookback=20, slope_min=0.0, entry_len=200
    )
    periods = uptrend_periods(is_up)

    # ---- EXPORTS ----
    out_csv = "gld_uptrend_periods.csv"
    periods.to_csv(out_csv, index=False)

    # optional: also save a readable text file
    out_txt = "gld_uptrend_periods.txt"
    with open(out_txt, "w", encoding="utf-8") as f:
        f.write(periods.to_string(index=False))

    print(f"Saved: {out_csv}")
    print(f"Saved: {out_txt}")

    plot_uptrend(close, is_up, ma_len=200)


if __name__ == "__main__":
    main()

"""
SMURF-style backtest (Daily decisioning)
Step 1: Trend gate (channel breakout) -> SPY when trend ON, IEF when trend OFF
Optional Step 2: Mean reversion sleeve (RSI(2) dip buys) when trend ON
Optional Step 3: Calendar anomaly sleeve (turn-of-month) when trend ON

- Uses daily close data (yfinance auto_adjust=True).
- Avoids lookahead by applying today's signals to *tomorrow's* returns.
- Includes simple transaction costs via turnover.

Install:
  pip install yfinance pandas numpy

Run:
  python smurf_backtest.py
"""

import numpy as np
import pandas as pd
import yfinance as yf


# -----------------------------
# Data
# -----------------------------
def download_prices(tickers, start="2004-01-01", end=None):
    px = yf.download(
        tickers=tickers,
        start=start,
        end=end,
        auto_adjust=True,
        progress=False,
        threads=True,
    )["Close"]

    if isinstance(px, pd.Series):
        px = px.to_frame()

    px = px.dropna(how="all")
    px = px.ffill().dropna()
    px.index = pd.to_datetime(px.index)
    return px


# -----------------------------
# Indicators
# -----------------------------
def ema(s: pd.Series, span: int) -> pd.Series:
    return s.ewm(span=span, adjust=False).mean()


def rsi(close: pd.Series, n: int = 14) -> pd.Series:
    delta = close.diff()
    up = delta.clip(lower=0.0)
    down = (-delta).clip(lower=0.0)
    roll_up = up.ewm(alpha=1 / n, adjust=False).mean()
    roll_down = down.ewm(alpha=1 / n, adjust=False).mean()
    rs = roll_up / roll_down.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def donchian_bands(close: pd.Series, n: int):
    upper = close.rolling(n).max()
    lower = close.rolling(n).min()
    return upper, lower


def keltner_bands_close_only(close: pd.Series, n: int, k: float):
    """
    Close-only Keltner approximation:
      ATR ~= 1.4 * mean(|Δclose|, n)
      Upper = EMA(close, n) + 1.4 * k * mean_abs_change
      Lower = EMA(close, n) - 1.4 * k * mean_abs_change
    """
    mean_abs_change = close.diff().abs().rolling(n).mean()
    mid = ema(close, span=n)
    upper = mid + 1.4 * k * mean_abs_change
    lower = mid - 1.4 * k * mean_abs_change
    return upper, lower


def channel_breakout_trend_state(close: pd.Series) -> pd.DataFrame:
    """
    Trend ON/OFF using a channel breakout + monotone trailing stop:

      UpperBand = min(DonchianHigh(20), KeltnerUpper(20,2))
      Enter trend when close[t] >= UpperBand[t-1]

      LowerBand = max(DonchianLow(40), KeltnerLower(40,2))
      TrailingStop is monotone:
         TS[t] = max(TS[t-1], LowerBand[t]) while trend is ON
      Exit trend when close[t] <= TS[t-1]

    Returns a dataframe with:
      trend_state in {0,1}, trailing_stop, upper_band, lower_band
    """
    don_up20, _ = donchian_bands(close, 20)
    kel_up20, _ = keltner_bands_close_only(close, n=20, k=2.0)
    upper_band = pd.concat([don_up20, kel_up20], axis=1).min(axis=1)

    _, don_dn40 = donchian_bands(close, 40)
    _, kel_dn40 = keltner_bands_close_only(close, n=40, k=2.0)
    lower_band = pd.concat([don_dn40, kel_dn40], axis=1).max(axis=1)

    entry = close >= upper_band.shift(1)

    trend = pd.Series(0, index=close.index, dtype=int)
    trailing_stop = pd.Series(np.nan, index=close.index, dtype=float)

    in_trend = False
    ts = np.nan

    for i, t in enumerate(close.index):
        lb = lower_band.iloc[i]
        if np.isnan(lb):
            trend.iloc[i] = 0
            trailing_stop.iloc[i] = np.nan
            continue

        if not in_trend:
            # If not in trend, we haven't "armed" a trailing stop yet.
            # Only turn on when entry triggers.
            if entry.iloc[i]:
                in_trend = True
                ts = lb  # initialize stop at lower band
        else:
            # Update monotone trailing stop (never decreases)
            ts = max(ts, lb)
            # Exit if price crosses yesterday's trailing stop
            if close.iloc[i] <= (trailing_stop.iloc[i - 1] if i > 0 else ts):
                in_trend = False
                ts = np.nan

        trend.iloc[i] = 1 if in_trend else 0
        trailing_stop.iloc[i] = ts if in_trend else np.nan

    return pd.DataFrame(
        {
            "upper_band": upper_band,
            "lower_band": lower_band,
            "trailing_stop": trailing_stop,
            "trend_state": trend,
        },
        index=close.index,
    )


# -----------------------------
# Calendar anomaly helper
# -----------------------------
def turn_of_month_window(index: pd.DatetimeIndex, after_days: int = 3) -> pd.Series:
    """
    True on:
      - last trading day of each month
      - first `after_days` trading days of the next month
    """
    df = pd.DataFrame(index=index)
    df["date"] = df.index
    df["ym"] = df["date"].dt.to_period("M")

    # last trading day per month
    last_day = df.groupby("ym")["date"].max()
    last_day_set = set(pd.to_datetime(last_day.values))
    is_last = df["date"].isin(last_day_set)

    # first N trading days per month
    df["day_in_month"] = df.groupby("ym").cumcount() + 1
    is_first_n = df["day_in_month"] <= after_days

    return (is_last | is_first_n).astype(bool)


# -----------------------------
# SMURF-like weights builder
# -----------------------------
def build_smurf_weights(
    prices: pd.DataFrame,
    equity="SPY",
    safe="IEF",
    use_mean_reversion=True,
    use_calendar=True,
    core_weight=0.85,
    mr_weight=0.15,
    cal_weight=0.10,
    rsi_n=2,
    rsi_entry=10,
    rsi_exit=50,
    mr_max_hold=5,
    cal_after_days=3,
):
    """
    Returns weights dataframe with columns [equity, safe].

    Core idea:
      - If trend OFF: 100% safe
      - If trend ON: core_weight in equity + optional MR sleeve + optional calendar sleeve
      - MR sleeve: enter on oversold, exit on rebound/time/trend-off
      - Calendar sleeve: add weight during turn-of-month window (only when trend ON)
      - Caps equity weight at 1.0 (no leverage)
    """
    px_eq = prices[equity]
    trend_df = channel_breakout_trend_state(px_eq)
    trend = trend_df["trend_state"].fillna(0).astype(int)

    # Mean reversion signal state machine
    rsi2 = rsi(px_eq, n=rsi_n)
    mr_pos = pd.Series(0, index=prices.index, dtype=int)
    hold_days = 0
    in_mr = False

    for i, t in enumerate(prices.index):
        if trend.iloc[i] == 0:
            in_mr = False
            hold_days = 0
            mr_pos.iloc[i] = 0
            continue

        # trend ON
        if not use_mean_reversion:
            mr_pos.iloc[i] = 0
            continue

        if not in_mr:
            if rsi2.iloc[i] < rsi_entry:
                in_mr = True
                hold_days = 0
        else:
            hold_days += 1
            # exit on rebound, time stop
            if (rsi2.iloc[i] > rsi_exit) or (hold_days >= mr_max_hold):
                in_mr = False
                hold_days = 0

        mr_pos.iloc[i] = 1 if in_mr else 0

    # Calendar window (turn-of-month)
    cal_on = (
        turn_of_month_window(prices.index, after_days=cal_after_days)
        if use_calendar
        else pd.Series(False, index=prices.index)
    )

    w = pd.DataFrame(0.0, index=prices.index, columns=[equity, safe])

    for i, t in enumerate(prices.index):
        if trend.iloc[i] == 0:
            w.loc[t, safe] = 1.0
            continue

        # trend ON
        eq_w = core_weight

        # Add MR sleeve only when MR signal is ON
        if use_mean_reversion and mr_pos.iloc[i] == 1:
            eq_w += mr_weight

        # Add calendar sleeve only during window
        if use_calendar and cal_on.iloc[i]:
            eq_w += cal_weight

        eq_w = min(1.0, eq_w)  # cap (no leverage)
        w.loc[t, equity] = eq_w
        w.loc[t, safe] = 1.0 - eq_w

    extras = pd.DataFrame(
        {
            "trend_state": trend,
            "mr_state": mr_pos,
            "cal_state": cal_on.astype(int),
            "rsi": rsi2,
        },
        index=prices.index,
    )
    return w, trend_df, extras


# -----------------------------
# Backtest engine
# -----------------------------
def backtest_from_weights(
    prices: pd.DataFrame,
    weights: pd.DataFrame,
    cost_bps=5.0,
    trading_days=252,
):
    """
    - Uses close-to-close returns.
    - Applies weights with a 1-day delay to avoid lookahead:
        portfolio_ret[t] uses weights[t-1] * returns[t]
    - Transaction cost charged on rebal days:
        cost = (cost_rate) * turnover
        turnover = 0.5 * sum(|w_t - w_{t-1}|)

    Returns dict with returns, equity_curve, metrics, turnover series.
    """
    rets = prices.pct_change().fillna(0.0)

    w = weights.copy().reindex(rets.index).fillna(method="ffill").fillna(0.0)

    # Turnover based on today's target vs yesterday's target
    dw = w.diff().abs().fillna(0.0)
    turnover = 0.5 * dw.sum(axis=1)

    cost_rate = cost_bps / 10000.0
    costs = cost_rate * turnover

    # Apply yesterday's weights to today's returns
    port_ret_gross = (w.shift(1).fillna(0.0) * rets).sum(axis=1)
    port_ret = port_ret_gross - costs

    equity = (1.0 + port_ret).cumprod()

    # Metrics
    n = len(port_ret)
    years = n / trading_days if trading_days > 0 else np.nan
    cagr = equity.iloc[-1] ** (1 / years) - 1 if years and years > 0 else np.nan
    vol = port_ret.std() * np.sqrt(trading_days)
    sharpe = (
        (port_ret.mean() * trading_days) / (port_ret.std() * np.sqrt(trading_days))
        if port_ret.std() > 0
        else np.nan
    )

    peak = equity.cummax()
    dd = equity / peak - 1.0
    max_dd = dd.min()

    avg_turnover = turnover.mean() * trading_days

    # Exposure to equity asset(s)
    exposure = w.iloc[:, 0].mean()  # assumes first column is equity

    metrics = {
        "CAGR": float(cagr),
        "Volatility": float(vol),
        "Sharpe": float(sharpe),
        "MaxDrawdown": float(max_dd),
        "AvgAnnualTurnover": float(avg_turnover),
        "AvgEquityWeight": float(exposure),
        "EndingEquity": float(equity.iloc[-1]),
    }

    return {
        "returns": port_ret,
        "equity": equity,
        "drawdown": dd,
        "turnover": turnover,
        "costs": costs,
        "metrics": metrics,
    }


# -----------------------------
# Main
# -----------------------------
if __name__ == "__main__":
    # --- Choose tickers ---
    EQUITY = "SPY"
    SAFE = "IEF"

    # --- Download data ---
    prices = download_prices([EQUITY, SAFE], start="2004-01-01")

    # --- Step 1 only (trend gate) ---
    # Set MR/Calendar off to validate baseline behavior first.
    weights_step1, trend_df, extras = build_smurf_weights(
        prices,
        equity=EQUITY,
        safe=SAFE,
        use_mean_reversion=False,
        use_calendar=False,
        core_weight=1.0,  # 100% equity when trend ON
        mr_weight=0.0,
        cal_weight=0.0,
    )
    bt1 = backtest_from_weights(prices[[EQUITY, SAFE]], weights_step1, cost_bps=5.0)
    print("\n=== STEP 1: Trend gate only ===")
    for k, v in bt1["metrics"].items():
        print(f"{k:>16}: {v:.4f}")

    # --- Full “starter SMURF” (trend + MR + calendar) ---
    weights_smurf, trend_df2, extras2 = build_smurf_weights(
        prices,
        equity=EQUITY,
        safe=SAFE,
        use_mean_reversion=True,
        use_calendar=True,
        core_weight=0.85,
        mr_weight=0.15,
        cal_weight=0.10,
        rsi_n=2,
        rsi_entry=10,
        rsi_exit=50,
        mr_max_hold=5,
        cal_after_days=3,
    )
    bt = backtest_from_weights(prices[[EQUITY, SAFE]], weights_smurf, cost_bps=5.0)
    print("\n=== STARTER SMURF: Trend + MR + Calendar ===")
    for k, v in bt["metrics"].items():
        print(f"{k:>16}: {v:.4f}")

    # --- Save outputs for inspection ---
    out = pd.concat(
        [
            prices[[EQUITY, SAFE]].rename(
                columns={EQUITY: "px_equity", SAFE: "px_safe"}
            ),
            trend_df2.add_prefix("trend_"),
            extras2.add_prefix("extra_"),
            weights_smurf.add_prefix("w_"),
            bt["returns"].rename("port_ret"),
            bt["equity"].rename("equity_curve"),
            bt["drawdown"].rename("drawdown"),
            bt["turnover"].rename("turnover"),
            bt["costs"].rename("costs"),
        ],
        axis=1,
    )
    out.to_csv("smurf_backtest_output.csv")
    print("\nWrote: smurf_backtest_output.csv")

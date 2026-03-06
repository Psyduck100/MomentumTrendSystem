from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Iterable, Optional, Dict, Any, Tuple

import numpy as np
import pandas as pd
import yfinance as yf


# ----------------------------
# Metrics
# ----------------------------
def max_drawdown(equity: pd.Series) -> float:
    peak = equity.cummax()
    dd = equity / peak - 1.0
    return float(dd.min())


def cagr_from_equity(equity: pd.Series) -> float:
    equity = equity.dropna()
    if len(equity) < 2:
        return np.nan
    years = (equity.index[-1] - equity.index[0]).days / 365.25
    return float((equity.iloc[-1] / equity.iloc[0]) ** (1 / years) - 1)


def sharpe_annualized_monthly(rets_m: pd.Series) -> float:
    r = rets_m.dropna()
    if len(r) < 24:
        return np.nan
    sd = r.std(ddof=1)
    if sd == 0:
        return np.nan
    return float((r.mean() / sd) * np.sqrt(12))


def turnover_rate(holdings: pd.Series) -> float:
    """Fraction of months where the held ticker changed at rebalance."""
    h = holdings.dropna().astype(str)
    if len(h) < 2:
        return np.nan
    changes = (h != h.shift(1)).sum()
    return float(changes / (len(h) - 1))


# ----------------------------
# Data
# ----------------------------
def download_prices(
    tickers: Iterable[str],
    start_date: str,
    end_date: Optional[str] = None,
    auto_adjust: bool = True,
) -> pd.DataFrame:
    tickers = list(dict.fromkeys([t.strip().upper() for t in tickers if str(t).strip()]))
    if not tickers:
        raise ValueError("No tickers provided.")

    if end_date is None:
        end_date = datetime.now().strftime("%Y-%m-%d")

    data = yf.download(
        tickers=tickers,
        start=start_date,
        end=end_date,
        auto_adjust=auto_adjust,
        progress=False,
        threads=True,
    )

    # Normalize to a simple columns=tickers DataFrame of prices
    if isinstance(data.columns, pd.MultiIndex):
        fields0 = data.columns.get_level_values(0)
        if "Close" in fields0:
            px = data["Close"]
        elif "Adj Close" in fields0:
            px = data["Adj Close"]
        else:
            swapped = data.swaplevel(0, 1, axis=1).sort_index(axis=1)
            fields1 = swapped.columns.get_level_values(1)
            if "Close" in fields1:
                px = swapped.xs("Close", level=1, axis=1)
            elif "Adj Close" in fields1:
                px = swapped.xs("Adj Close", level=1, axis=1)
            else:
                raise ValueError("Could not find Close/Adj Close in yfinance response.")
    else:
        # single ticker
        if "Close" in data.columns:
            px = pd.DataFrame({tickers[0]: data["Close"]})
        elif "Adj Close" in data.columns:
            px = pd.DataFrame({tickers[0]: data["Adj Close"]})
        else:
            raise ValueError("Could not find Close/Adj Close for single-ticker response.")

    # require all tickers present after alignment
    px = px.dropna(how="all").ffill().dropna(how="any")
    px.index = pd.to_datetime(px.index)
    px = px.reindex(columns=[t for t in tickers if t in px.columns])
    return px


def to_monthly_prices(daily_prices: pd.DataFrame) -> pd.DataFrame:
    """
    Month-end prices from daily (last trading day in month).
    IMPORTANT: drops the last month if it's not yet complete (partial month).
    """
    monthly = daily_prices.resample("ME").last().dropna(how="all")

    if len(monthly) == 0:
        return monthly

    last_daily = pd.Timestamp(daily_prices.index[-1]).normalize()
    last_label = pd.Timestamp(monthly.index[-1]).normalize()  # calendar month-end label

    # If we are not actually at month-end yet, the last monthly bin is partial -> drop it
    if last_daily < last_label:
        monthly = monthly.iloc[:-1]

    return monthly


# ----------------------------
# Momentum logic
# ----------------------------
def compute_monthly_returns(monthly_prices: pd.DataFrame) -> pd.DataFrame:
    return monthly_prices.pct_change().dropna(how="all")


def compute_lookback_returns(
    monthly_prices: pd.DataFrame, months: Tuple[int, int]
) -> Dict[int, pd.DataFrame]:
    """Returns dict {lookback_months: returns_df}."""
    out = {}
    for m in months:
        out[m] = monthly_prices.pct_change(m)
    return out


def rank_universe_on_date(
    lookback_rets: Dict[int, pd.DataFrame],
    weights: Tuple[float, float],
    universe: list[str],
    asof: pd.Timestamp,
) -> pd.DataFrame:
    """
    Returns DataFrame: ticker, score, ret_6m, ret_12m, abs_pass, rank
    """
    m1, m2 = sorted(list(lookback_rets.keys()))  # expect (6,12)
    w1, w2 = weights

    r1 = lookback_rets[m1].loc[asof, universe]
    r2 = lookback_rets[m2].loc[asof, universe]
    score = (w1 * r1) + (w2 * r2)

    df = pd.DataFrame(
        {
            "ticker": universe,
            "score": pd.to_numeric(score, errors="coerce").values,
            f"ret_{m1}m": pd.to_numeric(r1, errors="coerce").values,
            f"ret_{m2}m": pd.to_numeric(r2, errors="coerce").values,
        }
    ).dropna()

    df["abs_pass"] = df[f"ret_{m2}m"] > 0
    df = df.sort_values("score", ascending=False).reset_index(drop=True)
    df["rank"] = np.arange(1, len(df) + 1)
    return df


def pick_monthly_holding(
    rank_table: pd.DataFrame,
    defensive: str,
    abs_months: int = 12,
) -> str:
    if rank_table.empty:
        return defensive
    top = rank_table.iloc[0]
    return str(top["ticker"]) if bool(top["abs_pass"]) else defensive


def build_monthly_holdings(
    monthly_prices: pd.DataFrame,
    universe: list[str],
    defensive: str,
    lookbacks: Tuple[int, int] = (6, 12),
    weights: Tuple[float, float] = (0.5, 0.5),
) -> Tuple[pd.Series, Dict[pd.Timestamp, pd.DataFrame]]:
    """
    Returns:
      holdings_next: Series indexed by month-end (signal date), value=ticker to hold NEXT month
      rank_tables: dict[asof_date] -> rank_table DataFrame
    """
    lookback_rets = compute_lookback_returns(monthly_prices, lookbacks)

    max_lb = max(lookbacks)
    valid_idx = monthly_prices.index[max_lb:]  # ensures lookback returns exist

    holdings = []
    dates = []
    rank_tables = {}

    for asof in valid_idx:
        rt = rank_universe_on_date(lookback_rets, weights, universe, asof)
        rank_tables[asof] = rt
        pick = pick_monthly_holding(rt, defensive=defensive, abs_months=max_lb)
        dates.append(asof)
        holdings.append(pick)

    holdings_next = pd.Series(holdings, index=pd.DatetimeIndex(dates), name="hold_next_month")
    return holdings_next, rank_tables


# ----------------------------
# Backtest
# ----------------------------
def backtest_monthly_rotation(
    monthly_prices: pd.DataFrame,
    holdings_next: pd.Series,
    cost_bps: float = 0.0,
) -> Dict[str, Any]:
    """
    No-lookahead:
      - signal at month-end t -> choose holding for (t -> t+1)
      - apply holding to next month return by shifting 1.
    """
    rets_m = compute_monthly_returns(monthly_prices)

    # Apply holding chosen at t-1 to return over month t
    hold_applied = holdings_next.shift(1).reindex(rets_m.index)

    # Drop months before we have a holding
    valid = hold_applied.dropna().index
    rets_m = rets_m.loc[valid]
    hold_applied = hold_applied.loc[valid]

    # Portfolio return each month from held ticker
    port_ret = pd.Series(index=valid, dtype=float)
    for tkr in hold_applied.unique():
        mask = (hold_applied == tkr)
        if tkr not in rets_m.columns:
            raise ValueError(f"Holding ticker {tkr} not in monthly returns columns.")
        port_ret.loc[mask] = rets_m.loc[mask, tkr]

    port_ret = port_ret.fillna(0.0).rename("port_ret")

    # Transaction costs on switches (applied on months where holding changes)
    if cost_bps and cost_bps > 0:
        cost = cost_bps / 10000.0
        switched = (hold_applied != hold_applied.shift(1)).fillna(False)
        port_ret = (port_ret - switched.astype(float) * cost).rename("port_ret")

    equity = (1.0 + port_ret).cumprod().rename("equity")

    summary = pd.DataFrame(
        {
            "CAGR": [cagr_from_equity(equity)],
            "Sharpe(0rf)": [sharpe_annualized_monthly(port_ret)],
            "MaxDD": [max_drawdown(equity)],
            "Turnover": [turnover_rate(hold_applied)],
            "Months": [len(port_ret)],
        },
        index=["blend_6_12_top1_abs12_else_DEF"],
    )

    out = pd.concat([hold_applied.rename("holding"), port_ret, equity], axis=1)
    return {
        "out": out,
        "summary": summary,
        "monthly_returns": rets_m,
        "holdings_applied": hold_applied,
    }


def spy_buyhold_same_window(prices_monthly: pd.DataFrame, idx: pd.DatetimeIndex) -> pd.DataFrame:
    """
    SPY buy & hold summary over the exact index window used by the strategy.
    """
    rets_m = prices_monthly.pct_change().dropna(how="all")
    spy_ret = rets_m.reindex(idx)["SPY"].fillna(0.0).rename("SPY_ret")
    spy_eq = (1.0 + spy_ret).cumprod().rename("SPY_equity")

    spy_summary = pd.DataFrame(
        {
            "CAGR": [cagr_from_equity(spy_eq)],
            "Sharpe(0rf)": [sharpe_annualized_monthly(spy_ret)],
            "MaxDD": [max_drawdown(spy_eq)],
            "Months": [len(spy_ret)],
        },
        index=["SPY_BH_same_window"],
    )
    return spy_summary


# ----------------------------
# Config + main
# ----------------------------
@dataclass(frozen=True)
class StrategyConfig:
    universe: Tuple[str, ...] = ("OEF", "QQQ", "SPY", "IWN", "VTI")
    defensive: str = "IEF"
    lookbacks: Tuple[int, int] = (6, 12)
    weights: Tuple[float, float] = (0.5, 0.5)
    start_date: str = "2000-01-01"
    end_date: Optional[str] = None
    cost_bps: float = 0.0


def run_backtest(cfg: StrategyConfig) -> Dict[str, Any]:
    tickers = list(cfg.universe) + [cfg.defensive]
    px_daily = download_prices(
        tickers, start_date=cfg.start_date, end_date=cfg.end_date, auto_adjust=True
    )
    px_monthly = to_monthly_prices(px_daily)

    # Common history across all tickers (no missing months)
    px_monthly = px_monthly.dropna(how="any")

    holdings_next, rank_tables = build_monthly_holdings(
        px_monthly,
        universe=list(cfg.universe),
        defensive=cfg.defensive,
        lookbacks=cfg.lookbacks,
        weights=cfg.weights,
    )

    bt = backtest_monthly_rotation(
        px_monthly,
        holdings_next=holdings_next,
        cost_bps=cfg.cost_bps,
    )

    return {
        "cfg": cfg,
        "prices_daily": px_daily,
        "prices_monthly": px_monthly,
        "holdings_next": holdings_next,
        "rank_tables": rank_tables,
        **bt,
    }


def main():
    cfg = StrategyConfig(
        universe=("QQQ", "SPY"),
        defensive="IEF",
        lookbacks=(6, 12),
        weights=(0.5, 0.5),
        start_date="2000-01-01",
        end_date=None,
        cost_bps=5.0,
    )

    res = run_backtest(cfg)

    print("=" * 70)
    print("US EQUITIES MOMENTUM (blend_6_12) - Monthly Rebalance Backtest")
    print("=" * 70)
    print(f"Universe: {list(cfg.universe)} | Defensive: {cfg.defensive}")
    print(f"Score: {cfg.weights} on {cfg.lookbacks} months | Abs filter: 12M > 0")
    print(f"Monthly rebalance | Cost: {cfg.cost_bps} bps on switches")
    print(
        f"Monthly history used: {res['prices_monthly'].index.min().date()} -> {res['prices_monthly'].index.max().date()}"
    )
    print()

    print(res["summary"].to_string())

    # SPY buy & hold over the exact same months as the strategy returns
    idx = res["out"].index
    spy_summary = spy_buyhold_same_window(res["prices_monthly"], idx)
    print("\n" + "=" * 70)
    print("SPY buy & hold (same window as strategy)")
    print("=" * 70)
    print(spy_summary.to_string())

    strat_cagr = float(res["summary"]["CAGR"].iloc[0])
    spy_cagr = float(spy_summary["CAGR"].iloc[0])
    print(f"\nCAGR difference (strategy - SPY): {strat_cagr - spy_cagr:.2%}")

    # Last 12 months
    tail = res["out"][["holding", "port_ret", "equity"]].tail(12).copy()
    tail["port_ret"] = tail["port_ret"].map(lambda x: f"{x:.2%}")
    tail["equity"] = tail["equity"].map(lambda x: f"{x:.3f}")
    print("\nLast 12 months:")
    print(tail.to_string())

    # Rank table for most recent signal month-end
    asof = res["holdings_next"].index.max()
    rt = res["rank_tables"][asof].copy()
    rt["score"] = rt["score"].map(lambda x: f"{x:.2%}")
    rt[f"ret_{cfg.lookbacks[0]}m"] = rt[f"ret_{cfg.lookbacks[0]}m"].map(lambda x: f"{x:.2%}")
    rt[f"ret_{cfg.lookbacks[1]}m"] = rt[f"ret_{cfg.lookbacks[1]}m"].map(lambda x: f"{x:.2%}")
    print(f"\nRank table as-of {asof.date()} (signal month-end):")
    print(rt.to_string(index=False))
    # ----------------------------
    # Holding time breakdown (months held per ticker)
    # ----------------------------
    hold = res["out"]["holding"].dropna().astype(str)

    counts = hold.value_counts().sort_values(ascending=False)
    pct = (counts / len(hold))

    held_summary = pd.DataFrame(
        {
            "months_held": counts,
            "pct_months": pct,
            "first_held": hold.groupby(hold).apply(lambda s: s.index.min()),
            "last_held": hold.groupby(hold).apply(lambda s: s.index.max()),
        }
    ).sort_values("months_held", ascending=False)

    # Pretty formatting
    held_summary["pct_months"] = held_summary["pct_months"].map(lambda x: f"{x:.1%}")
    held_summary["first_held"] = pd.to_datetime(held_summary["first_held"]).dt.date
    held_summary["last_held"] = pd.to_datetime(held_summary["last_held"]).dt.date

    print("\n" + "=" * 70)
    print("Months held by ticker (strategy holdings)")
    print("=" * 70)
    print(held_summary.to_string())
     # ----------------------------
    # Return contribution by held ticker (log-return attribution)
    # ----------------------------
    rets = res["out"]["port_ret"].dropna()
    hold_applied = res["out"]["holding"].reindex(rets.index).astype(str)

    # avg monthly return while held + count months
    by_ticker = rets.groupby(hold_applied).agg(months="count", avg_monthly_ret="mean")

    # Contribution to total compounded return using log(1+r)
    # (log-returns add up cleanly across time and partition by ticker)
    logret = np.log1p(rets)
    sum_logret = logret.groupby(hold_applied).sum().rename("sum_logret")

    by_ticker = by_ticker.join(sum_logret)

    total_log = by_ticker["sum_logret"].sum()
    by_ticker["pct_total_logret"] = by_ticker["sum_logret"] / total_log if total_log != 0 else np.nan

    # Pretty formatting
    by_ticker["avg_monthly_ret"] = by_ticker["avg_monthly_ret"].map(lambda x: f"{x:.2%}")
    by_ticker["pct_total_logret"] = by_ticker["pct_total_logret"].map(lambda x: f"{x:.1%}")

    print("\n" + "=" * 70)
    print("Return contribution by ticker (based on months held)")
    print("=" * 70)
    print(by_ticker.sort_values("sum_logret", ascending=False).to_string())

    # ----------------------------
    # Average holding streak length (how long you keep a ticker once selected)
    # ----------------------------
    h = res["out"]["holding"].dropna().astype(str)

    # Identify streak boundaries
    streak_id = (h != h.shift(1)).cumsum()
    streaks = pd.DataFrame({"ticker": h, "streak_id": streak_id})
    streak_lengths = streaks.groupby(["ticker", "streak_id"]).size().rename("streak_len").reset_index()

    streak_stats = streak_lengths.groupby("ticker")["streak_len"].agg(
        streaks="count",
        avg_streak="mean",
        median_streak="median",
        max_streak="max",
    ).sort_values("avg_streak", ascending=False)

    # Pretty formatting
    streak_stats["avg_streak"] = streak_stats["avg_streak"].map(lambda x: f"{x:.2f}")
    streak_stats["median_streak"] = streak_stats["median_streak"].map(lambda x: f"{x:.0f}")
    streak_stats["max_streak"] = streak_stats["max_streak"].map(lambda x: f"{x:.0f}")

    print("\n" + "=" * 70)
    print("Holding streak stats (months per continuous hold)")
    print("=" * 70)
    print(streak_stats.to_string())
if __name__ == "__main__":
    main()
    
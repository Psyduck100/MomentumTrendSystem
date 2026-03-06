from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Iterable, Optional, Dict, Any, Tuple, List

import numpy as np
import pandas as pd
import yfinance as yf


# ============================================================
# Metrics
# ============================================================
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
    """Fraction of months where the held ticker changed."""
    h = holdings.dropna().astype(str)
    if len(h) < 2:
        return np.nan
    changes = (h != h.shift(1)).sum()
    return float(changes / (len(h) - 1))


# ============================================================
# Data
# ============================================================
def download_prices(
    tickers: Iterable[str],
    start_date: str,
    end_date: Optional[str] = None,
    auto_adjust: bool = True,
) -> pd.DataFrame:
    tickers = list(dict.fromkeys([str(t).strip().upper() for t in tickers if str(t).strip()]))
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

    # Normalize to columns=tickers DataFrame of prices
    if isinstance(data.columns, pd.MultiIndex):
        lvl0 = data.columns.get_level_values(0)
        if "Close" in lvl0:
            px = data["Close"]
        elif "Adj Close" in lvl0:
            px = data["Adj Close"]
        else:
            swapped = data.swaplevel(0, 1, axis=1).sort_index(axis=1)
            lvl1 = swapped.columns.get_level_values(1)
            if "Close" in lvl1:
                px = swapped.xs("Close", level=1, axis=1)
            elif "Adj Close" in lvl1:
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

    px = px.dropna(how="all").ffill()
    px.index = pd.to_datetime(px.index)
    px = px.reindex(columns=[t for t in tickers if t in px.columns])
    return px


def to_monthly_prices(daily_prices: pd.DataFrame) -> pd.DataFrame:
    """
    Month-end prices from daily (last trading day in month).
    Drops the last month if it's not complete yet (partial month).
    """
    monthly = daily_prices.resample("ME").last().dropna(how="all")
    if len(monthly) == 0:
        return monthly

    last_daily = pd.Timestamp(daily_prices.index[-1]).normalize()
    last_label = pd.Timestamp(monthly.index[-1]).normalize()  # calendar month-end label

    # If we are not at (calendar) month-end yet, last monthly bin is partial -> drop it
    if last_daily < last_label:
        monthly = monthly.iloc[:-1]

    return monthly


# ============================================================
# Momentum logic (choose between QQQ vs VTI, else defensive)
# ============================================================
def compute_monthly_returns(monthly_prices: pd.DataFrame) -> pd.DataFrame:
    return monthly_prices.pct_change().dropna(how="all")


def compute_lookback_returns(
    monthly_prices: pd.DataFrame, months: Tuple[int, ...]
) -> Dict[int, pd.DataFrame]:
    out: Dict[int, pd.DataFrame] = {}
    for m in months:
        out[m] = monthly_prices.pct_change(m)
    return out


def rank_universe_on_date_blend(
    lookback_rets: Dict[int, pd.DataFrame],
    weights: Tuple[float, ...],
    universe: List[str],
    asof: pd.Timestamp,
) -> pd.DataFrame:
    """
    Returns a table with: ticker, score, ret_{lb}m..., abs_pass(12m), rank
    """
    lbs = sorted(list(lookback_rets.keys()))
    if len(lbs) != len(weights):
        raise ValueError("lookbacks and weights must have same length.")

    # Pull lookback returns for universe
    cols = {}
    score = None
    for lb, w in zip(lbs, weights):
        r = lookback_rets[lb].loc[asof, universe]
        r = pd.to_numeric(r, errors="coerce")
        cols[f"ret_{lb}m"] = r
        score = (w * r) if score is None else (score + w * r)

    df = pd.DataFrame({"ticker": universe, "score": score.values})
    for k, ser in cols.items():
        df[k] = ser.values

    df = df.dropna(subset=["score"]).copy()
    df = df.sort_values("score", ascending=False).reset_index(drop=True)
    df["rank"] = np.arange(1, len(df) + 1)

    # abs filter uses 12m if present, else uses max lookback
    abs_lb = 12 if "ret_12m" in df.columns else max(lbs)
    df["abs_pass"] = df[f"ret_{abs_lb}m"] > 0
    return df


def build_monthly_holdings_top1_with_absfilter(
    monthly_prices: pd.DataFrame,
    universe: List[str],
    defensive: str,
    lookbacks: Tuple[int, ...] = (6, 12),
    weights: Tuple[float, ...] = (0.5, 0.5),
    abs_months: int = 12,
) -> Tuple[pd.Series, Dict[pd.Timestamp, pd.DataFrame]]:
    """
    Each month-end t:
      - compute blend score across universe
      - pick top ticker
      - if top's abs_months return > 0 => hold top next month
        else hold defensive next month

    Returns:
      holdings_next: indexed by signal month-end, value=ticker to hold NEXT month
      rank_tables: dict[asof] -> rank table
    """
    lookback_rets = compute_lookback_returns(monthly_prices, lookbacks)

    max_lb = max(max(lookbacks), abs_months)
    valid_idx = monthly_prices.index[max_lb:]  # ensures lookback returns exist

    holdings: List[str] = []
    dates: List[pd.Timestamp] = []
    rank_tables: Dict[pd.Timestamp, pd.DataFrame] = {}

    for asof in valid_idx:
        rt = rank_universe_on_date_blend(lookback_rets, weights, universe, asof)
        rank_tables[asof] = rt

        if rt.empty:
            pick = defensive
        else:
            top = rt.iloc[0]
            # abs check: use abs_months if we computed it; else fallback to max lookback
            abs_col = f"ret_{abs_months}m"
            if abs_col in rt.columns:
                abs_val = float(top[abs_col])
                pick = str(top["ticker"]) if abs_val > 0 else defensive
            else:
                # fallback: use max lookback return sign
                fallback_col = f"ret_{max(lookbacks)}m"
                abs_val = float(top[fallback_col])
                pick = str(top["ticker"]) if abs_val > 0 else defensive

        dates.append(asof)
        holdings.append(pick)

    holdings_next = pd.Series(holdings, index=pd.DatetimeIndex(dates), name="hold_next_month")
    return holdings_next, rank_tables


# ============================================================
# Backtest
# ============================================================
def backtest_monthly_rotation(
    monthly_prices: pd.DataFrame,
    holdings_next: pd.Series,
    cost_bps: float = 0.0,
    strategy_name: str = "top1_blend_abs_else_DEF",
) -> Dict[str, Any]:
    """
    No-lookahead:
      - signal at month-end t chooses holding for month (t -> t+1)
      - apply holding to next month's return via shift(1)
    """
    rets_m = compute_monthly_returns(monthly_prices)

    hold_applied = holdings_next.shift(1).reindex(rets_m.index)

    valid = hold_applied.dropna().index
    rets_m = rets_m.loc[valid]
    hold_applied = hold_applied.loc[valid]

    port_ret = pd.Series(index=valid, dtype=float)
    for tkr in hold_applied.unique():
        mask = hold_applied == tkr
        if tkr not in rets_m.columns:
            raise ValueError(f"Holding ticker {tkr} not in monthly returns columns.")
        port_ret.loc[mask] = rets_m.loc[mask, tkr]

    port_ret = port_ret.fillna(0.0).rename("port_ret")

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
        index=[strategy_name],
    )

    out = pd.concat([hold_applied.rename("holding"), port_ret, equity], axis=1)
    return {"out": out, "summary": summary, "monthly_returns": rets_m, "holdings_applied": hold_applied}


def buyhold_same_window(prices_monthly: pd.DataFrame, idx: pd.DatetimeIndex, ticker: str) -> pd.DataFrame:
    rets_m = prices_monthly.pct_change().dropna(how="all")
    r = rets_m.reindex(idx)[ticker].fillna(0.0).rename(f"{ticker}_ret")
    eq = (1.0 + r).cumprod().rename(f"{ticker}_equity")
    return pd.DataFrame(
        {
            "CAGR": [cagr_from_equity(eq)],
            "Sharpe(0rf)": [sharpe_annualized_monthly(r)],
            "MaxDD": [max_drawdown(eq)],
            "Months": [len(r)],
        },
        index=[f"{ticker}_BH_same_window"],
    )


# ============================================================
# Config + Runner
# ============================================================
@dataclass(frozen=True)
class StrategyConfig:
    # THIS is what you asked for: choose between QQQ or VTI
    universe: Tuple[str, ...] = ("QQQ", "VTI")
    defensive: str = "IEF"
    lookbacks: Tuple[int, ...] = (6, 12)
    weights: Tuple[float, ...] = (0.5, 0.5)
    abs_months: int = 12
    start_date: str = "2000-01-01"
    end_date: Optional[str] = None
    cost_bps: float = 5.0
    benchmark: str = "SPY"


def run_backtest(cfg: StrategyConfig) -> Dict[str, Any]:
    tickers = list(cfg.universe) + [cfg.defensive, cfg.benchmark]
    px_daily = download_prices(tickers, start_date=cfg.start_date, end_date=cfg.end_date, auto_adjust=True)
    px_monthly = to_monthly_prices(px_daily)

    # Common history across all tickers (no missing months)
    px_monthly = px_monthly.dropna(how="any")

    holdings_next, rank_tables = build_monthly_holdings_top1_with_absfilter(
        px_monthly,
        universe=list(cfg.universe),
        defensive=cfg.defensive,
        lookbacks=cfg.lookbacks,
        weights=cfg.weights,
        abs_months=cfg.abs_months,
    )

    bt = backtest_monthly_rotation(
        px_monthly,
        holdings_next=holdings_next,
        cost_bps=cfg.cost_bps,
        strategy_name=f"top1_blend_{cfg.lookbacks}_abs{cfg.abs_months}_else_{cfg.defensive}",
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
        universe=("QQQ", "VTI", "SPY"),  # <-- choose between these two
        defensive="IEF",
        lookbacks=(6, 12),
        weights=(0.5, 0.5),
        abs_months=12,
        start_date="2000-01-01",
        end_date=None,
        cost_bps=5.0,
        benchmark="SPY",
    )

    res = run_backtest(cfg)

    print("=" * 70)
    print("US EQUITIES MOMENTUM (choose between QQQ vs VTI) - Monthly Backtest")
    print("=" * 70)
    print(f"Universe: {list(cfg.universe)} | Defensive: {cfg.defensive} | Benchmark: {cfg.benchmark}")
    print(f"Score: weights={cfg.weights} on lookbacks={cfg.lookbacks} months | Abs filter: {cfg.abs_months}M > 0")
    print(f"Monthly rebalance | Cost: {cfg.cost_bps} bps on switches")
    print(f"Monthly history used: {res['prices_monthly'].index.min().date()} -> {res['prices_monthly'].index.max().date()}")
    print()

    print(res["summary"].to_string())

    # Benchmark buy & hold over exact same months as strategy
    idx = res["out"].index
    bench_summary = buyhold_same_window(res["prices_monthly"], idx, cfg.benchmark)
    print("\n" + "=" * 70)
    print(f"{cfg.benchmark} buy & hold (same window as strategy)")
    print("=" * 70)
    print(bench_summary.to_string())

    strat_cagr = float(res["summary"]["CAGR"].iloc[0])
    bench_cagr = float(bench_summary["CAGR"].iloc[0])
    print(f"\nCAGR difference (strategy - {cfg.benchmark}): {strat_cagr - bench_cagr:.2%}")

    # Last 12 months
    tail = res["out"][["holding", "port_ret", "equity"]].tail(12).copy()
    tail["port_ret"] = tail["port_ret"].map(lambda x: f"{x:.2%}")
    tail["equity"] = tail["equity"].map(lambda x: f"{x:.3f}")
    print("\nLast 12 months:")
    print(tail.to_string())

    # Most recent rank table (signal at month-end)
    asof = res["holdings_next"].index.max()
    rt = res["rank_tables"][asof].copy()
    rt["score"] = rt["score"].map(lambda x: f"{x:.2%}")
    for lb in cfg.lookbacks:
        col = f"ret_{lb}m"
        if col in rt.columns:
            rt[col] = rt[col].map(lambda x: f"{x:.2%}")
    print(f"\nRank table as-of {asof.date()} (signal month-end):")
    print(rt.to_string(index=False))

    # Months held by ticker
    hold = res["out"]["holding"].dropna().astype(str)
    counts = hold.value_counts().sort_values(ascending=False)
    pct = counts / len(hold)

    held_summary = pd.DataFrame(
        {
            "months_held": counts,
            "pct_months": pct,
            "first_held": hold.groupby(hold).apply(lambda s: s.index.min()),
            "last_held": hold.groupby(hold).apply(lambda s: s.index.max()),
        }
    ).sort_values("months_held", ascending=False)

    held_summary["pct_months"] = held_summary["pct_months"].map(lambda x: f"{x:.1%}")
    held_summary["first_held"] = pd.to_datetime(held_summary["first_held"]).dt.date
    held_summary["last_held"] = pd.to_datetime(held_summary["last_held"]).dt.date

    print("\n" + "=" * 70)
    print("Months held by ticker (strategy holdings)")
    print("=" * 70)
    print(held_summary.to_string())

    # Return contribution by ticker (log-return attribution)
    rets = res["out"]["port_ret"].dropna()
    hold_applied = res["out"]["holding"].reindex(rets.index).astype(str)

    by_ticker = rets.groupby(hold_applied).agg(months="count", avg_monthly_ret="mean")
    logret = np.log1p(rets)
    sum_logret = logret.groupby(hold_applied).sum().rename("sum_logret")
    by_ticker = by_ticker.join(sum_logret)

    total_log = by_ticker["sum_logret"].sum()
    by_ticker["pct_total_logret"] = by_ticker["sum_logret"] / total_log if total_log != 0 else np.nan

    by_ticker["avg_monthly_ret"] = by_ticker["avg_monthly_ret"].map(lambda x: f"{x:.2%}")
    by_ticker["pct_total_logret"] = by_ticker["pct_total_logret"].map(lambda x: f"{x:.1%}")

    print("\n" + "=" * 70)
    print("Return contribution by ticker (based on months held)")
    print("=" * 70)
    print(by_ticker.sort_values("sum_logret", ascending=False).to_string())

    # Holding streak stats
    h = res["out"]["holding"].dropna().astype(str)
    streak_id = (h != h.shift(1)).cumsum()
    streaks = pd.DataFrame({"ticker": h, "streak_id": streak_id})
    streak_lengths = streaks.groupby(["ticker", "streak_id"]).size().rename("streak_len").reset_index()

    streak_stats = streak_lengths.groupby("ticker")["streak_len"].agg(
        streaks="count",
        avg_streak="mean",
        median_streak="median",
        max_streak="max",
    ).sort_values("avg_streak", ascending=False)

    streak_stats["avg_streak"] = streak_stats["avg_streak"].map(lambda x: f"{x:.2f}")
    streak_stats["median_streak"] = streak_stats["median_streak"].map(lambda x: f"{x:.0f}")
    streak_stats["max_streak"] = streak_stats["max_streak"].map(lambda x: f"{x:.0f}")

    print("\n" + "=" * 70)
    print("Holding streak stats (months per continuous hold)")
    print("=" * 70)
    print(streak_stats.to_string())


if __name__ == "__main__":
    main()

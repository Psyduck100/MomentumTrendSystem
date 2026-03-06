from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Iterable, Optional, Dict, Any, Tuple, List

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
    tickers = list(
        dict.fromkeys([t.strip().upper() for t in tickers if str(t).strip()])
    )
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
            raise ValueError(
                "Could not find Close/Adj Close for single-ticker response."
            )

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
    out: Dict[int, pd.DataFrame] = {}
    for m in months:
        out[m] = monthly_prices.pct_change(m)
    return out


def rank_universe_on_date(
    lookback_rets: Dict[int, pd.DataFrame],
    weights: Tuple[float, float],
    universe: List[str],
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


def pick_monthly_holding(rank_table: pd.DataFrame, defensive: str) -> str:
    if rank_table.empty:
        return defensive
    top = rank_table.iloc[0]
    return str(top["ticker"]) if bool(top["abs_pass"]) else defensive


def build_monthly_holdings(
    monthly_prices: pd.DataFrame,
    universe: List[str],
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
    rank_tables: Dict[pd.Timestamp, pd.DataFrame] = {}

    for asof in valid_idx:
        rt = rank_universe_on_date(lookback_rets, weights, universe, asof)
        rank_tables[asof] = rt
        pick = pick_monthly_holding(rt, defensive=defensive)
        dates.append(asof)
        holdings.append(pick)

    holdings_next = pd.Series(
        holdings, index=pd.DatetimeIndex(dates), name="hold_next_month"
    )
    return holdings_next, rank_tables


# ----------------------------
# Backtest
# ----------------------------
def backtest_monthly_rotation(
    monthly_prices: pd.DataFrame,
    holdings_next: pd.Series,
    cost_bps: float = 0.0,
    label: str = "blend_6_12_top1_abs12_else_DEF",
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
        mask = hold_applied == tkr
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
        index=[label],
    )

    out = pd.concat([hold_applied.rename("holding"), port_ret, equity], axis=1)
    return {"out": out, "summary": summary, "holdings_applied": hold_applied}


def spy_buyhold_same_window(
    prices_monthly: pd.DataFrame, idx: pd.DatetimeIndex
) -> pd.DataFrame:
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


def holdings_breakdown(out_df: pd.DataFrame) -> pd.DataFrame:
    hold = out_df["holding"].dropna().astype(str)
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
    return held_summary


# ----------------------------
# Compare universes
# ----------------------------
@dataclass(frozen=True)
class StrategyConfig:
    universe: Tuple[str, ...]
    defensive: str = "IEF"
    lookbacks: Tuple[int, int] = (6, 12)
    weights: Tuple[float, float] = (0.5, 0.5)
    start_date: str = "2000-01-01"
    end_date: Optional[str] = None
    cost_bps: float = 5.0


def run_one(cfg: StrategyConfig) -> Dict[str, Any]:
    # include SPY always (benchmark) + defensive + universe
    tickers = list(dict.fromkeys([*cfg.universe, cfg.defensive, "SPY"]))
    px_daily = download_prices(
        tickers, start_date=cfg.start_date, end_date=cfg.end_date, auto_adjust=True
    )
    px_monthly = to_monthly_prices(px_daily)

    # Common history across ALL tickers used in this run (this is what “overlapping period” means)
    px_monthly = px_monthly.dropna(how="any")
    if len(px_monthly) == 0:
        raise ValueError("No overlapping monthly history for these tickers.")

    holdings_next, rank_tables = build_monthly_holdings(
        px_monthly,
        universe=list(cfg.universe),
        defensive=cfg.defensive,
        lookbacks=cfg.lookbacks,
        weights=cfg.weights,
    )

    label = f"U={'-'.join(cfg.universe)}"
    bt = backtest_monthly_rotation(
        px_monthly, holdings_next, cost_bps=cfg.cost_bps, label=label
    )

    # benchmark over exact same strategy months
    idx = bt["out"].index
    spy_sum = spy_buyhold_same_window(px_monthly, idx)

    return {
        "cfg": cfg,
        "prices_monthly": px_monthly,
        "rank_tables": rank_tables,
        "holdings_next": holdings_next,
        "out": bt["out"],
        "summary": bt["summary"],
        "spy_summary": spy_sum,
    }


def main():
    cfgs = [
        StrategyConfig(universe=("QQQ", "SPY", "IWN", "VTI", "OEF")),
        StrategyConfig(universe=("QQQ", "SPY", "IWN", "VTI", "XLG")),
    ]

    runs = []
    for cfg in cfgs:
        res = run_one(cfg)
        runs.append(res)

    print("=" * 78)
    print("OEF vs XLG swap test (same strategy, each on its overlapping history)")
    print("=" * 78)

    # side-by-side summary table
    rows = []
    for r in runs:
        strat = r["summary"].iloc[0]
        spy = r["spy_summary"].iloc[0]
        months = int(strat["Months"])
        start = r["out"].index.min().date()
        end = r["out"].index.max().date()
        rows.append(
            {
                "universe": "-".join(r["cfg"].universe),
                "window": f"{start} -> {end}",
                "months": months,
                "CAGR": float(strat["CAGR"]),
                "Sharpe": float(strat["Sharpe(0rf)"]),
                "MaxDD": float(strat["MaxDD"]),
                "Turnover": float(strat["Turnover"]),
                "SPY_CAGR": float(spy["CAGR"]),
                "CAGR_diff_vs_SPY": float(strat["CAGR"] - spy["CAGR"]),
            }
        )

    comp = pd.DataFrame(rows).set_index("universe")
    # pretty format
    comp_fmt = comp.copy()
    comp_fmt["CAGR"] = comp_fmt["CAGR"].map(lambda x: f"{x:.2%}")
    comp_fmt["SPY_CAGR"] = comp_fmt["SPY_CAGR"].map(lambda x: f"{x:.2%}")
    comp_fmt["CAGR_diff_vs_SPY"] = comp_fmt["CAGR_diff_vs_SPY"].map(
        lambda x: f"{x:.2%}"
    )
    comp_fmt["Sharpe"] = comp_fmt["Sharpe"].map(lambda x: f"{x:.2f}")
    comp_fmt["MaxDD"] = comp_fmt["MaxDD"].map(lambda x: f"{x:.2%}")
    comp_fmt["Turnover"] = comp_fmt["Turnover"].map(lambda x: f"{x:.3f}")

    print("\nSUMMARY (each run uses its own overlapping history):")
    print(comp_fmt.to_string())

    # holdings breakdown for each run
    for r in runs:
        uni = "-".join(r["cfg"].universe)
        print("\n" + "=" * 78)
        print(f"Months held by ticker: {uni}")
        print("=" * 78)
        print(holdings_breakdown(r["out"]).to_string())

        print("\nLast 12 months:")
        tail = r["out"][["holding", "port_ret", "equity"]].tail(12).copy()
        tail["port_ret"] = tail["port_ret"].map(lambda x: f"{x:.2%}")
        tail["equity"] = tail["equity"].map(lambda x: f"{x:.3f}")
        print(tail.to_string())


if __name__ == "__main__":
    main()

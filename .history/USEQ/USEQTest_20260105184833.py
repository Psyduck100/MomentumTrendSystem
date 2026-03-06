from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Iterable, Optional, Dict, Any, Tuple

import numpy as np
import pandas as pd
import yfinance as yf
def build_monthly_holdings_top_of_two_blend612(
    monthly_prices: pd.DataFrame,
    risky_pair: Tuple[str, str],
    defensive: str,
    weights: Tuple[float, float] = (0.5, 0.5),
    lookbacks: Tuple[int, int] = (6, 12),
) -> Tuple[pd.Series, Dict[pd.Timestamp, pd.DataFrame]]:
    """
    Choose between two risky assets (e.g., QQQ vs VTI) using blend score:
        score = w6*ret_6m + w12*ret_12m

    Then apply absolute filter on the *chosen* asset:
        if chosen ret_12m > 0 -> hold chosen next month
        else -> hold defensive next month

    Returns:
      holdings_next: Series indexed by month-end (signal date), value=ticker to hold NEXT month
      rank_tables: dict[asof_date] -> small table with scores/returns/decision
    """
    r1, r2 = risky_pair
    w6, w12 = weights
    lb6, lb12 = lookbacks
    if lb6 != 6 or lb12 != 12:
        # Not required, but keeps naming consistent. Remove if you want flexible lookbacks.
        pass

    lookback_rets = compute_lookback_returns(monthly_prices, lookbacks)
    ret6 = lookback_rets[lb6]
    ret12 = lookback_rets[lb12]

    max_lb = max(lookbacks)
    valid_idx = monthly_prices.index[max_lb:]  # ensures ret_12m exists

    holdings = []
    dates = []
    rank_tables: Dict[pd.Timestamp, pd.DataFrame] = {}

    for asof in valid_idx:
        r6 = ret6.loc[asof, [r1, r2]].astype(float)
        r12 = ret12.loc[asof, [r1, r2]].astype(float)
        score = (w6 * r6) + (w12 * r12)

        # pick top by score
        top = score.idxmax()
        top_abs = float(r12[top])
        abs_pass = bool(top_abs > 0)

        hold_next = top if abs_pass else defensive

        rt = pd.DataFrame(
            {
                "ticker": [r1, r2],
                "score": [float(score[r1]), float(score[r2])],
                "ret_6m": [float(r6[r1]), float(r6[r2])],
                "ret_12m": [float(r12[r1]), float(r12[r2])],
            }
        )
        rt["rank"] = rt["score"].rank(ascending=False, method="first").astype(int)
        rt = rt.sort_values("score", ascending=False).reset_index(drop=True)
        rt["top_picked"] = (rt["ticker"] == top)
        rt["abs_pass_top"] = abs_pass
        rt["decision_hold_next"] = hold_next

        rank_tables[asof] = rt
        dates.append(asof)
        holdings.append(hold_next)

    holdings_next = pd.Series(holdings, index=pd.DatetimeIndex(dates), name="hold_next_month")
    return holdings_next, rank_tables


# ----------------------------
# Config + run_backtest + main (replace your current ones)
# ----------------------------
@dataclass(frozen=True)
class StrategyConfig:
    risky_pair: Tuple[str, str] = ("QQQ", "VTI")
    defensive: str = "IEF"
    lookbacks: Tuple[int, int] = (6, 12)
    weights: Tuple[float, float] = (0.5, 0.5)
    start_date: str = "2000-01-01"
    end_date: Optional[str] = None
    cost_bps: float = 0.0


def run_backtest(cfg: StrategyConfig) -> Dict[str, Any]:
    tickers = [cfg.risky_pair[0], cfg.risky_pair[1], cfg.defensive, "SPY"]  # SPY for benchmark
    px_daily = download_prices(
        tickers, start_date=cfg.start_date, end_date=cfg.end_date, auto_adjust=True
    )
    px_monthly = to_monthly_prices(px_daily)

    # common history for all tickers used (QQQ/VTI/IEF/SPY)
    px_monthly = px_monthly.dropna(how="any")

    holdings_next, rank_tables = build_monthly_holdings_top_of_two_blend612(
        px_monthly,
        risky_pair=cfg.risky_pair,
        defensive=cfg.defensive,
        weights=cfg.weights,
        lookbacks=cfg.lookbacks,
    )

    bt = backtest_monthly_rotation(
        px_monthly,
        holdings_next=holdings_next,
        cost_bps=cfg.cost_bps,
    )

    # rename the strategy row label to match this variant (optional)
    bt["summary"].index = ["top1_blend612_of_QQQ_VTI_abs12_else_IEF"]

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
        risky_pair=("QQQ", "VTI"),
        defensive="IEF",
        lookbacks=(6, 12),
        weights=(0.5, 0.5),
        start_date="2000-01-01",
        end_date=None,
        cost_bps=5.0,
    )

    res = run_backtest(cfg)

    print("=" * 70)
    print("QQQ vs VTI (blend_6_12) + abs(12M) else IEF - Monthly Backtest")
    print("=" * 70)
    print(f"Risky pair: {list(cfg.risky_pair)} | Defensive: {cfg.defensive}")
    print(f"Score: {cfg.weights} on {cfg.lookbacks} months | Abs filter: 12M > 0 on chosen")
    print(f"Monthly rebalance | Cost: {cfg.cost_bps} bps on switches")
    print(
        f"Monthly history used: {res['prices_monthly'].index.min().date()} -> {res['prices_monthly'].index.max().date()}"
    )
    print()

    print(res["summary"].to_string())

    # SPY buy & hold over the exact same months as strategy
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

    # Show latest signal table
    asof = res["holdings_next"].index.max()
    rt = res["rank_tables"][asof].copy()
    rt["score"] = rt["score"].map(lambda x: f"{x:.2%}")
    rt["ret_6m"] = rt["ret_6m"].map(lambda x: f"{x:.2%}")
    rt["ret_12m"] = rt["ret_12m"].map(lambda x: f"{x:.2%}")
    print(f"\nRank table as-of {asof.date()} (signal month-end):")
    print(rt.to_string(index=False))

    # Months held breakdown
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

    held_summary["pct_months"] = held_summary["pct_months"].map(lambda x: f"{x:.1%}")
    held_summary["first_held"] = pd.to_datetime(held_summary["first_held"]).dt.date
    held_summary["last_held"] = pd.to_datetime(held_summary["last_held"]).dt.date

    print("\n" + "=" * 70)
    print("Months held by ticker (strategy holdings)")
    print("=" * 70)
    print(held_summary.to_string())


if __name__ == "__main__":
    main()

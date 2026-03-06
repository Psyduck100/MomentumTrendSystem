"""
Sector Rotation (Optimized 170D Momentum) - Modular Rebalance Module

Strategy:
- Universe: XLK, XLV, XLI, XLE, XAR (XLF removed, XLV kept)
- Score: 170 trading-day return  
- Selection: top-ranked ticker (rank_gap=0, always switch to leader)
- Rebalancing: monthly (month-end recommendation)

Performance reference is evaluated in dedicated backtest scripts.

Design goals:
- NO prints inside library functions (prints only in main)
- Each function returns useful objects (DataFrames / dicts)
- Output shape mirrors strategy/USEQ.py for easy integration
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Iterable, Optional

import numpy as np
import pandas as pd
import yfinance as yf


DEFAULT_UNIVERSE = ("XLK", "XLV", "XLI", "XLE", "XAR")
DEFAULT_LOOKBACK_DAYS = 170
DEFAULT_START_DATE = "2011-09-29"


@dataclass(frozen=True)
class StrategyConfig:
    universe: tuple[str, ...] = DEFAULT_UNIVERSE
    lookback_days: int = DEFAULT_LOOKBACK_DAYS
    start_date: str = DEFAULT_START_DATE


def download_prices(
    tickers: Iterable[str],
    start_date: str,
    end_date: Optional[str] = None,
    auto_adjust: bool = True,
) -> pd.DataFrame:
    tickers = list(dict.fromkeys(str(t).strip() for t in tickers if str(t).strip()))
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

    if isinstance(data.columns, pd.MultiIndex):
        fields = data.columns.get_level_values(0)
        if "Close" in fields:
            px = data["Close"]
        elif "Adj Close" in fields:
            px = data["Adj Close"]
        else:
            swapped = data.swaplevel(0, 1, axis=1).sort_index(axis=1)
            subfields = swapped.columns.get_level_values(1)
            if "Close" in subfields:
                px = swapped.xs("Close", level=1, axis=1)
            elif "Adj Close" in subfields:
                px = swapped.xs("Adj Close", level=1, axis=1)
            else:
                raise ValueError("Could not find Close/Adj Close in yfinance response.")
    else:
        if "Close" in data.columns:
            px = pd.DataFrame({tickers[0]: data["Close"]})
        elif "Adj Close" in data.columns:
            px = pd.DataFrame({tickers[0]: data["Adj Close"]})
        else:
            raise ValueError("Could not find Close/Adj Close in single-ticker response.")

    px = px.dropna(how="all").ffill().dropna(how="all")
    px.index = pd.to_datetime(px.index)
    return px.reindex(columns=[t for t in tickers if t in px.columns])


def last_complete_month_end(daily_index: pd.DatetimeIndex) -> pd.Timestamp:
    if len(daily_index) == 0:
        raise ValueError("Empty price index.")

    last_day = pd.Timestamp(daily_index[-1])
    cal_me = (last_day + pd.offsets.MonthEnd(0)).normalize()
    bus_me = (last_day + pd.offsets.BMonthEnd(0)).normalize()

    if last_day.normalize() >= bus_me:
        return cal_me
    return (last_day + pd.offsets.MonthEnd(-1)).normalize()


def _resolve_asof_on_daily_index(
    daily_prices: pd.DataFrame,
    asof: Optional[pd.Timestamp],
) -> pd.Timestamp:
    if asof is None:
        asof = last_complete_month_end(daily_prices.index)

    asof = pd.Timestamp(asof)
    eligible = daily_prices.index[daily_prices.index <= asof]
    if len(eligible) == 0:
        raise ValueError("No eligible daily bars at or before asof.")
    return pd.Timestamp(eligible[-1])


def compute_170d_scores(
    daily_prices: pd.DataFrame,
    asof: pd.Timestamp,
    lookback_days: int,
    universe: list[str],
) -> pd.Series:
    px = daily_prices.loc[:asof, universe].dropna(how="all")
    if len(px) <= lookback_days:
        raise ValueError(
            f"Not enough history for {lookback_days} trading-day lookback at {asof.date()}."
        )

    start = px.iloc[-(lookback_days + 1)]
    end = px.iloc[-1]
    scores = ((end / start) - 1.0).dropna().sort_values(ascending=False)
    return pd.to_numeric(scores, errors="coerce").dropna()


def build_rank_table(scores: pd.Series) -> pd.DataFrame:
    if scores.empty:
        return pd.DataFrame(columns=["rank", "ticker", "score_170d"])

    df = pd.DataFrame(
        {
            "ticker": scores.index,
            "score_170d": scores.values,
        }
    )
    df = df.sort_values("score_170d", ascending=False).reset_index(drop=True)
    df["rank"] = np.arange(1, len(df) + 1)
    return df[["rank", "ticker", "score_170d"]]


def pick_recommendation(rank_table: pd.DataFrame) -> dict:
    if rank_table.empty:
        return {
            "recommendation": None,
            "reason": "No valid 170D scores available.",
            "top_ranked": None,
            "top_score_170d": None,
        }

    top = rank_table.iloc[0]
    ticker = str(top["ticker"])
    score = float(top["score_170d"])
    return {
        "recommendation": ticker,
        "reason": f"Top-ranked by 170D momentum ({score:.2%}).",
        "top_ranked": ticker,
        "top_score_170d": score,
    }


def get_recommendations(
    config: StrategyConfig,
    asof: Optional[pd.Timestamp] = None,
) -> dict:
    universe = list(config.universe)
    prices_daily = download_prices(universe, start_date=config.start_date, auto_adjust=True)
    asof_bar = _resolve_asof_on_daily_index(prices_daily, asof=asof)

    scores = compute_170d_scores(
        daily_prices=prices_daily,
        asof=asof_bar,
        lookback_days=int(config.lookback_days),
        universe=universe,
    )
    rank_table = build_rank_table(scores)
    decision = pick_recommendation(rank_table)

    return {
        "asof_date": asof_bar.strftime("%Y-%m-%d"),
        "universe": universe,
        "prices_daily": prices_daily,
        "rank_table": rank_table,
        "decision": decision,
        "lookback_days": int(config.lookback_days),
    }


def main() -> dict:
    cfg = StrategyConfig()
    result = get_recommendations(cfg)

    asof = result["asof_date"]
    universe = result["universe"]
    rank_table = result["rank_table"].copy()
    decision = result["decision"]
    prices_daily = result["prices_daily"]
    lookback_days = result["lookback_days"]

    winner = decision["recommendation"]
    winner_price = None
    if winner is not None and winner in prices_daily.columns and len(prices_daily.index):
        winner_price = float(prices_daily[winner].iloc[-1])

    print("=" * 60)
    print("SECTOR ROTATION - STATIC 170D MOMENTUM")
    print("=" * 60)
    print(f"As-of: {asof}")
    print(f"Universe size: {len(universe)}")
    print(f"Lookback: {lookback_days} trading days")
    if winner_price is not None:
        print(f"Current price ({winner}): {winner_price:.2f}")
    print()

    print("ALL Rankings:")
    if rank_table.empty:
        print("No rankings available.")
    else:
        rank_table["score_170d"] = rank_table["score_170d"].map(lambda x: f"{x:.2%}")
        print(rank_table.to_string(index=False))

    print("\n" + "-" * 60)
    print(f"RECOMMENDED: {decision['recommendation']}")
    print(f"Reason: {decision['reason']}")
    print("-" * 60)

    return result


if __name__ == "__main__":
    main()

"""
US Equities Momentum (blend_6_12) - Modular Rebalance Module

Strategy:
- Universe: ../CSVs/US_equities.csv (column: 'ticker')
- Score: 0.5 * 6M return + 0.5 * 12M return
- Absolute filter: 12M return must be > 0, else go defensive
- Defensive: IEF
- Selection: single top-ranked ticker

Design goals:
- NO prints inside library functions (prints only in main)
- Each function returns useful objects (DataFrames / dicts)
- Easy to call from layered strategies (e.g., "get_recommendations" returns ranked table)
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd
import yfinance as yf


# ----------------------------
# Config
# ----------------------------
DEFAULT_UNIVERSE_CSV = Path(__file__).parent.parent / "CSVs" / "US_equities.csv"
DEFAULT_DEFENSIVE_SYMBOL = "IEF"

DEFAULT_LOOKBACK_MONTHS = (6, 12)
DEFAULT_LOOKBACK_WEIGHTS = (0.5, 0.5)


@dataclass(frozen=True)
class StrategyConfig:
    universe_csv: Path = DEFAULT_UNIVERSE_CSV
    defensive_symbol: str = DEFAULT_DEFENSIVE_SYMBOL
    lookback_months: tuple[int, ...] = DEFAULT_LOOKBACK_MONTHS
    lookback_weights: tuple[float, ...] = DEFAULT_LOOKBACK_WEIGHTS
    start_date: str = "2001-01-01"


# ----------------------------
# Universe
# ----------------------------
def load_universe(universe_csv: Path) -> list[str]:
    """Load tickers from CSV file with a 'ticker' column."""
    df = pd.read_csv(universe_csv, encoding="latin-1")
    if "ticker" not in df.columns:
        raise ValueError(
            f"Universe CSV missing required column 'ticker': {universe_csv}"
        )
    tickers = df["ticker"].dropna().astype(str).str.strip().tolist()
    # de-dupe while keeping order
    seen = set()
    out = []
    for t in tickers:
        if t and t not in seen:
            seen.add(t)
            out.append(t)
    return out


# ----------------------------
# Prices
# ----------------------------
def download_prices(
    tickers: Iterable[str],
    start_date: str,
    end_date: Optional[str] = None,
    auto_adjust: bool = True,
) -> pd.DataFrame:
    """
    Download daily prices via yfinance.

    Returns: DataFrame indexed by date, columns=tickers, values=price (Close if auto_adjust=True).
    No prints.
    """
    tickers = list(dict.fromkeys(tickers))  # de-dupe, preserve order
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

    # yfinance returns either:
    # - multiindex columns (field, ticker) or (ticker, field)
    # - or single-level columns for 1 ticker
    if isinstance(data.columns, pd.MultiIndex):
        # Normalize to columns=tickers using Close (auto_adjust=True) or Adj Close / Close fallback
        # Try common patterns: ("Close", ticker) or (ticker, "Close")
        fields = data.columns.get_level_values(0)
        if "Close" in fields:
            px = data["Close"]
        elif "Adj Close" in fields:
            px = data["Adj Close"]
        else:
            # Swap levels if needed and retry
            swapped = data.swaplevel(0, 1, axis=1).sort_index(axis=1)
            subfields = swapped.columns.get_level_values(1)
            if "Close" in subfields:
                px = swapped.xs("Close", level=1, axis=1)
            elif "Adj Close" in subfields:
                px = swapped.xs("Adj Close", level=1, axis=1)
            else:
                raise ValueError("Could not find Close/Adj Close in yfinance response.")
    else:
        # single ticker: columns likely include Close/Adj Close
        if "Close" in data.columns:
            px = pd.DataFrame({tickers[0]: data["Close"]})
        elif "Adj Close" in data.columns:
            px = pd.DataFrame({tickers[0]: data["Adj Close"]})
        else:
            raise ValueError(
                "Could not find Close/Adj Close for single-ticker response."
            )

    px = px.dropna(how="all").ffill().dropna()
    px.index = pd.to_datetime(px.index)
    # Ensure column order matches requested tickers (when present)
    px = px.reindex(columns=[t for t in tickers if t in px.columns])
    return px


def last_complete_month_end(daily_index: pd.DatetimeIndex) -> pd.Timestamp:
    """
    Returns the last completed month-end date relative to the last daily bar.
    If last bar is not on month-end, we use previous month-end.
    """
    if len(daily_index) == 0:
        raise ValueError("Empty price index.")
    last_day = pd.Timestamp(daily_index[-1])
    month_end = last_day + pd.offsets.MonthEnd(0)
    if last_day.normalize() == month_end.normalize():
        return month_end.normalize()
    return (last_day + pd.offsets.MonthEnd(-1)).normalize()


def to_monthly_prices(daily_prices: pd.DataFrame) -> pd.DataFrame:
    """Month-end prices from daily (uses last available trading day in each month)."""
    monthly = daily_prices.resample("ME").last()
    return monthly.dropna(how="all")


# ----------------------------
# Momentum + Rankings
# ----------------------------
def compute_lookback_returns(
    monthly_prices: pd.DataFrame, months: Iterable[int]
) -> pd.DataFrame:
    """
    Returns a DataFrame with MultiIndex columns: (ret_{m}m, ticker)
    """
    months = list(months)
    parts = []
    for m in months:
        r = monthly_prices.pct_change(m)
        r.columns = pd.MultiIndex.from_product([[f"ret_{m}m"], r.columns])
        parts.append(r)
    out = pd.concat(parts, axis=1).sort_index(axis=1)
    return out


def compute_blend_scores(
    monthly_returns: pd.DataFrame,
    lookback_months: Iterable[int],
    lookback_weights: Iterable[float],
    asof: pd.Timestamp,
) -> pd.Series:
    """
    monthly_returns: columns MultiIndex (ret_{m}m, ticker)
    returns a Series indexed by ticker with the blended score as-of 'asof'.
    """
    lookback_months = list(lookback_months)
    lookback_weights = list(lookback_weights)
    if len(lookback_months) != len(lookback_weights):
        raise ValueError("lookback_months and lookback_weights must have same length.")

    if asof not in monthly_returns.index:
        raise ValueError(f"asof date {asof} not in monthly_returns index.")

    score = None
    for m, w in zip(lookback_months, lookback_weights):
        block = monthly_returns.loc[asof, f"ret_{m}m"]  # Series by ticker
        score = (w * block) if score is None else (score + w * block)

    # Ensure numeric and drop NaNs
    score = pd.to_numeric(score, errors="coerce")
    return score.dropna()


def build_rank_table(
    monthly_returns: pd.DataFrame,
    scores: pd.Series,
    asof: pd.Timestamp,
    abs_months: int = 12,
) -> pd.DataFrame:
    """
    Returns a ranking table with:
      ticker, score, ret_6m, ret_12m, abs_pass, rank
    """
    ret6 = (
        monthly_returns.loc[asof, "ret_6m"]
        if ("ret_6m" in monthly_returns.columns.get_level_values(0))
        else None
    )
    ret12 = monthly_returns.loc[asof, f"ret_{abs_months}m"]

    df = pd.DataFrame(
        {
            "ticker": scores.index,
            "score": scores.values,
            "ret_6m": ret6.reindex(scores.index).values if ret6 is not None else np.nan,
            f"ret_{abs_months}m": ret12.reindex(scores.index).values,
        }
    )

    df["abs_pass"] = df[f"ret_{abs_months}m"] > 0
    df = df.sort_values("score", ascending=False).reset_index(drop=True)
    df["rank"] = np.arange(1, len(df) + 1)
    return df


def pick_recommendation(
    rank_table: pd.DataFrame,
    defensive_symbol: str,
    abs_months: int = 12,
) -> dict:
    """
    Returns a dict with:
      recommendation, reason, top_ranked, top_score, top_abs_return, abs_pass
    """
    if rank_table.empty:
        return {
            "recommendation": defensive_symbol,
            "reason": "No valid scores available; defaulting defensive.",
            "top_ranked": None,
            "top_score": None,
            "top_abs_return": None,
            "abs_pass": False,
        }

    top = rank_table.iloc[0]
    top_ticker = str(top["ticker"])
    top_score = float(top["score"])
    top_abs = float(top[f"ret_{abs_months}m"])
    abs_pass = bool(top["abs_pass"])

    if abs_pass:
        return {
            "recommendation": top_ticker,
            "reason": f"Top-ranked with positive {abs_months}M return ({top_abs:.2%}).",
            "top_ranked": top_ticker,
            "top_score": top_score,
            "top_abs_return": top_abs,
            "abs_pass": True,
        }

    return {
        "recommendation": defensive_symbol,
        "reason": f"Defensive: top-ranked {top_ticker} has negative {abs_months}M return ({top_abs:.2%}).",
        "top_ranked": top_ticker,
        "top_score": top_score,
        "top_abs_return": top_abs,
        "abs_pass": False,
    }


def get_recommendations(
    config: StrategyConfig,
    asof: Optional[pd.Timestamp] = None,
) -> dict:
    """
    Main reusable function for your layered system.

    Returns dict with:
      - asof_date (YYYY-MM-DD)
      - universe (list)
      - prices_daily (DataFrame)
      - prices_monthly (DataFrame)
      - monthly_returns (DataFrame multiindex cols)
      - rank_table (DataFrame)
      - decision (dict)
    No prints.
    """
    universe = load_universe(config.universe_csv)
    all_tickers = universe + [config.defensive_symbol]

    prices_daily = download_prices(all_tickers, start_date=config.start_date)
    prices_monthly = to_monthly_prices(prices_daily)

    if asof is None:
        asof = last_complete_month_end(prices_daily.index)

    # Ensure asof exists in monthly index (if not, step back to prior month end)
    if asof not in prices_monthly.index:
        # pick most recent monthly date <= asof
        eligible = prices_monthly.index[prices_monthly.index <= asof]
        if len(eligible) == 0:
            raise ValueError("Not enough monthly data to compute lookbacks.")
        asof = eligible[-1]

    monthly_returns = compute_lookback_returns(prices_monthly, config.lookback_months)
    scores = compute_blend_scores(
        monthly_returns,
        lookback_months=config.lookback_months,
        lookback_weights=config.lookback_weights,
        asof=asof,
    )

    # Restrict to just universe (exclude defensive symbol from ranking)
    scores = scores.reindex([t for t in universe if t in scores.index]).dropna()

    rank_table = build_rank_table(monthly_returns, scores, asof=asof, abs_months=12)
    decision = pick_recommendation(
        rank_table, defensive_symbol=config.defensive_symbol, abs_months=12
    )

    return {
        "asof_date": asof.strftime("%Y-%m-%d"),
        "universe": universe,
        "prices_daily": prices_daily,
        "prices_monthly": prices_monthly,
        "monthly_returns": monthly_returns,
        "rank_table": rank_table,
        "decision": decision,
    }


# ----------------------------
# Main (prints live here only)
# ----------------------------
def main():
    cfg = StrategyConfig()

    result = get_recommendations(cfg)

    asof = result["asof_date"]
    universe = result["universe"]
    rank_table = result["rank_table"]
    decision = result["decision"]

    print("=" * 60)
    print("US EQUITIES MOMENTUM - MONTHLY REBALANCE (blend_6_12)")
    print("=" * 60)
    print(f"As-of (month-end): {asof}")
    print(f"Universe size: {len(universe)}")
    print(f"Defensive: {cfg.defensive_symbol}")
    print(f"Score: {cfg.lookback_weights} on {cfg.lookback_months} months")
    print()

    print("ALL Rankings:")
    show = rank_table.head(15).copy()
    show["score"] = show["score"].map(lambda x: f"{x:.2%}")
    show["ret_6m"] = show["ret_6m"].map(lambda x: f"{x:.2%}" if pd.notna(x) else "NA")
    show["ret_12m"] = show["ret_12m"].map(lambda x: f"{x:.2%}")
    print(
        show[["rank", "ticker", "score", "ret_6m", "ret_12m", "abs_pass"]].to_string(
            index=False
        )
    )

    print("\n" + "-" * 60)
    print(f"RECOMMENDED: {decision['recommendation']}")
    print(f"Reason: {decision['reason']}")
    print("-" * 60)

    return result


if __name__ == "__main__":
    main()

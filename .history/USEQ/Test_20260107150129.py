from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable, Optional, Literal, Dict

import numpy as np
import pandas as pd
import yfinance as yf

# ----------------------------
# Config
# ----------------------------
DEFAULT_UNIVERSE_CSV = Path(__file__).parent.parent / "CSVs" / "US_equities_alt.csv"
DEFAULT_DEFENSIVE_SYMBOL = "IEF"

DEFAULT_LOOKBACK_MONTHS = (6, 12)
DEFAULT_LOOKBACK_WEIGHTS = (0.5, 0.5)

AbsFilterKind = Literal["ret_12m_pos", "ma_200"]


@dataclass(frozen=True)
class AbsFilterConfig:
    kind: AbsFilterKind = "ret_12m_pos"
    # for MA filter
    ma_days: int = 200
    # for return filter (interprets "12m" in trading days)
    trading_days_per_month: int = 21


@dataclass(frozen=True)
class StrategyConfig:
    universe_csv: Path = DEFAULT_UNIVERSE_CSV
    defensive_symbol: str = DEFAULT_DEFENSIVE_SYMBOL
    lookback_months: tuple[int, ...] = DEFAULT_LOOKBACK_MONTHS
    lookback_weights: tuple[float, ...] = DEFAULT_LOOKBACK_WEIGHTS
    start_date: str = "2001-01-01"
    end_date: Optional[str] = None
    trading_days_per_month: int = 21


# ----------------------------
# Universe
# ----------------------------
def load_universe(universe_csv: Path) -> list[str]:
    df = pd.read_csv(universe_csv, encoding="latin-1")
    if "ticker" not in df.columns:
        raise ValueError(
            f"Universe CSV missing required column 'ticker': {universe_csv}"
        )
    tickers = df["ticker"].dropna().astype(str).str.strip().tolist()
    seen, out = set(), []
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
    tickers = list(dict.fromkeys(tickers))
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
            raise ValueError(
                "Could not find Close/Adj Close for single-ticker response."
            )

    px = px.dropna(how="all").ffill().dropna()
    px.index = pd.to_datetime(px.index)
    px = px.reindex(columns=[t for t in tickers if t in px.columns])
    return px


# ----------------------------
# Signals (daily)
# ----------------------------
def _lookback_days(months: int, tdpm: int) -> int:
    return int(months * tdpm)


def compute_daily_lookback_returns(
    daily_prices: pd.DataFrame,
    lookback_months: Iterable[int],
    tdpm: int,
) -> Dict[int, pd.DataFrame]:
    """
    Returns dict: {months: DataFrame of lookback returns at each day}
      ret_m[t] = price[t] / price[t - m*tdpm] - 1
    """
    out = {}
    for m in lookback_months:
        d = _lookback_days(m, tdpm)
        out[m] = daily_prices.pct_change(d)
    return out


def compute_blend_scores_daily(
    lookback_returns: Dict[int, pd.DataFrame],
    lookback_months: Iterable[int],
    lookback_weights: Iterable[float],
) -> pd.DataFrame:
    """
    Returns DataFrame (date x ticker) of blended scores.
    """
    months = list(lookback_months)
    weights = list(lookback_weights)
    if len(months) != len(weights):
        raise ValueError("lookback_months and lookback_weights must have same length.")

    score = None
    for m, w in zip(months, weights):
        block = lookback_returns[m]
        score = (w * block) if score is None else (score + w * block)
    return score


def abs_filter_pass(
    *,
    abs_cfg: AbsFilterConfig,
    daily_prices: pd.DataFrame,
    daily_returns: pd.DataFrame,
    date: pd.Timestamp,
    top_ticker: str,
) -> bool:
    """
    Decide whether we pass the absolute filter for the currently top-ranked ticker.
    Evaluated using information available at 'date' close.
    """
    if abs_cfg.kind == "ret_12m_pos":
        d12 = _lookback_days(12, abs_cfg.trading_days_per_month)
        if date not in daily_prices.index:
            return False
        idx = daily_prices.index.get_loc(date)
        if isinstance(idx, slice) or isinstance(idx, np.ndarray):
            return False
        if idx < d12:
            return False
        p_now = daily_prices.iloc[idx][top_ticker]
        p_then = daily_prices.iloc[idx - d12][top_ticker]
        if not np.isfinite(p_now) or not np.isfinite(p_then) or p_then == 0:
            return False
        ret_12m = p_now / p_then - 1.0
        return bool(ret_12m > 0)

    if abs_cfg.kind == "ma_200":
        ma = daily_prices[top_ticker].rolling(abs_cfg.ma_days).mean()
        val = daily_prices.at[date, top_ticker]
        mav = ma.at[date] if date in ma.index else np.nan
        return bool(np.isfinite(val) and np.isfinite(mav) and (val > mav))

    raise ValueError(f"Unknown abs filter kind: {abs_cfg.kind}")


def build_daily_positions(
    *,
    universe: list[str],
    defensive_symbol: str,
    daily_prices: pd.DataFrame,
    cfg: StrategyConfig,
    abs_cfg: AbsFilterConfig,
) -> pd.Series:
    """
    Returns desired holding *signal* at each date (evaluated at close).
    You should shift this by 1 day before applying returns to avoid lookahead.
    """
    # compute blend scores
    lookbacks = compute_daily_lookback_returns(
        daily_prices[universe], cfg.lookback_months, cfg.trading_days_per_month
    )
    scores = compute_blend_scores_daily(
        lookbacks, cfg.lookback_months, cfg.lookback_weights
    )

    # pick top each day (exclude NaNs)
    top = scores.idxmax(axis=1, skipna=True)  # date -> ticker (or NaN if all NaN)

    # compute abs pass + final signal
    signal = pd.Series(index=scores.index, dtype="object")

    daily_rets = daily_prices.pct_change()

    for dt in scores.index:
        t = top.at[dt]
        if not isinstance(t, str) or t not in universe:
            signal.at[dt] = defensive_symbol
            continue

        ok = abs_filter_pass(
            abs_cfg=abs_cfg,
            daily_prices=daily_prices,
            daily_returns=daily_rets,
            date=dt,
            top_ticker=t,
        )
        signal.at[dt] = t if ok else defensive_symbol

    return signal


# ----------------------------
# Backtest + Metrics
# ----------------------------
def backtest_from_positions(
    daily_prices: pd.DataFrame,
    positions_signal: pd.Series,
    *,
    transaction_cost_bps: float = 0.0,
) -> pd.DataFrame:
    """
    positions_signal: date->ticker decided at close(date), applied on next day (shifted).
    Returns DataFrame with columns: position, ret, equity
    """
    daily_rets = daily_prices.pct_change()

    # apply on next bar to avoid lookahead
    position = positions_signal.shift(1)

    # compute strategy return by selecting the held column each day
    ret = pd.Series(index=daily_prices.index, dtype="float64")

    for dt in daily_prices.index:
        sym = position.at[dt] if dt in position.index else None
        if isinstance(sym, str) and sym in daily_rets.columns:
            ret.at[dt] = daily_rets.at[dt, sym]
        else:
            ret.at[dt] = np.nan

    ret = ret.dropna()

    # transaction costs when position changes (on the applied position series)
    if transaction_cost_bps and transaction_cost_bps > 0:
        pos_applied = position.reindex(ret.index)
        trades = pos_applied.ne(pos_applied.shift(1)).fillna(False)
        cost = transaction_cost_bps / 10_000.0
        ret = ret - trades.astype(float) * cost

    equity = (1.0 + ret).cumprod()
    out = pd.DataFrame(
        {"position": position.reindex(ret.index), "ret": ret, "equity": equity}
    )
    return out


def max_drawdown(equity: pd.Series) -> float:
    peak = equity.cummax()
    dd = equity / peak - 1.0
    return float(dd.min())


def compute_metrics(bt: pd.DataFrame, annualization: int = 252) -> dict:
    ret = bt["ret"].dropna()
    if len(ret) < 2:
        return {"CAGR": np.nan, "Sharpe": np.nan, "MaxDD": np.nan, "Days": len(ret)}

    equity = bt["equity"].dropna()
    n = len(ret)
    cagr = float(equity.iloc[-1] ** (annualization / n) - 1.0)

    vol = float(ret.std(ddof=1))
    sharpe = float(np.sqrt(annualization) * ret.mean() / vol) if vol > 0 else np.nan

    mdd = max_drawdown(equity)

    return {"CAGR": cagr, "Sharpe": sharpe, "MaxDD": mdd, "Days": n}


def run_backtest(
    cfg: StrategyConfig,
    abs_cfg: AbsFilterConfig,
    *,
    transaction_cost_bps: float = 0.0,
) -> dict:
    """
    High-level wrapper: loads universe, downloads prices, builds positions, backtests, returns results.
    No prints.
    """
    universe = load_universe(cfg.universe_csv)
    all_tickers = universe + [cfg.defensive_symbol]

    prices = download_prices(
        all_tickers,
        start_date=cfg.start_date,
        end_date=cfg.end_date,
        auto_adjust=True,
    )

    # keep only columns that actually downloaded
    present_universe = [t for t in universe if t in prices.columns]
    if cfg.defensive_symbol not in prices.columns:
        raise ValueError(
            f"Defensive symbol {cfg.defensive_symbol} not found in downloaded prices."
        )

    positions_signal = build_daily_positions(
        universe=present_universe,
        defensive_symbol=cfg.defensive_symbol,
        daily_prices=prices,
        cfg=cfg,
        abs_cfg=abs_cfg,
    )

    bt = backtest_from_positions(
        prices, positions_signal, transaction_cost_bps=transaction_cost_bps
    )
    metrics = compute_metrics(bt)

    return {
        "config": cfg,
        "abs_filter": abs_cfg,
        "universe_used": present_universe,
        "prices": prices,
        "positions_signal": positions_signal,
        "bt": bt,
        "metrics": metrics,
    }


# ----------------------------
# Main (prints live here only)
# ----------------------------
def main():
    cfg = StrategyConfig()

    # Compare two absolute filters:
    res_ret12 = run_backtest(
        cfg,
        AbsFilterConfig(
            kind="ret_12m_pos", trading_days_per_month=cfg.trading_days_per_month
        ),
        transaction_cost_bps=0.0,
    )
    res_ma200 = run_backtest(
        cfg,
        AbsFilterConfig(kind="ma_200", ma_days=200),
        transaction_cost_bps=0.0,
    )

    rows = []
    for name, res in [
        ("abs=ret_12m_pos", res_ret12),
        ("abs=ma_200", res_ma200),
    ]:
        m = res["metrics"]
        rows.append(
            {
                "variant": name,
                "CAGR": m["CAGR"],
                "Sharpe": m["Sharpe"],
                "MaxDD": m["MaxDD"],
                "Days": m["Days"],
                "UniverseUsed": len(res["universe_used"]),
            }
        )

    comp = pd.DataFrame(rows).set_index("variant")
    # pretty print
    show = comp.copy()
    show["CAGR"] = show["CAGR"].map(lambda x: f"{x:.2%}" if pd.notna(x) else "NA")
    show["Sharpe"] = show["Sharpe"].map(lambda x: f"{x:.2f}" if pd.notna(x) else "NA")
    show["MaxDD"] = show["MaxDD"].map(lambda x: f"{x:.2%}" if pd.notna(x) else "NA")

    print("=" * 70)
    print("US EQUITIES MOMENTUM (blend_6_12) - DAILY BACKTEST COMPARISON")
    print("=" * 70)
    print(show.to_string())
    print("=" * 70)

    return res_ret12, res_ma200


if __name__ == "__main__":
    main()

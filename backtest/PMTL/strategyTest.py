from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any
import os
import sys
import numpy as np
import pandas as pd
import yfinance as yf

if __name__ == "__main__" and __package__ is None:
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

import strategy.PMTL as PMTL
import strategy.USEQ as USEQ


@dataclass(frozen=True)
class LayerConfig:
    gld_ticker: str = "GLD"
    useq_start_date: str = "2001-01-01"
    cost_bps_roundtrip: float = 0.0


def years_in_index(idx: pd.Index, periods_per_year: int = 252) -> float:
    return max(1e-9, len(idx) / periods_per_year)


def max_drawdown(returns: pd.Series) -> float:
    equity = (1.0 + returns.fillna(0.0)).cumprod()
    peak = equity.cummax()
    dd = equity / peak - 1.0
    return float(dd.min()) if len(dd) else np.nan


def perf_metrics(
    returns: pd.Series, *, periods_per_year: int = 252
) -> Dict[str, float]:
    returns = returns.fillna(0.0)
    total = float((1.0 + returns).prod() - 1.0) if len(returns) else np.nan
    yrs = years_in_index(returns.index, periods_per_year=periods_per_year)
    cagr = float((1.0 + total) ** (1.0 / yrs) - 1.0) if yrs > 0 else np.nan
    vol = float(returns.std(ddof=0) * np.sqrt(periods_per_year))
    sharpe = (
        float(returns.mean() / returns.std(ddof=0) * np.sqrt(periods_per_year))
        if returns.std(ddof=0) > 0
        else np.nan
    )
    mdd = max_drawdown(returns)
    return {
        "cagr": cagr,
        "sharpe": sharpe,
        "max_dd": mdd,
    }


def ann_over_mask(
    returns: pd.Series, mask: pd.Series, *, periods_per_year: int = 252
) -> float:
    mask = mask.reindex(returns.index).fillna(False).astype(bool)
    if not mask.any():
        return np.nan
    total = float((1.0 + returns[mask]).prod() - 1.0)
    years = float(mask.sum() / periods_per_year)
    return float((1.0 + total) ** (1.0 / years) - 1.0) if years > 0 else np.nan


def regime_stats(is_bull: pd.Series) -> Dict[str, float]:
    is_bull = is_bull.fillna(False).astype(bool)
    diff = is_bull.astype(int).diff().fillna(0)
    return {
        "bull_frac": float(is_bull.mean()),
        "bull_entries": float((diff == 1).sum()),
        "bull_exits": float((diff == -1).sum()),
    }


def slice_period(
    series: pd.Series, start: str, end: str | None
) -> pd.Series:
    return series.loc[start:] if end is None else series.loc[start:end]


def build_useq_recommendations(
    daily_prices: pd.DataFrame,
    config: USEQ.StrategyConfig,
    *,
    universe: list[str],
) -> pd.Series:
    monthly_prices = USEQ.to_monthly_prices(daily_prices)
    monthly_returns = USEQ.compute_lookback_returns(
        monthly_prices, config.lookback_months
    )
    recs: Dict[pd.Timestamp, str] = {}

    for asof in monthly_prices.index:
        scores = USEQ.compute_blend_scores(
            monthly_returns,
            lookback_months=config.lookback_months,
            lookback_weights=config.lookback_weights,
            asof=asof,
        )
        scores = scores.reindex([t for t in universe if t in scores.index]).dropna()
        rank_table = USEQ.build_rank_table(
            monthly_returns, scores, asof=asof, abs_months=12
        )
        decision = USEQ.pick_recommendation(
            rank_table, defensive_symbol=config.defensive_symbol, abs_months=12
        )
        recs[asof] = decision["recommendation"]

    return pd.Series(recs).sort_index()


def positions_from_monthly_recs(
    recs: pd.Series,
    daily_index: pd.DatetimeIndex,
    default_ticker: str,
) -> pd.Series:
    pos = pd.Series(index=daily_index, dtype=object)
    for asof, ticker in recs.items():
        future = daily_index[daily_index > asof]
        if len(future) == 0:
            continue
        pos.loc[future[0]] = str(ticker)
    pos = pos.ffill().fillna(default_ticker)
    return pos


def returns_from_positions(returns_df: pd.DataFrame, positions: pd.Series) -> pd.Series:
    pos = positions.reindex(returns_df.index).ffill()
    out = pd.Series(0.0, index=returns_df.index)
    for ticker in returns_df.columns:
        mask = pos == ticker
        if mask.any():
            out.loc[mask] = returns_df.loc[mask, ticker]
    return out


def main():
    cfg = LayerConfig()

    gld = yf.download(cfg.gld_ticker, period="max", auto_adjust=True, progress=False)
    close = PMTL.as_series(gld["Close"], "Close").dropna()

    feature_params_up_260 = dict(
        ma_len_entry=200,
        ma_len_exit=200,
        slope_lookback=20,
        entry_len=260,
        exit_len=90,
    )
    rule_params_up = dict(slope_min=0.0, ma_buffer=0.005)

    feature_params_down = dict(
        ma_len_entry=200,
        ma_len_exit=270,
        slope_lookback=20,
        entry_len=90,
        exit_len=90,
    )
    rule_params_down = dict(slope_min=0.0, ma_buffer=0.005)

    reg_260 = PMTL.build_regime(
        close,
        feature_params_up=feature_params_up_260,
        rule_params_up=rule_params_up,
        feature_params_down=feature_params_down,
        rule_params_down=rule_params_down,
        gate_up="BME",
        gate_down="W-FRI",
        down_enter_gates=2,
        down_exit_gates=2,
    )
    is_bull_260 = reg_260["is_up"].reindex(close.index).fillna(False)

    useq_cfg = USEQ.StrategyConfig(start_date=cfg.useq_start_date)
    universe = ["SPY", "QQQ", "VTI"]
    all_tickers = universe + [useq_cfg.defensive_symbol]

    useq_prices = USEQ.download_prices(all_tickers, start_date=useq_cfg.start_date)
    useq_recs = build_useq_recommendations(useq_prices, useq_cfg, universe=universe)

    positions_useq = positions_from_monthly_recs(
        useq_recs, useq_prices.index, useq_cfg.defensive_symbol
    )

    gld_returns = close.pct_change().fillna(0.0)
    useq_returns_df = useq_prices.pct_change().fillna(0.0)
    useq_returns = returns_from_positions(useq_returns_df, positions_useq)

    is_bull_260 = is_bull_260.reindex(gld_returns.index).fillna(False)
    useq_returns = useq_returns.reindex(gld_returns.index).fillna(0.0)

    combined_returns_260 = gld_returns.where(is_bull_260, useq_returns)

    gld_bull_returns_260 = gld_returns.where(is_bull_260, 0.0)
    useq_nonbull_returns_260 = useq_returns.where(~is_bull_260, 0.0)

    print("\n=== Full Period ===")
    print("GLD Bull Sleeve (260):", perf_metrics(gld_bull_returns_260))
    print("USEQ Non-Bull Sleeve (260):", perf_metrics(useq_nonbull_returns_260))
    print("Combined Layered (260):", perf_metrics(combined_returns_260))
    print("Bull stats (260):", regime_stats(is_bull_260))
    print(
        f"Bull ann over time (260): {ann_over_mask(gld_returns, is_bull_260):.6f}"
    )

    for label, (start, end) in {
        "Train_2004_2015": ("2004-01-01", "2015-12-31"),
        "Test_2015_2025": ("2015-01-01", "2025-12-31"),
    }.items():
        gld_r = slice_period(gld_returns, start, end)
        useq_r = slice_period(useq_returns, start, end)
        bull_260 = slice_period(is_bull_260, start, end)

        comb_260 = gld_r.where(bull_260, useq_r)

        print(f"\n=== {label} ===")
        print("Combined Layered (260):", perf_metrics(comb_260))


if __name__ == "__main__":
    main()

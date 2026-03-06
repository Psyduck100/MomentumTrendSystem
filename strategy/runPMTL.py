from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any
import os
import sys

import pandas as pd
import yfinance as yf

if __name__ == "__main__" and __package__ is None:
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

import strategy.PMTL as PMTL
import strategy.USEQ as USEQ


@dataclass(frozen=True)
class LiveConfig:
    gld_ticker: str = "GLD"
    useq_start_date: str = "2001-01-01"
    universe: tuple[str, ...] = ("SPY", "QQQ", "VTI")


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


def _format_pmtl_summary(snapshot: Dict[str, Any]) -> str:
    lines = []
    lines.append("PMTL Live Snapshot")
    lines.append("-" * 72)
    lines.append(f"As of: {snapshot['asof_date']}")
    lines.append(f"GLD close (latest): {snapshot['gld_price']:.2f}")
    lines.append(
        f"GLD close (gate {snapshot['gate_up']}): {snapshot['gld_gate_up_price']:.2f} "
        f"@ {snapshot['gld_gate_up_date']}"
    )
    lines.append(
        f"GLD close (gate {snapshot['gate_down']}): {snapshot['gld_gate_down_price']:.2f} "
        f"@ {snapshot['gld_gate_down_date']}"
    )
    lines.append(f"Regime: {snapshot['regime']}")
    lines.append(f"Recommendation: {snapshot['recommended_holding']}")
    lines.append(f"Why: {snapshot['reason']}")
    lines.append(f"Up details: {snapshot['up_reason']}")
    lines.append(f"Down details: {snapshot['down_reason']}")
    lines.append(f"USEQ month-end: {snapshot['useq_asof_month_end']}")
    lines.append(f"USEQ pick: {snapshot['useq_recommendation']}")
    lines.append("-" * 72)
    return "\n".join(lines)


def get_live_snapshot(cfg: LiveConfig) -> Dict[str, Any]:
    gld = yf.download(cfg.gld_ticker, period="max", auto_adjust=True, progress=False)
    close = PMTL.as_series(gld["Close"], "Close").dropna()

    feature_params_up = dict(
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

    reg = PMTL.build_regime(
        close,
        feature_params_up=feature_params_up,
        rule_params_up=rule_params_up,
        feature_params_down=feature_params_down,
        rule_params_down=rule_params_down,
        gate_up="BME",
        gate_down="W-FRI",
        down_enter_gates=2,
        down_exit_gates=2,
    )

    is_bull = reg["is_up"].reindex(close.index).fillna(False)
    is_down = reg["is_down"].reindex(close.index).fillna(False)
    bull_today = bool(is_bull.iloc[-1]) if len(is_bull) else False

    useq_cfg = USEQ.StrategyConfig(start_date=cfg.useq_start_date)
    universe = list(cfg.universe)
    all_tickers = universe + [useq_cfg.defensive_symbol]

    useq_prices = USEQ.download_prices(all_tickers, start_date=useq_cfg.start_date)
    useq_recs = build_useq_recommendations(useq_prices, useq_cfg, universe=universe)
    latest_month_end = USEQ.last_complete_month_end(useq_prices.index)
    useq_reco = useq_recs.loc[useq_recs.index <= latest_month_end].iloc[-1]

    recommended = cfg.gld_ticker if bull_today else str(useq_reco)
    if bull_today:
        reason = "Hold GLD because PMTL is in bull regime (uptrend)."
    elif recommended == useq_cfg.defensive_symbol:
        reason = (
            f"Hold {recommended} because PMTL is not bull and USEQ "
            "triggered its defensive filter."
        )
    else:
        reason = (
            f"Hold {recommended} because PMTL is not bull and USEQ "
            "monthly momentum ranked it highest."
        )

    gate_up = "BME"
    gate_down = "W-FRI"
    last_day = close.index[-1].normalize()
    gate_up_series = close.resample(gate_up).last()
    gate_down_series = close.resample(gate_down).last()
    gate_up_series = gate_up_series[gate_up_series.index <= last_day]
    gate_down_series = gate_down_series[gate_down_series.index <= last_day]
    gld_gate_up_date = gate_up_series.index[-1] if len(gate_up_series) else close.index[-1]
    gld_gate_down_date = gate_down_series.index[-1] if len(gate_down_series) else close.index[-1]

    detail = PMTL.describe_regime_snapshot(
        close,
        feature_params_up=feature_params_up,
        rule_params_up=rule_params_up,
        feature_params_down=feature_params_down,
        rule_params_down=rule_params_down,
        asof_up=gld_gate_up_date,
        asof_down=gld_gate_down_date,
    )

    regime_label = "UP" if bull_today else ("DOWN" if bool(is_down.iloc[-1]) else "CHOP")

    return {
        "asof_date": close.index[-1].strftime("%Y-%m-%d"),
        "gld_price": float(close.iloc[-1]),
        "gld_gate_up_price": float(gate_up_series.iloc[-1]) if len(gate_up_series) else float(close.iloc[-1]),
        "gld_gate_down_price": float(gate_down_series.iloc[-1]) if len(gate_down_series) else float(close.iloc[-1]),
        "gld_gate_up_date": gld_gate_up_date.strftime("%Y-%m-%d"),
        "gld_gate_down_date": gld_gate_down_date.strftime("%Y-%m-%d"),
        "gate_up": gate_up,
        "gate_down": gate_down,
        "bull_today": bull_today,
        "regime": regime_label,
        "recommended_holding": recommended,
        "reason": reason,
        "up_reason": detail["up_reason"],
        "down_reason": detail["down_reason"],
        "useq_asof_month_end": latest_month_end.strftime("%Y-%m-%d"),
        "useq_recommendation": str(useq_reco),
        "bull_frac": float(is_bull.mean()),
    }


def main() -> None:
    snapshot = get_live_snapshot(LiveConfig())
    print(_format_pmtl_summary(snapshot))


if __name__ == "__main__":
    main()

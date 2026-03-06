from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

TRADING_DAYS = 252


def _annualized_return(returns: pd.Series, periods_per_year: int = TRADING_DAYS) -> float:
    r = pd.Series(returns, dtype=float).dropna()
    if r.empty:
        return float("nan")
    growth = (1.0 + r).prod()
    years = len(r) / float(periods_per_year)
    if years <= 0:
        return float("nan")
    return float(growth ** (1.0 / years) - 1.0)


def _annualized_volatility(returns: pd.Series, periods_per_year: int = TRADING_DAYS) -> float:
    r = pd.Series(returns, dtype=float).dropna()
    if len(r) < 2:
        return float("nan")
    return float(r.std(ddof=1) * np.sqrt(periods_per_year))


def _max_drawdown(returns: pd.Series) -> float:
    r = pd.Series(returns, dtype=float).fillna(0.0)
    if r.empty:
        return float("nan")
    equity = (1.0 + r).cumprod()
    peak = equity.cummax()
    dd = equity / peak - 1.0
    return float(dd.min())


def _sharpe(returns: pd.Series, periods_per_year: int = TRADING_DAYS) -> float:
    r = pd.Series(returns, dtype=float).dropna()
    if len(r) < 2:
        return float("nan")
    sigma = r.std(ddof=1)
    if sigma <= 0.0:
        return float("nan")
    return float(np.sqrt(periods_per_year) * r.mean() / sigma)


def _sortino(returns: pd.Series, periods_per_year: int = TRADING_DAYS) -> float:
    r = pd.Series(returns, dtype=float).dropna()
    if len(r) < 2:
        return float("nan")
    downside = r[r < 0.0]
    if downside.empty:
        return float("nan")
    downside_vol = downside.std(ddof=1)
    if downside_vol <= 0.0 or np.isnan(downside_vol):
        return float("nan")
    return float(np.sqrt(periods_per_year) * r.mean() / downside_vol)


def _calmar(cagr: float, max_drawdown: float) -> float:
    if np.isnan(cagr) or np.isnan(max_drawdown) or max_drawdown >= 0.0:
        return float("nan")
    return float(cagr / abs(max_drawdown))


def _cagr_over_mask(
    returns: pd.Series,
    mask: Optional[pd.Series],
    periods_per_year: int = TRADING_DAYS,
) -> float:
    r = pd.Series(returns, dtype=float)
    if mask is None:
        return float("nan")
    m = pd.Series(mask).reindex(r.index).fillna(False).astype(bool)
    r_pos = r[m].dropna()
    if r_pos.empty:
        return float("nan")
    return _annualized_return(r_pos, periods_per_year=periods_per_year)


def compute_return_metrics(
    returns: pd.Series,
    positions: Optional[pd.Series] = None,
    periods_per_year: int = TRADING_DAYS,
) -> dict[str, float]:
    r = pd.Series(returns, dtype=float).dropna()
    if r.empty:
        return {
            "cagr": float("nan"),
            "cagr_in_position": float("nan"),
            "sharpe": float("nan"),
            "sortino": float("nan"),
            "volatility": float("nan"),
            "max_drawdown": float("nan"),
            "calmar": float("nan"),
            "total_return": float("nan"),
            "hit_rate": float("nan"),
            "days": 0.0,
            "years": float("nan"),
            "time_in_position": float("nan"),
        }

    cagr = _annualized_return(r, periods_per_year=periods_per_year)
    max_dd = _max_drawdown(r)

    time_in_position = float("nan")
    cagr_in_position = float("nan")
    if positions is not None:
        pos = pd.Series(positions).reindex(r.index).fillna(0.0)
        in_pos = pos > 0.0
        time_in_position = float(in_pos.mean())
        cagr_in_position = _cagr_over_mask(r, in_pos, periods_per_year=periods_per_year)

    return {
        "cagr": cagr,
        "cagr_in_position": cagr_in_position,
        "sharpe": _sharpe(r, periods_per_year=periods_per_year),
        "sortino": _sortino(r, periods_per_year=periods_per_year),
        "volatility": _annualized_volatility(r, periods_per_year=periods_per_year),
        "max_drawdown": max_dd,
        "calmar": _calmar(cagr, max_dd),
        "total_return": float((1.0 + r).prod() - 1.0),
        "hit_rate": float((r > 0.0).mean()),
        "days": float(len(r)),
        "years": float(len(r) / periods_per_year),
        "time_in_position": time_in_position,
    }


def compute_relative_metrics(
    strategy_returns: pd.Series,
    benchmark_returns: pd.Series,
    periods_per_year: int = TRADING_DAYS,
) -> dict[str, float]:
    aligned = pd.concat(
        [
            pd.Series(strategy_returns, dtype=float).rename("strategy"),
            pd.Series(benchmark_returns, dtype=float).rename("benchmark"),
        ],
        axis=1,
        join="inner",
    ).dropna()

    if aligned.empty:
        return {
            "active_return": float("nan"),
            "tracking_error": float("nan"),
            "information_ratio": float("nan"),
            "beta_to_benchmark": float("nan"),
            "correlation": float("nan"),
            "aligned_days": 0.0,
        }

    active = aligned["strategy"] - aligned["benchmark"]
    te = _annualized_volatility(active, periods_per_year=periods_per_year)
    active_ann = float(active.mean() * periods_per_year)
    ir = float(active_ann / te) if te and te > 0 and np.isfinite(te) else float("nan")

    bench_var = float(aligned["benchmark"].var(ddof=1)) if len(aligned) > 1 else float("nan")
    cov = float(aligned["strategy"].cov(aligned["benchmark"])) if len(aligned) > 1 else float("nan")
    beta = float(cov / bench_var) if bench_var and np.isfinite(bench_var) and bench_var > 0 else float("nan")
    corr = float(aligned["strategy"].corr(aligned["benchmark"])) if len(aligned) > 1 else float("nan")

    return {
        "active_return": active_ann,
        "tracking_error": te,
        "information_ratio": ir,
        "beta_to_benchmark": beta,
        "correlation": corr,
        "aligned_days": float(len(aligned)),
    }


def _fmt_pct(v: float) -> str:
    return "NA" if pd.isna(v) else f"{v:.2%}"


def _fmt_num(v: float, digits: int = 2) -> str:
    return "NA" if pd.isna(v) else f"{v:.{digits}f}"


def format_metrics_block(metrics: dict[str, float], *, include_relative: bool = False) -> str:
    lines = [
        f"CAGR: {_fmt_pct(metrics.get('cagr', float('nan')))}",
        f"CAGR in Position: {_fmt_pct(metrics.get('cagr_in_position', float('nan')))}",
        f"Sharpe: {_fmt_num(metrics.get('sharpe', float('nan')))}",
        f"Sortino: {_fmt_num(metrics.get('sortino', float('nan')))}",
        f"Volatility: {_fmt_pct(metrics.get('volatility', float('nan')))}",
        f"MaxDD: {_fmt_pct(metrics.get('max_drawdown', float('nan')))}",
        f"Total Return: {_fmt_pct(metrics.get('total_return', float('nan')))}",
    ]
    if not pd.isna(metrics.get("time_in_position", float("nan"))):
        lines.append(f"Time in Position: {_fmt_pct(metrics.get('time_in_position', float('nan')))}")
    if include_relative:
        lines.extend(
            [
                f"Information Ratio: {_fmt_num(metrics.get('information_ratio', float('nan')), 3)}",
                f"Active Return: {_fmt_pct(metrics.get('active_return', float('nan')))}",
            ]
        )
    return "\n".join(lines)

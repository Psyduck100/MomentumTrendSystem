from collections.abc import Sequence
import numpy as np

from momentum_program.analytics.constants import (
    SCORE_MODE_12M_MINUS_1M,
    SCORE_MODE_BLEND_6_12,
    SCORE_MODE_BLEND_LAMBDA,
    SCORE_MODE_RW_3_6_9_12,
    SCORE_MODE_RW_ALPHA,
)
from momentum_program.data_models import MomentumScore, PriceSeries


WINDOW_LONG = 252  # ~12 months of trading days
WINDOW_1M = 21  # ~1 month of trading days
RECENCY_MONTHS = (3, 6, 9, 12)


def _window_return(closes: list[float], window: int) -> float:
    if len(closes) <= window:
        return 0.0
    start = closes[-window]
    end = closes[-1]
    if start == 0:
        return 0.0
    return (end - start) / start


def _rolling_volatility(closes: list[float], window: int = WINDOW_LONG) -> float:
    """Compute annualized volatility over the last window days."""
    if len(closes) <= window:
        return 0.0
    recent_closes = closes[-window:]
    returns = []
    for i in range(1, len(recent_closes)):
        if recent_closes[i - 1] > 0:
            returns.append(
                (recent_closes[i] - recent_closes[i - 1]) / recent_closes[i - 1]
            )
    if not returns:
        return 0.0
    # Annualize daily volatility (252 trading days)
    return np.std(returns) * np.sqrt(252)


def _recency_weights(alpha: float | None) -> dict[int, float]:
    """Return normalized recency weights for the configured alpha."""
    value = 0.0 if alpha is None else float(alpha)
    raw = np.array([month ** (-value) for month in RECENCY_MONTHS], dtype=float)
    if not np.isfinite(raw).all() or raw.sum() == 0:
        return {month: 1.0 / len(RECENCY_MONTHS) for month in RECENCY_MONTHS}
    normalized = raw / raw.sum()
    return {month: float(weight) for month, weight in zip(RECENCY_MONTHS, normalized)}


def calc_momentum(
    series: PriceSeries,
    vol_adjusted: bool = False,
    score_mode: str = SCORE_MODE_RW_3_6_9_12,
    score_param: float | None = None,
) -> float:
    closes = series.closes()
    ret_12 = _window_return(closes, WINDOW_LONG)
    ret_9 = _window_return(closes, 189)  # ~9 months
    ret_6 = _window_return(closes, 126)  # ~6 months
    ret_3 = _window_return(closes, 63)  # ~3 months
    ret_1m = _window_return(closes, WINDOW_1M)

    if score_mode == SCORE_MODE_12M_MINUS_1M:
        momentum = ret_12 - ret_1m
    elif score_mode == SCORE_MODE_BLEND_6_12:
        momentum = 0.5 * ret_6 + 0.5 * ret_12
    elif score_mode == SCORE_MODE_RW_3_6_9_12:
        momentum = 0.4 * ret_3 + 0.2 * ret_6 + 0.2 * ret_9 + 0.2 * ret_12
    elif score_mode == SCORE_MODE_RW_ALPHA:
        weights = _recency_weights(score_param)
        horizon_returns = {3: ret_3, 6: ret_6, 9: ret_9, 12: ret_12}
        momentum = sum(weights[month] * horizon_returns[month] for month in RECENCY_MONTHS)
    elif score_mode == SCORE_MODE_BLEND_LAMBDA:
        lam = score_param if score_param is not None else 0.5
        lam = min(max(lam, 0.0), 1.0)
        momentum = lam * ret_6 + (1 - lam) * ret_12
    else:
        momentum = ret_12 - ret_1m

    if vol_adjusted:
        vol = _rolling_volatility(closes, WINDOW_LONG)
        if vol > 0:
            return momentum / vol
        return 0.0
    return momentum


def rank_momentum(
    series_list: Sequence[PriceSeries],
    vol_adjusted: bool = False,
    score_mode: str = SCORE_MODE_RW_3_6_9_12,
    score_param: float | None = None,
) -> list[MomentumScore]:
    scores = [
        MomentumScore(
            symbol=series.symbol,
            score=calc_momentum(
                series,
                vol_adjusted=vol_adjusted,
                score_mode=score_mode,
                score_param=score_param,
            ),
        )
        for series in series_list
    ]
    scores.sort(key=lambda score: score.score, reverse=True)
    return scores

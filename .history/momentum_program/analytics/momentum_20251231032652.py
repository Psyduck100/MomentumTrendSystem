from collections.abc import Sequence
import numpy as np

from momentum_program.data_models import MomentumScore, PriceSeries


WINDOW_6M = 126  # ~6 months of trading days
WINDOW_1M = 21  # ~1 month of trading days


def _window_return(closes: list[float], window: int) -> float:
    if len(closes) <= window:
        return 0.0
    start = closes[-window]
    end = closes[-1]
    if start == 0:
        return 0.0
    return (end - start) / start


def _rolling_volatility(closes: list[float], window: int = WINDOW_6M) -> float:
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


def calc_momentum(series: PriceSeries, vol_adjusted: bool = False) -> float:
    closes = series.closes()
    # Score = 6M return minus 1M return
    ret_6m = _window_return(closes, WINDOW_6M)
    ret_1m = _window_return(closes, WINDOW_1M)
    momentum = ret_6m - ret_1m

    if vol_adjusted:
        vol = _rolling_volatility(closes, WINDOW_6M)
        if vol > 0:
            return momentum / vol
        return 0.0
    return momentum


def rank_momentum(
    series_list: Sequence[PriceSeries], vol_adjusted: bool = False
) -> list[MomentumScore]:
    scores = [
        MomentumScore(symbol=series.symbol, score=calc_momentum(series, vol_adjusted))
        for series in series_list
    ]
    scores.sort(key=lambda score: score.score, reverse=True)
    return scores

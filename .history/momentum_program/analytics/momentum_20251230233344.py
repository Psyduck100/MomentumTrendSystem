from collections.abc import Sequence

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


def calc_momentum(series: PriceSeries) -> float:
    closes = series.closes()
    # Score = 6M return minus 1M return
    ret_6m = _window_return(closes, WINDOW_6M)
    ret_1m = _window_return(closes, WINDOW_1M)
    return ret_6m - ret_1m


def rank_momentum(series_list: Sequence[PriceSeries]) -> list[MomentumScore]:
    scores = [
        MomentumScore(symbol=series.symbol, score=calc_momentum(series))
        for series in series_list
    ]
    scores.sort(key=lambda score: score.score, reverse=True)
    return scores

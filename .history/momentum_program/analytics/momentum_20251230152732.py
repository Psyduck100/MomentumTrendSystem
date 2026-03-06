from collections.abc import Sequence

from momentum_program.data_models import MomentumScore, PriceSeries


def calc_total_return(series: PriceSeries) -> float:
    closes = series.closes()
    if len(closes) < 2:
        return 0.0
    return (closes[-1] - closes[0]) / closes[0]


def rank_momentum(series_list: Sequence[PriceSeries]) -> list[MomentumScore]:
    scores = [MomentumScore(symbol=series.symbol, score=calc_total_return(series)) for series in series_list]
    scores.sort(key=lambda score: score.score, reverse=True)
    return scores

from datetime import datetime

from momentum_program.analytics.filters import top_n
from momentum_program.analytics.momentum import rank_momentum
from momentum_program.config import StrategyConfig
from momentum_program.data_models import MomentumScore, PortfolioTarget, PriceSeries


def build_targets(
    price_series: list[PriceSeries],
    cfg: StrategyConfig,
    as_of: datetime,
    bucket_map: dict[str, str] | None = None,
) -> PortfolioTarget:
    bucket_map = bucket_map or {}
    momentum_scores = rank_momentum(price_series)

    # Group scores by bucket; default bucket if none supplied.
    grouped: dict[str, list[MomentumScore]] = {}
    for score in momentum_scores:
        bucket = bucket_map.get(score.symbol, "default")
        grouped.setdefault(bucket, []).append(score)

    winners: list[MomentumScore] = []
    top_n_per_bucket = cfg.top_n_per_bucket or cfg.top_n
    for scores in grouped.values():
        winners.extend(top_n(scores, top_n_per_bucket))

    if not winners:
        return PortfolioTarget(weights={}, as_of=as_of)

    weight = 1.0 / len(winners)
    weights = {score.symbol: weight for score in winners}
    return PortfolioTarget(weights=weights, as_of=as_of)

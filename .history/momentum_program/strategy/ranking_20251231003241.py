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
    bucket_rankings: dict[str, list[str]] = {}
    top_n_per_bucket = cfg.top_n_per_bucket or cfg.top_n
    for bucket, scores in grouped.items():
        sorted_scores = sorted(scores, key=lambda s: s.score, reverse=True)
        top_scores = sorted_scores[:top_n_per_bucket]
        bucket_rankings[bucket] = [s.symbol for s in top_scores]
        winners.extend(top_scores)

    if not winners:
        return PortfolioTarget(weights={}, as_of=as_of, bucket_rankings=bucket_rankings)

    weight = 1.0 / len(winners)
    weights = {score.symbol: weight for score in winners}
    return PortfolioTarget(weights=weights, as_of=as_of, bucket_rankings=bucket_rankings)

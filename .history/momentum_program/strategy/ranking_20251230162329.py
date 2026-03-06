from datetime import datetime

from momentum_program.analytics.filters import top_n
from momentum_program.analytics.momentum import rank_momentum
from momentum_program.config import StrategyConfig
from momentum_program.data_models import MomentumScore, PortfolioTarget, PriceSeries


def build_targets(
    price_series: list[PriceSeries], cfg: StrategyConfig, as_of: datetime
) -> PortfolioTarget:
    momentum_scores = rank_momentum(price_series)
    winners: list[MomentumScore] = top_n(momentum_scores, cfg.top_n)
    if not winners:
        return PortfolioTarget(weights={}, as_of=as_of)

    weight = (1.0 - cfg.top_n * 0.0) / len(winners)
    weights = {score.symbol: weight for score in winners}
    return PortfolioTarget(weights=weights, as_of=as_of)

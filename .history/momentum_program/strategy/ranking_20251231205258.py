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
    prev_bucket_selection: dict[str, str | None] | None = None,
) -> PortfolioTarget:
    bucket_map = bucket_map or {}
    momentum_scores = rank_momentum(price_series, vol_adjusted=cfg.vol_adjusted)
    excluded_buckets = set(cfg.excluded_buckets or [])
    prev_bucket_selection = prev_bucket_selection or {}

    # Market filter is forced off per user preference.
    market_filter_enabled = False

    # Check market absolute momentum if enabled
    market_is_positive = True
    if market_filter_enabled:
        market_series = next(
            (s for s in price_series if s.symbol == cfg.market_ticker), None
        )
        if market_series and len(market_series.bars) >= 252:
            # 12M return
            market_12m_ret = (
                market_series.bars[-1].close - market_series.bars[-252].close
            ) / market_series.bars[-252].close
            market_is_positive = market_12m_ret > 0

    # Group scores by bucket; default bucket if none supplied.
    grouped: dict[str, list[MomentumScore]] = {}
    for score in momentum_scores:
        bucket = bucket_map.get(score.symbol, "default")
        if bucket in excluded_buckets:
            continue
        grouped.setdefault(bucket, []).append(score)

    winners: list[MomentumScore] = []
    bucket_rankings: dict[str, list[str]] = {}
    bucket_scores: dict[str, list[tuple[str, float]]] = {}
    bucket_selection: dict[str, str | None] = {}
    top_n_per_bucket = cfg.top_n_per_bucket or cfg.top_n
    for bucket, scores in grouped.items():
        # Apply market filter: if market negative, only select from defensive bucket
        if (
            market_filter_enabled
            and not market_is_positive
            and bucket != cfg.defensive_bucket
        ):
            top_sorted = sorted(scores, key=lambda s: s.score, reverse=True)[
                :top_n_per_bucket
            ]
            bucket_rankings[bucket] = [s.symbol for s in top_sorted]
            bucket_scores[bucket] = [(s.symbol, s.score) for s in top_sorted]
            bucket_selection[bucket] = None
            continue

        sorted_scores = sorted(scores, key=lambda s: s.score, reverse=True)
        top_scores = sorted_scores[:top_n_per_bucket]
        bucket_rankings[bucket] = [s.symbol for s in top_scores]
        bucket_scores[bucket] = [(s.symbol, s.score) for s in top_scores]
        # Pick leader with rank-gap turnover control
        leader = top_scores[0] if top_scores else None
        current = prev_bucket_selection.get(bucket)
        ranked_symbols = [s.symbol for s in sorted_scores]
        if (
            leader
            and current
            and cfg.rank_gap_threshold > 0
            and current in ranked_symbols
        ):
            leader_rank = ranked_symbols.index(leader)
            current_rank = ranked_symbols.index(current)
            if leader_rank > current_rank - cfg.rank_gap_threshold:
                leader = current
        # If prior pick isn't in the current ranking (e.g., dropped from universe), don't keep it.
        if leader and leader not in ranked_symbols:
            leader = None

        bucket_selection[bucket] = leader
        if leader:
            match = next((s for s in sorted_scores if s.symbol == leader), None)
            if match:
                winners.append(match)

    if not winners:
        return PortfolioTarget(
            weights={},
            as_of=as_of,
            bucket_rankings=bucket_rankings,
            bucket_selection=bucket_selection,
            bucket_scores=bucket_scores,
        )

    weight = 1.0 / len(winners)
    weights = {score.symbol: weight for score in winners}
    return PortfolioTarget(
        weights=weights,
        as_of=as_of,
        bucket_rankings=bucket_rankings,
        bucket_selection=bucket_selection,
        bucket_scores=bucket_scores,
    )

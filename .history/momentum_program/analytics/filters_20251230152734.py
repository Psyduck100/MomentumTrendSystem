from collections.abc import Sequence

from momentum_program.data_models import MomentumScore


def top_n(scores: Sequence[MomentumScore], n: int) -> list[MomentumScore]:
    return list(scores[:n])

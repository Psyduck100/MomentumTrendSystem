import numpy as np
import pandas as pd


def compute_metrics(returns: pd.Series) -> dict:
    """Compute CAGR, volatility, Sharpe, max drawdown, Sortino for a return series."""
    if len(returns) == 0:
        return {
            "cagr": 0.0,
            "volatility": 0.0,
            "sharpe": 0.0,
            "max_drawdown": 0.0,
            "sortino": 0.0,
            "total_return": 0.0,
        }

    cumulative = (1 + returns).cumprod()
    total_return = cumulative.iloc[-1] - 1

    # CAGR
    years = len(returns) / 12
    cagr = (cumulative.iloc[-1]) ** (1 / years) - 1 if years > 0 else 0.0

    # Volatility (annualized)
    volatility = returns.std() * np.sqrt(12)

    # Sharpe Ratio (assuming 0% risk-free rate)
    sharpe = cagr / volatility if volatility > 0 else 0.0

    # Max Drawdown
    max_drawdown = (cumulative / cumulative.cummax() - 1).min()

    # Sortino (penalize only downside vol)
    downside_returns = returns[returns < 0]
    downside_vol = downside_returns.std() * np.sqrt(12) if len(downside_returns) > 0 else 0.0
    sortino = cagr / downside_vol if downside_vol > 0 else 0.0

    return {
        "cagr": cagr,
        "volatility": volatility,
        "sharpe": sharpe,
        "max_drawdown": max_drawdown,
        "sortino": sortino,
        "total_return": total_return,
    }


def compute_turnover(positions_history: list[list[str]]) -> float:
    """Compute average turnover: average fraction of portfolio that changes each period."""
    if len(positions_history) < 2:
        return 0.0

    turnovers = []
    for i in range(1, len(positions_history)):
        prev_set = set(positions_history[i - 1])
        curr_set = set(positions_history[i])
        # Turnover = (additions + deletions) / 2 / portfolio_size
        additions = len(curr_set - prev_set)
        deletions = len(prev_set - curr_set)
        portfolio_size = max(len(prev_set), len(curr_set), 1)
        turnover = (additions + deletions) / 2 / portfolio_size
        turnovers.append(turnover)

    return np.mean(turnovers) if turnovers else 0.0

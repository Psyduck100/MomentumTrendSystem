import pandas as pd


def validate_prices(prices: pd.DataFrame) -> None:

    if not isinstance(prices.index, pd.DatetimeIndex):
        raise ValueError("Prices DataFrame must have a DatetimeIndex.")

    if prices.index.has_duplicates:
        raise ValueError("Prices DataFrame index contains duplicate dates.")
    if prices.isna().all().all():
        raise ValueError("Prices DataFrame contains only NaN values.")
    if not prices.index.is_monotonic_increasing:
        raise ValueError("Prices DataFrame index must be sorted ascending.")


def compute_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Compute returns from price DataFrame."""
    validate_prices(prices)
    rets = prices.pct_change()
    return rets.fillna(0.0)

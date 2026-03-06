import numpy as np
import pandas as pd
from pandas.testing import assert_series_equal

from pmtl_backtest_engine import PMTLBacktestEngine
from pmtl_fallback_strategies import CashFallback


def test_sma_signal_is_lagged_one_period():
    # Construct synthetic weekly prices to verify decisions use prior data
    dates = pd.date_range("2020-01-03", periods=4, freq="W-FRI")
    prices = pd.Series([100.0, 110.0, 90.0, 120.0], index=dates)

    # Manually build engine without downloads
    engine = PMTLBacktestEngine.__new__(PMTLBacktestEngine)
    engine.main_ticker = "TEST"
    engine.start_date = "2020-01-01"
    engine.end_date = "2020-12-31"
    engine.frequency = "W"
    engine.prices = prices

    fallback = CashFallback(engine.start_date, engine.end_date)

    # Run SMA with window=2; first period should stay in fallback due to shift
    result = engine.backtest_sma(window=2, fallback=fallback)

    # Expected: signal shifts by one period, so week3 bears the week2 buy signal
    expected = pd.Series(
        [np.nan, 0.0, -0.1818181818, 0.0], index=dates, name="SMA_2"
    )

    assert_series_equal(result.round(8), expected.round(8), check_names=True)


def test_ema_signal_is_lagged_one_period():
    # Same synthetic series; EMA span=2 with prior-period shift should behave like SMA in timing
    dates = pd.date_range("2020-01-03", periods=4, freq="W-FRI")
    prices = pd.Series([100.0, 110.0, 90.0, 120.0], index=dates)

    engine = PMTLBacktestEngine.__new__(PMTLBacktestEngine)
    engine.main_ticker = "TEST"
    engine.start_date = "2020-01-01"
    engine.end_date = "2020-12-31"
    engine.frequency = "W"
    engine.prices = prices

    fallback = CashFallback(engine.start_date, engine.end_date)

    result = engine.backtest_ema(window=2, fallback=fallback)

    expected = pd.Series(
        [np.nan, 0.0, -0.1818181818, 0.0], index=dates, name="EMA_2"
    )

    assert_series_equal(result.round(8), expected.round(8), check_names=True)
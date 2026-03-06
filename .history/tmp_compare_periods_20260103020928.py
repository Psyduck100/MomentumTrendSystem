from pathlib import Path

from momentum_program.backtest.metrics import compute_metrics
from moving_average_switch import download_prices, run_strategy

periods = [
    ("2002-01-01", "2025-12-31"),
    ("2002-01-01", "2017-12-31"),
    ("2002-01-01", "2012-12-31"),
]

cache_dir = Path("backtest_cache")

for start, end in periods:
    ma_returns = run_strategy(
        stock_ticker="SPY",
        start=start,
        end=end,
        ma_length=200,
        cash_rate=0.03,
        defensive_ticker="IEF",
        cache_dir=cache_dir,
    )
    ma_metrics = compute_metrics(ma_returns)

    prices = download_prices(["SPY"], start, end, cache_dir).sort_index()["SPY"].dropna()
    monthly = prices.resample("ME").last().dropna()
    spy_returns = monthly.pct_change().dropna()
    spy_metrics = compute_metrics(spy_returns)

    print(f"\nPeriod {start} – {end}")
    print("MA Switch (SPY↔IEF)")
    print(
        f"  CAGR {ma_metrics['cagr']*100:5.2f}% | Sharpe {ma_metrics['sharpe']:4.2f} | MaxDD {ma_metrics['max_drawdown']*100:6.2f}% | Total {(ma_metrics['total_return']*100):6.2f}%"
    )
    print("SPY Buy & Hold")
    print(
        f"  CAGR {spy_metrics['cagr']*100:5.2f}% | Sharpe {spy_metrics['sharpe']:4.2f} | MaxDD {spy_metrics['max_drawdown']*100:6.2f}% | Total {(spy_metrics['total_return']*100):6.2f}%"
    )

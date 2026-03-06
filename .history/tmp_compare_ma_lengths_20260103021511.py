from pathlib import Path

from momentum_program.backtest.metrics import compute_metrics
from moving_average_switch import download_prices, run_strategy

start = "2015-01-01"
end = "2025-12-31"
ma_lengths = [200, 100]
cache_dir = Path("backtest_cache")

prices = download_prices(["SPY"], start, end, cache_dir).sort_index()["SPY"].dropna()
monthly = prices.resample("ME").last().dropna()
spy_returns = monthly.pct_change().dropna()
spy_metrics = compute_metrics(spy_returns)

print("SPY Buy & Hold")
print(
    f"  Period {spy_returns.index[0].date()} – {spy_returns.index[-1].date()} | CAGR {spy_metrics['cagr']*100:5.2f}% | "
    f"Sharpe {spy_metrics['sharpe']:4.2f} | MaxDD {spy_metrics['max_drawdown']*100:6.2f}% | Total {(spy_metrics['total_return']*100):6.2f}%"
)

for ma_length in ma_lengths:
    ma_returns = run_strategy(
        stock_ticker="SPY",
        start=start,
        end=end,
        ma_length=ma_length,
        cash_rate=0.03,
        defensive_ticker="IEF",
        cache_dir=cache_dir,
    )
    ma_metrics = compute_metrics(ma_returns)
    print(f"\nMA Switch (length {ma_length}d)")
    print(
        f"  Period {ma_returns.index[0].date()} – {ma_returns.index[-1].date()} | CAGR {ma_metrics['cagr']*100:5.2f}% | "
        f"Sharpe {ma_metrics['sharpe']:4.2f} | MaxDD {ma_metrics['max_drawdown']*100:6.2f}% | Total {(ma_metrics['total_return']*100):6.2f}%"
    )

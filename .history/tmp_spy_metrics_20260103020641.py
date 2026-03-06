from pathlib import Path

from momentum_program.backtest.metrics import compute_metrics
from moving_average_switch import download_prices

start = "2015-01-01"
end = "2025-12-31"
cache_dir = Path("backtest_cache")

data = download_prices(["SPY"], start, end, cache_dir).sort_index()["SPY"].dropna()
monthly = data.resample("ME").last().dropna()
returns = monthly.pct_change().dropna()
metrics = compute_metrics(returns)

print("SPY Buy-and-Hold")
print(f"Period: {returns.index[0].date()} – {returns.index[-1].date()}")
print(
    f"CAGR {metrics['cagr']*100:5.2f}% | Sharpe {metrics['sharpe']:4.2f} | MaxDD {metrics['max_drawdown']*100:6.2f}%"
)
print(f"Total Return {(metrics['total_return']*100):6.2f}%")

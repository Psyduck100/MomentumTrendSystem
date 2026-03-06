import yfinance as yf
from momentum_program.backtest.metrics import compute_metrics

start = "2015-01-01"
end = "2025-12-31"

data = yf.download("SPY", start=start, end=end, progress=False)["Adj Close"]
monthly = data.resample("ME").last().dropna()
returns = monthly.pct_change().dropna()
metrics = compute_metrics(returns)

print("SPY Buy-and-Hold")
print(f"Period: {returns.index[0].date()} – {returns.index[-1].date()}")
print(
    f"CAGR {metrics['cagr']*100:5.2f}% | Sharpe {metrics['sharpe']:4.2f} | MaxDD {metrics['max_drawdown']*100:6.2f}%"
)
print(f"Total Return {(metrics['total_return']*100):6.2f}%")

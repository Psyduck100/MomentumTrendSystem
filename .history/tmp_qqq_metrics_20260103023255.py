from pathlib import Path

from momentum_program.backtest.metrics import compute_metrics
from moving_average_switch import download_prices

WINDOWS = [
    ("2001-01-01", "2025-12-31"),
    ("2001-01-01", "2013-12-31"),
    ("2013-01-01", "2025-12-31"),
    ("2006-01-01", "2020-12-31"),
]

def describe(label: str, metrics: dict[str, float]) -> str:
    return (
        f"{label}: CAGR {metrics['cagr']*100:5.2f}% | Sharpe {metrics['sharpe']:5.2f} | "
        f"MaxDD {metrics['max_drawdown']*100:6.2f}% | Total {metrics['total_return']*100:6.2f}%"
    )


def compute_buy_hold(ticker: str, start: str, end: str) -> dict[str, float]:
    prices = (
        download_prices([ticker], start, end, Path("backtest_cache"))
        .sort_index()[ticker]
        .dropna()
    )
    monthly = prices.resample("ME").last().dropna()
    returns = monthly.pct_change().dropna()
    return compute_metrics(returns)


def main() -> None:
    for start, end in WINDOWS:
        print(f"\nWindow {start} → {end}")
        qqq = compute_buy_hold("QQQ", start, end)
        spy = compute_buy_hold("SPY", start, end)
        print("  " + describe("QQQ", qqq))
        print("  " + describe("SPY", spy))


if __name__ == "__main__":
    main()

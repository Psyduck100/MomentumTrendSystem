from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd
import yfinance as yf

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backtest.tester.Config import EngineConfig
from backtest.tester.Data_model import compute_returns
from backtest.tester.Defensive import ConstantReturnDefensive
from backtest.tester.Engine import run_engine
from backtest.tester.RebalanceGate import MonthlyRebalanceGate
from backtest.tester.Rules import AlwaysEnterRule, NeverExitRule
from backtest.tester.Selector import TopMomentumSelector
from backtest.tester.UniverseProvider import StaticUniverse


UNIVERSE = ["XLK", "XLV", "XLF", "XLI", "XLY", "XLE", "XAR"]
BENCHMARK = "SPY"


def _download_prices(tickers: list[str], start: str, end: str | None = None) -> pd.DataFrame:
    data = yf.download(
        tickers=tickers,
        start=start,
        end=end,
        auto_adjust=True,
        progress=False,
        threads=True,
    )

    if isinstance(data.columns, pd.MultiIndex):
        if "Close" in data.columns.get_level_values(0):
            px = data["Close"]
        elif "Adj Close" in data.columns.get_level_values(0):
            px = data["Adj Close"]
        else:
            swapped = data.swaplevel(0, 1, axis=1).sort_index(axis=1)
            if "Close" in swapped.columns.get_level_values(1):
                px = swapped.xs("Close", level=1, axis=1)
            elif "Adj Close" in swapped.columns.get_level_values(1):
                px = swapped.xs("Adj Close", level=1, axis=1)
            else:
                raise ValueError("Could not find Close/Adj Close in downloaded data.")
    else:
        if "Close" in data.columns:
            px = pd.DataFrame({tickers[0]: data["Close"]})
        elif "Adj Close" in data.columns:
            px = pd.DataFrame({tickers[0]: data["Adj Close"]})
        else:
            raise ValueError("Could not find Close/Adj Close in single-ticker data.")

    px = px.dropna(how="all").ffill().dropna(how="all")
    px.index = pd.to_datetime(px.index)
    return px.reindex(columns=[t for t in tickers if t in px.columns])


def _metrics_from_daily(daily: pd.DataFrame, ann: int = 252) -> dict[str, float]:
    ret = daily["ret"].dropna()
    eq = daily["equity"].dropna()
    if len(ret) < 2:
        return {"cagr": np.nan, "sharpe": np.nan, "maxdd": np.nan, "total_return": np.nan}

    cagr = float(eq.iloc[-1] ** (ann / len(ret)) - 1.0)
    vol = float(ret.std(ddof=1))
    sharpe = float(np.sqrt(ann) * ret.mean() / vol) if vol > 0 else np.nan
    peak = eq.cummax()
    maxdd = float((eq / peak - 1.0).min())
    total_return = float(eq.iloc[-1] - 1.0)
    return {
        "cagr": cagr,
        "sharpe": sharpe,
        "maxdd": maxdd,
        "total_return": total_return,
    }


def _buy_hold_metrics(price: pd.Series, ann: int = 252) -> dict[str, float]:
    if isinstance(price, pd.DataFrame):
        if price.shape[1] != 1:
            raise ValueError("Expected a single price series for buy-and-hold metrics.")
        price = price.iloc[:, 0]
    ret = price.pct_change().fillna(0.0)
    eq = (1.0 + ret).cumprod()
    cagr = float(eq.iloc[-1] ** (ann / len(eq)) - 1.0)
    vol = float(ret.std(ddof=1))
    sharpe = float(np.sqrt(ann) * ret.mean() / vol) if vol > 0 else np.nan
    peak = eq.cummax()
    maxdd = float((eq / peak - 1.0).min())
    total_return = float(eq.iloc[-1] - 1.0)
    return {
        "cagr": cagr,
        "sharpe": sharpe,
        "maxdd": maxdd,
        "total_return": total_return,
    }


def _download_raw_close(ticker: str, start: str, end: str | None = None) -> pd.Series:
    data = yf.download(
        tickers=ticker,
        start=start,
        end=end,
        auto_adjust=False,
        progress=False,
        threads=True,
    )
    if "Close" not in data.columns:
        raise ValueError(f"Raw close not found for {ticker}.")
    s = data["Close"]
    if isinstance(s, pd.DataFrame):
        if ticker in s.columns:
            s = s[ticker]
        elif s.shape[1] == 1:
            s = s.iloc[:, 0]
        else:
            raise ValueError(f"Unexpected close shape for {ticker}: {s.shape}")
    s = s.dropna()
    s.index = pd.to_datetime(s.index)
    s.name = ticker
    return s


def main() -> None:
    start_date = "2012-01-01"
    end_date = None
    lookback_days = 126
    top_n = 1

    tickers = UNIVERSE + [BENCHMARK]
    prices = _download_prices(tickers, start=start_date, end=end_date)

    strategy_prices = prices[UNIVERSE].dropna(how="all")
    benchmark_prices = prices[BENCHMARK].dropna()
    benchmark_raw_close = _download_raw_close(BENCHMARK, start=start_date, end=end_date)

    cfg = EngineConfig(
        cost_bps=2.0,
        slippage_bps=1.0,
        initial_capital=1.0,
        trade_delay=1,
    )

    selector = TopMomentumSelector(lookback_days=lookback_days, top_n=top_n)

    result = run_engine(
        prices=strategy_prices,
        config=cfg,
        universe_provider=StaticUniverse(UNIVERSE),
        rebalance_gate=MonthlyRebalanceGate(when="last"),
        entry_rule=AlwaysEnterRule(),
        exit_rule=NeverExitRule(),
        selector=selector,
        defensive_asset=ConstantReturnDefensive(rate_annual=0.0),
    )

    strat_metrics = _metrics_from_daily(result.daily)
    bench_tr_metrics = _buy_hold_metrics(
        benchmark_prices.loc[result.daily.index.min() : result.daily.index.max()]
    )
    bench_price_metrics = _buy_hold_metrics(
        benchmark_raw_close.loc[result.daily.index.min() : result.daily.index.max()]
    )

    returns = compute_returns(strategy_prices)
    latest_date = strategy_prices.index[-1]
    latest_weights = selector.select(latest_date, strategy_prices, returns, UNIVERSE)
    latest_pick = max(latest_weights, key=latest_weights.get) if latest_weights else "NONE"

    print("Sector Rotation Backtest (Monthly Top-1, 6M Momentum)")
    print(f"Universe: {', '.join(UNIVERSE)}")
    print(f"Period: {result.daily.index.min().date()} -> {result.daily.index.max().date()}")
    print(f"Trades logged: {len(result.trades)}")
    print(f"Current recommended holding: {latest_pick} (as of {latest_date.date()})")

    print("\nStrategy metrics:")
    print(
        f"  CAGR: {strat_metrics['cagr']:.2%} | Sharpe: {strat_metrics['sharpe']:.2f} "
        f"| MaxDD: {strat_metrics['maxdd']:.2%} | Total: {strat_metrics['total_return']:.2%}"
    )

    print(f"\nBenchmark ({BENCHMARK}) buy-and-hold (price-only):")
    print(
        f"  CAGR: {bench_price_metrics['cagr']:.2%} | Sharpe: {bench_price_metrics['sharpe']:.2f} "
        f"| MaxDD: {bench_price_metrics['maxdd']:.2%} | Total: {bench_price_metrics['total_return']:.2%}"
    )

    print(f"\nBenchmark ({BENCHMARK}) buy-and-hold (total return, dividends reinvested):")
    print(
        f"  CAGR: {bench_tr_metrics['cagr']:.2%} | Sharpe: {bench_tr_metrics['sharpe']:.2f} "
        f"| MaxDD: {bench_tr_metrics['maxdd']:.2%} | Total: {bench_tr_metrics['total_return']:.2%}"
    )

    if not result.trades.empty:
        print("\nLast 10 trade events:")
        print(result.trades.tail(10).to_string(index=False))


if __name__ == "__main__":
    main()

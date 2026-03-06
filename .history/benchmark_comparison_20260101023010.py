"""Compare strategy returns to market benchmarks overall and per bucket."""
from pathlib import Path

import pandas as pd
import yfinance as yf
import numpy as np

from momentum_program.config import AppConfig
from momentum_program.universe.bucket_csv_provider import BucketedCsvUniverseProvider
from momentum_program.universe.csv_provider import CsvUniverseProvider
from momentum_program.universe.tradingview_provider import TradingViewUniverseProvider
from momentum_program.backtest.engine import backtest_momentum
from momentum_program.backtest.metrics import compute_metrics, compute_turnover
from momentum_program.analytics.constants import (
    SCORE_MODE_12M_MINUS_1M,
    SCORE_MODE_RW_3_6_9_12,
)


# Benchmark map: bucket -> representative ETF
BENCHMARK_MAP = {
    "US_equities": "SPY",
    "US_small_mid_cap": "IWM",
    "Emerging_Markets": "EEM",
    "Intl_developed": "EFA",
    "Bonds": "AGG",
    "Commodities": "DBC",
    "REITs": "VNQ",
}


def download_benchmark(ticker: str, start: str, end: str) -> pd.Series:
    """Download benchmark returns for a given period."""
    data = yf.download(ticker, start=start, end=end, progress=False)
    if data.empty or "Adj Close" not in data.columns:
        return pd.Series(dtype=float)
    prices = data["Adj Close"].resample("ME").last()
    returns = prices.pct_change()
    return returns


def compute_benchmark_cagr(ticker: str, start: str, end: str) -> float:
    """Compute annualized return for a benchmark."""
    data = yf.download(ticker, start=start, end=end, progress=False)
    if data.empty or "Adj Close" not in data.columns:
        return float("nan")
    
    if isinstance(data.columns, pd.MultiIndex):
        prices = data["Adj Close"][ticker] if ticker in data["Adj Close"].columns else data["Adj Close"].iloc[:, 0]
    else:
        prices = data["Adj Close"]
    
    prices = prices.dropna()
    if len(prices) < 2:
        return float("nan")
    
    start_price = prices.iloc[0]
    end_price = prices.iloc[-1]
    years = len(prices) / 252  # approximate trading days
    
    if start_price <= 0:
        return float("nan")
    
    total_return = (end_price / start_price) - 1
    cagr = (1 + total_return) ** (1 / years) - 1
    return cagr


def main() -> None:
    cfg = AppConfig()
    bucket_folder = Path("CSVs")
    bucket_csvs = list(bucket_folder.glob("*.csv")) if bucket_folder.exists() else []
    csv_path = Path("etfs.csv")

    if bucket_csvs:
        universe = BucketedCsvUniverseProvider(bucket_folder)
    elif csv_path.exists():
        universe = CsvUniverseProvider(csv_path)
    else:
        universe = TradingViewUniverseProvider(chunk=50)

    tickers = universe.get_tickers()
    bucket_map = universe.get_bucket_map()

    # Run strategy for key windows with both modes
    windows = [
        ("2012-01-01", "2022-12-31"),
        ("2015-01-01", "2025-12-31"),
        ("2018-01-01", "2022-12-31"),
        ("2019-01-01", "2023-12-31"),
        ("2020-01-01", "2024-12-31"),
    ]

    results = []

    for start_date, end_date in windows:
        print(f"\nProcessing {start_date} -> {end_date}")
        
        # Run RW gap 0
        data_rw = backtest_momentum(
            tickers=tickers,
            bucket_map=bucket_map,
            start_date=start_date,
            end_date=end_date,
            top_n_per_bucket=cfg.strategy.top_n_per_bucket or cfg.strategy.top_n,
            lookback_long=12,
            lookback_short=1,
            vol_adjusted=False,
            vol_lookback=6,
            market_filter=False,
            market_ticker="SPY",
            defensive_bucket="Bonds",
            market_threshold=0.0,
            rank_gap_threshold=0,
            score_mode=SCORE_MODE_RW_3_6_9_12,
        )
        
        # Run 12m-1m gap 2
        data_12m = backtest_momentum(
            tickers=tickers,
            bucket_map=bucket_map,
            start_date=start_date,
            end_date=end_date,
            top_n_per_bucket=cfg.strategy.top_n_per_bucket or cfg.strategy.top_n,
            lookback_long=12,
            lookback_short=1,
            vol_adjusted=False,
            vol_lookback=6,
            market_filter=False,
            market_ticker="SPY",
            defensive_bucket="Bonds",
            market_threshold=0.0,
            rank_gap_threshold=2,
            score_mode=SCORE_MODE_12M_MINUS_1M,
        )
        
        if data_rw["overall_returns"].empty or data_12m["overall_returns"].empty:
            continue
        
        # Strategy metrics
        rw_metrics = compute_metrics(data_rw["overall_returns"]["return"])
        m12_metrics = compute_metrics(data_12m["overall_returns"]["return"])
        
        # Market benchmark
        spy_cagr = compute_benchmark_cagr("SPY", start_date, end_date)
        
        # Per-bucket benchmarks
        bucket_benchmarks = {}
        for bucket, benchmark_ticker in BENCHMARK_MAP.items():
            if bucket in data_rw["bucket_returns"] and not data_rw["bucket_returns"][bucket].empty:
                bucket_cagr = compute_metrics(data_rw["bucket_returns"][bucket]["return"])["cagr"]
                bench_cagr = compute_benchmark_cagr(benchmark_ticker, start_date, end_date)
                bucket_benchmarks[bucket] = {
                    "strategy": bucket_cagr,
                    "benchmark": bench_cagr,
                    "ticker": benchmark_ticker,
                }
        
        results.append({
            "window": f"{start_date}->{end_date}",
            "rw_cagr": rw_metrics["cagr"],
            "12m_cagr": m12_metrics["cagr"],
            "spy_cagr": spy_cagr,
            "rw_alpha": rw_metrics["cagr"] - spy_cagr,
            "12m_alpha": m12_metrics["cagr"] - spy_cagr,
            "buckets": bucket_benchmarks,
        })
    
    # Print comparison
    print("\n" + "=" * 100)
    print("STRATEGY vs MARKET COMPARISON")
    print("=" * 100)
    print(f"\n{'Window':<25} {'RW gap=0':<12} {'12m gap=2':<12} {'SPY':<12} {'RW Alpha':<12} {'12m Alpha':<12}")
    print("-" * 100)
    
    for r in results:
        print(
            f"{r['window']:<25} "
            f"{r['rw_cagr']*100:>10.2f}%  "
            f"{r['12m_cagr']*100:>10.2f}%  "
            f"{r['spy_cagr']*100:>10.2f}%  "
            f"{r['rw_alpha']*100:>10.2f}%  "
            f"{r['12m_alpha']*100:>10.2f}%"
        )
    
    # Print bucket-level comparison for most recent full window
    print("\n" + "=" * 100)
    print("BUCKET-LEVEL COMPARISON (2015-2025 window, RW gap=0)")
    print("=" * 100)
    recent = [r for r in results if "2015-01-01->2025-12-31" in r["window"]]
    if recent:
        print(f"\n{'Bucket':<20} {'Strategy CAGR':<15} {'Benchmark':<10} {'Bench CAGR':<15} {'Alpha':<12}")
        print("-" * 100)
        for bucket, data in sorted(recent[0]["buckets"].items()):
            alpha = data["strategy"] - data["benchmark"]
            print(
                f"{bucket:<20} "
                f"{data['strategy']*100:>13.2f}%  "
                f"{data['ticker']:<10} "
                f"{data['benchmark']*100:>13.2f}%  "
                f"{alpha*100:>10.2f}%"
            )
    
    print("\n" + "=" * 100)


if __name__ == "__main__":
    main()

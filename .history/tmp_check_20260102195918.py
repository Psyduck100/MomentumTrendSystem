from pathlib import Path

from comprehensive_walkforward import generate_configs, run_backtest_config, WINDOW_TICKER_CACHE
from momentum_program.universe.bucket_csv_provider import BucketedCsvUniverseProvider

provider = BucketedCsvUniverseProvider(Path("CSVs"))
tickers = provider.get_tickers()
bucket_map = provider.get_bucket_map()

print(f"Loaded {len(tickers)} tickers across {len(set(bucket_map.values()))} buckets")
if not tickers:
    raise SystemExit("No tickers loaded; check CSV folder")

config = generate_configs()[0]
metrics = run_backtest_config(
    tickers,
    bucket_map,
    start_date="2015-01-01",
    end_date="2018-12-31",
    config=config,
    lookback_buffer_months=12,
)

if not metrics:
    raise SystemExit("Backtest returned no metrics; investigate data availability")

print(f"Sharpe {metrics['sharpe']:.3f} | CAGR {metrics['cagr']*100:.2f}% | MaxDD {metrics['max_dd']*100:.2f}%")
print(f"Months of returns: {metrics['n_months']}")
print("First few returns:")
print(metrics["returns"].head())
print("Ticker cache keys (sample):", list(WINDOW_TICKER_CACHE.keys())[:3])

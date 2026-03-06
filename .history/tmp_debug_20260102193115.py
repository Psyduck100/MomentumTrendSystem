from pathlib import Path
from comprehensive_walkforward import run_backtest_config, generate_configs
import yfinance as yf
import pandas as pd
from momentum_program.universe.bucket_csv_provider import BucketedCsvUniverseProvider

bucket_folder = Path('CSVs')
universe = BucketedCsvUniverseProvider(bucket_folder)
tickers = universe.get_tickers()
bucket_map = universe.get_bucket_map()
print('tickers', len(tickers))
print(tickers)
df = yf.download(tickers, start='2015-01-01', end='2017-12-31', progress=False)
print('raw columns sample', df.columns[:10])
print('is_multi', isinstance(df.columns, pd.MultiIndex))
configs = generate_configs()
print('configs', len(configs))
config = configs[0]
print('first', config)
metrics = run_backtest_config(tickers, bucket_map, '2015-01-01', '2017-12-31', config)
print(metrics)

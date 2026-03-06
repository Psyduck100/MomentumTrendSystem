import pandas as pd

df = pd.read_csv('pmtl_ma_sweep_annual_returns_tb3ms.csv', index_col=0)

# Show GLD benchmark vs top strategies
strategies = ['GLD_benchmark', 'EMA_100', 'SMA_100', 'EMA_110', 'SMA_110']
years = [str(y) for y in range(2005, 2026)]

selected = df.loc[strategies, years]

print("="*100)
print("ANNUAL RETURNS: TB3MS FALLBACK STRATEGIES")
print("="*100)
print("\n" + selected.to_string())

print("\n" + "="*100)
print("SUMMARY STATISTICS")
print("="*100)

for strategy in strategies:
    returns = df.loc[strategy, years].astype(float)
    print(f"\n{strategy:15} - Mean: {returns.mean():.2%}, Std: {returns.std():.2%}, Min: {returns.min():.2%}, Max: {returns.max():.2%}")

print("\n" + "="*100)
print("POSITIVE YEARS COUNT (out of 21 years: 2005-2025)")
print("="*100)
for strategy in strategies:
    returns = df.loc[strategy, years].astype(float)
    positive_count = (returns > 0).sum()
    print(f"{strategy:15} - {positive_count}/21 positive years ({positive_count/21*100:.1f}%)")

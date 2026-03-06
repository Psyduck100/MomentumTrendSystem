import pandas as pd

df = pd.read_csv('CSVs/TB3MS.csv')
print("First 10 rows:")
print(df.head(10))
print("\n...Last 10 rows:")
print(df.tail(10))
print(f"\nMean TB3MS: {df['TB3MS'].mean():.2f}")
print(f"Max TB3MS: {df['TB3MS'].max():.2f}")
print(f"Min TB3MS: {df['TB3MS'].min():.2f}")

# Check monthly return conversion
df['observation_date'] = pd.to_datetime(df['observation_date'])
df['TB3MS_monthly_ret'] = df['TB3MS'] / 100 / 12
print(f"\nAverage monthly TB3MS return (annualized /12): {df['TB3MS_monthly_ret'].mean():.6f}")
print(f"Max monthly TB3MS return: {df['TB3MS_monthly_ret'].max():.6f}")

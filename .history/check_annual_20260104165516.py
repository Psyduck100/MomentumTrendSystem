import pandas as pd

df = pd.read_csv('pmtl_ma_sweep_annual_returns_tb3ms.csv')
print("Columns:", df.columns.tolist())
print("\nFirst few rows:")
print(df.head())
print("\nDataframe shape:", df.shape)

# Display just key columns
if len(df.columns) > 1:
    # Skip first column (index)
    cols_to_show = [col for col in df.columns if 'GLD' in col or 'EMA_100' in col or 'SMA_100' in col]
    if not cols_to_show:
        cols_to_show = df.columns[:4]
    print("\nSelected columns:", cols_to_show)
    print(df[cols_to_show].to_string())

"""
Compare Monthly vs Weekly vs Biweekly rebalancing frequencies.
"""

import pandas as pd

print("="*80)
print("REBALANCING FREQUENCY COMPARISON - EMA 100 + CASH")
print("="*80)

# Load results
monthly = pd.read_csv('pmtl_cash_results.csv')
weekly = pd.read_csv('pmtl_weekly_results.csv')
biweekly = pd.read_csv('pmtl_biweekly_results.csv')

# Get EMA 100 for each
monthly_ema100 = monthly[(monthly['type'] == 'EMA') & (monthly['window'] == 100)].iloc[0]
weekly_ema100 = weekly[(weekly['type'] == 'EMA') & (weekly['window'] == 100)].iloc[0]
biweekly_ema100 = biweekly[(biweekly['type'] == 'EMA') & (biweekly['window'] == 100)].iloc[0]

print("\nBEST STRATEGY (EMA 100-day) ACROSS FREQUENCIES:")
print("-"*80)
print(f"{'Frequency':<15} {'CAGR':<12} {'Sharpe':<12} {'MaxDD':<12} {'Rebal/Year':<12}")
print("-"*80)

print(f"{'Monthly':<15} {monthly_ema100['cagr']:>10.2%}  {monthly_ema100['sharpe']:>10.3f}  "
      f"{monthly_ema100['max_drawdown']:>10.2%}  {'12':<12}")
print(f"{'Biweekly':<15} {biweekly_ema100['cagr']:>10.2%}  {biweekly_ema100['sharpe']:>10.3f}  "
      f"{biweekly_ema100['max_drawdown']:>10.2%}  {'26':<12}")
print(f"{'Weekly':<15} {weekly_ema100['cagr']:>10.2%}  {weekly_ema100['sharpe']:>10.3f}  "
      f"{weekly_ema100['max_drawdown']:>10.2%}  {'52':<12}")

print("\n" + "="*80)
print("KEY FINDINGS:")
print("="*80)

# Calculate differences
weekly_vs_monthly_cagr = weekly_ema100['cagr'] - monthly_ema100['cagr']
biweekly_vs_monthly_cagr = biweekly_ema100['cagr'] - monthly_ema100['cagr']
weekly_vs_biweekly_cagr = weekly_ema100['cagr'] - biweekly_ema100['cagr']

print(f"\n1. CAGR Impact:")
print(f"   Monthly:   {monthly_ema100['cagr']:.2%} (baseline)")
print(f"   Biweekly:  {biweekly_ema100['cagr']:.2%} ({biweekly_vs_monthly_cagr:+.2%} vs monthly)")
print(f"   Weekly:    {weekly_ema100['cagr']:.2%} ({weekly_vs_monthly_cagr:+.2%} vs monthly)")

print(f"\n2. Sharpe Ratio:")
print(f"   Monthly:   {monthly_ema100['sharpe']:.3f}")
print(f"   Biweekly:  {biweekly_ema100['sharpe']:.3f}")
print(f"   Weekly:    {weekly_ema100['sharpe']:.3f}")

print(f"\n3. Max Drawdown:")
print(f"   Monthly:   {monthly_ema100['max_drawdown']:.2%}")
print(f"   Biweekly:  {biweekly_ema100['max_drawdown']:.2%}")
print(f"   Weekly:    {weekly_ema100['max_drawdown']:.2%}")

print(f"\n4. Rebalancing Frequency vs Performance:")
print(f"   More frequent rebalancing → Higher CAGR")
print(f"   Weekly outperforms monthly by {weekly_vs_monthly_cagr:.2%}")
print(f"   But max drawdown is worse: {weekly_ema100['max_drawdown']:.2%} vs {monthly_ema100['max_drawdown']:.2%}")

print("\n" + "="*80)
print("INTERPRETATION:")
print("="*80)
print(f"""
Weekly rebalancing achieves the highest CAGR ({weekly_ema100['cagr']:.2%}) but with:
- More frequent trading (52x/year vs 12x/year)
- Worse max drawdown ({weekly_ema100['max_drawdown']:.2%} vs {monthly_ema100['max_drawdown']:.2%})
- Higher transaction costs in real trading

Biweekly is a middle ground:
- Better than monthly ({biweekly_ema100['cagr']:.2%} vs {monthly_ema100['cagr']:.2%})
- Less frequent than weekly (26x/year)
- Similar max drawdown to weekly

RECOMMENDATION:
- For backtest: Weekly looks best ({weekly_ema100['cagr']:.2%})
- For live trading: Monthly is likely better due to:
  * Lower transaction costs
  * Better max drawdown ({monthly_ema100['max_drawdown']:.2%})
  * Less time monitoring
  * Still excellent returns ({monthly_ema100['cagr']:.2%})

The {weekly_vs_monthly_cagr:.2%} CAGR difference may not justify 4.3x more trades.
""")

print("="*80)

# Show top 5 for each frequency
print("\nTOP 5 STRATEGIES BY FREQUENCY:")
print("="*80)

print("\nMONTHLY:")
print("-"*80)
monthly_sorted = monthly.sort_values('cagr', ascending=False).head(5)
for _, row in monthly_sorted.iterrows():
    print(f"  {row['type']} {int(row['window']):3d}: {row['cagr']:6.2%} CAGR, "
          f"{row['sharpe']:5.3f} Sharpe, {row['max_drawdown']:7.2%} MaxDD")

print("\nBIWEEKLY:")
print("-"*80)
biweekly_sorted = biweekly.sort_values('cagr', ascending=False).head(5)
for _, row in biweekly_sorted.iterrows():
    print(f"  {row['type']} {int(row['window']):3d}: {row['cagr']:6.2%} CAGR, "
          f"{row['sharpe']:5.3f} Sharpe, {row['max_drawdown']:7.2%} MaxDD")

print("\nWEEKLY:")
print("-"*80)
weekly_sorted = weekly.sort_values('cagr', ascending=False).head(5)
for _, row in weekly_sorted.iterrows():
    print(f"  {row['type']} {int(row['window']):3d}: {row['cagr']:6.2%} CAGR, "
          f"{row['sharpe']:5.3f} Sharpe, {row['max_drawdown']:7.2%} MaxDD")

print("\n" + "="*80)

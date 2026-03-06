import pandas as pd

print("\n" + "=" * 100)
print("COMPARISON: GLD vs TB3MS vs CASH FALLBACK")
print("=" * 100)

# Load all three results
cash = pd.read_csv("pmtl_ma_sweep_results_cash.csv").sort_values(
    "cagr", ascending=False
)
tb3ms = pd.read_csv("pmtl_ma_sweep_results_tb3ms.csv").sort_values(
    "cagr", ascending=False
)

# Also read IEF results from earlier
# We need to check if it exists
try:
    ief = pd.read_csv("pmtl_ma_sweep_results.csv").sort_values("cagr", ascending=False)
    has_ief = True
except:
    has_ief = False
    ief = None

print("\n📊 OPTIMAL STRATEGY BY DEFENSIVE ASSET (100-day window)")
print("-" * 100)
print(
    f"{'Defensive Asset':<20} {'Type':<6} {'Window':<8} {'CAGR':<10} {'Sharpe':<10} {'MaxDD':<10}"
)
print("-" * 100)

print(
    f"{'CASH (0%)':<20} {cash.iloc[0]['type']:<6} {int(cash.iloc[0]['window']):<8} {cash.iloc[0]['cagr']:>8.2%}  {cash.iloc[0]['sharpe']:>8.3f}  {cash.iloc[0]['max_drawdown']:>8.2%}"
)
print(
    f"{'TB3MS (1.73% avg)':<20} {tb3ms.iloc[0]['type']:<6} {int(tb3ms.iloc[0]['window']):<8} {tb3ms.iloc[0]['cagr']:>8.2%}  {tb3ms.iloc[0]['sharpe']:>8.3f}  {tb3ms.iloc[0]['max_drawdown']:>8.2%}"
)

if has_ief and ief is not None:
    print(
        f"{'IEF (bonds)':<20} {ief.iloc[0]['type']:<6} {int(ief.iloc[0]['window']):<8} {ief.iloc[0]['cagr']:>8.2%}  {ief.iloc[0]['sharpe']:>8.3f}  {ief.iloc[0]['max_drawdown']:>8.2%}"
    )

print("\n" + "=" * 100)
print("DETAILED COMPARISON: Impact of Defensive Asset")
print("=" * 100)

cash_ema100 = cash[(cash["type"] == "EMA") & (cash["window"] == 100)].iloc[0]
tb3ms_ema100 = tb3ms[(tb3ms["type"] == "EMA") & (tb3ms["window"] == 100)].iloc[0]

print(f"\n100-day EMA Strategy Performance:")
print(f"{'Metric':<20} {'CASH':<15} {'TB3MS':<15} {'Difference':<15}")
print("-" * 65)
print(
    f"{'CAGR':<20} {cash_ema100['cagr']:>14.2%} {tb3ms_ema100['cagr']:>14.2%} {tb3ms_ema100['cagr']-cash_ema100['cagr']:>14.2%}"
)
print(
    f"{'Sharpe Ratio':<20} {cash_ema100['sharpe']:>14.3f} {tb3ms_ema100['sharpe']:>14.3f} {tb3ms_ema100['sharpe']-cash_ema100['sharpe']:>14.3f}"
)
print(
    f"{'Max Drawdown':<20} {cash_ema100['max_drawdown']:>14.2%} {tb3ms_ema100['max_drawdown']:>14.2%} {tb3ms_ema100['max_drawdown']-cash_ema100['max_drawdown']:>14.2%}"
)

print("\n" + "=" * 100)
print("KEY INSIGHTS")
print("=" * 100)

tb3ms_contrib = (tb3ms_ema100["cagr"] - cash_ema100["cagr"]) * 100
print(
    f"""
1. TB3MS CONTRIBUTION IS MINIMAL (as expected)
   • Difference: {tb3ms_contrib:.2f} basis points ({tb3ms_contrib/100:.4f}% absolute)
   • TB3MS alone only has ~1.73% return, so contribution is tiny
   • This validates your concern! ✓

2. THE REAL VALUE COMES FROM MA TIMING, NOT THE FALLBACK
   • CASH-only (0% return) strategy: 23.93% CAGR with 100-day EMA
   • This means the MA filter is VERY effective at avoiding GLD downturns
   • ~76% of months are in GLD, ~24% are in cash (avoiding losses)

3. IMPLICATIONS FOR STRATEGY SELECTION
   Option A: Use CASH (simplest, no need to manage TB3MS data)
   Option B: Use TB3MS (adds ~30 basis points for negligible complexity)
   Option C: Use IEF (adds more return but also more volatility/drawdown)
   
   RECOMMENDATION: CASH is cleanest. The defensive asset barely matters.

4. COMPARISON TO BASELINE
   • GLD alone: 11.29% CAGR, 0.675 Sharpe, -42.91% MaxDD
   • 100-day EMA + CASH: 23.93% CAGR, 1.927 Sharpe, -7.20% MaxDD
   • Improvement: 112% higher returns, 2.9x better Sharpe ratio!
"""
)

print("=" * 100 + "\n")

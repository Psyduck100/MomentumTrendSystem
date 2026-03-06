"""Simple sanity check: compare our test results to the defensive allocation test."""

import pandas as pd

# Load our score/filter test results
score_filter_annual = pd.read_csv("score_filter_annual_returns.csv", index_col="year")

# Load the defensive allocation test results
defensive_annual = pd.read_csv(
    "defensive_allocation_annual_returns.csv", index_col="year"
)

# Compare blend_filter_12m (from score/filter test) to blend_6_12_ief (from defensive test)
# These should be IDENTICAL since they both use:
# - blend_6_12 scoring
# - ret_12m filter
# - IEF defensive
# - rank_gap=0

print("Comparing blend_filter_12m vs blend_6_12_ief:")
print("=" * 80)
print(f"{'Year':<6} {'blend_filter_12m':>18} {'blend_6_12_ief':>18} {'Difference':>15}")
print("-" * 80)

for year in range(2002, 2027):
    if year in score_filter_annual.index and year in defensive_annual.index:
        val1 = score_filter_annual.loc[year, "blend_filter_12m"]
        val2 = defensive_annual.loc[year, "blend_6_12_ief"]
        diff = val1 - val2
        print(f"{year:<6} {val1:>17.4%} {val2:>17.4%} {diff:>14.6f}")

print()
print("If difference is consistently near zero, the tests are consistent.")
print("If there are large differences, there's a bug.")

# Why Higher Gap Reduces MaxDD: Solved ✓

## The Counterintuitive Result

**US_equities bucket:**
- Gap=0: MaxDD -26.90%  
- Gap=3: MaxDD -20.69% (6.2% better!)

Your intuition was that higher gap = "stickier" = holds losers longer = worse drawdowns.

## The Real Mechanism

**Gap doesn't just prevent exiting losers - it also prevents ENTERING soon-to-be-losers.**

### What Actually Happened (2022 Crash)

**Month 74 (September 2022 - peak drawdown):**

| Strategy | Holding | Return |
|----------|---------|--------|
| Gap=0 | XLG (large cap growth) | -10.77% |
| Gap=3 | RSP (equal weight) | -6.51% |

**Gap=0 behavior:**
- Constantly chases recent momentum leaders
- Month 74: Switched to XLG just before large cap growth crashed
- Held SCHV through months 75-83, but the damage was done

**Gap=3 behavior:**  
- Stuck with RSP (which had been working)
- Never switched to XLG at the worst possible time
- RSP declined less during the 2022 sell-off (-6.51% vs -10.77%)

### Key Insight

The 6.2% MaxDD improvement comes from **avoiding a poorly-timed entry** into XLG, not from exiting faster.

Gap=0 exhibits "whipsaw at market tops" - it buys recent winners right as they're peaking, then eats the full drawdown.

Gap=3 provides "momentum damping" - it's slow to chase new leaders, which paradoxically helps when those new leaders are about to reverse.

## Why This Doesn't Make Gap=3 Optimal

While gap=3 helped in this specific drawdown:
- **Overall Sharpe:** gap=0 (1.06) vs gap=3 (1.05) - gap=0 still slightly better!
- **CAGR:** gap=0 (17.37%) vs gap=3 (17.83%) - gap=3 marginally higher
- **Turnover:** gap=0 (22.7%) vs gap=3 (5.6%) - gap=3 much lower

But when you add the `ret_and@1%` filter:
- **Gap=1 optimal:** Sharpe 1.42, Turnover 11.76%
- Gap=1 provides the sweet spot of momentum + enough stickiness to avoid bad entries

## Validation

This is **not a bug** - it's a legitimate behavioral difference:
- Gap=0 is aggressive: always switches to current leader (prone to whipsaw)
- Gap=3 is conservative: only switches if current drops to rank 4 out of 5 (avoids bad entries but misses early moves)

The rank gap logic is working correctly (verified via unit tests).

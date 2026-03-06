# Gap=3 MaxDD "Improvement": Curve-Fitting Analysis

## The Question

Is gap=3's better MaxDD (-20.69% vs -26.90%) a systematic advantage or just lucky curve-fitting?

## Evidence of Curve-Fitting

### 1. Performance at Actual MaxDD Points

**At the worst drawdown moment (Sep 2022, -26.90%):**

- Gap=0: -8.71% monthly return
- Gap=3: -9.16% monthly return ← **WORSE**

Gap=3 didn't perform better when it mattered most.

### 2. Analysis of ALL Major Drawdowns

**Drawdown counts:**

- Gap=0: 9 drawdowns > -10%
- Gap=3: 6 drawdowns > -10%

**Performance during gap=0's worst 3 drawdowns:**

| Date     | Gap=0 DD | Gap=0 Holding | Gap=0 Return | Gap=3 Holding | Gap=3 Return | Difference            |
| -------- | -------- | ------------- | ------------ | ------------- | ------------ | --------------------- |
| Sep 2022 | -26.90%  | SCHV          | -8.71%       | RSP           | -9.16%       | **-0.46% worse**      |
| Jun 2022 | -22.51%  | SCHV          | -8.67%       | RSP           | -9.39%       | **-0.72% worse**      |
| Dec 2018 | -17.51%  | SCHV          | -9.26%       | SCHV          | -9.26%       | **0.00% (identical)** |

Gap=3 did NOT consistently protect during drawdowns.

### 3. Where the Difference Actually Came From

**Month 74 (June 2022) - NOT the MaxDD month:**

- Gap=0: Switched to XLG, lost -10.77%
- Gap=3: Held RSP, lost -6.51%
- **4.3% advantage from this single event**

The entire MaxDD improvement traces to avoiding ONE poorly-timed switch, 5 months before the actual MaxDD.

### 4. Overall Performance Metrics

| Metric   | Gap=0   | Gap=3   | Winner           |
| -------- | ------- | ------- | ---------------- |
| Sharpe   | 1.06    | 1.05    | Gap=0            |
| CAGR     | 17.37%  | 17.83%  | Gap=3 (marginal) |
| MaxDD    | -26.90% | -20.69% | Gap=3            |
| Turnover | 22.7%   | 5.6%    | Gap=3 (lower)    |

Gap=0 still has slightly better risk-adjusted returns (Sharpe).

## Conclusion: This Is Curve-Fitting

**Why this is NOT a systematic advantage:**

1. ✗ Gap=3 didn't protect at other major drawdowns
2. ✗ Gap=3 actually did worse at the MaxDD month itself
3. ✗ All benefit comes from avoiding ONE specific switch
4. ✗ No theoretical reason this would persist out-of-sample
5. ✗ Overall Sharpe is actually lower

**What actually happened:**

- Gap=0 made an unlucky switch (to XLG) in June 2022
- Gap=3 happened to avoid that specific switch
- This single event accounts for the entire MaxDD difference
- It's not a reliable, repeatable edge

## Proper Strategy Selection

Don't optimize for gap=3 based on this one lucky outcome.

**Best choice:** Gap=1 + ret_and@1% filter

- Sharpe: 1.42
- Turnover: 11.76%
- Balances momentum capture with turnover control

The gap=3 MaxDD "advantage" is **sample-specific noise**, not a robust feature.

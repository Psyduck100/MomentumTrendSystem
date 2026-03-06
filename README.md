# US Equities Momentum Strategy

A data-driven monthly rebalancing momentum strategy for US equity ETFs with defensive allocation.

## Strategy Overview

**Scoring Method:** Blend of 6-month and 12-month total returns (50/50 weight)  
**Ranking:** Top-ranked ticker by blend score  
**Absolute Filter:** Positive 12-month return required; else hold IEF (Treasury ETF)  
**Rebalancing:** Monthly (end-of-month)  
**Universe:** SCHB, XLG, SCHV, QQQ, RSP (US equity ETFs)  
**Defensive Asset:** IEF (iShares 7-10 Year Treasury ETF)

## Historical Performance

**Backtest Period:** 2002-02-28 to 2026-01-31 (24 years)

| Metric       | Value   | vs SPY   |
| ------------ | ------- | -------- |
| CAGR         | 13.28%  | +4.73pp  |
| Sharpe Ratio | 0.92    | +0.30    |
| Max Drawdown | -27.18% | -23.60pp |

**Benchmarks:**

- SPY: 8.55% CAGR, 0.62 Sharpe, -50.78% MaxDD
- QQQ: 10.15% CAGR, 0.56 Sharpe, -56.33% MaxDD

## Files

### Core Strategy Files

- **`UsEquitiesRebalance.py`** - Monthly recommendation script. Run at month-end to get the recommended position for next month.
- **`compare_defensive_choices.py`** - Validation that IEF is superior to TB3MS cash as the defensive asset.
- **`CSVs/US_equities.csv`** - Universe of 5 US equity ETFs
- **`CSVs/TB3MS.csv`** - 3-month T-bill rates (historical risk-free rates)

### Core Package

- **`momentum_program/`** - Core backtesting engine and analytics
  - `backtest/engine.py` - Main backtest engine supporting blend scoring, absolute filters, and defensive allocation
  - `backtest/metrics.py` - Performance metrics computation
  - `analytics/momentum.py` - Momentum scoring calculations
  - `config.py`, `constants.py` - Configuration and constants

## Usage

### Get Next Month's Recommendation

```bash
python UsEquitiesRebalance.py
```

Output: Recommended ticker and rationale (e.g., "SCHB" or "IEF")

### Validate Strategy (Optional)

```bash
python compare_defensive_choices.py
```

Runs a full backtest comparing IEF vs TB3MS cash defensive allocation.

## Strategy Logic

1. **Score Calculation (Monthly)**
   - 6-month return = (Price_now / Price_6m_ago) - 1
   - 12-month return = (Price_now / Price_12m_ago) - 1
   - Blend score = 0.5 × return_6m + 0.5 × return_12m

2. **Ranking**
   - Rank tickers by blend score (descending)
   - Select top-ranked ticker

3. **Absolute Filter**
   - If top ticker's 12-month return > 0: **Hold it**
   - Else: **Hold IEF** (Treasury ETF for defensive allocation)

4. **Rebalance**
   - On month-end, recompute scores and apply filter
   - If recommendation changes, transition to new position

## Why This Strategy Works

- **Momentum Effect:** Strong empirical evidence that positive-return assets tend to outperform
- **Blend Scoring:** Combines medium-term (6M) and long-term (12M) trends, more robust than single lookback
- **Defensive Filter:** Exits equities when 12-month momentum is negative, reducing drawdown exposure
- **IEF Defensive:** Treasury ETF provides equity hedge and positive carry in downturns
- **Simplicity:** Single position + monthly rebalance = low transaction costs and easy to execute

## Backtesting Notes

- **Start Date:** 2001-01-01 (price data cached)
- **End Date:** Latest month-end
- **Price Source:** Yahoo Finance (via yfinance)
- **Rebalancing:** Last trading day of each calendar month
- **Slippage:** 3 basis points per trade
- **Expense Ratio:** 0.1% per year (blended across holdings)

## Configuration

All strategy parameters are hardcoded in the scripts:

- **Universe:** `UsEquitiesRebalance.py` line ~19 (loads from `CSVs/US_equities.csv`)
- **Weights:** Lines ~37-38 (50/50 for 6M/12M)
- **Defensive Symbol:** Line ~20 (`"IEF"`)
- **Backtest Parameters:** `momentum_program/backtest/engine.py` (slippage, expense ratio, etc.)

# Momentum Program (Multi-Model)
<img width="2160" height="1260" alt="PMTL vs SPY equity curve" src="https://github.com/user-attachments/assets/36934329-d2d4-4fe4-af4a-4994dcc5252d" />
A multi-model momentum research and live-signal project covering CRYP, PMTL, USEQ, and SECTOR170.

# Motivation

The momentum factor is one of the most persistent return premia in asset pricing, consistently documented across equities, sectors, and asset classes. MSCI's factor research identifies momentum as producing among the strongest risk-adjusted returns across multiple decades. While the raw premium has compressed since the early 2000s, it remains statistically significant particularly when implemented through dual momentum or cross-sectional frameworks, and when combined with trend-following systems.
This motivates the us to explores whether combining absolute and cross-sectional momentum signals with regime filters and breakout confirmation can produce stable, out-of-sample alpha across equities, sectors, gold, and Bitcoin. The goal is not to find the optimal parameters, but to identify robust configurations whose performance holds across a wide parameter neighborhood.

Sources:
Antonacci, Gary. Dual Momentum Investing: An Innovative Strategy for Higher Returns with Lower Risk. 1st ed., McGraw‑Hill Education, 2014.

Chen, Linda H. and Jiang, George and Zhu, Xingnong, Do Style and Sector Indexes Carry Momentum? (August 28, 2011). The Journal of Investment Strategies, vol1(3), Summer 2012, 67-89., Available at SSRN: https://ssrn.com/abstract=2139210 or http://dx.doi.org/10.2139/ssrn.2139210

RAntonacci, Gary, Risk Premia Harvesting Through Dual Momentum (October 1, 2016). Journal of Management & Entrepreneurship, vol.2, no.1 (Mar 2017), 27-55, Available at SSRN: https://ssrn.com/abstract=2042750 or http://dx.doi.org/10.2139/ssrn.2042750

Zarattini, Carlo and Antonacci, Gary, A Century of Profitable Industry Trends (June 07, 2024). 2025 Charles H. Dow Award Winner, Available at SSRN: https://ssrn.com/abstract=4857230 or http://dx.doi.org/10.2139/ssrn.4857230

Abhishek Gupta, Stuart Doole, MSCI Factor Indexing Through the Decades
(July 29, 2025) https://www.msci.com/downloads/web/msci-com/research-and-insights/paper/factor-indexing-through-the-decades/factor-indexing-through-the-decades.pdf

## Model Overviews

### USEQ (US Equities Momentum)

**Scoring Method:** Blend of 6-month and 12-month total returns (50/50 weight)  
**Ranking:** Top-ranked ticker by blend score  
**Absolute Filter:** Positive 12-month return required; else hold IEF (Treasury ETF)  
**Rebalancing:** Monthly (end-of-month)  
**Universe:** SPY, QQQ, IVE (pruned long-history proxy basket for US equity exposures)  
**Defensive Asset:** IEF (iShares 7-10 Year Treasury ETF)

### CRYP (Crypto Trend/Gate Model)

**Signal Inputs:** MA + Donchian components (primary and alternate configs)  
**Execution Calendar:** US market sessions by default (ETF-realistic mode)  
**Positioning:** Binary risk-on/off signal with one-bar execution lag in backtests  
**Asset Base:** BTC proxy return stream with fee-adjusted net returns  
**Use Case:** Daily directional timing for Bitcoin-linked exposure

### PMTL (Gold Regime + USEQ Sleeve)

**Core Asset:** GLD regime model (UP/DOWN/CHOP)  
**Regime Logic:** Decision-gated trend rules using moving averages, slope, and channel levels  
**Bull Regime:** Hold GLD sleeve  
**Non-Bull Regime:** Route to USEQ sleeve recommendation (which may itself go defensive to IEF)  
**Use Case:** Regime-switching between precious metals momentum and US equities momentum

### SECTOR170 (Sector Rotation)

**Signal:** Relative momentum ranking on a 175-trading-day lookback  
**Universe:** XLK, XLV, XLI, XLE, XAR  
**Selection:** Top-ranked ticker at each rebalance decision  
**Rebalancing:** Month-end decision cadence with next-session implementation in backtest runner  
**Benchmarking:** Compared against SPY with information ratio and active return

## Historical Performance

The project includes dedicated modular backtests for all active models under `backtest/`.
Headline stats below come from current backtest runners as of 2026-03-05.

| Model     | Window                  | CAGR   | Sharpe | MaxDD   | IR vs SPY | Active Return vs SPY |
| --------- | ----------------------- | ------ | ------ | ------- | --------- | -------------------- |
| CRYP      | 2013-01-02 → 2026-03-05 | 63.04% | 1.362  | -50.57% | NA        | NA                   |
| PMTL      | 2004-11-18 → 2026-03-05 | 18.12% | 1.069  | -22.80% | NA        | NA                   |
| USEQ      | 2002-07-31 → 2026-03-05 | 13.48% | 0.811  | -28.56% | 0.138     | 2.01%                |
| SECTOR170 | 2013-07-01 → 2026-03-04 | 21.89% | 0.968  | -34.37% | 0.516     | 7.88%                |

For metric definitions, calibration notes, and run commands for each model, see `backtest/README.md`.

## Project Files

### Legacy USEQ-Focused Scripts

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

### USEQ Next-Month Recommendation

```bash
python UsEquitiesRebalance.py
```

Output: Recommended ticker and rationale (e.g., "SCHB" or "IEF")

### Validate USEQ Defensive Choice (Optional)

```bash
python compare_defensive_choices.py
```

Runs a full backtest comparing IEF vs TB3MS cash defensive allocation.

### Run Modular Backtests (All Models)

```bash
pipenv run python backtest/CRYP/run_backtest.py
pipenv run python backtest/PMTL/run_backtest.py
pipenv run python backtest/USEQ/run_backtest.py
pipenv run python backtest/SECTOR170/run_backtest.py
```

## Model Mechanics (High Level)

### USEQ

1. Compute 6M and 12M returns monthly
2. Blend scores with fixed weights
3. Rank candidates and apply absolute filter
4. Hold top asset or IEF until next rebalance

### CRYP

1. Build BTC proxy price/return stream
2. Compute MA/Donchian components by selected config
3. Combine to binary signal with confirmation rules
4. Apply one-bar lag and transaction costs in backtest

### PMTL

1. Detect GLD regime state (UP/DOWN/CHOP)
2. If UP, allocate to GLD sleeve
3. If not UP, allocate by USEQ sleeve recommendation
4. Aggregate sleeve returns into layered portfolio return

### SECTOR170

1. Compute 175-day momentum scores at month-end
2. Select top-ranked sector ETF
3. Hold selected sector until next decision
4. Evaluate against SPY on absolute and active metrics

## USEQ Strategy Logic

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

## USEQ Backtesting Notes

- **USEQ Backtest Universe:** SPY, QQQ, IVE
- **USEQ Effective Start:** 2002-07-31 (based on combined ETF history + lookback warmup)
- **End Date:** Latest month-end
- **Price Source:** Yahoo Finance (via yfinance)
- **Rebalancing:** Last trading day of each calendar month
- **Slippage:** 3 basis points per trade
- **Expense Ratio:** 0.1% per year (blended across holdings)

## USEQ Configuration

All strategy parameters are hardcoded in the scripts:

- **Universe:** `UsEquitiesRebalance.py` line ~19 (loads from `CSVs/US_equities.csv`)
- **Weights:** Lines ~37-38 (50/50 for 6M/12M)
- **Defensive Symbol:** Line ~20 (`"IEF"`)
- **Backtest Parameters:** `momentum_program/backtest/engine.py` (slippage, expense ratio, etc.)

# UPDATES

locked in USEQ final params (for now) dual momentum system
started playing around with trend breakout systems
seems like gold and BTC benefits most from such systems, start finalizing parameters for gold first
finalized backtested parameters for gold trend breakout system
backteseting overlaying USEQ model with gold model during gold risk off periods
locked in PMTL final params and system (for now)
start finalizing paramters for BTC model
locked in CRYP final params after backtesting
exploring sector rotation via cross sectional momentum
backtesting final params for such model and optimizing etf universe
locked in final etf universe and params for SECTOR170 model
organizing models for git repo push
trimmed many backtest files before pushing to repo, kept most basic ones for demonstration as file directory was too messy


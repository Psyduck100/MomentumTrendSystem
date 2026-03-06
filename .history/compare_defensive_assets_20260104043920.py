#!/usr/bin/env python3
"""
Test blend_filter_12m and excess_rf with defensive asset being IEF vs TB3MS (T-bills).
Compares 4 configurations:
  1. blend_filter_12m + IEF
  2. blend_filter_12m_excess_rf + IEF
  3. blend_filter_12m + TB3MS
  4. blend_filter_12m_excess_rf + TB3MS
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from momentum_program.backtest.engine import backtest_momentum
from momentum_program.analytics.constants import SCORE_MODE_BLEND_6_12

# Load universe (equities only)
UNIVERSE_FILE = Path("CSVs/US_equities.csv")
assert UNIVERSE_FILE.exists(), f"Missing {UNIVERSE_FILE}"
universe_df = pd.read_csv(UNIVERSE_FILE, index_col=0)
equities = universe_df.index.tolist()
print(f"Universe: {len(equities)} equities")

# Create bucket map (all equities in one bucket)
bucket_map = {ticker: "US_Equities" for ticker in equities}

# Load risk-free rate (TB3MS) for excess-rf filter
TB3MS_FILE = Path("CSVs/TB3MS.csv")
assert TB3MS_FILE.exists(), f"Missing {TB3MS_FILE}"
tb3ms = pd.read_csv(TB3MS_FILE, index_col="observation_date", parse_dates=True)
tb3ms.columns = ["TB3MS"]
tb3ms["TB3MS"] = pd.to_numeric(tb3ms["TB3MS"], errors="coerce")

def load_risk_free_band(tb3ms_df: pd.DataFrame) -> pd.Series:
    """
    Convert TB3MS annualized % to 12-month compounded risk-free return.
    Input: TB3MS % (annualized 3-month bill rate)
    Output: Series of 12-month compounded risk-free returns aligned to month-end rebalance dates
    """
    # Convert % to decimal
    rf_decimal = tb3ms["TB3MS"] / 100.0
    # Convert annual rate to monthly
    rf_monthly = rf_decimal / 12.0
    # Compound to 12 months: (1 + r_monthly)^12 - 1
    rf_12m_compound = (1 + rf_monthly) ** 12 - 1
    # Resample to month-end to align with rebalance dates
    rf_band = rf_12m_compound.resample("ME").last()
    return rf_band

print("Loading risk-free band...")
rf_band = load_risk_free_band(tb3ms)
print(f"Risk-free band: {len(rf_band)} months, {rf_band.index[0]} to {rf_band.index[-1]}")

# Configuration list: (name, score_mode, score_param, abs_filter_mode, abs_filter_band_series, defensive_symbol)
configs = [
    ("blend_filter_12m_ief", SCORE_MODE_BLEND_6_12, None, "ret_12m", pd.Series(0.0, index=rf_band.index), "IEF"),
    ("blend_filter_12m_excess_rf_ief", SCORE_MODE_BLEND_6_12, None, "ret_12m", rf_band, "IEF"),
    ("blend_filter_12m_tb3ms", SCORE_MODE_BLEND_6_12, None, "ret_12m", pd.Series(0.0, index=rf_band.index), "TB3MS"),
    ("blend_filter_12m_excess_rf_tb3ms", SCORE_MODE_BLEND_6_12, None, "ret_12m", rf_band, "TB3MS"),
]

results = {}
print("\n" + "=" * 80)
print("Running backtest comparisons...")
print("=" * 80)

for config_name, score_mode, score_param, abs_filter_mode, abs_filter_band_series, defensive_symbol in configs:
    print(f"\n[{config_name}]")
    print(f"  Filter: {abs_filter_mode}, Defensive: {defensive_symbol}")
    
    result = backtest_momentum(
        tickers=equities,
        bucket_map=bucket_map,
        start_date="2002-01-01",
        end_date="2026-01-04",
        top_n_per_bucket=1,
        cache_dir=Path("backtest_cache"),
        slippage_bps=3.0,
        expense_ratio=0.001,
        vol_adjusted=False,
        score_mode=score_mode,
        score_param=score_param,
        abs_filter_mode=abs_filter_mode,
        abs_filter_band=0.0,
        abs_filter_band_series=abs_filter_band_series,
        abs_filter_cash_annual=0.04,
        defensive_symbol=defensive_symbol,
        rank_gap_threshold=0,
    )
    
    results[config_name] = result
    
    if result["status"] == "success":
        annual_returns = result["annual_returns"]
        summary = {
            "CAGR": result["cagr"],
            "Annual Return": result["annual_return"],
            "Sharpe": result["sharpe"],
            "Sortino": result.get("sortino", np.nan),
            "Max DD": result["max_drawdown"],
            "Win Rate": result["win_rate"],
            "Avg Win": result["avg_win"],
            "Avg Loss": result["avg_loss"],
        }
        for key, val in summary.items():
            if isinstance(val, float):
                print(f"  {key:15s}: {val:10.2%}" if key != "Sharpe" and key != "Sortino" else f"  {key:15s}: {val:10.2f}")
            else:
                print(f"  {key:15s}: {val}")
    else:
        print(f"  ERROR: {result.get('error', 'Unknown error')}")

# Create comparison summary
print("\n" + "=" * 80)
print("SUMMARY COMPARISON")
print("=" * 80)

summary_data = []
for config_name in [c[0] for c in configs]:
    result = results.get(config_name, {})
    if result.get("status") == "success":
        summary_data.append({
            "Config": config_name,
            "CAGR": result["cagr"],
            "Sharpe": result["sharpe"],
            "Max Drawdown": result["max_drawdown"],
            "Avg Annual Return": result["annual_return"],
        })

summary_df = pd.DataFrame(summary_data)
print(summary_df.to_string(index=False))

# Save to CSV
output_file = Path("defensive_asset_comparison.csv")
summary_df.to_csv(output_file, index=False)
print(f"\nSaved to {output_file}")

# Detailed analysis
print("\n" + "=" * 80)
print("DETAILED ANALYSIS")
print("=" * 80)

ief_12m = results["blend_filter_12m_ief"]["cagr"]
ief_excess = results["blend_filter_12m_excess_rf_ief"]["cagr"]
tb3ms_12m = results["blend_filter_12m_tb3ms"]["cagr"]
tb3ms_excess = results["blend_filter_12m_excess_rf_tb3ms"]["cagr"]

print(f"\nStandard filter (ret_12m > 0):")
print(f"  IEF defensive:   {ief_12m:7.2%} CAGR")
print(f"  TB3MS defensive: {tb3ms_12m:7.2%} CAGR (diff: {tb3ms_12m - ief_12m:+7.2%})")

print(f"\nExcess-RF filter (ret_12m > RF_12m):")
print(f"  IEF defensive:   {ief_excess:7.2%} CAGR")
print(f"  TB3MS defensive: {tb3ms_excess:7.2%} CAGR (diff: {tb3ms_excess - ief_excess:+7.2%})")

print(f"\nIEF advantage by filter:")
print(f"  Standard vs Excess-RF (IEF):  {ief_12m - ief_excess:+7.2%}")
print(f"  Standard vs Excess-RF (TB3MS): {tb3ms_12m - tb3ms_excess:+7.2%}")

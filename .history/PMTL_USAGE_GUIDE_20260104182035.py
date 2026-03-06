"""
PMTL Refactored System - Usage Guide

This refactored system makes it trivial to run different tests without code changes.
Just modify pmtl_config.py and run pmtl_runner.py
"""

# ==============================================================================
# EXAMPLE 1: Switch Fallback Strategy
# ==============================================================================

# File: pmtl_config.py (last line)

# BEFORE: Using CASH
# ACTIVE_CONFIG = CASH_ONLY_CONFIG

# AFTER: Switch to TB3MS with one line change
# ACTIVE_CONFIG = TBILLS_CONFIG

# AFTER: Switch to IEF
# ACTIVE_CONFIG = IEF_CONFIG

# Then run:
# $ python pmtl_runner.py


# ==============================================================================
# EXAMPLE 2: Change Window Range
# ==============================================================================

# File: pmtl_config.py (modify WindowConfig)

# BEFORE: Test 100-200 days in steps of 10
# windows=WindowConfig(start_window=100, end_window=200, step=10)

# AFTER: Test 80-180 days in steps of 5 (finer resolution)
# windows=WindowConfig(start_window=80, end_window=180, step=5)

# AFTER: Quick test with only 100-130 window range
# windows=WindowConfig(start_window=100, end_window=130, step=10)

# AFTER: EMA-only (no SMA)
# windows=WindowConfig(start_window=100, end_window=200, step=10, use_sma=False)


# ==============================================================================
# EXAMPLE 3: Change Date Range
# ==============================================================================

# File: pmtl_config.py (modify DateConfig)

# BEFORE: Full range 2005-2025
# dates=DateConfig(start_date="2005-01-01", end_date="2025-12-31")

# AFTER: Recent data only 2020-2025
# dates=DateConfig(start_date="2020-01-01", end_date="2025-12-31")

# AFTER: 2008 crisis period
# dates=DateConfig(start_date="2008-01-01", end_date="2008-12-31")


# ==============================================================================
# EXAMPLE 4: Create Custom Configuration
# ==============================================================================

# File: pmtl_config.py (add new config)

# Custom: Fine-tuned test with TB3MS
CUSTOM_TBILLS_FINE = BacktestConfig(
    fallback_type=FallbackType.TBILLS,
    fallback_csv="CSVs/TB3MS.csv",
    windows=WindowConfig(
        start_window=95, end_window=105, step=1
    ),  # Fine sweep around 100
    dates=DateConfig(start_date="2005-01-01", end_date="2025-12-31"),
    output_prefix="pmtl_tbills_finesweep",
    primary_metric=MetricType.SHARPE,  # Optimize for Sharpe instead of CAGR
)

# Then set:
# ACTIVE_CONFIG = CUSTOM_TBILLS_FINE


# ==============================================================================
# ARCHITECTURE
# ==============================================================================

# pmtl_config.py
#   └─ Centralized configuration parameters
#   └─ Easy to add new FallbackType, MetricType, WindowConfig presets

# pmtl_fallback_strategies.py
#   └─ CashFallback: 0% return
#   └─ TBillsFallback: TB3MS from CSV
#   └─ YFinanceFallback: Any ticker (IEF, SPY, TLT, AGG)
#   └─ Factory function: get_fallback_strategy()
#   └─ Easy to add new strategies (e.g., custom blends, crypto, etc.)

# pmtl_backtest_engine.py
#   └─ PMTLBacktestEngine: Core logic
#   └─ backtest_sma(): SMA-based test
#   └─ backtest_ema(): EMA-based test
#   └─ run_window_sweep(): Batch test multiple windows
#   └─ No hardcoded fallback logic - everything is pluggable

# pmtl_runner.py
#   └─ Main entry point
#   └─ Reads ACTIVE_CONFIG
#   └─ Creates engine, fallback, runs sweep
#   └─ Exports results
#   └─ All workflow in ~50 lines (DRY!)


# ==============================================================================
# ADVANTAGES OF REFACTORED DESIGN
# ==============================================================================

# ✅ DRY: No duplicated backtest logic
# ✅ Flexible: Change tests without touching code
# ✅ Extensible: Add new fallbacks/metrics without modifying existing code
# ✅ Testable: Each module can be tested independently
# ✅ Maintainable: Single source of truth for each concept
# ✅ Composable: Mix and match configs, strategies, metrics


# ==============================================================================
# COMMON WORKFLOWS
# ==============================================================================

# Test 1: Compare all three fallbacks with same window range
print(
    """
Test multiple fallbacks:

1. pmtl_config.py: ACTIVE_CONFIG = CASH_ONLY_CONFIG
   python pmtl_runner.py

2. pmtl_config.py: ACTIVE_CONFIG = TBILLS_CONFIG
   python pmtl_runner.py

3. pmtl_config.py: ACTIVE_CONFIG = IEF_CONFIG
   python pmtl_runner.py

Results files:
  - pmtl_cash_results.csv
  - pmtl_tbills_results.csv
  - pmtl_ief_results.csv
"""
)

# Test 2: Optimize window with cash fallback
print(
    """
Fine-tune window around optimal:

1. pmtl_config.py: ACTIVE_CONFIG = FINE_SWEEP_CONFIG
   python pmtl_runner.py

Results file: pmtl_finesweep_results.csv
"""
)

# Test 3: Test period comparison
print(
    """
Compare performance in different market periods:

1. pmtl_config.py: DateConfig(start_date="2005-01-01", end_date="2008-12-31")
   python pmtl_runner.py

2. pmtl_config.py: DateConfig(start_date="2020-01-01", end_date="2025-12-31")
   python pmtl_runner.py

Compare results to understand regime-dependent performance
"""
)


# ==============================================================================
# ADDING NEW FALLBACK STRATEGIES
# ==============================================================================

# Example: Add VanEck Gold ETF (GDX - junior miners) as fallback

# 1. pmtl_fallback_strategies.py - it's already supported!
#    Just use: get_fallback_strategy("gdx", ...)
#    Or create custom class:


class GDXFallback(YFinanceFallback):
    def __init__(self, start_date: str, end_date: str):
        super().__init__(start_date, end_date, "GDX")

    @property
    def name(self) -> str:
        return "GDX (Junior Miners)"


# 2. pmtl_config.py - add to FallbackType enum if you want:
# class FallbackType(Enum):
#     GDX = "gdx"

# 3. pmtl_config.py - create preset:
# GDX_CONFIG = BacktestConfig(
#     fallback_type=FallbackType.GDX,
#     ...
# )

# 4. Run it:
# ACTIVE_CONFIG = GDX_CONFIG
# python pmtl_runner.py


print("\nFor questions or updates, see pmtl_config.py, pmtl_fallback_strategies.py")

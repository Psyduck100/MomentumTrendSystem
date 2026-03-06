"""PMTL Strategy Configuration Module

Centralized configuration for all backtest parameters.
Easy to modify without changing core logic.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Optional


class FallbackType(Enum):
    """Supported defensive fallback strategies."""
    CASH = "cash"           # 0% return
    TBILLS = "tbills"       # TB3MS (3-month T-Bills)
    IEF = "ief"             # Intermediate-term bonds
    SPY = "spy"             # Stock market
    # Easy to add more: TLT, AGG, VGIT, etc.


class MetricType(Enum):
    """Supported performance metrics."""
    CAGR = "cagr"
    SHARPE = "sharpe"
    MAX_DRAWDOWN = "max_drawdown"
    SORTINO = "sortino"
    CALMAR = "calmar"


@dataclass
class WindowConfig:
    """Configuration for MA window sweep."""
    start_window: int = 50         # Starting trading days
    end_window: int = 150          # Ending trading days
    step: int = 10                 # Increment between windows
    use_sma: bool = True           # Include Simple MA tests
    use_ema: bool = True           # Include Exponential MA tests
    
    def get_windows(self):
        """Generate list of windows to test."""
        return list(range(self.start_window, self.end_window + 1, self.step))


@dataclass
class DateConfig:
    """Configuration for backtest dates."""
    start_date: str = "2005-01-01"
    end_date: str = "2025-12-31"
    frequency: str = "ME"          # Month-end resampling


@dataclass
class BacktestConfig:
    """Master configuration for backtests."""
    # Core strategy
    main_ticker: str = "GLD"
    fallback_type: FallbackType = FallbackType.CASH
    fallback_ticker: Optional[str] = None  # For IEF, SPY, etc.
    fallback_csv: Optional[str] = None     # For TB3MS data
    
    # Parameters
    windows: WindowConfig = None
    dates: DateConfig = None
    
    # Output
    output_prefix: str = "pmtl"  # Files: pmtl_results.csv, etc.
    primary_metric: MetricType = MetricType.CAGR
    
    # Other
    verbose: bool = True
    
    def __post_init__(self):
        """Set defaults for nested configs."""
        if self.windows is None:
            self.windows = WindowConfig()
        if self.dates is None:
            self.dates = DateConfig()
        
        # Auto-set fallback ticker based on type
        if self.fallback_ticker is None:
            fallback_defaults = {
                FallbackType.CASH: None,
                FallbackType.TBILLS: None,
                FallbackType.IEF: "IEF",
                FallbackType.SPY: "SPY",
            }
            self.fallback_ticker = fallback_defaults.get(self.fallback_type)
        
        # Auto-set CSV path for TB3MS
        if self.fallback_type == FallbackType.TBILLS and self.fallback_csv is None:
            self.fallback_csv = "CSVs/TB3MS.csv"
    
    def get_output_filename(self, suffix: str) -> str:
        """Generate output filename."""
        fallback_name = self.fallback_type.value
        return f"{self.output_prefix}_results_{fallback_name}_{suffix}.csv"


# ============================================================================
# PRESET CONFIGURATIONS (easy to switch between)
# ============================================================================

# Test 1: Cash-only (fastest, baseline)
CASH_ONLY_CONFIG = BacktestConfig(
    fallback_type=FallbackType.CASH,
    output_prefix="pmtl_cash"
)

# Test 2: TB3MS fallback
TBILLS_CONFIG = BacktestConfig(
    fallback_type=FallbackType.TBILLS,
    fallback_csv="CSVs/TB3MS.csv",
    output_prefix="pmtl_tbills"
)

# Test 3: IEF fallback
IEF_CONFIG = BacktestConfig(
    fallback_type=FallbackType.IEF,
    fallback_ticker="IEF",
    output_prefix="pmtl_ief"
)

# Test 4: Quick test (smaller window range for development)
QUICK_TEST_CONFIG = BacktestConfig(
    fallback_type=FallbackType.CASH,
    windows=WindowConfig(start_window=100, end_window=130, step=10),
    output_prefix="pmtl_quicktest"
)

# Test 5: Fine-grained sweep (more windows for final optimization)
FINE_SWEEP_CONFIG = BacktestConfig(
    fallback_type=FallbackType.CASH,
    windows=WindowConfig(start_window=80, end_window=180, step=5),
    output_prefix="pmtl_finesweep"
)

# Test 6: Weekly rebalancing
WEEKLY_CONFIG = BacktestConfig(
    fallback_type=FallbackType.CASH,
    dates=DateConfig(frequency='W'),
    output_prefix="pmtl_weekly"
)

# Test 7: Biweekly rebalancing
BIWEEKLY_CONFIG = BacktestConfig(
    fallback_type=FallbackType.CASH,
    dates=DateConfig(frequency='2W'),
    output_prefix="pmtl_biweekly"
)

# Test 8: Monthly rebalancing (explicit)
MONTHLY_CONFIG = BacktestConfig(
    fallback_type=FallbackType.CASH,
    dates=DateConfig(frequency='ME'),
    output_prefix="pmtl_monthly"
)

# Test 9: Every 3 days rebalancing
THREEDAY_CONFIG = BacktestConfig(
    fallback_type=FallbackType.CASH,
    dates=DateConfig(frequency='3D'),
    output_prefix="pmtl_3day"
)

# Active configuration (change this to switch tests)
ACTIVE_CONFIG = WEEKLY_CONFIG


if __name__ == "__main__":
    # Print active config for verification
    print("Active Configuration:")
    print(f"  Main Ticker: {ACTIVE_CONFIG.main_ticker}")
    print(f"  Fallback Type: {ACTIVE_CONFIG.fallback_type.value}")
    print(f"  Windows: {ACTIVE_CONFIG.windows.start_window}-{ACTIVE_CONFIG.windows.end_window} (step {ACTIVE_CONFIG.windows.step})")
    print(f"  Dates: {ACTIVE_CONFIG.dates.start_date} to {ACTIVE_CONFIG.dates.end_date}")
    print(f"  Output Prefix: {ACTIVE_CONFIG.output_prefix}")
    print(f"  Primary Metric: {ACTIVE_CONFIG.primary_metric.value}")

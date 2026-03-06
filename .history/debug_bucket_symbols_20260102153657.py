"""Debug script to check how many symbols are in bucket_symbols"""

from momentum_program.backtest.engine import run_backtest
from momentum_program.scoring import SCORE_MODE_RW_3_6_9_12
import json

# Patch the engine to print bucket_symbols
original_backtest = run_backtest


def debug_backtest(*args, **kwargs):
    # Add instrumentation
    import momentum_program.backtest.engine as engine_module

    original_code = engine_module.run_backtest

    # Run and capture
    result = original_code(*args, **kwargs)
    return result


# Quick test
from universe import UNIVERSE_SPEC

result = run_backtest(
    universe_spec=UNIVERSE_SPEC,
    top_n_per_bucket=5,
    rank_gap_threshold=0,
    market_filter=False,
    abs_filter_mode="none",
    score_mode=SCORE_MODE_RW_3_6_9_12,
)

# Check US_equities bucket positions
us_eq_positions = result["bucket_positions"]["US_equities"]
print("\nFirst 10 US_equities positions (gap=0, no filter):")
for i, pos in enumerate(us_eq_positions[:10]):
    print(f"  Month {i}: {pos} (count={len(pos)})")

# Check returns structure
us_eq_returns = result["bucket_returns"]["US_equities"]
print("\nFirst 5 US_equities returns:")
for i, ret_data in enumerate(us_eq_returns[:5]):
    print(f"  Month {i}: symbols={ret_data['symbols']} return={ret_data['return']:.4f}")

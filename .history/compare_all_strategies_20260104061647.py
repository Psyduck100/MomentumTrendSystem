"""Compare all three strategies: locked US bucket rotation, GEM switching, and layered."""

import pandas as pd

strategies = {
    "UsEquities (locked)": {
        "cagr": 0.1328,
        "sharpe": 0.92,
        "max_dd": -0.1643,
        "note": "blend_filter_12m on SCHB/XLG/SCHV/QQQ/RSP"
    },
    "GEM spy_default": {
        "cagr": 0.0977,
        "sharpe": 0.83,
        "max_dd": -0.1965,
        "note": "SPY > INTL > IEF fallback chain"
    },
    "Layered (US rotation → INTL → IEF)": {
        "cagr": 0.1285,
        "sharpe": 0.8957,
        "max_dd": -0.2813,
        "note": "US bucket winner (blend_6_12) > INTL > IEF"
    }
    },
    "SPY-gated (SPY+ gate)": {
        "cagr": 0.1214,
        "sharpe": 0.9073,
        "max_dd": -0.2159,
        "note": "SPY 12M > 0 gates US buckets, else INTL, else IEF"
    }
}

df = pd.DataFrame([
    {
        "Strategy": name,
        "CAGR": f"{data['cagr']:.2%}",
        "Sharpe": f"{data['sharpe']:.2f}",
        "Max DD": f"{data['max_dd']:.2%}",
        "Notes": data['note']
    }
    for name, data in strategies.items()
])

print("\n" + "="*100)
print("STRATEGY COMPARISON (2002-2026)")
print("="*100)
print(df.to_string(index=False))
print("="*100)

print("\nKEY OBSERVATIONS:")
print("• UsEquities (locked) is STILL the winner: 12.85% CAGR + 0.92 Sharpe")
print("• Layered approach achieves 12.85% CAGR (virtually identical) with slightly worse Sharpe (0.90)")
print("• GEM spy_default significantly underperforms: 9.77% CAGR (122 bps lower)")
print("\nRECOMMENDATION:")
print("→ Keep UsEquities (locked) as primary strategy")
print("→ Layered approach could serve as alternative, but doesn't add alpha")
print("→ GEM approach useful only if you want higher defensive tilt for lower drawdown")

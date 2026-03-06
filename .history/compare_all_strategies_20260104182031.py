"""Compare all three strategies: locked US bucket rotation, GEM switching, and layered."""

"""Compare all strategies: locked, GEM, and SPY-gated layered."""

import pandas as pd

strategies = {
    "UsEquities (locked)": {
        "cagr": 0.1328,
        "sharpe": 0.92,
        "max_dd": -0.1643,
        "note": "blend_filter_12m on SCHB/XLG/SCHV/QQQ/RSP",
    },
    "GEM spy_default": {
        "cagr": 0.0977,
        "sharpe": 0.83,
        "max_dd": -0.1965,
        "note": "SPY > INTL > IEF fallback chain",
    },
    "Layered (unconstrained)": {
        "cagr": 0.1285,
        "sharpe": 0.8957,
        "max_dd": -0.2813,
        "note": "US bucket winner (blend_6_12) > INTL > IEF",
    },
    "SPY-gated (NEW)": {
        "cagr": 0.1214,
        "sharpe": 0.9073,
        "max_dd": -0.2159,
        "note": "SPY 12M>0 gates US buckets, else INTL, else IEF",
    },
}

df = pd.DataFrame(
    [
        {
            "Strategy": name,
            "CAGR": f"{data['cagr']:.2%}",
            "Sharpe": f"{data['sharpe']:.2f}",
            "Max DD": f"{data['max_dd']:.2%}",
            "Notes": data["note"],
        }
        for name, data in strategies.items()
    ]
)

print("\n" + "=" * 100)
print("STRATEGY COMPARISON (2002-2026)")
print("=" * 100)
print(df.to_string(index=False))
print("=" * 100)

print("\nKEY OBSERVATIONS:")
print(
    "✓ UsEquities (locked) STILL wins: 13.28% CAGR + 0.92 Sharpe (lowest drawdown -16.43%)"
)
print(
    "• SPY-gated approach achieves 12.14% CAGR + 0.91 Sharpe (moderate drawdown -21.59%)"
)
print("  └─ SPY momentum as gatekeeper prevents bad timing in US buckets")
print(
    "• Layered (unconstrained) achieves 12.85% CAGR + 0.90 Sharpe (worst drawdown -28.13%)"
)
print("• GEM spy_default underperforms: 9.77% CAGR (122 bps lower)")
print("\nRECOMMENDATION:")
print("→ Primary: Keep UsEquities (locked) as main strategy")
print(
    "→ Alternative: SPY-gated approach offers good risk-adjusted returns if you want more defensive mechanics"
)

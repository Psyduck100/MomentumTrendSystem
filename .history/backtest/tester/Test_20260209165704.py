from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backtest.tester.Config import EngineConfig
from backtest.tester.Defensive import ConstantReturnDefensive
from backtest.tester.Engine import run_engine
from backtest.tester.RebalanceGate import MonthlyRebalanceGate
from backtest.tester.Rules import NDaysMomentumEntryRule, NDaysMomentumExitRule
from backtest.tester.Selector import TopMomentumSelector
from backtest.tester.UniverseProvider import StaticUniverse


def _synthetic_prices() -> pd.DataFrame:
    idx = pd.bdate_range("2015-01-01", "2025-12-31")
    n = len(idx)
    rng = np.random.default_rng(7)

    # Three sleeves with different drifts to exercise selector/rules.
    r1 = rng.normal(0.00045, 0.012, n)
    r2 = rng.normal(0.00025, 0.010, n)
    r3 = rng.normal(0.00015, 0.009, n)

    p1 = 100 * np.cumprod(1 + r1)
    p2 = 100 * np.cumprod(1 + r2)
    p3 = 100 * np.cumprod(1 + r3)
    return pd.DataFrame({"A": p1, "B": p2, "C": p3}, index=idx)


def _metrics(daily: pd.DataFrame, ann: int = 252) -> dict[str, float]:
    ret = daily["ret"].dropna()
    eq = daily["equity"].dropna()
    if len(ret) < 2:
        return {"cagr": np.nan, "sharpe": np.nan, "maxdd": np.nan}
    cagr = float(eq.iloc[-1] ** (ann / len(ret)) - 1.0)
    vol = float(ret.std(ddof=1))
    sharpe = float(np.sqrt(ann) * ret.mean() / vol) if vol > 0 else np.nan
    peak = eq.cummax()
    maxdd = float((eq / peak - 1.0).min())
    return {"cagr": cagr, "sharpe": sharpe, "maxdd": maxdd}


def main() -> None:
    prices = _synthetic_prices()

    cfg = EngineConfig(
        cost_bps=1.0,
        slippage_bps=1.0,
        initial_capital=1.0,
        trade_delay=1,
    )

    result = run_engine(
        prices=prices,
        config=cfg,
        universe_provider=StaticUniverse(["A", "B", "C"]),
        rebalance_gate=MonthlyRebalanceGate(when="last"),
        entry_rule=NDaysMomentumEntryRule(lookback_days=126),
        exit_rule=NDaysMomentumExitRule(lookback_days=126),
        selector=TopMomentumSelector(lookback_days=126, top_n=1),
        defensive_asset=ConstantReturnDefensive(rate_annual=0.04),
    )

    m = _metrics(result.daily)
    print("Modular Tester Smoke Run")
    print(f"Rows: {len(result.daily)} | Trades: {len(result.trades)}")
    print(
        f"CAGR: {m['cagr']:.2%} | Sharpe: {m['sharpe']:.2f} | MaxDD: {m['maxdd']:.2%}"
    )
    if not result.trades.empty:
        print("\nLast 5 trade events:")
        print(result.trades.tail(5).to_string(index=False))


if __name__ == "__main__":
    main()

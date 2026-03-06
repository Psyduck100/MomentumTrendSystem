from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
import sys

import pandas as pd
import yfinance as yf

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backtest.USEQ.Test import AbsFilterConfig, StrategyConfig, run_backtest


BASE_UNIVERSE = ["QQQ", "SCHB", "RSP"]
DEFENSIVE = "IEF"


def _prepare_yf_cache() -> None:
    cache_dir = Path("backtest") / "cache" / "yf"
    cache_dir.mkdir(parents=True, exist_ok=True)
    yf.set_tz_cache_location(str(cache_dir))


def _common_start_date() -> str:
    tickers = BASE_UNIVERSE + ["SCHV", "VLUE", DEFENSIVE]
    px = yf.download(
        tickers=tickers,
        start="2001-01-01",
        end="2026-12-31",
        auto_adjust=True,
        progress=False,
        threads=False,
    )
    close = px["Close"] if isinstance(px.columns, pd.MultiIndex) else px
    first_valid = close[tickers].dropna().index.min()
    if first_valid is None:
        raise RuntimeError("Could not determine shared start date for SCHV/VLUE test.")
    return pd.Timestamp(first_valid).strftime("%Y-%m-%d")


def _run_variant(
    *,
    value_ticker: str,
    start_date: str,
    abs_filter: str,
    tc_bps: float,
) -> dict:
    cfg = StrategyConfig(
        universe_override=BASE_UNIVERSE + [value_ticker],
        defensive_symbol=DEFENSIVE,
        start_date=start_date,
        end_date=None,
        rebalance_freq="M",
    )
    abs_cfg = AbsFilterConfig(kind=abs_filter, ma_days=200, trading_days_per_month=21)
    res = run_backtest(cfg, abs_cfg, transaction_cost_bps=tc_bps)
    bt = res["bt"]
    m = res["metrics"]
    switches = int(bt["position"].ne(bt["position"].shift(1)).sum())
    return {
        "variant": value_ticker,
        "start": str(bt.index.min().date()),
        "end": str(bt.index.max().date()),
        "abs_filter": abs_filter,
        "tc_bps": tc_bps,
        "CAGR": float(m["CAGR"]),
        "Sharpe": float(m["Sharpe"]),
        "MaxDD": float(m["MaxDD"]),
        "TotalReturn": float(bt["equity"].iloc[-1] - 1.0),
        "Switches": switches,
        "Config": asdict(cfg),
    }


def main() -> None:
    _prepare_yf_cache()
    start = _common_start_date()
    rows = []

    for abs_filter in ["ret_12m_pos", "ma_200"]:
        for tc_bps in [0.0, 3.0]:
            for value_ticker in ["SCHV", "VLUE"]:
                rows.append(
                    _run_variant(
                        value_ticker=value_ticker,
                        start_date=start,
                        abs_filter=abs_filter,
                        tc_bps=tc_bps,
                    )
                )

    df = pd.DataFrame(rows)
    out_path = Path("backtest") / "USEQ" / "schv_vs_vlue_results.csv"
    df.to_csv(out_path, index=False)

    shown = df[
        [
            "abs_filter",
            "tc_bps",
            "variant",
            "start",
            "end",
            "CAGR",
            "Sharpe",
            "MaxDD",
            "TotalReturn",
            "Switches",
        ]
    ].copy()
    shown["CAGR"] = shown["CAGR"].map(lambda x: f"{x:.2%}")
    shown["Sharpe"] = shown["Sharpe"].map(lambda x: f"{x:.2f}")
    shown["MaxDD"] = shown["MaxDD"].map(lambda x: f"{x:.2%}")
    shown["TotalReturn"] = shown["TotalReturn"].map(lambda x: f"{x:.2%}")

    print(f"Common SCHV/VLUE start date: {start}")
    print()
    print(shown.to_string(index=False))
    print()
    print(f"Saved raw results: {out_path}")


if __name__ == "__main__":
    main()

from __future__ import annotations

from datetime import datetime
import os
import sys

if __name__ == "__main__" and __package__ is None:
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

import pandas as pd

from CRYP.backtest import run_backtest
from CRYP.calendar import get_trading_days
from CRYP.data import build_btc_proxy, fetch_close
from CRYP.signals import apply_gate, donchian_signal, sma_signal
from CRYP.experiments.sweeps import (
    combine_entry_exit_signals,
    sweep_combined_entry_exit_grid,
)
from CRYP.validate_proxy import validate_proxy


REGIME_SPLITS = {
    "2013-2016": ("2013-01-01", "2016-12-31"),
    "2017-2019": ("2017-01-01", "2019-12-31"),
    "2020-2021": ("2020-01-01", "2021-12-31"),
    "2022": ("2022-01-01", "2022-12-31"),
    "2023-2025": ("2023-01-01", "2025-12-31"),
}


def _print_metrics(label: str, metrics: dict) -> None:
    print(label, metrics)


def _write_csv(df: pd.DataFrame, name: str) -> None:
    if df.empty:
        return
    out_dir = os.path.dirname(os.path.abspath(__file__))
    os.makedirs(out_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(out_dir, f"{name}_{ts}.csv")
    df.to_csv(path, index=False)
    print(f"Wrote CSV: {path}")


def _write_excel_by_logic(df: pd.DataFrame, name: str) -> None:
    if df.empty:
        return
    out_dir = os.path.dirname(os.path.abspath(__file__))
    os.makedirs(out_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(out_dir, f"{name}_{ts}.xlsx")

    def _drop_redundant_columns(
        frame: pd.DataFrame, entry_logic: str, exit_logic: str
    ) -> pd.DataFrame:
        drop_cols = []
        entry_logic = str(entry_logic).upper()
        exit_logic = str(exit_logic).upper()

        if entry_logic == "DONCHIAN":
            drop_cols += ["entry_len", "ma_entry_buffer"]
        if exit_logic == "DONCHIAN":
            drop_cols += ["exit_len", "ma_exit_buffer"]

        if entry_logic == "MA" and exit_logic == "MA":
            drop_cols += [
                "high_len",
                "low_len",
                "donchian_entry_buffer",
                "donchian_exit_buffer",
            ]

        return frame.drop(columns=[c for c in drop_cols if c in frame.columns])

    try:
        with pd.ExcelWriter(path) as writer:
            for (entry_logic, exit_logic), group in df.groupby(
                ["enter_logic", "exit_logic"]
            ):
                sheet_name = f"{entry_logic}_{exit_logic}"[:31]
                trimmed = _drop_redundant_columns(group.copy(), entry_logic, exit_logic)
                trimmed.to_excel(writer, sheet_name=sheet_name, index=False)
        print(f"Wrote Excel: {path}")
    except ImportError as exc:
        print(f"Excel export skipped (missing dependency): {exc}")


def _metrics_2022(returns, signal) -> dict:
    res_2022 = run_backtest(
        returns.loc["2022-01-01":"2022-12-31"],
        signal.loc["2022-01-01":"2022-12-31"],
        cost_bps=10.0,
        slippage_bps=0.0,
        vol_target=None,
        vol_lookback=20,
    )
    return res_2022.metrics


def _run_regimes(
    returns,
    signal,
    label,
    cost_bps=0.0,
    slippage_bps=0.0,
    vol_target=0.2,
    cash_returns=None,
):
    print(f"{label} regime splits:")
    for name, (start, end) in REGIME_SPLITS.items():
        sub_ret = returns.loc[start:end]
        sub_sig = signal.loc[start:end]
        if sub_ret.empty:
            continue
        res = run_backtest(
            sub_ret,
            sub_sig,
            cost_bps=cost_bps,
            slippage_bps=slippage_bps,
            vol_target=vol_target,
            vol_lookback=20,
            cash_returns=cash_returns,
        )
        _print_metrics(f"  {name}", res.metrics)


def main() -> None:
    start = datetime(2013, 1, 1)
    end = datetime(2025, 12, 31)

    trading_days = get_trading_days(start, end, symbol="SPY")
    btc_close = fetch_close("BTC-USD", start, end)
    proxy = build_btc_proxy(btc_close, trading_days, fee_annual=0.0025)

    price = proxy["btc_td"]
    returns = proxy["ret_net"]

    print("Combined SMA + Donchian sweep (entry/exit lengths and k confirms):")
    entry_lengths = (30,)
    exit_lengths = (
        24,
        25,
    )
    ma_entry_buffers = (0.0,)
    ma_exit_buffers = (0.006,)
    donchian_entry_buffers = (0.0015,)
    donchian_exit_buffers = (0.0,)
    highs = (46,)
    lows = (38,)
    gates = ("D",)
    k_confirms_entry = (1,)
    k_confirms_exit = (1,)
    enter_logics = ("DONCHIAN",)
    exit_logics = ("MA",)

    total_configs = (
        len(entry_lengths)
        * len(exit_lengths)
        * len(ma_entry_buffers)
        * len(ma_exit_buffers)
        * len(donchian_entry_buffers)
        * len(donchian_exit_buffers)
        * len(highs)
        * len(lows)
        * len(gates)
        * len(k_confirms_entry)
        * len(k_confirms_exit)
        * len(enter_logics)
        * len(exit_logics)
    )
    n_jobs = 1 if total_configs < 1000 else max((os.cpu_count() or 2) - 1, 1)
    print(f"Starting sweep with n_jobs={n_jobs} (configs={total_configs})")

    combined = sweep_combined_entry_exit_grid(
        price=price,
        returns=returns,
        entry_lengths=entry_lengths,
        exit_lengths=exit_lengths,
        ma_entry_buffers=ma_entry_buffers,
        ma_exit_buffers=ma_exit_buffers,
        donchian_entry_buffers=donchian_entry_buffers,
        donchian_exit_buffers=donchian_exit_buffers,
        highs=highs,
        lows=lows,
        gates=gates,
        k_confirms_entry=k_confirms_entry,
        k_confirms_exit=k_confirms_exit,
        enter_logics=enter_logics,
        exit_logics=exit_logics,
        cost_bps=10.0,
        slippage_bps=0.0,
        vol_target=None,
        vol_lookback=20,
        progress_every=100,
        n_jobs=n_jobs,
        chunk_size=50,
    )
    if not combined.empty:
        combined = combined.copy()
        combined["satellite_score"] = (
            combined["cagr"] - 0.001 * combined["switches_per_year"]
        )

        top_for_2022 = combined.copy()

        sma_entry_cache = {}
        sma_exit_cache = {}
        don_cache = {}
        time_in_mkt_2022 = []
        count_2022 = 0
        total_2022 = len(top_for_2022)
        for _, row in top_for_2022.iterrows():
            entry_len = int(row["entry_len"])
            exit_len = int(row["exit_len"])
            ma_entry_buffer = float(row["ma_entry_buffer"])
            ma_exit_buffer = float(row["ma_exit_buffer"])
            don_entry_buffer = float(row["donchian_entry_buffer"])
            don_exit_buffer = float(row["donchian_exit_buffer"])
            don_key = (
                int(row["high_len"]),
                int(row["low_len"]),
                don_entry_buffer,
                don_exit_buffer,
            )
            if entry_len not in sma_entry_cache:
                sma_entry_cache[entry_len] = sma_signal(
                    price, length=entry_len, buffer=ma_entry_buffer
                )
            if exit_len not in sma_exit_cache:
                sma_exit_cache[exit_len] = sma_signal(
                    price, length=exit_len, buffer=ma_exit_buffer
                )
            if don_key not in don_cache:
                don_cache[don_key] = donchian_signal(
                    price,
                    high_len=don_key[0],
                    low_len=don_key[1],
                    entry_buffer=don_entry_buffer,
                    exit_buffer=don_exit_buffer,
                )
            sig = combine_entry_exit_signals(
                sma_entry_cache[entry_len],
                sma_exit_cache[exit_len],
                don_cache[don_key],
                enter_logic=str(row["enter_logic"]),
                exit_logic=str(row["exit_logic"]),
                gate=str(row["gate"]),
                k_confirm_entry=int(row["k_confirm_entry"]),
                k_confirm_exit=int(row["k_confirm_exit"]),
            )
            metrics_2022 = _metrics_2022(returns, sig)
            time_in_mkt_2022.append(metrics_2022["time_in_market"])
            count_2022 += 1
            if total_2022 and count_2022 % 100 == 0:
                pct = (count_2022 / total_2022) * 100.0
                print(f"2022 metrics progress: {count_2022}/{total_2022} ({pct:.1f}%)")

        top_for_2022["time_in_market_2022"] = time_in_mkt_2022

        combined = combined.sort_values(
            by=["cagr"],
            ascending=False,
        )
        print(combined.head(10).to_string(index=False))
        _write_csv(combined, "cryp_combined_sweep")
        _write_excel_by_logic(combined, "cryp_combined_sweep_by_logic")
    else:
        print("  no combined rows")


if __name__ == "__main__":
    main()

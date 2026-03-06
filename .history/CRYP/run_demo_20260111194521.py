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

    raw = sma_signal(price, length=200, buffer=0.005)
    gated = apply_gate(raw, gate="W-FRI", k_confirm=1)

    buffers = (0.0, 0.005, 0.01)
    print("Buffer n_diff check (gated vs buffer=0):")
    base_signal = apply_gate(
        sma_signal(price, length=200, buffer=buffers[0]), gate="W-FRI", k_confirm=1
    )
    for buf in buffers[1:]:
        sig = apply_gate(
            sma_signal(price, length=200, buffer=buf), gate="W-FRI", k_confirm=1
        )
        diff_count = int((sig != base_signal).sum())
        diff_pct = float((sig != base_signal).mean()) * 100.0
        print(f"  buffer={buf:.3f}: n_diff={diff_count} ({diff_pct:.2f}%)")

    res = run_backtest(
        returns,
        gated,
        cost_bps=10.0,
        slippage_bps=0.0,
        vol_target=0.2,
        vol_lookback=20,
    )
    _print_metrics("SMA metrics:", res.metrics)

    raw_d = donchian_signal(price, high_len=200, low_len=90)
    gated_d = apply_gate(raw_d, gate="W-FRI", k_confirm=1)
    res_d = run_backtest(
        returns,
        gated_d,
        cost_bps=10.0,
        slippage_bps=0.0,
        vol_target=0.2,
        vol_lookback=20,
    )
    _print_metrics("Donchian metrics:", res_d.metrics)

    buy_hold_signal = pd.Series(1.0, index=returns.index)
    buy_hold = run_backtest(
        returns,
        signal=buy_hold_signal,
        cost_bps=0.0,
        slippage_bps=0.0,
        vol_target=None,
    )
    _print_metrics("Buy & Hold metrics:", buy_hold.metrics)

    ibit_start = datetime(2024, 1, 1)
    ibit_close = fetch_close("IBIT", ibit_start, end)
    proxy_2024 = proxy.loc[ibit_start:, "ret_net"]
    val = validate_proxy(proxy_2024, ibit_close)
    print("Proxy validation:", val)
    mean_diff_bps = val.diff_mean_daily * 10000.0
    mean_diff_pct = val.diff_mean_daily * 100.0
    mean_diff_annual_pct = val.mean_diff * 100.0
    te_annual_pct = val.tracking_error * 100.0
    print(
        f"mean_return_diff_daily = {mean_diff_pct:.5f}% ({mean_diff_bps:.3f} bps/day)"
    )
    print(f"mean_return_diff_annualized = {mean_diff_annual_pct:.3f}%/yr")
    print(f"tracking_error_annualized = {te_annual_pct:.2f}%/yr")
    print(f"corr = {val.correlation:.3f}")

    print("Cost sensitivity (SMA):")
    for cost in (0.0, 5.0, 10.0, 20.0):
        res_cost = run_backtest(
            returns,
            gated,
            cost_bps=cost,
            slippage_bps=0.0,
            vol_target=0.2,
            vol_lookback=20,
        )
        _print_metrics(f"  cost_bps={cost}", res_cost.metrics)

    _run_regimes(
        returns,
        gated,
        "SMA",
        cost_bps=10.0,
        slippage_bps=0.0,
        vol_target=0.2,
    )
    _run_regimes(
        returns,
        gated_d,
        "Donchian",
        cost_bps=10.0,
        slippage_bps=0.0,
        vol_target=0.2,
    )
    _run_regimes(
        returns,
        buy_hold_signal,
        "Buy & Hold",
        cost_bps=0.0,
        slippage_bps=0.0,
        vol_target=None,
    )

    n_jobs = max((os.cpu_count() or 2) - 1, 1)
    print("Combined SMA + Donchian sweep (entry/exit lengths and k confirms):")
    ma_entry_buffers = (0.0,)
    ma_exit_buffers = (0, 0.0025, 0.005, 0.0075, 0.001)
    donchian_entry_buffers = (0, 0.0025, 0.005, 0.0075, 0.01, 0.0125)
    donchian_exit_buffers = (0.0,)
    combined_parts = []
    for ma_entry_buffer in ma_entry_buffers:
        for ma_exit_buffer in ma_exit_buffers:
            for donchian_entry_buffer in donchian_entry_buffers:
                for donchian_exit_buffer in donchian_exit_buffers:
                    part = sweep_combined_entry_exit_grid(
                        price=price,
                        returns=returns,
                        entry_lengths=(20, 25, 30),
                        exit_lengths=(25, 30, 35),
                        ma_entry_buffer=ma_entry_buffer,
                        ma_exit_buffer=ma_exit_buffer,
                        donchian_entry_buffer=donchian_entry_buffer,
                        donchian_exit_buffer=donchian_exit_buffer,
                        highs=(40, 45, 50),
                        lows=(35, 40, 45),
                        gates=("D",),
                        k_confirms_entry=(1,),
                        k_confirms_exit=(1,),
                        enter_logics=("DONCHIAN", "AND"),
                        exit_logics=("MA",),
                        cost_bps=10.0,
                        slippage_bps=0.0,
                        vol_target=None,
                        vol_lookback=20,
                        progress_every=100,
                        n_jobs=n_jobs,
                        chunk_size=50,
                    )
                    combined_parts.append(part)
    combined = (
        pd.concat(combined_parts, ignore_index=True)
        if combined_parts
        else pd.DataFrame()
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

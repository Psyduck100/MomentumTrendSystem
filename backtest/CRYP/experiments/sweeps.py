from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import multiprocessing as mp

import pandas as pd

from backtest.CRYP.backtest import run_backtest
from CRYP.trading_calendar import gate_days
from CRYP.signals import apply_gate, donchian_signal, sma_signal


REGIME_SPLITS = {
    "2013-2016": ("2013-01-01", "2016-12-31"),
    "2017-2019": ("2017-01-01", "2019-12-31"),
    "2020-2021": ("2020-01-01", "2021-12-31"),
    "2022": ("2022-01-01", "2022-12-31"),
    "2023-2025": ("2023-01-01", "2025-12-31"),
}


@dataclass(frozen=True)
class SweepConfig:
    gate: str = "W-FRI"
    k_confirm: int = 1
    cost_bps: float = 0.0
    slippage_bps: float = 0.0
    vol_target: float | None = None
    vol_lookback: int = 20
    cash_returns: pd.Series | None = None


def _eval_splits(returns: pd.Series, signal: pd.Series, cfg: SweepConfig) -> dict:
    split_metrics = {}
    for name, (start, end) in REGIME_SPLITS.items():
        sub_ret = returns.loc[start:end]
        sub_sig = signal.loc[start:end]
        if sub_ret.empty:
            continue
        res = run_backtest(
            sub_ret,
            sub_sig,
            cost_bps=cfg.cost_bps,
            slippage_bps=cfg.slippage_bps,
            vol_target=cfg.vol_target,
            vol_lookback=cfg.vol_lookback,
            cash_returns=cfg.cash_returns,
        )
        split_metrics[name] = res.metrics
    return split_metrics


def _robust_min(metric_by_split: dict, key: str) -> float:
    values = [m[key] for m in metric_by_split.values() if key in m]
    return float(min(values)) if values else 0.0


def _worst_maxdd(metric_by_split: dict) -> float:
    values = [m["max_drawdown"] for m in metric_by_split.values() if "max_drawdown" in m]
    return float(min(values)) if values else 0.0


def combine_signals(
    sma_signal_series: pd.Series,
    donchian_signal_series: pd.Series,
    enter_logic: str,
    exit_logic: str,
) -> pd.Series:
    enter_logic = enter_logic.upper()
    exit_logic = exit_logic.upper()
    if enter_logic not in {"AND", "OR", "SMA", "DONCHIAN"}:
        raise ValueError(f"enter_logic must be AND, OR, SMA, or DONCHIAN, got {enter_logic}")
    if exit_logic not in {"AND", "OR", "SMA", "DONCHIAN"}:
        raise ValueError(f"exit_logic must be AND, OR, SMA, or DONCHIAN, got {exit_logic}")

    idx = sma_signal_series.index.union(donchian_signal_series.index)
    sma = sma_signal_series.reindex(idx).fillna(0.0)
    don = donchian_signal_series.reindex(idx).fillna(0.0)

    combined = pd.Series(index=idx, dtype=float)
    current = 0.0
    for ts in idx:
        sma_on = sma.loc[ts] > 0.0
        don_on = don.loc[ts] > 0.0
        if current == 0.0:
            if enter_logic == "AND":
                if sma_on and don_on:
                    current = 1.0
            elif enter_logic == "OR":
                if sma_on or don_on:
                    current = 1.0
            elif enter_logic == "SMA":
                if sma_on:
                    current = 1.0
            else:
                if don_on:
                    current = 1.0
        else:
            if exit_logic == "AND":
                if (not sma_on) and (not don_on):
                    current = 0.0
            elif exit_logic == "OR":
                if (not sma_on) or (not don_on):
                    current = 0.0
            elif exit_logic == "SMA":
                if not sma_on:
                    current = 0.0
            else:
                if not don_on:
                    current = 0.0
        combined.loc[ts] = current
    return combined


def combine_entry_exit_signals(
    sma_entry: pd.Series,
    sma_exit: pd.Series,
    donchian_signal_series: pd.Series,
    enter_logic: str,
    exit_logic: str,
    gate: str,
    k_confirm_entry: int,
    k_confirm_exit: int,
) -> pd.Series:
    enter_logic = enter_logic.upper()
    exit_logic = exit_logic.upper()
    if enter_logic not in {"AND", "OR", "MA", "DONCHIAN"}:
        raise ValueError(
            f"enter_logic must be AND, OR, MA, or DONCHIAN, got {enter_logic}"
        )
    if exit_logic not in {"OR", "MA", "DONCHIAN"}:
        raise ValueError(f"exit_logic must be OR, MA, or DONCHIAN, got {exit_logic}")
    if k_confirm_entry <= 0 or k_confirm_exit <= 0:
        raise ValueError("k_confirm_entry and k_confirm_exit must be > 0")

    idx = sma_entry.index.union(sma_exit.index).union(donchian_signal_series.index)
    sma_entry = sma_entry.reindex(idx).fillna(0.0)
    sma_exit = sma_exit.reindex(idx).fillna(0.0)
    don = donchian_signal_series.reindex(idx).fillna(0.0)

    gate_idx = gate_days(idx, gate)
    combined = pd.Series(index=idx, dtype=float)
    current = 0.0
    entry_count = 0
    exit_count = 0
    for ts in idx:
        if ts in gate_idx:
            sma_entry_on = sma_entry.loc[ts] > 0.0
            sma_exit_on = sma_exit.loc[ts] > 0.0
            don_on = don.loc[ts] > 0.0

            if enter_logic == "AND":
                entry_cond = sma_entry_on and don_on
            elif enter_logic == "OR":
                entry_cond = sma_entry_on or don_on
            elif enter_logic == "MA":
                entry_cond = sma_entry_on
            else:
                entry_cond = don_on

            if exit_logic == "OR":
                exit_cond = (not sma_exit_on) or (not don_on)
            elif exit_logic == "MA":
                exit_cond = not sma_exit_on
            else:
                exit_cond = not don_on

            if current == 0.0:
                if entry_cond:
                    entry_count += 1
                    if entry_count >= k_confirm_entry:
                        current = 1.0
                        entry_count = 0
                else:
                    entry_count = 0
            else:
                if exit_cond:
                    exit_count += 1
                    if exit_count >= k_confirm_exit:
                        current = 0.0
                        exit_count = 0
                else:
                    exit_count = 0
        combined.loc[ts] = current
    return combined


def compute_entry_exit_trigger_counts(
    sma_entry: pd.Series,
    sma_exit: pd.Series,
    donchian_signal_series: pd.Series,
    enter_logic: str,
    exit_logic: str,
    gate: str,
    k_confirm_entry: int,
    k_confirm_exit: int,
) -> dict:
    enter_logic = enter_logic.upper()
    exit_logic = exit_logic.upper()
    idx = sma_entry.index.union(sma_exit.index).union(donchian_signal_series.index)
    sma_entry = sma_entry.reindex(idx).fillna(0.0)
    sma_exit = sma_exit.reindex(idx).fillna(0.0)
    don = donchian_signal_series.reindex(idx).fillna(0.0)
    gate_idx = gate_days(idx, gate)

    current = 0.0
    entry_count = 0
    exit_count = 0
    entry_trig_ma = 0
    entry_trig_don = 0
    exit_trig_ma = 0
    exit_trig_don = 0
    exit_ma_only_gates = 0
    exit_don_only_gates = 0
    exit_both_off_gates = 0
    entries_total = 0
    exits_total = 0

    for ts in idx:
        if ts not in gate_idx:
            continue
        sma_entry_on = sma_entry.loc[ts] > 0.0
        sma_exit_on = sma_exit.loc[ts] > 0.0
        don_on = don.loc[ts] > 0.0

        if enter_logic == "AND":
            entry_cond = sma_entry_on and don_on
        elif enter_logic == "OR":
            entry_cond = sma_entry_on or don_on
        elif enter_logic == "MA":
            entry_cond = sma_entry_on
        else:
            entry_cond = don_on

        if exit_logic == "OR":
            exit_cond = (not sma_exit_on) or (not don_on)
        elif exit_logic == "MA":
            exit_cond = not sma_exit_on
        elif exit_logic == "DONCHIAN":
            exit_cond = not don_on
        else:
            exit_cond = (not sma_exit_on) and (not don_on)

        if current == 0.0:
            if entry_cond:
                entry_count += 1
                if entry_count >= k_confirm_entry:
                    current = 1.0
                    entry_count = 0
                    entries_total += 1
                    if sma_entry_on:
                        entry_trig_ma += 1
                    if don_on:
                        entry_trig_don += 1
            else:
                entry_count = 0
        else:
            if (not sma_exit_on) and don_on:
                exit_ma_only_gates += 1
            if sma_exit_on and (not don_on):
                exit_don_only_gates += 1
            if (not sma_exit_on) and (not don_on):
                exit_both_off_gates += 1
            if exit_cond:
                exit_count += 1
                if exit_count >= k_confirm_exit:
                    current = 0.0
                    exit_count = 0
                    exits_total += 1
                    if not sma_exit_on:
                        exit_trig_ma += 1
                    if not don_on:
                        exit_trig_don += 1
            else:
                exit_count = 0

    return {
        "entries_total": entries_total,
        "exits_total": exits_total,
        "entry_trig_ma": entry_trig_ma,
        "entry_trig_donchian": entry_trig_don,
        "exit_trig_ma": exit_trig_ma,
        "exit_trig_donchian": exit_trig_don,
        "exit_ma_only_gates": exit_ma_only_gates,
        "exit_don_only_gates": exit_don_only_gates,
        "exit_both_off_gates": exit_both_off_gates,
    }


def sweep_sma(
    price: pd.Series,
    returns: pd.Series,
    lengths: Iterable[int],
    buffers: Iterable[float],
    cfg: SweepConfig,
) -> pd.DataFrame:
    rows = []
    for length in lengths:
        for buffer in buffers:
            raw = sma_signal(price, length, buffer=buffer)
            gated = apply_gate(raw, gate=cfg.gate, k_confirm=cfg.k_confirm)
            res = run_backtest(
                returns,
                gated,
                cost_bps=cfg.cost_bps,
                slippage_bps=cfg.slippage_bps,
                vol_target=cfg.vol_target,
                vol_lookback=cfg.vol_lookback,
                cash_returns=cfg.cash_returns,
            )
            split_metrics = _eval_splits(returns, gated, cfg)
            rows.append(
                {
                    "length": length,
                    "buffer": buffer,
                    **res.metrics,
                    "robust_min_sharpe": _robust_min(split_metrics, "sharpe"),
                    "robust_min_cagr": _robust_min(split_metrics, "cagr"),
                    "worst_maxdd": _worst_maxdd(split_metrics),
                }
            )
    return pd.DataFrame(rows)


def sweep_donchian(
    price: pd.Series,
    returns: pd.Series,
    highs: Iterable[int],
    lows: Iterable[int],
    cfg: SweepConfig,
) -> pd.DataFrame:
    rows = []
    for high_len in highs:
        for low_len in lows:
            raw = donchian_signal(price, high_len, low_len)
            gated = apply_gate(raw, gate=cfg.gate, k_confirm=cfg.k_confirm)
            res = run_backtest(
                returns,
                gated,
                cost_bps=cfg.cost_bps,
                slippage_bps=cfg.slippage_bps,
                vol_target=cfg.vol_target,
                vol_lookback=cfg.vol_lookback,
                cash_returns=cfg.cash_returns,
            )
            split_metrics = _eval_splits(returns, gated, cfg)
            rows.append(
                {
                    "high_len": high_len,
                    "low_len": low_len,
                    **res.metrics,
                    "robust_min_sharpe": _robust_min(split_metrics, "sharpe"),
                    "robust_min_cagr": _robust_min(split_metrics, "cagr"),
                    "worst_maxdd": _worst_maxdd(split_metrics),
                }
            )
    return pd.DataFrame(rows)


def sweep_sma_grid(
    price: pd.Series,
    returns: pd.Series,
    lengths: Iterable[int],
    buffers: Iterable[float],
    gates: Iterable[str],
    k_confirms: Iterable[int],
    cost_bps: float = 0.0,
    slippage_bps: float = 0.0,
    vol_target: float | None = None,
    vol_lookback: int = 20,
    cash_returns: pd.Series | None = None,
) -> pd.DataFrame:
    rows = []
    for gate in gates:
        for k_confirm in k_confirms:
            cfg = SweepConfig(
                gate=gate,
                k_confirm=k_confirm,
                cost_bps=cost_bps,
                slippage_bps=slippage_bps,
                vol_target=vol_target,
                vol_lookback=vol_lookback,
                cash_returns=cash_returns,
            )
            df = sweep_sma(price, returns, lengths, buffers, cfg)
            df["gate"] = gate
            df["k_confirm"] = k_confirm
            rows.append(df)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def sweep_donchian_grid(
    price: pd.Series,
    returns: pd.Series,
    highs: Iterable[int],
    lows: Iterable[int],
    gates: Iterable[str],
    k_confirms: Iterable[int],
    cost_bps: float = 0.0,
    slippage_bps: float = 0.0,
    vol_target: float | None = None,
    vol_lookback: int = 20,
    cash_returns: pd.Series | None = None,
) -> pd.DataFrame:
    rows = []
    for gate in gates:
        for k_confirm in k_confirms:
            cfg = SweepConfig(
                gate=gate,
                k_confirm=k_confirm,
                cost_bps=cost_bps,
                slippage_bps=slippage_bps,
                vol_target=vol_target,
                vol_lookback=vol_lookback,
                cash_returns=cash_returns,
            )
            df = sweep_donchian(price, returns, highs, lows, cfg)
            df["gate"] = gate
            df["k_confirm"] = k_confirm
            rows.append(df)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def sweep_combined_grid(
    price: pd.Series,
    returns: pd.Series,
    lengths: Iterable[int],
    buffers: Iterable[float],
    highs: Iterable[int],
    lows: Iterable[int],
    gates: Iterable[str],
    k_confirms: Iterable[int],
    enter_logics: Iterable[str],
    exit_logics: Iterable[str],
    cost_bps: float = 0.0,
    slippage_bps: float = 0.0,
    vol_target: float | None = None,
    vol_lookback: int = 20,
    cash_returns: pd.Series | None = None,
    progress_every: int = 100,
) -> pd.DataFrame:
    rows = []
    lengths = list(lengths)
    buffers = list(buffers)
    highs = list(highs)
    lows = list(lows)
    gates = list(gates)
    k_confirms = list(k_confirms)
    enter_logics = list(enter_logics)
    exit_logics = list(exit_logics)
    total = (
        len(lengths)
        * len(buffers)
        * len(highs)
        * len(lows)
        * len(gates)
        * len(k_confirms)
        * len(enter_logics)
        * len(exit_logics)
    )
    count = 0
    for gate in gates:
        for k_confirm in k_confirms:
            cfg = SweepConfig(
                gate=gate,
                k_confirm=k_confirm,
                cost_bps=cost_bps,
                slippage_bps=slippage_bps,
                vol_target=vol_target,
                vol_lookback=vol_lookback,
                cash_returns=cash_returns,
            )
            for length in lengths:
                for buffer in buffers:
                    sma_sig = sma_signal(price, length, buffer=buffer)
                    for high_len in highs:
                        for low_len in lows:
                            don_sig = donchian_signal(price, high_len, low_len)
                            for enter_logic in enter_logics:
                                for exit_logic in exit_logics:
                                    raw = combine_signals(
                                        sma_sig,
                                        don_sig,
                                        enter_logic=enter_logic,
                                        exit_logic=exit_logic,
                                    )
                                    gated = apply_gate(raw, gate=cfg.gate, k_confirm=cfg.k_confirm)
                                    res = run_backtest(
                                        returns,
                                        gated,
                                        cost_bps=cfg.cost_bps,
                                        slippage_bps=cfg.slippage_bps,
                                        vol_target=cfg.vol_target,
                                        vol_lookback=cfg.vol_lookback,
                                        cash_returns=cfg.cash_returns,
                                    )
                                    split_metrics = _eval_splits(returns, gated, cfg)
                                    split_cagrs = {}
                                    for split_name, metrics in split_metrics.items():
                                        if split_name == "2022":
                                            continue
                                        col = f"cagr_{split_name.replace('-', '_')}"
                                        split_cagrs[col] = metrics.get("cagr", 0.0)
                                    rows.append(
                                        {
                                            "length": length,
                                            "buffer": buffer,
                                            "high_len": high_len,
                                            "low_len": low_len,
                                            "enter_logic": enter_logic,
                                            "exit_logic": exit_logic,
                                            "gate": gate,
                                            "k_confirm": k_confirm,
                                            **res.metrics,
                                            **split_cagrs,
                                        }
                                    )
                                    count += 1
                                    if progress_every > 0 and count % progress_every == 0:
                                        pct = (count / total) * 100.0 if total else 100.0
                                        print(
                                            f"combined sweep progress: {count}/{total} "
                                            f"({pct:.1f}%)"
                                        )
    return pd.DataFrame(rows)


def sweep_combined_entry_exit_grid(
    price: pd.Series,
    returns: pd.Series,
    entry_lengths: Iterable[int],
    exit_lengths: Iterable[int],
    ma_entry_buffers: Iterable[float],
    ma_exit_buffers: Iterable[float],
    donchian_entry_buffers: Iterable[float],
    donchian_exit_buffers: Iterable[float],
    highs: Iterable[int],
    lows: Iterable[int],
    gates: Iterable[str],
    k_confirms_entry: Iterable[int],
    k_confirms_exit: Iterable[int],
    enter_logics: Iterable[str],
    exit_logics: Iterable[str],
    cost_bps: float = 0.0,
    slippage_bps: float = 0.0,
    vol_target: float | None = None,
    vol_lookback: int = 20,
    cash_returns: pd.Series | None = None,
    progress_every: int = 100,
    n_jobs: int = 1,
    chunk_size: int = 50,
) -> pd.DataFrame:
    rows = []
    entry_lengths = list(entry_lengths)
    exit_lengths = list(exit_lengths)
    ma_entry_buffers = list(ma_entry_buffers)
    ma_exit_buffers = list(ma_exit_buffers)
    donchian_entry_buffers = list(donchian_entry_buffers)
    donchian_exit_buffers = list(donchian_exit_buffers)
    highs = list(highs)
    lows = list(lows)
    gates = list(gates)
    k_confirms_entry = list(k_confirms_entry)
    k_confirms_exit = list(k_confirms_exit)
    enter_logics = list(enter_logics)
    exit_logics = list(exit_logics)
    total = (
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
    count = 0
    if n_jobs <= 1:
        sma_entry_cache: dict[int, pd.Series] = {}
        sma_exit_cache: dict[int, pd.Series] = {}
        don_cache: dict[tuple[int, int], pd.Series] = {}

    if n_jobs <= 1:
        for gate in gates:
            for k_entry in k_confirms_entry:
                for k_exit in k_confirms_exit:
                    cfg = SweepConfig(
                        gate=gate,
                        k_confirm=1,
                        cost_bps=cost_bps,
                        slippage_bps=slippage_bps,
                        vol_target=vol_target,
                        vol_lookback=vol_lookback,
                        cash_returns=cash_returns,
                    )
                    for entry_len in entry_lengths:
                        for exit_len in exit_lengths:
                            for ma_entry_buffer in ma_entry_buffers:
                                if (entry_len, ma_entry_buffer) not in sma_entry_cache:
                                    sma_entry_cache[(entry_len, ma_entry_buffer)] = sma_signal(
                                        price, length=entry_len, buffer=ma_entry_buffer
                                    )
                                for ma_exit_buffer in ma_exit_buffers:
                                    if (exit_len, ma_exit_buffer) not in sma_exit_cache:
                                        sma_exit_cache[(exit_len, ma_exit_buffer)] = sma_signal(
                                            price, length=exit_len, buffer=ma_exit_buffer
                                        )
                                    for donchian_entry_buffer in donchian_entry_buffers:
                                        for donchian_exit_buffer in donchian_exit_buffers:
                                            for high_len in highs:
                                                for low_len in lows:
                                                    don_key = (
                                                        high_len,
                                                        low_len,
                                                        donchian_entry_buffer,
                                                        donchian_exit_buffer,
                                                    )
                                                    if don_key not in don_cache:
                                                        don_cache[don_key] = donchian_signal(
                                                            price,
                                                            high_len=high_len,
                                                            low_len=low_len,
                                                            entry_buffer=donchian_entry_buffer,
                                                            exit_buffer=donchian_exit_buffer,
                                                        )
                                                    for enter_logic in enter_logics:
                                                        for exit_logic in exit_logics:
                                                            sma_entry_series = sma_entry_cache[
                                                                (entry_len, ma_entry_buffer)
                                                            ]
                                                            sma_exit_series = sma_exit_cache[
                                                                (exit_len, ma_exit_buffer)
                                                            ]
                                                            raw = combine_entry_exit_signals(
                                                                sma_entry_series,
                                                                sma_exit_series,
                                                                don_cache[don_key],
                                                                enter_logic=enter_logic,
                                                                exit_logic=exit_logic,
                                                                gate=gate,
                                                                k_confirm_entry=k_entry,
                                                                k_confirm_exit=k_exit,
                                                            )
                                                            trigger_counts = (
                                                                compute_entry_exit_trigger_counts(
                                                                    sma_entry_series,
                                                                    sma_exit_series,
                                                                    don_cache[don_key],
                                                                    enter_logic=enter_logic,
                                                                    exit_logic=exit_logic,
                                                                    gate=gate,
                                                                    k_confirm_entry=k_entry,
                                                                    k_confirm_exit=k_exit,
                                                                )
                                                            )
                                                            res = run_backtest(
                                                                returns,
                                                                raw,
                                                                cost_bps=cfg.cost_bps,
                                                                slippage_bps=cfg.slippage_bps,
                                                                vol_target=cfg.vol_target,
                                                                vol_lookback=cfg.vol_lookback,
                                                                cash_returns=cfg.cash_returns,
                                                            )
                                                            split_metrics = _eval_splits(
                                                                returns, raw, cfg
                                                            )
                                                            split_cagrs = {}
                                                            for (
                                                                split_name,
                                                                metrics,
                                                            ) in split_metrics.items():
                                                                if split_name == "2022":
                                                                    continue
                                                                col = (
                                                                    f"cagr_{split_name.replace('-', '_')}"
                                                                )
                                                                split_cagrs[col] = metrics.get(
                                                                    "cagr", 0.0
                                                                )
                                                            rows.append(
                                                                {
                                                                    "entry_len": entry_len,
                                                                    "exit_len": exit_len,
                                                                    "ma_entry_buffer": ma_entry_buffer,
                                                                    "ma_exit_buffer": ma_exit_buffer,
                                                                    "donchian_entry_buffer": donchian_entry_buffer,
                                                                    "donchian_exit_buffer": donchian_exit_buffer,
                                                                    "high_len": high_len,
                                                                    "low_len": low_len,
                                                                    "enter_logic": enter_logic,
                                                                    "exit_logic": exit_logic,
                                                                    "gate": gate,
                                                                    "k_confirm_entry": k_entry,
                                                                    "k_confirm_exit": k_exit,
                                                                    **res.metrics,
                                                                    **split_cagrs,
                                                                    **trigger_counts,
                                                                }
                                                            )
                                                            count += 1
                                                            if (
                                                                progress_every > 0
                                                                and count % progress_every == 0
                                                            ):
                                                                pct = (
                                                                    (count / total) * 100.0
                                                                    if total
                                                                    else 100.0
                                                                )
                                                                print(
                                                                    "combined sweep progress: "
                                                                    f"{count}/{total} "
                                                                    f"({pct:.1f}%)"
                                                                )
        return pd.DataFrame(rows)

    tasks = []
    for gate in gates:
        for k_entry in k_confirms_entry:
            for k_exit in k_confirms_exit:
                for entry_len in entry_lengths:
                    for exit_len in exit_lengths:
                        for ma_entry_buffer in ma_entry_buffers:
                            for ma_exit_buffer in ma_exit_buffers:
                                for donchian_entry_buffer in donchian_entry_buffers:
                                    for donchian_exit_buffer in donchian_exit_buffers:
                                        for high_len in highs:
                                            for low_len in lows:
                                                for enter_logic in enter_logics:
                                                    for exit_logic in exit_logics:
                                                        tasks.append(
                                                            (
                                                                gate,
                                                                k_entry,
                                                                k_exit,
                                                                entry_len,
                                                                exit_len,
                                                                ma_entry_buffer,
                                                                ma_exit_buffer,
                                                                donchian_entry_buffer,
                                                                donchian_exit_buffer,
                                                                high_len,
                                                                low_len,
                                                                enter_logic,
                                                                exit_logic,
                                                            )
                                                        )
    chunks = [tasks[i : i + chunk_size] for i in range(0, len(tasks), chunk_size)]

    ctx = mp.get_context("spawn")
    with ctx.Pool(processes=n_jobs) as pool:
        args_iter = (
            (
                price,
                returns,
                cost_bps,
                slippage_bps,
                vol_target,
                vol_lookback,
                cash_returns,
                chunk,
            )
            for chunk in chunks
        )
        for chunk_rows in pool.imap_unordered(_combined_entry_exit_worker, args_iter):
            rows.extend(chunk_rows)
            count += len(chunk_rows)
            if progress_every > 0 and count % progress_every == 0:
                pct = (count / total) * 100.0 if total else 100.0
                print(f"combined sweep progress: {count}/{total} ({pct:.1f}%)")
    return pd.DataFrame(rows)


def _combined_entry_exit_worker(args: tuple) -> list[dict]:
    (
        price,
        returns,
        cost_bps,
        slippage_bps,
        vol_target,
        vol_lookback,
        cash_returns,
        chunk,
    ) = args
    local_rows = []
    sma_entry_cache: dict[int, pd.Series] = {}
    sma_exit_cache: dict[int, pd.Series] = {}
    don_cache: dict[tuple[int, int], pd.Series] = {}
    for (
        gate,
        k_entry,
        k_exit,
        entry_len,
        exit_len,
        ma_entry_buffer,
        ma_exit_buffer,
        donchian_entry_buffer,
        donchian_exit_buffer,
        high_len,
        low_len,
        enter_logic,
        exit_logic,
    ) in chunk:
        if (entry_len, ma_entry_buffer) not in sma_entry_cache:
            sma_entry_cache[(entry_len, ma_entry_buffer)] = sma_signal(
                price, length=entry_len, buffer=ma_entry_buffer
            )
        if (exit_len, ma_exit_buffer) not in sma_exit_cache:
            sma_exit_cache[(exit_len, ma_exit_buffer)] = sma_signal(
                price, length=exit_len, buffer=ma_exit_buffer
            )
        don_key = (high_len, low_len, donchian_entry_buffer, donchian_exit_buffer)
        if don_key not in don_cache:
            don_cache[don_key] = donchian_signal(
                price,
                high_len=high_len,
                low_len=low_len,
                entry_buffer=donchian_entry_buffer,
                exit_buffer=donchian_exit_buffer,
            )
        sma_entry_series = sma_entry_cache[(entry_len, ma_entry_buffer)]
        sma_exit_series = sma_exit_cache[(exit_len, ma_exit_buffer)]
        raw = combine_entry_exit_signals(
            sma_entry_series,
            sma_exit_series,
            don_cache[don_key],
            enter_logic=enter_logic,
            exit_logic=exit_logic,
            gate=gate,
            k_confirm_entry=k_entry,
            k_confirm_exit=k_exit,
        )
        trigger_counts = compute_entry_exit_trigger_counts(
            sma_entry_series,
            sma_exit_series,
            don_cache[don_key],
            enter_logic=enter_logic,
            exit_logic=exit_logic,
            gate=gate,
            k_confirm_entry=k_entry,
            k_confirm_exit=k_exit,
        )
        res = run_backtest(
            returns,
            raw,
            cost_bps=cost_bps,
            slippage_bps=slippage_bps,
            vol_target=vol_target,
            vol_lookback=vol_lookback,
            cash_returns=cash_returns,
        )
        cfg = SweepConfig(
            gate=gate,
            k_confirm=1,
            cost_bps=cost_bps,
            slippage_bps=slippage_bps,
            vol_target=vol_target,
            vol_lookback=vol_lookback,
            cash_returns=cash_returns,
        )
        split_metrics = _eval_splits(returns, raw, cfg)
        split_cagrs = {}
        for split_name, metrics in split_metrics.items():
            if split_name == "2022":
                continue
            col = f"cagr_{split_name.replace('-', '_')}"
            split_cagrs[col] = metrics.get("cagr", 0.0)
        local_rows.append(
            {
                "entry_len": entry_len,
                "exit_len": exit_len,
                "ma_entry_buffer": ma_entry_buffer,
                "ma_exit_buffer": ma_exit_buffer,
                "donchian_entry_buffer": donchian_entry_buffer,
                "donchian_exit_buffer": donchian_exit_buffer,
                "high_len": high_len,
                "low_len": low_len,
                "enter_logic": enter_logic,
                "exit_logic": exit_logic,
                "gate": gate,
                "k_confirm_entry": k_entry,
                "k_confirm_exit": k_exit,
                **res.metrics,
                **split_cagrs,
                **trigger_counts,
            }
        )
    return local_rows

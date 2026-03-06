from __future__ import annotations

from dataclasses import dataclass
import pandas as pd

from CRYP.trading_calendar import gate_days


@dataclass(frozen=True)
class SignalConfig:
    gate: str = "W-FRI"
    k_confirm: int = 1


def _apply_hysteresis(raw_on: pd.Series, raw_off: pd.Series) -> pd.Series:
    signal = pd.Series(index=raw_on.index, dtype=float)
    current = 0.0
    for ts in raw_on.index:
        if raw_on.loc[ts]:
            current = 1.0
        elif raw_off.loc[ts]:
            current = 0.0
        signal.loc[ts] = current
    return signal


def sma_signal(price: pd.Series, length: int, buffer: float = 0.0) -> pd.Series:
    if length <= 0:
        raise ValueError("length must be > 0")
    sma = price.rolling(length).mean()
    on = price > sma * (1.0 + buffer)
    off = price < sma * (1.0 - buffer)
    return _apply_hysteresis(on, off)


def donchian_signal(
    price: pd.Series,
    high_len: int,
    low_len: int,
    entry_buffer: float = 0.0,
    exit_buffer: float = 0.0,
) -> pd.Series:
    if high_len <= 0 or low_len <= 0:
        raise ValueError("lengths must be > 0")
    shifted = price.shift(1)
    high_n = shifted.rolling(high_len).max()
    low_m = shifted.rolling(low_len).min()
    on = price > high_n * (1.0 + entry_buffer)
    off = price < low_m * (1.0 - exit_buffer)
    return _apply_hysteresis(on, off)


def apply_gate(
    signal: pd.Series,
    gate: str = "W-FRI",
    k_confirm: int = 1,
) -> pd.Series:
    if k_confirm <= 0:
        raise ValueError("k_confirm must be > 0")
    gated = pd.Series(index=signal.index, dtype=float)
    gate_idx = gate_days(signal.index, gate)
    current = 0.0
    last_desired = current
    confirm = 0
    for ts in signal.index:
        if ts in gate_idx:
            desired = float(signal.loc[ts])
            if desired != current:
                if desired == last_desired:
                    confirm += 1
                else:
                    confirm = 1
                    last_desired = desired
                if confirm >= k_confirm:
                    current = desired
                    confirm = 0
            else:
                confirm = 0
                last_desired = desired
        gated.loc[ts] = current
    return gated


def gated_signal(signal: pd.Series, config: SignalConfig) -> pd.Series:
    return apply_gate(signal, gate=config.gate, k_confirm=config.k_confirm)


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

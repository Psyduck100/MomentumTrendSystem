from __future__ import annotations

from datetime import datetime
import os
import sys

if __name__ == "__main__" and __package__ is None:
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

import pandas as pd

from CRYP.trading_calendar import get_trading_days, trading_days_from_series
from CRYP.data import build_btc_proxy, fetch_close
from CRYP.signals import combine_entry_exit_signals, donchian_signal, sma_signal


GATE = "D"
K_CONFIRM_ENTRY = 1
K_CONFIRM_EXIT = 1


CONFIG_PRIMARY = {
    "entry_len": 30,
    "exit_len": 32,
    "ma_entry_buffer": 0.0,
    "ma_exit_buffer": 0.006,
    "donchian_entry_buffer": 0.0015,
    "donchian_exit_buffer": 0.0,
    "high_len": 40,
    "low_len": 45,
    "enter_logic": "AND",
    "exit_logic": "MA",
}

CONFIG_ALT = {
    "entry_len": None,
    "exit_len": (24, 25),
    "ma_entry_buffer": None,
    "ma_exit_buffer": 0.005,
    "donchian_entry_buffer": 0.0075,
    "donchian_exit_buffer": 0.0,
    "high_len": 46,
    "low_len": 38,
    "enter_logic": "DONCHIAN",
    "exit_logic": "MA",
}


def _ma_signal_or_off(price: pd.Series, length: int | None, buffer: float | None) -> pd.Series:
    if length is None:
        return pd.Series(0.0, index=price.index)
    return sma_signal(price, length=length, buffer=buffer or 0.0)


def _avg_ma_signal(price: pd.Series, lengths: tuple[int, int], buffer: float | None) -> pd.Series:
    left = sma_signal(price, length=int(lengths[0]), buffer=buffer or 0.0)
    right = sma_signal(price, length=int(lengths[1]), buffer=buffer or 0.0)
    return (left + right) / 2.0


def _latest_btc_price(
    btc_close: pd.Series,
    override_price: float | None = None,
) -> float:
    if override_price is not None:
        return float(override_price)
    return float(btc_close.iloc[-1])


def load_proxy_data(
    start: datetime,
    end: datetime,
    use_btc_calendar: bool = False,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    btc_close = fetch_close("BTC-USD", start, end)
    if use_btc_calendar:
        trading_days = trading_days_from_series(btc_close)
    else:
        trading_days = get_trading_days(start, end, symbol="SPY")
    proxy = build_btc_proxy(btc_close, trading_days, fee_annual=0.0025)
    return proxy["btc_td"], proxy["ret_net"], btc_close


def build_components_for_config(
    price: pd.Series, cfg: dict
) -> tuple[pd.Series, pd.Series, pd.Series]:
    entry_len = cfg["entry_len"]
    exit_len = cfg["exit_len"]

    sma_entry = _ma_signal_or_off(price, entry_len, cfg["ma_entry_buffer"])
    if exit_len is None:
        sma_exit = pd.Series(0.0, index=price.index)
    elif isinstance(exit_len, (tuple, list)):
        lengths = [int(v) for v in exit_len]
        if len(lengths) != 2:
            raise ValueError("exit_len must be a single int or a 2-length tuple/list")
        sma_exit = _avg_ma_signal(price, (lengths[0], lengths[1]), cfg["ma_exit_buffer"])
    else:
        sma_exit = _ma_signal_or_off(price, int(exit_len), cfg["ma_exit_buffer"])

    don = donchian_signal(
        price,
        high_len=cfg["high_len"],
        low_len=cfg["low_len"],
        entry_buffer=cfg["donchian_entry_buffer"] or 0.0,
        exit_buffer=cfg["donchian_exit_buffer"] or 0.0,
    )

    return sma_entry, sma_exit, don


def build_signal_for_config(price: pd.Series, cfg: dict) -> pd.Series:
    sma_entry, sma_exit, don = build_components_for_config(price, cfg)
    return combine_entry_exit_signals(
        sma_entry,
        sma_exit,
        don,
        enter_logic=cfg["enter_logic"],
        exit_logic=cfg["exit_logic"],
        gate=GATE,
        k_confirm_entry=K_CONFIRM_ENTRY,
        k_confirm_exit=K_CONFIRM_EXIT,
    )


def run_daily_configs(
    start: datetime | None = None,
    end: datetime | None = None,
    use_btc_calendar: bool = False,
) -> dict:
    if start is None:
        start = datetime(2013, 1, 1)
    if end is None:
        end = datetime.today()

    price, returns, btc_close = load_proxy_data(
        start, end, use_btc_calendar=use_btc_calendar
    )
    results = {}
    for label, cfg in (("primary", CONFIG_PRIMARY), ("alt", CONFIG_ALT)):
        sma_entry, sma_exit, don = build_components_for_config(price, cfg)
        signal = combine_entry_exit_signals(
            sma_entry,
            sma_exit,
            don,
            enter_logic=cfg["enter_logic"],
            exit_logic=cfg["exit_logic"],
            gate=GATE,
            k_confirm_entry=K_CONFIRM_ENTRY,
            k_confirm_exit=K_CONFIRM_EXIT,
        )
        results[label] = {
            "config": cfg,
            "signal": signal,
            "latest_signal": int(signal.iloc[-1]),
            "latest_date": signal.index[-1],
            "btc_close": btc_close,
            "price": price,
            "sma_entry": sma_entry,
            "sma_exit": sma_exit,
            "don": don,
        }
    return results


def summarize_signal(signal: pd.Series) -> dict:
    if signal.empty:
        return {"action": "NO DATA", "reason": "No signal values available."}
    latest = int(signal.iloc[-1])
    prev = int(signal.iloc[-2]) if len(signal) > 1 else latest
    if latest == 1 and prev == 0:
        return {"action": "BUY", "reason": "Signal flipped ON versus the prior day."}
    if latest == 0 and prev == 1:
        return {"action": "SELL", "reason": "Signal flipped OFF versus the prior day."}
    if latest == 1:
        return {"action": "HOLD", "reason": "Signal remains ON; stay invested."}
    return {"action": "HOLD", "reason": "Signal remains OFF; stay in cash."}


def _price_position(price: float, low: float | None, high: float | None) -> str:
    if low is None or high is None:
        return "unknown"
    if price > high:
        return "above"
    if price < low:
        return "below"
    return "between"


def _fmt_level(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.2f}"


def _sma_levels(
    price: pd.Series,
    length: int,
    buffer: float,
    ts: pd.Timestamp,
) -> tuple[float | None, float | None]:
    sma = price.rolling(length).mean()
    if ts not in sma.index:
        return None, None
    base = sma.loc[ts]
    if pd.isna(base):
        return None, None
    return float(base * (1.0 - buffer)), float(base * (1.0 + buffer))


def _donchian_levels(
    price: pd.Series,
    high_len: int,
    low_len: int,
    entry_buffer: float,
    exit_buffer: float,
    ts: pd.Timestamp,
) -> tuple[float | None, float | None]:
    shifted = price.shift(1)
    high_n = shifted.rolling(high_len).max()
    low_m = shifted.rolling(low_len).min()
    if ts not in high_n.index or ts not in low_m.index:
        return None, None
    high_val = high_n.loc[ts]
    low_val = low_m.loc[ts]
    if pd.isna(high_val) or pd.isna(low_val):
        return None, None
    entry_level = float(high_val * (1.0 + entry_buffer))
    exit_level = float(low_val * (1.0 - exit_buffer))
    return entry_level, exit_level


def describe_signal_reason(
    price: pd.Series,
    signal: pd.Series,
    sma_entry: pd.Series,
    sma_exit: pd.Series,
    don: pd.Series,
    cfg: dict,
) -> dict:
    if signal.empty:
        return {"action": "NO DATA", "reason": "No signal values available."}
    ts = signal.index[-1]
    last_price = float(price.loc[ts])
    latest = int(signal.iloc[-1])
    prev = int(signal.iloc[-2]) if len(signal) > 1 else latest

    sma_entry_on = bool(sma_entry.loc[ts] > 0.0)
    sma_exit_on = bool(sma_exit.loc[ts] > 0.0)
    don_on = bool(don.loc[ts] > 0.0)

    enter_logic = cfg["enter_logic"].upper()
    exit_logic = cfg["exit_logic"].upper()

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

    if latest == 1 and prev == 0:
        action_label = "Entry triggered"
    elif latest == 0 and prev == 1:
        action_label = "Exit triggered"
    elif latest == 1:
        action_label = "Signal ON"
    else:
        action_label = "Signal OFF"

    entry_len = cfg["entry_len"]
    exit_len = cfg["exit_len"]

    entry_parts = []
    entry_trigger_ma = None
    entry_trigger_don = None
    if entry_len is not None:
        entry_low, entry_high = _sma_levels(
            price, int(entry_len), float(cfg["ma_entry_buffer"] or 0.0), ts
        )
        entry_pos = _price_position(last_price, entry_low, entry_high)
        entry_trigger_ma = entry_high is not None and last_price > entry_high
        entry_parts.append(
            f"MA entry {'ON' if sma_entry_on else 'OFF'} "
            f"(band {_fmt_level(entry_low)} to {_fmt_level(entry_high)})"
        )
    if enter_logic in {"AND", "OR", "DONCHIAN"}:
        don_entry, _ = _donchian_levels(
            price,
            int(cfg["high_len"]),
            int(cfg["low_len"]),
            float(cfg["donchian_entry_buffer"] or 0.0),
            float(cfg["donchian_exit_buffer"] or 0.0),
            ts,
        )
        if don_entry is None:
            don_pos = "unknown"
            entry_trigger_don = False
        else:
            don_pos = "above" if last_price > don_entry else "below"
            entry_trigger_don = last_price > don_entry
        entry_parts.append(
            f"Donchian {'ON' if don_on else 'OFF'} "
            f"(breakout {_fmt_level(don_entry)})"
        )

    exit_parts = []
    exit_trigger_ma = None
    exit_trigger_don = None
    if exit_logic in {"MA", "OR"}:
        if isinstance(exit_len, (tuple, list)):
            lengths = [int(v) for v in exit_len]
            low_0, high_0 = _sma_levels(
                price, lengths[0], float(cfg["ma_exit_buffer"] or 0.0), ts
            )
            low_1, high_1 = _sma_levels(
                price, lengths[1], float(cfg["ma_exit_buffer"] or 0.0), ts
            )
            pos_0 = _price_position(last_price, low_0, high_0)
            pos_1 = _price_position(last_price, low_1, high_1)
            exit_trigger_ma = (
                low_0 is not None
                and low_1 is not None
                and last_price < low_0
                and last_price < low_1
            )
            exit_parts.append(
                f"MA exit {'ON' if sma_exit_on else 'OFF'} "
                f"({lengths[0]}d band {_fmt_level(low_0)} to {_fmt_level(high_0)}, "
                f"{lengths[1]}d band {_fmt_level(low_1)} to {_fmt_level(high_1)})"
            )
        elif exit_len is not None:
            exit_low, exit_high = _sma_levels(
                price, int(exit_len), float(cfg["ma_exit_buffer"] or 0.0), ts
            )
            exit_pos = _price_position(last_price, exit_low, exit_high)
            exit_trigger_ma = exit_low is not None and last_price < exit_low
            exit_parts.append(
                f"MA exit {'ON' if sma_exit_on else 'OFF'} "
                f"(band {_fmt_level(exit_low)} to {_fmt_level(exit_high)})"
            )
    if exit_logic in {"DONCHIAN", "OR"}:
        _, don_exit = _donchian_levels(
            price,
            int(cfg["high_len"]),
            int(cfg["low_len"]),
            float(cfg["donchian_entry_buffer"] or 0.0),
            float(cfg["donchian_exit_buffer"] or 0.0),
            ts,
        )
        if don_exit is None:
            don_exit_pos = "unknown"
            exit_trigger_don = False
        else:
            don_exit_pos = "below" if last_price < don_exit else "above"
            exit_trigger_don = last_price < don_exit
        exit_parts.append(
            f"Donchian {'ON' if don_on else 'OFF'} "
            f"(breakdown {_fmt_level(don_exit)})"
        )

    entry_status = "met" if entry_cond else "not met"
    exit_status = "met" if exit_cond else "not met"
    entry_detail = "; ".join(entry_parts) if entry_parts else "Entry logic unavailable."
    exit_detail = "; ".join(exit_parts) if exit_parts else "Exit logic unavailable."

    if enter_logic == "AND":
        entry_trigger = bool(entry_trigger_ma) and bool(entry_trigger_don)
    elif enter_logic == "OR":
        entry_trigger = bool(entry_trigger_ma) or bool(entry_trigger_don)
    elif enter_logic == "MA":
        entry_trigger = bool(entry_trigger_ma)
    else:
        entry_trigger = bool(entry_trigger_don)

    if exit_logic == "OR":
        exit_trigger = bool(exit_trigger_ma) or bool(exit_trigger_don)
    elif exit_logic == "MA":
        exit_trigger = bool(exit_trigger_ma)
    else:
        exit_trigger = bool(exit_trigger_don)

    if exit_trigger:
        action = "SELL"
        reason = (
            f"{action_label}: exit trigger met. "
            f"{exit_detail}"
        )
    elif entry_trigger:
        action = "BUY"
        reason = (
            f"{action_label}: entry trigger met. "
            f"{entry_detail}"
        )
    else:
        action = "HOLD"
        reason = (
            f"{action_label}: entry {entry_status}, exit {exit_status}. "
            f"Entry: {entry_detail} "
            f"Exit: {exit_detail}"
        )
    return {"action": action, "reason": reason}


def last_trading_day(signal: pd.Series) -> pd.Timestamp | None:
    if signal.empty:
        return None
    return pd.Timestamp(signal.index[-1]).tz_localize(None)


def get_daily_recommendations(
    start: datetime | None = None,
    end: datetime | None = None,
    use_btc_calendar: bool = False,
    live_price: float | None = None,
) -> dict:
    results = run_daily_configs(start, end, use_btc_calendar=use_btc_calendar)
    for payload in results.values():
        payload["decision"] = describe_signal_reason(
            payload["price"],
            payload["signal"],
            payload["sma_entry"],
            payload["sma_exit"],
            payload["don"],
            payload["config"],
        )
        payload["live_price"] = _latest_btc_price(payload["btc_close"], live_price)
    return results


def main() -> None:
    start = datetime(2013, 1, 1)
    end = datetime.today()
    live_price = None
    if "--live-price" in sys.argv:
        idx = sys.argv.index("--live-price")
        if idx + 1 >= len(sys.argv):
            raise ValueError("--live-price requires a numeric value")
        live_price = float(sys.argv[idx + 1])

    results = get_daily_recommendations(
        start, end, use_btc_calendar=True, live_price=live_price
    )

    def _fmt(value) -> str:
        if isinstance(value, (tuple, list)):
            return f"({value[1]}+{value[0]})/2"
        return "N/A" if value is None else str(value)

    print("Daily Signal Summary")
    print("-" * 72)
    for label, payload in results.items():
        config_name = "Primary (MA+Donchian)" if label == "primary" else "Alt (Donchian+MA)"
        decision = payload["decision"]
        signal_day = last_trading_day(payload["signal"])
        print(f"{config_name} -> {decision['action']}")
        print(f"Why: {decision['reason']}")
        if signal_day is not None:
            print(f"As of: {signal_day:%Y-%m-%d %H:%M} (BTC day)")
        print(f"BTC: {payload['live_price']:.2f}")
        cfg = payload["config"]
        print(
            "Config: "
            f"entry={_fmt(cfg['entry_len'])}, exit={_fmt(cfg['exit_len'])}, "
            f"ma_buf=({_fmt(cfg['ma_entry_buffer'])},{_fmt(cfg['ma_exit_buffer'])}), "
            f"don_buf=({_fmt(cfg['donchian_entry_buffer'])},{_fmt(cfg['donchian_exit_buffer'])}), "
            f"high/low=({cfg['high_len']},{cfg['low_len']}), "
            f"logic=({cfg['enter_logic']}/{cfg['exit_logic']}), gate={GATE}, "
            f"k=({K_CONFIRM_ENTRY},{K_CONFIRM_EXIT})"
        )
        print("-" * 72)


if __name__ == "__main__":
    main()

from __future__ import annotations

from datetime import datetime

from CRYP.backtest import run_backtest
from CRYP.calendar import get_trading_days
from CRYP.data import build_btc_proxy, fetch_close
from CRYP.signals import SignalConfig, apply_gate, donchian_signal, sma_signal
from CRYP.validate_proxy import validate_proxy


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

    res = run_backtest(
        returns,
        gated,
        cost_bps=10.0,
        slippage_bps=0.0,
        vol_target=0.2,
        vol_lookback=20,
    )
    print("SMA metrics:", res.metrics)

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
    print("Donchian metrics:", res_d.metrics)

    ibit_start = datetime(2024, 1, 1)
    ibit_close = fetch_close("IBIT", ibit_start, end)
    proxy_2024 = proxy.loc[ibit_start:, "ret_net"]
    val = validate_proxy(proxy_2024, ibit_close)
    print("Proxy validation:", val)


if __name__ == "__main__":
    main()

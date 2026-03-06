from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ProxyValidation:
    correlation: float
    tracking_error: float
    mean_diff: float
    diff_mean_daily: float
    diff_std_daily: float
    tracking_error_annualized: bool


def validate_proxy(
    proxy_returns: pd.Series,
    ibit_close: pd.Series,
) -> ProxyValidation:
    ibit = ibit_close.pct_change().fillna(0.0)
    proxy, ibit = proxy_returns.align(ibit, join="inner")
    diff = proxy - ibit
    corr = float(proxy.corr(ibit))
    diff_mean_daily = float(diff.mean())
    diff_std_daily = float(diff.std())
    tracking_error = float(diff_std_daily * np.sqrt(252.0))
    mean_diff = float(diff_mean_daily * 252.0)
    return ProxyValidation(
        correlation=corr,
        tracking_error=tracking_error,
        mean_diff=mean_diff,
        diff_mean_daily=diff_mean_daily,
        diff_std_daily=diff_std_daily,
        tracking_error_annualized=True,
    )

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import yfinance as yf

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


UNIVERSE = ["XLK", "XLV", "XLF", "XLI", "XLY", "XLE", "XAR"]
BENCHMARK = "SPY"
RISK_OFF = "IEF"


@dataclass(frozen=True)
class RotationResult:
    daily: pd.DataFrame
    decisions: pd.DataFrame
    trades: pd.DataFrame
    metrics: dict[str, float]


def _download_prices(
    tickers: list[str],
    start: str,
    end: str | None = None,
    auto_adjust: bool = True,
) -> pd.DataFrame:
    data = yf.download(
        tickers=tickers,
        start=start,
        end=end,
        auto_adjust=auto_adjust,
        progress=False,
        threads=True,
    )

    if isinstance(data.columns, pd.MultiIndex):
        if "Close" in data.columns.get_level_values(0):
            px = data["Close"]
        elif "Adj Close" in data.columns.get_level_values(0):
            px = data["Adj Close"]
        else:
            swapped = data.swaplevel(0, 1, axis=1).sort_index(axis=1)
            if "Close" in swapped.columns.get_level_values(1):
                px = swapped.xs("Close", level=1, axis=1)
            elif "Adj Close" in swapped.columns.get_level_values(1):
                px = swapped.xs("Adj Close", level=1, axis=1)
            else:
                raise ValueError("Could not find Close/Adj Close in downloaded data.")
    else:
        if "Close" in data.columns:
            px = pd.DataFrame({tickers[0]: data["Close"]})
        elif "Adj Close" in data.columns:
            px = pd.DataFrame({tickers[0]: data["Adj Close"]})
        else:
            raise ValueError("Could not find Close/Adj Close in single-ticker data.")

    px = px.dropna(how="all").ffill().dropna(how="all")
    px.index = pd.to_datetime(px.index)
    return px.reindex(columns=[t for t in tickers if t in px.columns])


def _monthly_last_trading_days(index: pd.DatetimeIndex) -> set[pd.Timestamp]:
    s = pd.Series(index, index=index)
    target = s.resample("ME").last().dropna()
    return set(pd.DatetimeIndex(target.values))


def _metrics_from_returns(ret: pd.Series, ann: int = 252) -> dict[str, float]:
    ret = ret.fillna(0.0)
    eq = (1.0 + ret).cumprod()
    if len(ret) < 2:
        return {
            "cagr": np.nan,
            "sharpe": np.nan,
            "maxdd": np.nan,
            "total_return": np.nan,
            "vol": np.nan,
            "time_in_market": np.nan,
        }

    cagr = float(eq.iloc[-1] ** (ann / len(ret)) - 1.0)
    vol = float(ret.std(ddof=1) * np.sqrt(ann))
    sharpe = (
        float(np.sqrt(ann) * ret.mean() / ret.std(ddof=1))
        if ret.std(ddof=1) > 0
        else np.nan
    )
    peak = eq.cummax()
    maxdd = float((eq / peak - 1.0).min())
    total_return = float(eq.iloc[-1] - 1.0)
    return {
        "cagr": cagr,
        "sharpe": sharpe,
        "maxdd": maxdd,
        "total_return": total_return,
        "vol": vol,
        "time_in_market": np.nan,
    }


def _momentum_scores(
    prices: pd.DataFrame,
    date: pd.Timestamp,
    lookback_days: int,
) -> pd.Series:
    px = prices.loc[:date].dropna(how="all")
    if len(px) <= lookback_days:
        return pd.Series(dtype=float)
    start = px.iloc[-(lookback_days + 1)]
    end = px.iloc[-1]
    return ((end / start) - 1.0).dropna().sort_values(ascending=False)


def _pick_with_rank_gap(scores: pd.Series, current_asset: str | None, rank_gap: int) -> str:
    leader = str(scores.index[0])
    if current_asset is None or current_asset not in scores.index:
        return leader
    current_rank = int(scores.index.get_loc(current_asset)) + 1
    return current_asset if current_rank <= 1 + int(rank_gap) else leader


def _time_in_market_summary(daily: pd.DataFrame) -> dict[str, float]:
    n = len(daily)
    if n == 0:
        return {
            "risk_on_%": np.nan,
            "bonds_%": np.nan,
            "cash_%": np.nan,
        }
    held = daily["held_asset"]
    risk_on = float(held.isin(UNIVERSE).mean() * 100.0)
    bonds = float((held == RISK_OFF).mean() * 100.0)
    cash = float(held.isna().mean() * 100.0)
    return {
        "risk_on_%": risk_on,
        "bonds_%": bonds,
        "cash_%": cash,
    }


def run_rotation(
    prices: pd.DataFrame,
    lookback_days: int,
    rank_gap: int,
    cost_bps: float,
    slippage_bps: float,
    trade_delay: int = 1,
    enable_defensive: bool = True,
) -> RotationResult:
    prices = prices.copy().dropna(how="all")
    risk_on_universe = [t for t in UNIVERSE if t in prices.columns]
    if RISK_OFF not in prices.columns:
        raise ValueError(f"{RISK_OFF} must be included in prices for risk-off behavior.")
    if len(risk_on_universe) == 0:
        raise ValueError("No risk-on assets available in prices.")
    returns = prices.pct_change().fillna(0.0)

    dates = prices.index
    rebal_days = _monthly_last_trading_days(dates)
    total_cost_bps = float(cost_bps) + float(slippage_bps)

    current_asset: str | None = None
    pending_asset: str | None = None
    pending_exec_date: pd.Timestamp | None = None

    equity = 1.0
    daily_rows: list[dict] = []
    decision_rows: list[dict] = []
    trade_rows: list[dict] = []

    for i, date in enumerate(dates):
        date = pd.Timestamp(date)
        turnover = 0.0
        cost = 0.0
        trade_event = ""

        if pending_exec_date is not None and date == pending_exec_date:
            prev_asset = current_asset
            next_asset = pending_asset

            if prev_asset == next_asset:
                turnover = 0.0
            elif prev_asset is None or next_asset is None:
                turnover = 1.0
            else:
                turnover = 2.0

            cost = turnover * (total_cost_bps / 10000.0)
            current_asset = next_asset
            trade_event = "EXECUTE"

            trade_rows.append(
                {
                    "date": date,
                    "event": "EXECUTE",
                    "asset_after": current_asset,
                    "turnover": turnover,
                    "cost": cost,
                }
            )

            pending_asset = None
            pending_exec_date = None

        asset_ret = 0.0 if current_asset is None else float(returns.at[date, current_asset])
        daily_ret = asset_ret - cost
        equity *= 1.0 + daily_ret

        daily_rows.append(
            {
                "date": date,
                "ret": daily_ret,
                "asset_ret": asset_ret,
                "equity": equity,
                "held_asset": current_asset,
                "turnover": turnover,
                "cost": cost,
                "trade_event": trade_event,
            }
        )

        if pending_exec_date is not None or date not in rebal_days:
            continue

        scores = _momentum_scores(prices[risk_on_universe], date, lookback_days=lookback_days)
        if scores.empty:
            continue

        leader = str(scores.index[0])
        leader_mom = float(scores.iloc[0])

        if enable_defensive and leader_mom <= 0.0:
            chosen = RISK_OFF
            kept_non_leader = False
            incumbent_rank = None
            chosen_mom = np.nan
            risk_mode = "RISK_OFF"
        else:
            chosen = _pick_with_rank_gap(
                scores,
                current_asset=current_asset if current_asset in UNIVERSE else None,
                rank_gap=rank_gap,
            )
            kept_non_leader = bool(chosen != leader)
            incumbent_rank = (
                int(scores.index.get_loc(current_asset)) + 1
                if current_asset in scores.index
                else None
            )
            chosen_mom = float(scores.loc[chosen])
            risk_mode = "RISK_ON"

        next_idx = i + int(trade_delay)
        if chosen != current_asset and next_idx < len(dates):
            pending_asset = chosen
            pending_exec_date = pd.Timestamp(dates[next_idx])
            trade_rows.append(
                {
                    "date": date,
                    "event": "SCHEDULE",
                    "asset_after": chosen,
                    "turnover": np.nan,
                    "cost": np.nan,
                }
            )

        decision_rows.append(
            {
                "date": date,
                "leader": leader,
                "chosen": chosen,
                "incumbent": current_asset,
                "incumbent_rank": incumbent_rank,
                "leader_mom": leader_mom,
                "chosen_mom": chosen_mom,
                "kept_non_leader": kept_non_leader,
                "leader_minus_chosen_mom": (leader_mom - chosen_mom) if pd.notna(chosen_mom) else np.nan,
                "rank_gap": int(rank_gap),
                "risk_mode": risk_mode,
            }
        )

    daily = pd.DataFrame(daily_rows).set_index("date")
    decisions = pd.DataFrame(decision_rows)
    trades = pd.DataFrame(trade_rows)
    metrics = _metrics_from_returns(daily["ret"])
    metrics.update(_time_in_market_summary(daily))
    metrics["time_in_market"] = metrics["risk_on_%"] / 100.0
    return RotationResult(daily=daily, decisions=decisions, trades=trades, metrics=metrics)


def _hold_stats(daily: pd.DataFrame, prices: pd.DataFrame) -> pd.DataFrame:
    total_days = len(daily)
    invested_days = int(daily["held_asset"].notna().sum())
    rets = prices.pct_change().reindex(daily.index).fillna(0.0)

    rows = []
    for asset in prices.columns:
        mask = daily["held_asset"] == asset
        hold_days = int(mask.sum())
        if hold_days == 0:
            avg_asset_ret = np.nan
            ann_asset_ret = np.nan
            avg_strat_ret = np.nan
        else:
            avg_asset_ret = float(rets.loc[mask, asset].mean())
            ann_asset_ret = float((1.0 + avg_asset_ret) ** 252 - 1.0)
            avg_strat_ret = float(daily.loc[mask, "ret"].mean())

        rows.append(
            {
                "asset": asset,
                "hold_days": hold_days,
                "hold_%_all_days": (hold_days / total_days) * 100.0 if total_days else np.nan,
                "hold_%_invested_days": (hold_days / invested_days) * 100.0 if invested_days else np.nan,
                "avg_daily_asset_ret_%": avg_asset_ret * 100.0 if pd.notna(avg_asset_ret) else np.nan,
                "avg_daily_strat_ret_%": avg_strat_ret * 100.0 if pd.notna(avg_strat_ret) else np.nan,
                "ann_asset_ret_when_held_%": ann_asset_ret * 100.0 if pd.notna(ann_asset_ret) else np.nan,
            }
        )

    return pd.DataFrame(rows).sort_values("hold_%_all_days", ascending=False).reset_index(drop=True)


def _leave_one_out(
    prices: pd.DataFrame,
    lookback_days: int,
    rank_gap: int,
    cost_bps: float,
    slippage_bps: float,
) -> pd.DataFrame:
    base = run_rotation(
        prices=prices,
        lookback_days=lookback_days,
        rank_gap=rank_gap,
        cost_bps=cost_bps,
        slippage_bps=slippage_bps,
        trade_delay=1,
    )

    rows = []
    for asset in UNIVERSE:
        reduced_cols = [c for c in prices.columns if c != asset]
        test = run_rotation(
            prices=prices[reduced_cols],
            lookback_days=lookback_days,
            rank_gap=rank_gap,
            cost_bps=cost_bps,
            slippage_bps=slippage_bps,
            trade_delay=1,
        )
        rows.append(
            {
                "removed_asset": asset,
                "cagr_without_%": test.metrics["cagr"] * 100.0,
                "delta_cagr_pp": (test.metrics["cagr"] - base.metrics["cagr"]) * 100.0,
                "delta_sharpe": test.metrics["sharpe"] - base.metrics["sharpe"],
                "delta_maxdd_pp": (test.metrics["maxdd"] - base.metrics["maxdd"]) * 100.0,
                "delta_risk_on_pp": test.metrics["risk_on_%"] - base.metrics["risk_on_%"],
                "delta_bonds_pp": test.metrics["bonds_%"] - base.metrics["bonds_%"],
                "hurt_if_removed_positive": (test.metrics["cagr"] > base.metrics["cagr"]),
            }
        )

    return pd.DataFrame(rows).sort_values("delta_cagr_pp", ascending=False).reset_index(drop=True)


def _benchmark_report(start: str, end: str | None, idx: pd.DatetimeIndex) -> tuple[dict, dict]:
    bench_total = _download_prices([BENCHMARK], start=start, end=end, auto_adjust=True)[BENCHMARK]
    bench_price = _download_prices([BENCHMARK], start=start, end=end, auto_adjust=False)[BENCHMARK]

    bench_total = bench_total.loc[idx.min() : idx.max()]
    bench_price = bench_price.loc[idx.min() : idx.max()]

    return (
        _metrics_from_returns(bench_price.pct_change().fillna(0.0)),
        _metrics_from_returns(bench_total.pct_change().fillna(0.0)),
    )


def _rank_gap_comparison(base: RotationResult, alt: RotationResult) -> dict[str, float]:
    idx = base.daily.index.intersection(alt.daily.index)
    base_ret = base.daily.loc[idx, "ret"]
    alt_ret = alt.daily.loc[idx, "ret"]
    active = alt_ret - base_ret

    rel = (1.0 + alt_ret).cumprod() / (1.0 + base_ret).cumprod()
    years = len(idx) / 252.0 if len(idx) else np.nan
    rel_cagr = float(rel.iloc[-1] ** (1.0 / years) - 1.0) if years and years > 0 else np.nan

    return {
        "active_cagr_vs_gap0_%": rel_cagr * 100.0,
        "avg_daily_active_bps": float(active.mean() * 10000.0),
        "tracking_error_%": float(active.std(ddof=1) * np.sqrt(252.0) * 100.0),
        "underperform_day_%": float((active < 0).mean() * 100.0),
        "delta_risk_on_pp": alt.metrics["risk_on_%"] - base.metrics["risk_on_%"],
        "delta_bonds_pp": alt.metrics["bonds_%"] - base.metrics["bonds_%"],
    }


def _regime_labels_from_spy(spy_total_return_price: pd.Series) -> pd.Series:
    px = spy_total_return_price.dropna().copy()
    sma200 = px.rolling(200).mean()
    slope = sma200.diff(20)
    regime = pd.Series("chop", index=px.index, dtype=object)
    regime[(px > sma200) & (slope > 0)] = "bull"
    regime[(px < sma200) & (slope < 0)] = "bear"
    return regime


def _regime_performance(daily: pd.DataFrame, regime: pd.Series) -> pd.DataFrame:
    aligned = daily.join(regime.rename("regime"), how="left")
    aligned["regime"] = aligned["regime"].fillna("chop")
    rows = []
    for label in ["bull", "bear", "chop"]:
        sub = aligned.loc[aligned["regime"] == label, "ret"]
        if sub.empty:
            rows.append({"regime": label, "days": 0, "cagr_%": np.nan, "avg_daily_bps": np.nan, "sharpe": np.nan})
            continue
        m = _metrics_from_returns(sub)
        rows.append(
            {
                "regime": label,
                "days": int(len(sub)),
                "cagr_%": m["cagr"] * 100.0,
                "avg_daily_bps": float(sub.mean() * 10000.0),
                "sharpe": m["sharpe"],
            }
        )
    return pd.DataFrame(rows)


def _decision_regret(decisions: pd.DataFrame, prices: pd.DataFrame, trade_delay: int = 1) -> pd.DataFrame:
    if decisions.empty:
        return pd.DataFrame(columns=["date", "chosen", "leader", "chosen_ret", "leader_ret", "leader_minus_chosen_ret"])

    idx = prices.index
    rows = []
    for i, row in decisions.reset_index(drop=True).iterrows():
        d = pd.Timestamp(row["date"])
        if d not in idx:
            continue
        d_pos = int(idx.get_loc(d))
        start_pos = d_pos + trade_delay
        if start_pos >= len(idx):
            continue
        start_date = idx[start_pos]

        if i + 1 < len(decisions):
            next_d = pd.Timestamp(decisions.iloc[i + 1]["date"])
            if next_d in idx:
                end_pos = int(idx.get_loc(next_d)) + trade_delay
            else:
                end_pos = len(idx) - 1
        else:
            end_pos = len(idx) - 1
        end_pos = min(end_pos, len(idx) - 1)
        if end_pos <= start_pos:
            continue

        end_date = idx[end_pos]
        chosen = str(row["chosen"])
        leader = str(row["leader"])
        if chosen not in prices.columns or leader not in prices.columns:
            continue

        chosen_ret = float(prices.at[end_date, chosen] / prices.at[start_date, chosen] - 1.0)
        leader_ret = float(prices.at[end_date, leader] / prices.at[start_date, leader] - 1.0)
        rows.append(
            {
                "date": d,
                "chosen": chosen,
                "leader": leader,
                "chosen_ret": chosen_ret,
                "leader_ret": leader_ret,
                "leader_minus_chosen_ret": leader_ret - chosen_ret,
                "risk_mode": row["risk_mode"],
                "kept_non_leader": bool(row["kept_non_leader"]),
            }
        )

    return pd.DataFrame(rows)


def _fmt_metrics(m: dict[str, float]) -> str:
    return (
        f"CAGR {m['cagr']:.2%} | Sharpe {m['sharpe']:.2f} | MaxDD {m['maxdd']:.2%} | "
        f"Total {m['total_return']:.2%} | RiskOn {m['risk_on_%']:.1f}% | Bonds {m['bonds_%']:.1f}% | Cash {m['cash_%']:.1f}%"
    )


def _information_ratio(strategy_ret: pd.Series, benchmark_ret: pd.Series, ann: int = 252) -> float:
    idx = strategy_ret.index.intersection(benchmark_ret.index)
    if len(idx) < 2:
        return np.nan
    active = strategy_ret.loc[idx].fillna(0.0) - benchmark_ret.loc[idx].fillna(0.0)
    te = float(active.std(ddof=1))
    if te <= 0.0:
        return np.nan
    return float(np.sqrt(ann) * active.mean() / te)


def main() -> None:
    start_date = "2012-01-01"
    end_date = None
    lookback_days = 126
    cost_bps = 2.0
    slippage_bps = 1.0

    tickers = UNIVERSE + [RISK_OFF]
    prices = _download_prices(tickers, start=start_date, end=end_date, auto_adjust=True)
    spy_total = _download_prices([BENCHMARK], start=start_date, end=end_date, auto_adjust=True)[BENCHMARK]

    run_gap0 = run_rotation(
        prices=prices,
        lookback_days=lookback_days,
        rank_gap=0,
        cost_bps=cost_bps,
        slippage_bps=slippage_bps,
        trade_delay=1,
    )
    run_gap1 = run_rotation(
        prices=prices,
        lookback_days=lookback_days,
        rank_gap=1,
        cost_bps=cost_bps,
        slippage_bps=slippage_bps,
        trade_delay=1,
    )

    prices_no_xly = prices[[c for c in prices.columns if c != "XLY"]]
    run_gap1_no_xly_def_on = run_rotation(
        prices=prices_no_xly,
        lookback_days=lookback_days,
        rank_gap=1,
        cost_bps=cost_bps,
        slippage_bps=slippage_bps,
        trade_delay=1,
        enable_defensive=True,
    )
    run_gap1_no_xly_def_off = run_rotation(
        prices=prices_no_xly,
        lookback_days=lookback_days,
        rank_gap=1,
        cost_bps=cost_bps,
        slippage_bps=slippage_bps,
        trade_delay=1,
        enable_defensive=False,
    )

    hold_gap0 = _hold_stats(run_gap0.daily, prices)
    hold_gap1 = _hold_stats(run_gap1.daily, prices)

    loo_gap0 = _leave_one_out(prices, lookback_days, rank_gap=0, cost_bps=cost_bps, slippage_bps=slippage_bps)
    loo_gap1 = _leave_one_out(prices, lookback_days, rank_gap=1, cost_bps=cost_bps, slippage_bps=slippage_bps)

    comp = _rank_gap_comparison(run_gap0, run_gap1)

    bench_price, bench_total = _benchmark_report(start_date, end_date, run_gap0.daily.index)
    bench_total_ret = (
        _download_prices([BENCHMARK], start=start_date, end=end_date, auto_adjust=True)[BENCHMARK]
        .pct_change()
        .fillna(0.0)
    )

    kept_non_leader_days = int(run_gap1.decisions["kept_non_leader"].sum()) if not run_gap1.decisions.empty else 0
    risk_off_switches0 = int((run_gap0.decisions["risk_mode"] == "RISK_OFF").sum()) if not run_gap0.decisions.empty else 0
    risk_off_switches1 = int((run_gap1.decisions["risk_mode"] == "RISK_OFF").sum()) if not run_gap1.decisions.empty else 0

    regime = _regime_labels_from_spy(spy_total).reindex(run_gap0.daily.index).ffill().bfill()
    regime_gap0 = _regime_performance(run_gap0.daily, regime)
    regime_gap1 = _regime_performance(run_gap1.daily, regime)

    regret_gap0 = _decision_regret(run_gap0.decisions, prices)
    regret_gap1 = _decision_regret(run_gap1.decisions, prices)

    avg_regret_all_gap0 = float(regret_gap0["leader_minus_chosen_ret"].mean()) if not regret_gap0.empty else np.nan
    avg_regret_all_gap1 = float(regret_gap1["leader_minus_chosen_ret"].mean()) if not regret_gap1.empty else np.nan
    avg_regret_non_leader_gap1 = (
        float(regret_gap1.loc[regret_gap1["kept_non_leader"], "leader_minus_chosen_ret"].mean())
        if not regret_gap1.empty and regret_gap1["kept_non_leader"].any()
        else np.nan
    )

    ir_gap1_no_xly_def_on_vs_spy = _information_ratio(
        run_gap1_no_xly_def_on.daily["ret"], bench_total_ret
    )
    ir_gap1_no_xly_def_off_vs_spy = _information_ratio(
        run_gap1_no_xly_def_off.daily["ret"], bench_total_ret
    )
    ir_def_off_vs_def_on = _information_ratio(
        run_gap1_no_xly_def_off.daily["ret"], run_gap1_no_xly_def_on.daily["ret"]
    )

    pd.set_option("display.max_columns", 30)
    pd.set_option("display.width", 180)

    print("Sector Rotation Diagnostics")
    print(f"Universe (risk-on): {', '.join(UNIVERSE)}")
    print(f"Risk-off asset: {RISK_OFF} (dividend-adjusted total return series)")
    print("Risk-off rule: if leader momentum <= 0 at rebalance decision, switch to IEF")
    print(f"Period: {run_gap0.daily.index.min().date()} -> {run_gap0.daily.index.max().date()}")
    print(f"Lookback days: {lookback_days} | Costs: {cost_bps + slippage_bps:.1f} bps turnover")

    print("\nPerformance by rank_gap")
    print(f"  rank_gap=0: {_fmt_metrics(run_gap0.metrics)}")
    print(f"  rank_gap=1: {_fmt_metrics(run_gap1.metrics)}")

    print("\nSPY benchmark")
    print(f"  Price-only: CAGR {bench_price['cagr']:.2%} | Sharpe {bench_price['sharpe']:.2f} | MaxDD {bench_price['maxdd']:.2%} | Total {bench_price['total_return']:.2%}")
    print(f"  Total-return: CAGR {bench_total['cagr']:.2%} | Sharpe {bench_total['sharpe']:.2f} | MaxDD {bench_total['maxdd']:.2%} | Total {bench_total['total_return']:.2%}")

    print("\nRank-gap (1 vs 0) relative diagnostics")
    print(
        f"  Active CAGR: {comp['active_cagr_vs_gap0_%']:.2f}% | Avg daily active: {comp['avg_daily_active_bps']:.3f} bps | "
        f"Tracking error: {comp['tracking_error_%']:.2f}% | Underperform days: {comp['underperform_day_%']:.1f}%"
    )
    print(
        f"  Delta time-in-market (1-0): RiskOn {comp['delta_risk_on_pp']:.1f} pp, Bonds {comp['delta_bonds_pp']:.1f} pp | "
        f"kept non-leader decisions: {kept_non_leader_days}"
    )

    print("\nRisk-off trigger counts")
    print(f"  rank_gap=0: {risk_off_switches0} rebalance decisions went to IEF")
    print(f"  rank_gap=1: {risk_off_switches1} rebalance decisions went to IEF")

    print("\nLeader regret diagnostics")
    print(
        f"  Avg leader-minus-chosen period return (gap0): {avg_regret_all_gap0:.3%} | "
        f"(gap1): {avg_regret_all_gap1:.3%} | non-leader only (gap1): {avg_regret_non_leader_gap1:.3%}"
    )

    print("\nRegime performance (SPY total-return regime labels)")
    print("  rank_gap=0")
    print(regime_gap0.to_string(index=False, float_format=lambda x: f"{x:,.3f}"))
    print("  rank_gap=1")
    print(regime_gap1.to_string(index=False, float_format=lambda x: f"{x:,.3f}"))

    print("\nHold stats (rank_gap=0)")
    print(hold_gap0.to_string(index=False, float_format=lambda x: f"{x:,.3f}"))

    print("\nHold stats (rank_gap=1)")
    print(hold_gap1.to_string(index=False, float_format=lambda x: f"{x:,.3f}"))

    print("\nLeave-one-out impact (rank_gap=0): sorted by delta_cagr_pp descending")
    print(loo_gap0.to_string(index=False, float_format=lambda x: f"{x:,.3f}"))

    print("\nLeave-one-out impact (rank_gap=1): sorted by delta_cagr_pp descending")
    print(loo_gap1.to_string(index=False, float_format=lambda x: f"{x:,.3f}"))

    print("\nRequested scenario: rank_gap=1, XLY removed")
    print("  Defensive ON")
    print(f"    {_fmt_metrics(run_gap1_no_xly_def_on.metrics)}")
    print(f"    IR vs SPY total-return: {ir_gap1_no_xly_def_on_vs_spy:.3f}")
    print("  Defensive OFF")
    print(f"    {_fmt_metrics(run_gap1_no_xly_def_off.metrics)}")
    print(f"    IR vs SPY total-return: {ir_gap1_no_xly_def_off_vs_spy:.3f}")

    cagr_delta = (run_gap1_no_xly_def_off.metrics["cagr"] - run_gap1_no_xly_def_on.metrics["cagr"]) * 100.0
    maxdd_delta = (run_gap1_no_xly_def_off.metrics["maxdd"] - run_gap1_no_xly_def_on.metrics["maxdd"]) * 100.0
    print("  Defensive OFF minus ON")
    print(
        f"    Delta CAGR: {cagr_delta:.2f} pp | Delta MaxDD: {maxdd_delta:.2f} pp | "
        f"IR (OFF over ON): {ir_def_off_vs_def_on:.3f}"
    )


if __name__ == "__main__":
    main()

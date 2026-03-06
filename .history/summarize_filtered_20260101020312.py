from pathlib import Path

import pandas as pd

from momentum_program.config import AppConfig
from momentum_program.analytics.constants import (
    SCORE_MODE_12M_MINUS_1M,
    SCORE_MODE_RW_3_6_9_12,
)
from momentum_program.universe.bucket_csv_provider import BucketedCsvUniverseProvider
from momentum_program.universe.csv_provider import CsvUniverseProvider
from momentum_program.universe.tradingview_provider import TradingViewUniverseProvider
from momentum_program.backtest.engine import backtest_momentum
from momentum_program.backtest.metrics import compute_metrics, compute_turnover


def main() -> None:
    cfg = AppConfig()
    bucket_folder = Path("CSVs")
    bucket_csvs = list(bucket_folder.glob("*.csv")) if bucket_folder.exists() else []
    csv_path = Path("etfs.csv")

    if bucket_csvs:
        universe = BucketedCsvUniverseProvider(bucket_folder)
    elif csv_path.exists():
        universe = CsvUniverseProvider(csv_path)
    else:
        universe = TradingViewUniverseProvider(chunk=50)

    tickers = universe.get_tickers()
    bucket_map = universe.get_bucket_map()

    thresholds = [None]
    rank_gaps = [0, 2]
    vol_flags = [False]
    score_modes = [SCORE_MODE_RW_3_6_9_12, SCORE_MODE_12M_MINUS_1M]

    # Same windows as backtest: two anchor ranges plus rolling 5-year windows
    ranges: list[tuple[str, str]] = [
        ("2012-01-01", "2022-12-31"),
        ("2015-01-01", "2025-12-31"),
    ]
    for start in range(2012, 2022):
        end = min(start + 4, 2025)
        ranges.append((f"{start}-01-01", f"{end}-12-31"))

    rows: list[dict[str, object]] = []
    for start_date, end_date in ranges:
        for vol_flag in vol_flags:
            for rank_gap in rank_gaps:
                for score_mode in score_modes:
                    data = backtest_momentum(
                        tickers=tickers,
                        bucket_map=bucket_map,
                        start_date=start_date,
                        end_date=end_date,
                        top_n_per_bucket=cfg.strategy.top_n_per_bucket
                        or cfg.strategy.top_n,
                        lookback_long=12,
                        lookback_short=1,
                        vol_adjusted=vol_flag,
                        vol_lookback=6,
                        market_filter=False,
                        market_ticker="SPY",
                        defensive_bucket="Bonds",
                        market_threshold=0.0,
                        rank_gap_threshold=rank_gap,
                        score_mode=score_mode,
                    )
                    if data["overall_returns"].empty:
                        continue
                    metrics = compute_metrics(data["overall_returns"]["return"])
                    turnover = compute_turnover(data["overall_positions"])
                    rows.append(
                        {
                            "window": f"{start_date}->{end_date}",
                            "score_mode": score_mode,
                            "rank_gap": rank_gap,
                            "cagr": metrics["cagr"],
                            "sharpe": metrics["sharpe"],
                            "max_dd": metrics["max_drawdown"],
                            "turnover": turnover,
                        }
                    )

    if not rows:
        print("No results to display.")
        return

    df = pd.DataFrame(rows)
    window_order = list(dict.fromkeys(r["window"] for r in rows))
    df["window"] = pd.Categorical(df["window"], categories=window_order, ordered=True)
    df = df.sort_values(["window", "score_mode", "rank_gap"])

    fmt = {
        "cagr": lambda x: f"{x * 100:.2f}%",
        "sharpe": lambda x: f"{x:.2f}",
        "max_dd": lambda x: f"{x * 100:.2f}%",
        "turnover": lambda x: f"{x * 100:.2f}%",
    }

    print("\nPer-fold (RW vs 12m-1m; gap 0 vs 2)")
    print(df.to_string(index=False, formatters=fmt))

    summary = (
        df.groupby(["score_mode", "rank_gap"])
        .agg({"cagr": "mean", "sharpe": "mean", "max_dd": "mean", "turnover": "mean"})
        .reset_index()
    )
    print("\nAcross all windows (mean)")
    print(summary.to_string(index=False, formatters=fmt))


if __name__ == "__main__":
    main()

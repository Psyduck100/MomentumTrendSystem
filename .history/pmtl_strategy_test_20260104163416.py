"""PMTL Strategy: GLD with 50/100/200-day trading day MA filters vs 12M return filter.

Compare six strategies:
1. GLD 50-day TRADING MA filter: hold GLD if above 50 trading-day MA, else IEF
2. GLD 100-day TRADING MA filter: hold GLD if above 100 trading-day MA, else IEF
3. GLD 200-day TRADING MA filter: hold GLD if above 200 trading-day MA, else IEF
4. GLD 6M return filter: hold GLD if 6M > 0, else IEF
5. GLD 12M return filter: hold GLD if 12M > 0, else IEF
6. GLD benchmark: hold GLD always

Note: 50 trading days ≈ 72 cal days, 100 trading days ≈ 145 cal days, 200 trading days ≈ 290 cal days
"""

from __future__ import annotations
import pandas as pd
import yfinance as yf
from pathlib import Path
from enum import Enum

from momentum_program.backtest.metrics import compute_metrics

TICKERS = ["GLD", "IEF"]
START_DATE = "2005-01-01"
END_DATE = pd.Timestamp.today().strftime("%Y-%m-%d")

# Trading day conversions: 252 trading days per year / 365 calendar days
TRADING_TO_CALENDAR = 365 / 252

class FilterType(Enum):
    """Enum for filter types."""
    MA_50 = ("50TMA", int(50 * TRADING_TO_CALENDAR))
    MA_100 = ("100TMA", int(100 * TRADING_TO_CALENDAR))
    MA_200 = ("200TMA", int(200 * TRADING_TO_CALENDAR))

BACKTEST_CACHE = Path("backtest_cache")
BACKTEST_CACHE.mkdir(exist_ok=True)

def download_prices(tickers: list[str], end_date: str = END_DATE, start_date: str = START_DATE) -> pd.DataFrame:
    """Download daily prices."""
    print(f"Downloading {len(tickers)} tickers from {start_date} to {end_date}...")
    data = yf.download(tickers, start=start_date, end=end_date, progress=False)

    if isinstance(data.columns, pd.MultiIndex):
        data = data.swaplevel(0, 1, axis=1).sort_index(axis=1)
        prices = pd.DataFrame()
        for ticker in tickers:
            if ticker in data.columns.get_level_values(0):
                sub = data[ticker]
                if "Adj Close" in sub.columns:
                    prices[ticker] = sub["Adj Close"]
                elif "Close" in sub.columns:
                    prices[ticker] = sub["Close"]
    else:
        prices = pd.DataFrame({tickers[0]: data["Adj Close"]})

    return prices

def backtest_moving_average(prices: pd.DataFrame, filter_type: FilterType) -> dict:
    """Backtest: GLD above moving average -> GLD, else IEF.
    
    Args:
        prices: DataFrame with GLD and IEF prices
        filter_type: FilterType enum specifying which MA to use
    
    Returns:
        Dictionary with 'returns' key containing return series
    """
    filter_name, calendar_window = filter_type.value
    ma_column = f"GLD_{filter_name}"
    prices[ma_column] = prices["GLD"].rolling(window=calendar_window).mean()
    monthly = prices.resample("ME").last()
    
    positions = []
    monthly_rets = []
    
    for i in range(1, len(monthly)):
        gld_price = monthly.iloc[i]["GLD"]
        gld_ma = monthly.iloc[i][ma_column]
        
        position = "GLD" if pd.notna(gld_ma) and gld_price > gld_ma else "IEF"
        positions.append(position)
        
        if i + 1 < len(monthly):
            price_now = monthly.iloc[i][position]
            price_next = monthly.iloc[i + 1][position]
            monthly_ret = (price_next - price_now) / price_now if price_now != 0 else 0
            monthly_rets.append(monthly_ret)
    
    return_series = pd.Series(monthly_rets, index=monthly.index[2:len(monthly)])
    return {"returns": return_series}

def backtest_exponential_moving_average(prices: pd.DataFrame) -> dict:
    """Backtest: GLD above 100-day EMA -> GLD, else IEF.
    
    Note: yfinance returns only trading days, so daily data IS trading-day frequency.
    span=100 on daily data = 100 trading day EMA.
    """
    ema_span = 100  # 100 trading days (since data is daily trading days only)
    ema_column = "GLD_100EMA"
    prices[ema_column] = prices["GLD"].ewm(span=ema_span, adjust=False).mean()
    monthly = prices.resample("ME").last()

    positions = []
    monthly_rets = []

    for i in range(1, len(monthly)):
        gld_price = monthly.iloc[i]["GLD"]
        gld_ema = monthly.iloc[i][ema_column]

        position = "GLD" if pd.notna(gld_ema) and gld_price > gld_ema else "IEF"
        positions.append(position)

        if i + 1 < len(monthly):
            price_now = monthly.iloc[i][position]
            price_next = monthly.iloc[i + 1][position]
            monthly_ret = (price_next - price_now) / price_now if price_now != 0 else 0
            monthly_rets.append(monthly_ret)

    return_series = pd.Series(monthly_rets, index=monthly.index[2:len(monthly)])
    return {"returns": return_series}

def backtest_trailing_return(prices: pd.DataFrame, months: int) -> dict:
    """Backtest: GLD trailing return > 0 -> GLD, else IEF."""
    monthly = prices.resample("ME").last()
    
    positions = []
    monthly_rets = []
    
    for i in range(1, len(monthly)):
        if i >= months:
            gld_return = (monthly.iloc[i]["GLD"] - monthly.iloc[i - months]["GLD"]) / monthly.iloc[i - months]["GLD"]
        else:
            gld_return = 0
        
        position = "GLD" if gld_return > 0 else "IEF"
        positions.append(position)
        
        if i + 1 < len(monthly):
            price_now = monthly.iloc[i][position]
            price_next = monthly.iloc[i + 1][position]
            monthly_ret = (price_next - price_now) / price_now if price_now != 0 else 0
            monthly_rets.append(monthly_ret)
    
    return_series = pd.Series(monthly_rets, index=monthly.index[2:len(monthly)])
    return {"returns": return_series}


def run_four_year_chunks() -> None:
    prices_full = download_prices(TICKERS, end_date="2025-12-31")
    first_year = prices_full.index.min().year
    last_year = prices_full.index.max().year
    start_years = range(first_year, last_year + 1, 4)
    for start_year in start_years:
        end_year = min(start_year + 3, last_year)
        start_date = f"{start_year}-01-01"
        end_date = f"{end_year}-12-31"
        chunk = prices_full.loc[start_date:end_date]
        if len(chunk) < 60:
            continue
        ma50 = backtest_moving_average(chunk.copy(), FilterType.MA_50)["returns"]
        ma100 = backtest_moving_average(chunk.copy(), FilterType.MA_100)["returns"]
        ma200 = backtest_moving_average(chunk.copy(), FilterType.MA_200)["returns"]
        ema100 = backtest_exponential_moving_average(chunk.copy())["returns"]
        ret6 = backtest_trailing_return(chunk, months=6)["returns"]
        ret12 = backtest_trailing_return(chunk, months=12)["returns"]
        gld = backtest_gld_only(chunk)["returns"]

        if not all(len(r) > 0 for r in [ma50, ma100, ma200, ema100, ret6, ret12, gld]):
            continue

        m50 = compute_metrics(ma50)
        m100 = compute_metrics(ma100)
        m200 = compute_metrics(ma200)
        m_ema100 = compute_metrics(ema100)
        m6 = compute_metrics(ret6)
        m12 = compute_metrics(ret12)
        mg = compute_metrics(gld)
        # Table showing absolute metrics and deltas vs GLD for all configs
        rows = []
        configs = [
            ("50TMA", m50),
            ("100TMA", m100),
            ("100EMA", m_ema100),
            ("200TMA", m200),
            ("6M Ret", m6),
            ("12M Ret", m12),
        ]
        for name, metrics in configs:
            rows.append(
                {
                    "name": name,
                    "cagr": metrics["cagr"],
                    "sharpe": metrics["sharpe"],
                    "dd": metrics["max_drawdown"],
                    "dcagr": metrics["cagr"] - mg["cagr"],
                    "dsharpe": metrics["sharpe"] - mg["sharpe"],
                }
            )

        def fmt_pct(x: float) -> str:
            return f"{x:>.2%}" if pd.notna(x) else "   n/a"

        def fmt_signed_pct(x: float) -> str:
            return f"{x:>+.2%}" if pd.notna(x) else "   n/a"

        def fmt_float(x: float) -> str:
            return f"{x:>.3f}" if pd.notna(x) else "   n/a"

        def fmt_signed_float(x: float) -> str:
            return f"{x:>+.3f}" if pd.notna(x) else "   n/a"

        print("\n" + "="*140)
        print(f"GLD FILTER COMPARISON {start_year}-{end_year} ({ma50.index[0].strftime('%Y-%m-%d')} to {ma50.index[-1].strftime('%Y-%m-%d')})")
        print("="*140)
        header = f"{'Config':<10} {'CAGR':>10} {'Sharpe':>10} {'MaxDD':>10} {'d_CAGR_vs_GLD':>15} {'d_Sharpe_vs_GLD':>15}"
        print(header)
        print("-"*140)
        for row in rows:
            print(
                f"{row['name']:<10} {fmt_pct(row['cagr']):>10} {fmt_float(row['sharpe']):>10} {fmt_pct(row['dd']):>10} "
                f"{fmt_signed_pct(row['dcagr']):>15} {fmt_signed_float(row['dsharpe']):>15}"
            )
        # GLD baseline
        print("-"*140)
        print(
            f"{'GLD':<10} {fmt_pct(mg['cagr']):>10} {fmt_float(mg['sharpe']):>10} {fmt_pct(mg['max_drawdown']):>10} "
            f"{'0.00%':>15} {'0.000':>15}"
        )
        print("="*140)


def backtest_gld_only(prices: pd.DataFrame) -> dict:
    """Benchmark: hold GLD always."""
    monthly = prices.resample("ME").last()
    monthly_rets = []
    
    for i in range(1, len(monthly)):
        if i + 1 < len(monthly):
            price_now = monthly.iloc[i]["GLD"]
            price_next = monthly.iloc[i + 1]["GLD"]
            monthly_ret = (price_next - price_now) / price_now if price_now != 0 else 0
            monthly_rets.append(monthly_ret)
    
    return_series = pd.Series(monthly_rets, index=monthly.index[2:len(monthly)])
    return {"returns": return_series}

def run_scenario(end_label: str, end_date: str) -> None:
    prices = download_prices(TICKERS, end_date=end_date)
    
    ma50_result = backtest_moving_average(prices.copy(), FilterType.MA_50)
    ma100_result = backtest_moving_average(prices.copy(), FilterType.MA_100)
    ma200_result = backtest_moving_average(prices.copy(), FilterType.MA_200)
    ema100_result = backtest_exponential_moving_average(prices.copy())
    ret6m_result = backtest_trailing_return(prices, months=6)
    ret12m_result = backtest_trailing_return(prices, months=12)
    gld_result = backtest_gld_only(prices)
    
    ma50_ret = ma50_result["returns"]
    ma100_ret = ma100_result["returns"]
    ma200_ret = ma200_result["returns"]
    ema100_ret = ema100_result["returns"]
    ret6m_ret = ret6m_result["returns"]
    ret12m_ret = ret12m_result["returns"]
    gld_ret = gld_result["returns"]
    
    if all(len(r) > 0 for r in [ma50_ret, ma100_ret, ma200_ret, ema100_ret, ret6m_ret, ret12m_ret, gld_ret]):
        ma50_metrics = compute_metrics(ma50_ret)
        ma100_metrics = compute_metrics(ma100_ret)
        ma200_metrics = compute_metrics(ma200_ret)
        ema100_metrics = compute_metrics(ema100_ret)
        ret6m_metrics = compute_metrics(ret6m_ret)
        ret12m_metrics = compute_metrics(ret12m_ret)
        gld_metrics = compute_metrics(gld_ret)
        
        cal_50 = FilterType.MA_50.value[1]
        cal_100 = FilterType.MA_100.value[1]
        cal_200 = FilterType.MA_200.value[1]
        
        print("\n" + "="*160)
        print(f"GLD FILTER COMPARISON {end_label} ({ma50_ret.index[0].strftime('%Y-%m-%d')} to {ma50_ret.index[-1].strftime('%Y-%m-%d')})")
        print(f"Trading days: 50~{cal_50}cal, 100~{cal_100}cal, 200~{cal_200}cal")
        print("="*160)
        print(f"{'Metric':<12} {'50TMA':<18} {'100TMA':<18} {'100EMA':<18} {'200TMA':<18} {'6M Ret':<18} {'12M Ret':<18} {'GLD Bench':<18}")
        print("-"*160)
        
        print(f"{'CAGR':<12} {ma50_metrics['cagr']:>16.2%} {ma100_metrics['cagr']:>16.2%} {ema100_metrics['cagr']:>16.2%} {ma200_metrics['cagr']:>16.2%} {ret6m_metrics['cagr']:>16.2%} {ret12m_metrics['cagr']:>16.2%} {gld_metrics['cagr']:>16.2%}")
        print(f"{'Sharpe':<12} {ma50_metrics['sharpe']:>16.3f} {ma100_metrics['sharpe']:>16.3f} {ema100_metrics['sharpe']:>16.3f} {ma200_metrics['sharpe']:>16.3f} {ret6m_metrics['sharpe']:>16.3f} {ret12m_metrics['sharpe']:>16.3f} {gld_metrics['sharpe']:>16.3f}")
        print(f"{'Max DD':<12} {ma50_metrics['max_drawdown']:>16.2%} {ma100_metrics['max_drawdown']:>16.2%} {ema100_metrics['max_drawdown']:>16.2%} {ma200_metrics['max_drawdown']:>16.2%} {ret6m_metrics['max_drawdown']:>16.2%} {ret12m_metrics['max_drawdown']:>16.2%} {gld_metrics['max_drawdown']:>16.2%}")
        print("="*160)
        
        print("\nVs GLD Benchmark:")
        print(f"  50TMA:    {ma50_metrics['cagr'] - gld_metrics['cagr']:>+.2%} CAGR, Sharpe {ma50_metrics['sharpe'] - gld_metrics['sharpe']:>+.3f}")
        print(f"  100TMA:   {ma100_metrics['cagr'] - gld_metrics['cagr']:>+.2%} CAGR, Sharpe {ma100_metrics['sharpe'] - gld_metrics['sharpe']:>+.3f}")
        print(f"  100EMA:   {ema100_metrics['cagr'] - gld_metrics['cagr']:>+.2%} CAGR, Sharpe {ema100_metrics['sharpe'] - gld_metrics['sharpe']:>+.3f}")
        print(f"  200TMA:   {ma200_metrics['cagr'] - gld_metrics['cagr']:>+.2%} CAGR, Sharpe {ma200_metrics['sharpe'] - gld_metrics['sharpe']:>+.3f}")
        print(f"  6M Ret:   {ret6m_metrics['cagr'] - gld_metrics['cagr']:>+.2%} CAGR, Sharpe {ret6m_metrics['sharpe'] - gld_metrics['sharpe']:>+.3f}")
        print(f"  12M Ret:  {ret12m_metrics['cagr'] - gld_metrics['cagr']:>+.2%} CAGR, Sharpe {ret12m_metrics['sharpe'] - gld_metrics['sharpe']:>+.3f}")

def main():
    scenarios = [
        ("(through 2023)", "2023-12-31"),
        ("(through 2025)", "2025-12-31"),
    ]
    for label, end_date in scenarios:
        run_scenario(label, end_date)

    run_four_year_chunks()

if __name__ == "__main__":
    main()

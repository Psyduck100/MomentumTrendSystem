"""Compare scoring methods (12M vs blend_6_12) with various filter configurations.

Tests combinations of:
- Score modes: 12M only, blend_6_12
- Filters: none, ret_6m, ret_12m
- Defensive: IEF
- Rank gap: 0 (fixed)
"""

from __future__ import annotations
import pandas as pd
from pathlib import Path
from us_rotation_custom import BUCKET_MAP, BACKTEST_CACHE
from momentum_program.backtest.engine import backtest_momentum
from momentum_program.backtest.metrics import compute_metrics
from momentum_program.analytics.constants import SCORE_MODE_BLEND_6_12
from pandas.tseries.offsets import MonthEnd

# Universe
TICKERS_WITH_BOND = ["SPTM", "SPY", "QQQ", "OEF", "IWD", "IEF"]

TBILL_CSV = Path("CSVs/TB3MS.csv")

# Strategy configurations to test
CONFIGS = [
    # 12M scoring with various filters
    {"label": "12m_no_filter", "score_mode": "ret_12m", "filter": "none", "band_series": None},
    {"label": "12m_filter_6m", "score_mode": "ret_12m", "filter": "ret_6m", "band_series": None},
    {"label": "12m_filter_12m", "score_mode": "ret_12m", "filter": "ret_12m", "band_series": None},
    {"label": "12m_filter_12m_excess_rf", "score_mode": "ret_12m", "filter": "ret_12m", "band_series": "rf_12m"},
    
    # blend_6_12 scoring with various filters
    {"label": "blend_no_filter", "score_mode": SCORE_MODE_BLEND_6_12, "filter": "none", "band_series": None},
    {"label": "blend_filter_6m", "score_mode": SCORE_MODE_BLEND_6_12, "filter": "ret_6m", "band_series": None},
    {"label": "blend_filter_12m", "score_mode": SCORE_MODE_BLEND_6_12, "filter": "ret_12m", "band_series": None},
    {"label": "blend_filter_12m_excess_rf", "score_mode": SCORE_MODE_BLEND_6_12, "filter": "ret_12m", "band_series": "rf_12m"},
]

START_DATE = "2001-01-01"
END_DATE = pd.Timestamp.today().strftime("%Y-%m-%d")
CASH_RATE = 0.025


def compound_by_year(monthly_returns: pd.Series) -> pd.DataFrame:
    """Compound monthly returns to annual."""
    df = monthly_returns.to_frame('ret')
    df['year'] = df.index.year
    annual = df.groupby('year')['ret'].apply(lambda x: (1 + x).prod() - 1)
    return annual


def load_risk_free_band() -> pd.Series:
    """Load TB3MS and compute 12-month risk-free return.

    TB3MS is quoted in annualized percent (e.g., 5.15 = 5.15% per year).
    Steps:
    1. Convert to decimal: y_t = TB3MS / 100
    2. Convert to monthly return: r_f,t = y_t / 12 (simple approximation)
    3. Compound 12 monthly returns to get 12M risk-free return: R_f(12,t)
    """
    tbill = pd.read_csv(TBILL_CSV, parse_dates=['observation_date'])
    tbill['observation_date'] = tbill['observation_date'] + MonthEnd(0)
    tbill.set_index('observation_date', inplace=True)
    
    # Convert annualized percent to decimal
    tbill['rf_annual_decimal'] = tbill['TB3MS'] / 100.0
    
    # Convert to monthly return (simple: annual / 12)
    tbill['rf_monthly'] = tbill['rf_annual_decimal'] / 12.0
    
    # Compound last 12 monthly returns to get 12M risk-free return
    # rolling(12) gives us 12 consecutive values; product of (1 + r_f) then subtract 1
    tbill['rf_12m'] = (1 + tbill['rf_monthly']).rolling(12).apply(
        lambda x: x.prod() - 1, raw=False
    )
    
    return tbill['rf_12m'].dropna()


def run_strategy(label: str, score_mode: str, filter_mode: str, band_series: pd.Series | None) -> dict:
    """Run backtest for a single configuration."""
    print(f"Running {label}: score={score_mode} filter={filter_mode}")
    
    result = backtest_momentum(
        tickers=TICKERS_WITH_BOND,
        bucket_map=BUCKET_MAP,
        start_date=START_DATE,
        end_date=END_DATE,
        top_n_per_bucket=1,
        cache_dir=BACKTEST_CACHE,
        slippage_bps=3.0,
        expense_ratio=0.001,
        rank_gap_threshold=0,
        score_mode=score_mode,
        abs_filter_mode=filter_mode,
        abs_filter_cash_annual=CASH_RATE,
        defensive_symbol="IEF",
        abs_filter_band_series=band_series,
    )
    
    # Compute metrics from overall returns
    overall_df = result['overall_returns']
    monthly_rets = overall_df['return']
    metrics = compute_metrics(monthly_rets)
    result['metrics'] = metrics
    result['monthly_returns'] = monthly_rets
    
    # Extract ticker-level monthly returns
    result['ticker_monthly_returns'] = result['monthly_prices'].pct_change()
    
    return result


def main():
    """Run all configurations and export results."""
    
    # Run all strategies
    results = {}
    rf_band = load_risk_free_band()

    for config in CONFIGS:
        result = run_strategy(
            label=config['label'],
            score_mode=config['score_mode'],
            filter_mode=config['filter'],
            band_series=rf_band if config.get('band_series') == 'rf_12m' else None,
        )
        results[config['label']] = result
    
    # Extract CAGR/Sharpe/MaxDD summary
    summary_rows = []
    for label, res in results.items():
        metrics = res['metrics']
        summary_rows.append({
            'strategy': label,
            'start': res['monthly_returns'].index[0].strftime('%Y-%m-%d'),
            'end': res['monthly_returns'].index[-1].strftime('%Y-%m-%d'),
            'cagr': metrics['cagr'],
            'sharpe': metrics['sharpe'],
            'max_drawdown': metrics['max_drawdown']
        })
    
    summary_df = pd.DataFrame(summary_rows)
    
    # Add SPY and QQQ benchmarks
    spy_result = results[CONFIGS[0]['label']]  # Use first strategy's date range
    for ticker in ['SPY', 'QQQ']:
        bench_rets = spy_result['ticker_monthly_returns'][ticker]
        bench_annual = (1 + bench_rets).prod() ** (12 / len(bench_rets)) - 1
        bench_sharpe = bench_rets.mean() / bench_rets.std() * (12 ** 0.5) if bench_rets.std() > 0 else 0
        cumulative = (1 + bench_rets).cumprod()
        running_max = cumulative.expanding().max()
        drawdowns = (cumulative - running_max) / running_max
        bench_maxdd = drawdowns.min()
        
        summary_rows.append({
            'strategy': ticker,
            'start': bench_rets.index[0].strftime('%Y-%m-%d'),
            'end': bench_rets.index[-1].strftime('%Y-%m-%d'),
            'cagr': bench_annual,
            'sharpe': bench_sharpe,
            'max_drawdown': bench_maxdd
        })
    
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv('score_filter_cagr.csv', index=False)
    print(f"\nSaved CAGR summary to score_filter_cagr.csv")
    
    # Export annual returns
    annual_data = {}
    for label, res in results.items():
        annual_data[label] = compound_by_year(res['monthly_returns'])
    
    # Add benchmarks
    for ticker in ['SPY', 'QQQ']:
        bench_rets = spy_result['ticker_monthly_returns'][ticker]
        annual_data[ticker] = compound_by_year(bench_rets)
    
    annual_df = pd.DataFrame(annual_data)
    annual_df.index.name = 'year'
    annual_df.to_csv('score_filter_annual_returns.csv')
    print(f"Saved annual returns to score_filter_annual_returns.csv")


if __name__ == "__main__":
    main()

"""PMTL Strategy Backtest Engine

Core logic for running MA-based backtests with pluggable fallback strategies.
"""

from typing import Dict, Tuple
import pandas as pd
import yfinance as yf

from momentum_program.backtest.metrics import compute_metrics
from pmtl_fallback_strategies import get_fallback_strategy


class PMTLBacktestEngine:
    """Backtest engine for PMTL (Precious Metals Tactical Logic) strategy."""
    
    def __init__(self, main_ticker: str, start_date: str, end_date: str, frequency: str = 'ME'):
        self.main_ticker = main_ticker
        self.start_date = start_date
        self.end_date = end_date
        self.frequency = frequency  # 'ME'=monthly, 'W'=weekly, '2W'=biweekly
        self.prices = None
        self._download_prices()
    
    def _download_prices(self):
        """Download main ticker prices."""
        print(f"Downloading {self.main_ticker} prices...")
        data = yf.download(
            self.main_ticker,
            start=self.start_date,
            end=self.end_date,
            progress=False
        )
        
        # Handle both single and multi-ticker downloads
        if isinstance(data.columns, pd.MultiIndex):
            self.prices = data[('Close', self.main_ticker)] if ('Close', self.main_ticker) in data.columns else data.iloc[:, 0]
        else:
            self.prices = data['Close'] if 'Close' in data.columns else data['Adj Close']
    
    def backtest_sma(self, window: int, fallback) -> pd.Series:
        """Run SMA-based backtest with fallback strategy.
        
        If price > SMA: hold main asset
        If price ≤ SMA: use fallback strategy
        
        Args:
            window: Trading days for SMA
            fallback: FallbackStrategy instance
            
        Returns:
            Series of period returns (monthly/weekly/biweekly)
        """
        # Calculate SMA
        ma = self.prices.rolling(window=window, min_periods=1).mean()
        
        # Get period-end prices and dates
        monthly_prices = self.prices.resample(self.frequency).last()
        monthly_dates = monthly_prices.index
        
        # Get MA values at month-end
        ma_at_month_end = ma.reindex(monthly_dates, method='ffill')
        
        # Generate signal: 1 if price > MA, 0 otherwise
        signal = (monthly_prices > ma_at_month_end).astype(int)
        
        # Calculate main asset monthly returns
        main_monthly_ret = monthly_prices.pct_change()
        
        # Get fallback returns
        fallback_ret = fallback.get_monthly_returns(monthly_dates)
        
        # Blend: signal% main + (1-signal%) fallback
        blended_ret = (signal.values * main_monthly_ret.values + 
                       (1 - signal.values) * fallback_ret.values)
        
        return pd.Series(blended_ret, index=monthly_dates, name=f"SMA_{window}")
    
    def backtest_ema(self, window: int, fallback) -> pd.Series:
        """Run EMA-based backtest with fallback strategy.
        
        If price > EMA: hold main asset
        If price ≤ EMA: use fallback strategy
        
        Args:
            window: Trading days for EMA
            fallback: FallbackStrategy instance
            
        Returns:
            Series of period returns (monthly/weekly/biweekly)
        """
        # Calculate EMA
        ma = self.prices.ewm(span=window, adjust=False).mean()
        
        # Get period-end prices and dates
        monthly_prices = self.prices.resample(self.frequency).last()
        monthly_dates = monthly_prices.index
        
        # Get MA values at month-end
        ma_at_month_end = ma.reindex(monthly_dates, method='ffill')
        
        # Generate signal: 1 if price > MA, 0 otherwise
        signal = (monthly_prices > ma_at_month_end).astype(int)
        
        # Calculate main asset monthly returns
        main_monthly_ret = monthly_prices.pct_change()
        
        # Get fallback returns
        fallback_ret = fallback.get_monthly_returns(monthly_dates)
        
        # Blend: signal% main + (1-signal%) fallback
        blended_ret = (signal.values * main_monthly_ret.values + 
                       (1 - signal.values) * fallback_ret.values)
        
        return pd.Series(blended_ret, index=monthly_dates, name=f"EMA_{window}")
    
    def backtest_benchmark(self) -> pd.Series:
        """Run benchmark: hold main asset always."""
        monthly_prices = self.prices.resample(self.frequency).last()
        return monthly_prices.pct_change()
    
    def run_window_sweep(
        self,
        fallback,
        windows: list,
        use_sma: bool = True,
        use_ema: bool = True
    ) -> Tuple[pd.DataFrame, Dict, Dict]:
        """Test all windows with SMA/EMA and fallback strategy.
        
        Args:
            fallback: FallbackStrategy instance
            windows: List of window sizes to test
            use_sma: Whether to test SMA windows
            use_ema: Whether to test EMA windows
            
        Returns:
            (results_dataframe, annual_returns_dict)
        """
        results = []
        annual_returns_all = {}
        monthly_returns_all = {}
        
        # Get benchmark
        benchmark_rets = self.backtest_benchmark()
        benchmark_annual = benchmark_rets.groupby(benchmark_rets.index.year).apply(lambda x: (1 + x).prod() - 1)
        annual_returns_all['benchmark'] = benchmark_annual
        monthly_returns_all['benchmark'] = benchmark_rets
        
        # Determine periods per year based on frequency
        periods_per_year = {'ME': 12, 'W': 52, '2W': 26}.get(self.frequency, 12)
        
        metrics = compute_metrics(benchmark_rets, periods_per_year)
        print(f"\nBenchmark ({self.main_ticker}): CAGR {metrics['cagr']:.2%}, "
              f"Sharpe {metrics['sharpe']:.3f}, MaxDD {metrics['max_drawdown']:.2%}")
        
        # Test SMA windows
        if use_sma:
            print(f"\nTesting SMA windows (fallback: {fallback.name})...")
            for window in windows:
                sma_rets = self.backtest_sma(window, fallback)
                metrics = compute_metrics(sma_rets)
                
                results.append({
                    'type': 'SMA',
                    'window': window,
                    'cagr': metrics['cagr'],
                    'sharpe': metrics['sharpe'],
                    'max_drawdown': metrics['max_drawdown'],
                })
                
                # Annual returns (compound monthly inside each year)
                annual = sma_rets.groupby(sma_rets.index.year).apply(lambda x: (1 + x).prod() - 1)
                annual_returns_all[f'SMA_{window}'] = annual
                monthly_returns_all[f'SMA_{window}'] = sma_rets
                
                print(f"  SMA {window:3d}: CAGR {metrics['cagr']:6.2%}, "
                      f"Sharpe {metrics['sharpe']:6.3f}, MaxDD {metrics['max_drawdown']:7.2%}")
        
        # Test EMA windows
        if use_ema:
            print(f"\nTesting EMA windows (fallback: {fallback.name})...")
            for window in windows:
                ema_rets = self.backtest_ema(window, fallback)
                metrics = compute_metrics(ema_rets)
                
                results.append({
                    'type': 'EMA',
                    'window': window,
                    'cagr': metrics['cagr'],
                    'sharpe': metrics['sharpe'],
                    'max_drawdown': metrics['max_drawdown'],
                })
                
                # Annual returns (compound monthly inside each year)
                annual = ema_rets.groupby(ema_rets.index.year).apply(lambda x: (1 + x).prod() - 1)
                annual_returns_all[f'EMA_{window}'] = annual
                monthly_returns_all[f'EMA_{window}'] = ema_rets
                
                print(f"  EMA {window:3d}: CAGR {metrics['cagr']:6.2%}, "
                      f"Sharpe {metrics['sharpe']:6.3f}, MaxDD {metrics['max_drawdown']:7.2%}")
        
        return pd.DataFrame(results), annual_returns_all, monthly_returns_all

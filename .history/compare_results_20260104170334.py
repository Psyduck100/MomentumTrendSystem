"""
COMPARE TEST RESULTS

After running multiple tests with different configs, use this to compare them.
"""

import pandas as pd
from pathlib import Path


def compare_fallbacks():
    """Compare results from different fallback strategies."""
    
    print("\n" + "="*80)
    print("FALLBACK COMPARISON")
    print("="*80)
    
    # Load results
    cash = pd.read_csv('pmtl_cash_results.csv').sort_values('cagr', ascending=False)
    
    # Try to load others (may not exist)
    try:
        tbills = pd.read_csv('pmtl_tbills_results.csv').sort_values('cagr', ascending=False)
        has_tbills = True
    except FileNotFoundError:
        has_tbills = False
    
    try:
        ief = pd.read_csv('pmtl_ief_results.csv').sort_values('cagr', ascending=False)
        has_ief = True
    except FileNotFoundError:
        has_ief = False
    
    # Compare best strategies
    print("\n📊 BEST STRATEGY BY FALLBACK")
    print("-" * 80)
    print(f"{'Fallback':<15} {'Type':<6} {'Window':<8} {'CAGR':<12} {'Sharpe':<12} {'MaxDD':<12}")
    print("-" * 80)
    
    c = cash.iloc[0]
    print(f"{'CASH':<15} {c['type']:<6} {int(c['window']):<8} {c['cagr']:>10.2%}  {c['sharpe']:>10.3f}  {c['max_drawdown']:>10.2%}")
    
    if has_tbills:
        t = tbills.iloc[0]
        print(f"{'TB3MS':<15} {t['type']:<6} {int(t['window']):<8} {t['cagr']:>10.2%}  {t['sharpe']:>10.3f}  {t['max_drawdown']:>10.2%}")
        
        cagr_diff = t['cagr'] - c['cagr']
        print(f"\n  TB3MS advantage: {cagr_diff:+.2%} CAGR")
    
    if has_ief:
        i = ief.iloc[0]
        print(f"{'IEF':<15} {i['type']:<6} {int(i['window']):<8} {i['cagr']:>10.2%}  {i['sharpe']:>10.3f}  {i['max_drawdown']:>10.2%}")
        
        cagr_diff = i['cagr'] - c['cagr']
        print(f"\n  IEF vs CASH: {cagr_diff:+.2%} CAGR")


def compare_windows():
    """Compare best windows across all tests."""
    
    print("\n" + "="*80)
    print("WINDOW COMPARISON (All strategies)")
    print("="*80)
    
    # Load all available results
    results_files = list(Path('.').glob('pmtl_*_results.csv'))
    
    if not results_files:
        print("No result files found!")
        return
    
    print(f"\nFound {len(results_files)} result files:\n")
    
    for results_file in sorted(results_files):
        df = pd.read_csv(results_file)
        df_sorted = df.sort_values('cagr', ascending=False)
        best = df_sorted.iloc[0]
        
        print(f"{results_file.stem:30} → "
              f"{best['type']:5} {int(best['window']):3d} "
              f"CAGR {best['cagr']:6.2%} "
              f"Sharpe {best['sharpe']:6.3f}")


def compare_annual_returns():
    """Compare year-by-year returns for best strategies."""
    
    print("\n" + "="*80)
    print("ANNUAL RETURNS COMPARISON")
    print("="*80)
    
    # Load cash as baseline
    try:
        cash_annual = pd.read_csv('pmtl_cash_annual_returns.csv', index_col=0)
        cash_best = 'EMA_100'  # Best strategy from cash test
        
        if cash_best in cash_annual.index:
            cash_returns = cash_annual.loc[cash_best, '2005':'2025'].astype(float)
            
            print(f"\nCASH Fallback - {cash_best} Annual Returns:")
            print("-" * 80)
            
            years = [str(y) for y in range(2005, 2026)]
            rets = []
            for year in years:
                if year in cash_annual.columns:
                    ret = cash_annual.loc[cash_best, year]
                    rets.append(ret)
                    print(f"{year}: {ret:>8.2%}", end="  ")
                    if (int(year) - 2004) % 5 == 0:
                        print()
            
            print(f"\n\nStatistics:")
            print(f"  Mean:      {cash_returns.mean():>8.2%}")
            print(f"  Std Dev:   {cash_returns.std():>8.2%}")
            print(f"  Min:       {cash_returns.min():>8.2%}")
            print(f"  Max:       {cash_returns.max():>8.2%}")
            print(f"  Positive:  {(cash_returns > 0).sum()}/21 years")
    
    except Exception as e:
        print(f"Could not load annual returns: {e}")


def summary_table():
    """Create summary table of all results."""
    
    print("\n" + "="*80)
    print("SUMMARY: Top 3 Strategies by Fallback")
    print("="*80)
    
    results_files = list(Path('.').glob('pmtl_*_results.csv'))
    
    for results_file in sorted(results_files):
        df = pd.read_csv(results_file)
        df_sorted = df.sort_values('cagr', ascending=False)
        
        print(f"\n{results_file.stem.upper()}")
        print("-" * 80)
        print(f"{'Rank':<5} {'Type':<6} {'Window':<8} {'CAGR':<12} {'Sharpe':<12} {'MaxDD':<12}")
        print("-" * 80)
        
        for rank, (idx, row) in enumerate(df_sorted.head(3).iterrows(), 1):
            print(f"{rank:<5} {row['type']:<6} {int(row['window']):<8} "
                  f"{row['cagr']:>10.2%}  {row['sharpe']:>10.3f}  {row['max_drawdown']:>10.2%}")


if __name__ == "__main__":
    print("\n" + "="*80)
    print("PMTL RESULTS COMPARISON")
    print("="*80)
    
    summary_table()
    compare_windows()
    compare_fallbacks()
    # compare_annual_returns()  # Uncomment to see detailed annual returns
    
    print("\n" + "="*80)
    print("USAGE:")
    print("  1. Run different configs: pmtl_config.py (change ACTIVE_CONFIG)")
    print("  2. Run backtest: python pmtl_runner.py")
    print("  3. Compare: python compare_results.py")
    print("="*80 + "\n")

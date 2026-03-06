"""Extract per-bucket gap sensitivity from backtest_output_by_bucket.txt"""
from pathlib import Path
import re

def main():
    input_file = Path("backtest_output_by_bucket.txt")
    output_file = Path("gap_sensitivity_by_bucket.txt")
    
    if not input_file.exists():
        print(f"{input_file} not found")
        return
    
    content = input_file.read_text()
    
    # Split by scenario blocks
    scenario_blocks = re.split(r'^-{100,}', content, flags=re.MULTILINE)
    
    # Dict to accumulate bucket metrics by gap value
    # bucket_gap_metrics[bucket][gap] = list of (cagr, sharpe, maxdd)
    bucket_gap_metrics = {}
    bucket_gap_metrics_ret_and = {}
    
    for block in scenario_blocks:
        if not block.strip():
            continue
        
        # Extract config line
        config_match = re.search(r'Config: (.+)', block)
        if not config_match:
            continue
        
        config_str = config_match.group(1)
        
        # Determine if ret_and filter
        is_ret_and = 'ret_and' in config_str
        metrics_dict = bucket_gap_metrics_ret_and if is_ret_and else bucket_gap_metrics
        
        # Extract gap value (uniform or dict)
        gap_match = re.search(r'gap=(\d+)', config_str)
        if gap_match:
            gap_val = int(gap_match.group(1))
            
            # Extract bucket metrics
            bucket_lines = re.findall(r'^\s+(\w+[\w\s]*?)\s+\|\s+CAGR:\s+([\d.%+-]+)\s+\|\s+Sharpe:\s+([\d.+-]+)\s+\|\s+MaxDD:\s+([\d.%+-]+)', 
                                     block, flags=re.MULTILINE)
            
            for bucket_name, cagr_str, sharpe_str, maxdd_str in bucket_lines:
                bucket_name = bucket_name.strip()
                
                try:
                    cagr = float(cagr_str.rstrip('%')) / 100
                    sharpe = float(sharpe_str)
                    maxdd = float(maxdd_str.rstrip('%')) / 100
                    
                    if bucket_name not in metrics_dict:
                        metrics_dict[bucket_name] = {}
                    
                    if gap_val not in metrics_dict[bucket_name]:
                        metrics_dict[bucket_name][gap_val] = []
                    
                    metrics_dict[bucket_name][gap_val].append((cagr, sharpe, maxdd))
                except (ValueError, IndexError):
                    pass
    
    # Write output
    with output_file.open("w") as f:
        f.write("=" * 100 + "\n")
        f.write("PER-BUCKET GAP SENSITIVITY ANALYSIS\n")
        f.write("How gap (0-3) affects each bucket's CAGR, Sharpe, MaxDD\n")
        f.write("=" * 100 + "\n\n")
        
        # No absolute filter
        f.write("SCENARIO 1: No Absolute Filter (ret_and = OFF)\n")
        f.write("-" * 100 + "\n\n")
        
        if bucket_gap_metrics:
            for bucket in sorted(bucket_gap_metrics.keys()):
                f.write(f"{bucket}:\n")
                f.write(f"  Gap |  CAGR  | Sharpe | MaxDD\n")
                f.write(f"  ----|--------|--------|--------\n")
                
                for gap_val in [0, 1, 2, 3]:
                    if gap_val in bucket_gap_metrics[bucket]:
                        metrics = bucket_gap_metrics[bucket][gap_val]
                        if metrics:
                            # Average if multiple scenarios with same gap
                            avg_cagr = sum(m[0] for m in metrics) / len(metrics)
                            avg_sharpe = sum(m[1] for m in metrics) / len(metrics)
                            avg_maxdd = sum(m[2] for m in metrics) / len(metrics)
                            f.write(f"   {gap_val}  | {avg_cagr:6.2%} | {avg_sharpe:6.2f} | {avg_maxdd:6.2%}\n")
                    else:
                        f.write(f"   {gap_val}  |   N/A  |   N/A  |   N/A\n")
                
                f.write("\n")
        
        # With ret_and filter
        f.write("\n" + "=" * 100 + "\n\n")
        f.write("SCENARIO 2: Absolute Filter (ret_and @ 1%)\n")
        f.write("-" * 100 + "\n\n")
        
        if bucket_gap_metrics_ret_and:
            for bucket in sorted(bucket_gap_metrics_ret_and.keys()):
                f.write(f"{bucket}:\n")
                f.write(f"  Gap |  CAGR  | Sharpe | MaxDD\n")
                f.write(f"  ----|--------|--------|--------\n")
                
                for gap_val in [0, 1, 2, 3]:
                    if gap_val in bucket_gap_metrics_ret_and[bucket]:
                        metrics = bucket_gap_metrics_ret_and[bucket][gap_val]
                        if metrics:
                            avg_cagr = sum(m[0] for m in metrics) / len(metrics)
                            avg_sharpe = sum(m[1] for m in metrics) / len(metrics)
                            avg_maxdd = sum(m[2] for m in metrics) / len(metrics)
                            f.write(f"   {gap_val}  | {avg_cagr:6.2%} | {avg_sharpe:6.2f} | {avg_maxdd:6.2%}\n")
                    else:
                        f.write(f"   {gap_val}  |   N/A  |   N/A  |   N/A\n")
                
                f.write("\n")
    
    print(f"Gap sensitivity analysis written to {output_file}")

if __name__ == "__main__":
    main()

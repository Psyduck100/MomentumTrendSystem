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
    scenario_blocks = re.split(r"^-{100,}", content, flags=re.MULTILINE)

    # Dict to store bucket metrics by (gap_value, is_ret_and)
    # bucket_gap_metrics[(gap_val, is_ret_and)][bucket] = (cagr, sharpe, maxdd)
    bucket_gap_metrics = {}

    for block in scenario_blocks:
        if not block.strip():
            continue

        # Extract config line
        config_match = re.search(r"Config: (.+)", block)
        if not config_match:
            continue

        config_str = config_match.group(1)

        # Determine if ret_and filter
        is_ret_and = "ret_and" in config_str

        # Extract gap value - look for "gap=X" where X is a single digit (uniform gaps only)
        gap_match = re.search(r"gap=(\d+)", config_str)
        if not gap_match:
            continue

        gap_val = int(gap_match.group(1))

        # Skip per-bucket dict configs (they have multiple values in gap)
        # We only want uniform gaps (gap=0, gap=1, gap=2, gap=3)
        # Check if this looks like a dict config by looking for colons in the full config
        if "Bonds:" in config_str or "gap={" in config_str:
            continue  # Skip per-bucket dict configs

        # Extract bucket metrics
        bucket_lines = re.findall(
            r"^\s+(\w+[\w\s]*?)\s+\|\s+CAGR:\s+([\d.%+-]+)\s+\|\s+Sharpe:\s+([\d.+-]+)\s+\|\s+MaxDD:\s+([\d.%+-]+)",
            block,
            flags=re.MULTILINE,
        )

        key = (gap_val, is_ret_and)
        if key not in bucket_gap_metrics:
            bucket_gap_metrics[key] = {}

        for bucket_name, cagr_str, sharpe_str, maxdd_str in bucket_lines:
            bucket_name = bucket_name.strip()

            try:
                cagr = float(cagr_str.rstrip("%")) / 100
                sharpe = float(sharpe_str)
                maxdd = float(maxdd_str.rstrip("%")) / 100

                bucket_gap_metrics[key][bucket_name] = (cagr, sharpe, maxdd)
            except (ValueError, IndexError):
                pass

    # Write output
    with output_file.open("w") as f:
        f.write("=" * 100 + "\n")
        f.write("PER-BUCKET GAP SENSITIVITY ANALYSIS (Uniform Gaps Only)\n")
        f.write("How gap (0-3) affects each bucket's CAGR, Sharpe, MaxDD\n")
        f.write("=" * 100 + "\n\n")

        # Collect all bucket names
        all_buckets = set()
        for (gap, is_ret), buckets in bucket_gap_metrics.items():
            all_buckets.update(buckets.keys())
        all_buckets = sorted(list(all_buckets))

        # No absolute filter
        f.write("SCENARIO 1: No Absolute Filter (none)\n")
        f.write("-" * 100 + "\n\n")

        for bucket in all_buckets:
            f.write(f"{bucket}:\n")
            f.write(f"  Gap |  CAGR  | Sharpe | MaxDD\n")
            f.write(f"  ----|--------|--------|--------\n")

            for gap_val in [0, 1, 2, 3]:
                key = (gap_val, False)
                if key in bucket_gap_metrics and bucket in bucket_gap_metrics[key]:
                    cagr, sharpe, maxdd = bucket_gap_metrics[key][bucket]
                    f.write(
                        f"   {gap_val}  | {cagr:6.2%} | {sharpe:6.2f} | {maxdd:6.2%}\n"
                    )
                else:
                    f.write(f"   {gap_val}  |   N/A  |   N/A  |   N/A\n")

            f.write("\n")

        # With ret_and filter
        f.write("\n" + "=" * 100 + "\n\n")
        f.write("SCENARIO 2: Absolute Filter (ret_and @ 1%)\n")
        f.write("-" * 100 + "\n\n")

        for bucket in all_buckets:
            f.write(f"{bucket}:\n")
            f.write(f"  Gap |  CAGR  | Sharpe | MaxDD\n")
            f.write(f"  ----|--------|--------|--------\n")

            for gap_val in [0, 1, 2, 3]:
                key = (gap_val, True)
                if key in bucket_gap_metrics and bucket in bucket_gap_metrics[key]:
                    cagr, sharpe, maxdd = bucket_gap_metrics[key][bucket]
                    f.write(
                        f"   {gap_val}  | {cagr:6.2%} | {sharpe:6.2f} | {maxdd:6.2%}\n"
                    )
                else:
                    f.write(f"   {gap_val}  |   N/A  |   N/A  |   N/A\n")

            f.write("\n")

    print(f"Gap sensitivity analysis written to {output_file}")


if __name__ == "__main__":
    main()

"""Analyze how many symbols per bucket to understand gap behavior"""
from pathlib import Path
from momentum_program.universe.bucket_csv_provider import BucketedCsvUniverseProvider

def main():
    bucket_folder = Path("CSVs")
    if not bucket_folder.exists():
        print(f"{bucket_folder} not found")
        return
    
    universe = BucketedCsvUniverseProvider(bucket_folder)
    tickers = universe.get_tickers()
    bucket_map = universe.get_bucket_map()
    
    # Count symbols per bucket
    bucket_counts = {}
    for ticker, bucket in bucket_map.items():
        if bucket not in bucket_counts:
            bucket_counts[bucket] = []
        bucket_counts[bucket].append(ticker)
    
    print("BUCKET SIZES (before data availability filtering)")
    print("=" * 60)
    for bucket in sorted(bucket_counts.keys()):
        symbols = bucket_counts[bucket]
        print(f"{bucket:25s}: {len(symbols):3d} symbols")
        print(f"  {', '.join(sorted(symbols)[:10])}")
        if len(symbols) > 10:
            print(f"  ... and {len(symbols) - 10} more")
    
    print(f"\nTotal: {len(tickers)} symbols across {len(bucket_counts)} buckets")

if __name__ == "__main__":
    main()

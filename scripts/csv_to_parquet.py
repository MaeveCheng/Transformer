#!/usr/bin/env python3

import os
import pandas as pd
import numpy as np
from pathlib import Path
import glob
from datetime import datetime
import pyarrow.parquet as pq
import pyarrow as pa

def process_directory(dir_path):
    """Process all CSV files in a directory and convert to parquet"""
    
    symbol = os.path.basename(dir_path).replace('crypto_', '')
    
    print(f"Processing {symbol}...")
    csv_files = sorted(glob.glob(os.path.join(dir_path, '*.csv')))
    if not csv_files:
        print(f"  No CSV files found in {dir_path}")
        return None
    
    all_data = []
    for csv_file in csv_files:
        try:
            df = pd.read_csv(csv_file, header=None)
            
            # Check if timestamp is in microseconds (2025+ data) or milliseconds (pre-2025 data)
            # Timestamps > 2e12 are likely in microseconds
            if df[0].iloc[0] > 2e12:
                timestamp_unit = 'us'
            else:
                timestamp_unit = 'ms'
            
            data = pd.DataFrame({
                'symbol': symbol,
                'data_type': 'Crypto',
                'datetime': pd.to_datetime(df[0], unit=timestamp_unit).dt.floor('s'),  # Convert to datetime, floor to seconds
                'open': df[1].astype(np.float64),
                'high': df[2].astype(np.float64),
                'low': df[3].astype(np.float64),
                'close': df[4].astype(np.float64),
                'volume': df[5].astype(np.float64),
                'transactions': df[8].astype(np.int64)  # Number of trades
            })
            all_data.append(data)
        except Exception as e:
            print(f"  Error processing {csv_file}: {e}")
            continue
    
    if all_data:
        combined_df = pd.concat(all_data, ignore_index=True)
        combined_df = combined_df.sort_values('datetime').reset_index(drop=True)
        output_file = f"{symbol}.parquet"
        combined_df.to_parquet(output_file, engine='pyarrow', compression='snappy', index=False)
        print(f"  Saved {len(combined_df)} rows to {output_file}")
        print(f"  Date range: {combined_df['datetime'].min()} to {combined_df['datetime'].max()}")
        return combined_df
    
    return None

def main():
    """Process all crypto_* directories"""

    crypto_dirs = sorted(glob.glob('crypto_*'))
    
    print(f"Found {len(crypto_dirs)} directories to process\n")
    
    processed = 0
    failed = []
    for dir_path in crypto_dirs:
        if os.path.isdir(dir_path):
            result = process_directory(dir_path)
            if result is not None:
                processed += 1
            else:
                failed.append(dir_path)
    
    print(f"\n{'='*50}")
    print(f"Processing complete!")
    print(f"Successfully processed: {processed}/{len(crypto_dirs)} directories")
    if failed:
        print(f"\nFailed directories:")
        for dir_name in failed:
            print(f"  - {dir_name}")

if __name__ == "__main__":
    main()
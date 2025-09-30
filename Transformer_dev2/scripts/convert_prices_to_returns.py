#!/usr/bin/env python3
"""
Convert price columns in order book parquet file to t-1 returns amplified by 1000x.

This script:
1. Reads order book parquet file
2. Identifies all price columns (ending with '_px')
3. Converts them to t-1 returns: (price[t] - price[t-1]) / price[t-1] * 1000
4. Excludes mid_price from conversion
5. Saves to new parquet file
"""

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import numpy as np
from pathlib import Path
import argparse
import time
from typing import List, Tuple


def identify_price_columns(df: pd.DataFrame, exclude_mid_price: bool = True) -> List[str]:
    """Identify all price columns to convert."""
    price_columns = [col for col in df.columns if col.endswith('_px')]
    
    if exclude_mid_price and 'mid_price' in price_columns:
        price_columns.remove('mid_price')
    
    return sorted(price_columns)


def calculate_returns(prices: pd.Series, amplification: float = 1000.0) -> pd.Series:
    """
    Calculate t-1 returns amplified by the given factor.
    
    Returns = (price[t] - price[t-1000]) / price[t-1000] * amplification
    First value is set to 0.
    """
    # Calculate percentage returns
    returns = prices.pct_change()
    
    # Fill first value with 0
    returns.iloc[0] = 0.0
    
    # Amplify by factor
    returns = returns * amplification
    
    return returns


def convert_prices_to_returns(
    input_file: str, 
    output_file: str, 
    amplification: float = 1000.0,
    show_examples: bool = True
) -> None:
    """
    Convert price columns to returns in parquet file.
    
    Args:
        input_file: Path to input parquet file
        output_file: Path to output parquet file
        amplification: Factor to amplify returns (default 1000x)
        show_examples: Whether to show example conversions
    """
    print(f"Reading parquet file: {input_file}")
    start_time = time.time()
    
    # Read the parquet file
    df = pd.read_parquet(input_file)
    print(f"Loaded {len(df):,} rows, {len(df.columns)} columns in {time.time() - start_time:.2f}s")
    
    # Identify price columns
    price_columns = identify_price_columns(df, exclude_mid_price=True)
    print(f"\nFound {len(price_columns)} price columns to convert (excluding mid_price)")
    
    # Show some example columns
    print("\nExample price columns:")
    for col in price_columns[:5]:
        print(f"  - {col}")
    if len(price_columns) > 5:
        print(f"  ... and {len(price_columns) - 5} more")
    
    # Create copy for conversion
    df_returns = df.copy()
    
    # Convert each price column
    print(f"\nConverting price columns to {amplification}x amplified returns...")
    conversion_start = time.time()
    
    examples_shown = 0
    for i, col in enumerate(price_columns):
        # Calculate returns
        df_returns[col] = calculate_returns(df[col], amplification)
        
        # Show examples for first few columns
        if show_examples and examples_shown < 3:
            print(f"\nExample: {col}")
            print(f"  Original prices: {df[col].iloc[:5].values}")
            print(f"  Returns (no amp): {(df[col].pct_change().iloc[:5] * 1).values}")
            print(f"  Returns ({amplification}x): {df_returns[col].iloc[:5].values}")
            examples_shown += 1
        
        # Progress indicator
        if (i + 1) % 10 == 0:
            print(f"  Processed {i + 1}/{len(price_columns)} columns...", end='\r')
    
    print(f"\nConversion completed in {time.time() - conversion_start:.2f}s")
    
    # Verify mid_price is unchanged
    if 'mid_price' in df.columns:
        assert df['mid_price'].equals(df_returns['mid_price']), "mid_price should not be modified"
        print("\nVerified: mid_price column unchanged")
    
    # Show summary statistics
    print("\nSummary of conversions:")
    sample_col = price_columns[0]
    print(f"  Sample column: {sample_col}")
    print(f"  Original range: [{df[sample_col].min():.2f}, {df[sample_col].max():.2f}]")
    print(f"  Returns range: [{df_returns[sample_col].min():.2f}, {df_returns[sample_col].max():.2f}]")
    print(f"  Returns std: {df_returns[sample_col].std():.4f}")
    
    # Save to parquet
    print(f"\nSaving to: {output_file}")
    save_start = time.time()
    df_returns.to_parquet(output_file, compression='snappy', index=False)
    print(f"Saved in {time.time() - save_start:.2f}s")
    
    # Show file sizes
    input_size = Path(input_file).stat().st_size / (1024 * 1024)
    output_size = Path(output_file).stat().st_size / (1024 * 1024)
    print(f"\nFile sizes:")
    print(f"  Input:  {input_size:.2f} MB")
    print(f"  Output: {output_size:.2f} MB")
    print(f"  Ratio:  {output_size/input_size:.2%}")
    
    print(f"\nTotal time: {time.time() - start_time:.2f}s")


def main():
    parser = argparse.ArgumentParser(description="Convert price columns to amplified returns")
    parser.add_argument(
        "input_file", 
        help="Input parquet file path"
    )
    parser.add_argument(
        "-o", "--output", 
        help="Output parquet file path (default: input_returns.parquet)",
        default=None
    )
    parser.add_argument(
        "-a", "--amplification", 
        type=float, 
        default=1000.0,
        help="Amplification factor for returns (default: 1000)"
    )
    parser.add_argument(
        "--no-examples", 
        action="store_true",
        help="Don't show example conversions"
    )
    
    args = parser.parse_args()
    
    # Generate output filename if not provided
    if args.output is None:
        input_path = Path(args.input_file)
        args.output = str(input_path.parent / f"{input_path.stem}_returns{input_path.suffix}")
    
    # Run conversion
    convert_prices_to_returns(
        args.input_file,
        args.output,
        amplification=args.amplification,
        show_examples=not args.no_examples
    )


if __name__ == "__main__":
    main()
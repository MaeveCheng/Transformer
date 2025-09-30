#!/usr/bin/env python3
"""
Analyze BTC price movement after 30 minutes to understand typical return distributions.
This helps determine appropriate thresholds for Hold/Buy/Sell classification.
"""

import os
import glob
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Dict
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def analyze_returns_distribution(
    data_path: str,
    predict_minutes: int = 30,
    symbol_filter: str = 'BTC',
    max_files: int = 10
):
    """
    Analyze the distribution of returns after a specified time period.

    Args:
        data_path: Path to data directory
        predict_minutes: Minutes to look ahead for returns
        symbol_filter: Symbol to analyze (e.g., 'BTC')
        max_files: Maximum number of files to process
    """

    # Find data files
    if os.path.isdir(data_path):
        pattern = os.path.join(data_path, '*.parquet')
        files = glob.glob(pattern)
    else:
        # Single file
        files = [data_path]

    # Filter for symbol if specified
    if symbol_filter:
        files = [f for f in files if symbol_filter.upper() in f.upper()]

    if not files:
        raise ValueError(f"No files found for symbol {symbol_filter} in {data_path}")

    # Limit files
    files = files[:max_files]
    logger.info(f"Processing {len(files)} files")

    all_returns = []

    for file_path in files:
        logger.info(f"Processing {os.path.basename(file_path)}")

        try:
            # Read parquet file
            df = pd.read_parquet(file_path)

            # Check if we have price column
            price_col = None
            for col in ['close', 'Close', 'price', 'Price']:
                if col in df.columns:
                    price_col = col
                    break

            if price_col is None:
                logger.warning(f"No price column found in {file_path}")
                continue

            # Calculate returns for the specified period
            # Assuming 1-minute data
            df['return'] = df[price_col].pct_change(predict_minutes)

            # Shift returns to align with prediction point
            df['future_return'] = df['return'].shift(-predict_minutes)

            # Remove NaN values
            returns = df['future_return'].dropna().values

            # Convert to percentage
            returns = returns * 100  # Convert to percentage

            all_returns.extend(returns)

            logger.info(f"  Found {len(returns)} return samples")

        except Exception as e:
            logger.error(f"Error processing {file_path}: {e}")
            continue

    all_returns = np.array(all_returns)

    if len(all_returns) == 0:
        logger.error("No returns data collected")
        return

    # Calculate statistics
    logger.info(f"\n{'='*60}")
    logger.info(f"RETURN DISTRIBUTION ANALYSIS ({predict_minutes} minutes ahead)")
    logger.info(f"{'='*60}")
    logger.info(f"Total samples: {len(all_returns):,}")
    logger.info(f"\nBasic Statistics (in %):")
    logger.info(f"  Mean:   {np.mean(all_returns):.4f}%")
    logger.info(f"  Median: {np.median(all_returns):.4f}%")
    logger.info(f"  Std:    {np.std(all_returns):.4f}%")
    logger.info(f"  Min:    {np.min(all_returns):.4f}%")
    logger.info(f"  Max:    {np.max(all_returns):.4f}%")

    # Calculate percentiles
    percentiles = [1, 5, 10, 25, 50, 75, 90, 95, 99]
    logger.info(f"\nPercentiles (in %):")
    for p in percentiles:
        value = np.percentile(all_returns, p)
        logger.info(f"  {p:3d}%: {value:7.4f}%")

    # Analyze with different thresholds
    thresholds = [0.01, 0.05, 0.1, 0.2, 0.5, 1.0]  # in percentage

    logger.info(f"\nClassification with different thresholds:")
    logger.info(f"{'Threshold':>10} | {'Hold':>6} | {'Buy':>6} | {'Sell':>6}")
    logger.info(f"{'-'*40}")

    for threshold in thresholds:
        hold = np.sum((all_returns >= -threshold) & (all_returns <= threshold))
        buy = np.sum(all_returns > threshold)
        sell = np.sum(all_returns < -threshold)
        total = len(all_returns)

        hold_pct = hold / total * 100
        buy_pct = buy / total * 100
        sell_pct = sell / total * 100

        logger.info(f"{threshold:9.2f}% | {hold_pct:5.1f}% | {buy_pct:5.1f}% | {sell_pct:5.1f}%")

    # Create visualization
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    # 1. Histogram of returns
    ax = axes[0, 0]
    ax.hist(all_returns, bins=100, alpha=0.7, color='blue', edgecolor='black')
    ax.axvline(0, color='red', linestyle='--', label='Zero return')
    ax.set_xlabel(f'Return after {predict_minutes} minutes (%)')
    ax.set_ylabel('Frequency')
    ax.set_title('Return Distribution')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 2. Log scale histogram
    ax = axes[0, 1]
    ax.hist(all_returns, bins=100, alpha=0.7, color='green', edgecolor='black')
    ax.set_yscale('log')
    ax.axvline(0, color='red', linestyle='--', label='Zero return')
    ax.set_xlabel(f'Return after {predict_minutes} minutes (%)')
    ax.set_ylabel('Frequency (log scale)')
    ax.set_title('Return Distribution (Log Scale)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 3. CDF
    ax = axes[1, 0]
    sorted_returns = np.sort(all_returns)
    cumprob = np.arange(1, len(sorted_returns) + 1) / len(sorted_returns)
    ax.plot(sorted_returns, cumprob, color='purple', linewidth=2)
    ax.axvline(0, color='red', linestyle='--', alpha=0.5)
    ax.axhline(0.5, color='gray', linestyle=':', alpha=0.5)
    ax.set_xlabel(f'Return after {predict_minutes} minutes (%)')
    ax.set_ylabel('Cumulative Probability')
    ax.set_title('Cumulative Distribution Function')
    ax.grid(True, alpha=0.3)

    # 4. Box plot with different threshold visualizations
    ax = axes[1, 1]
    ax.boxplot(all_returns, vert=False, patch_artist=True,
               boxprops=dict(facecolor='lightblue'),
               medianprops=dict(color='red', linewidth=2))

    # Add threshold lines
    colors = ['green', 'yellow', 'orange', 'red', 'darkred', 'black']
    for threshold, color in zip(thresholds, colors):
        ax.axvline(threshold, color=color, linestyle='--', alpha=0.5,
                  label=f'±{threshold}%')
        ax.axvline(-threshold, color=color, linestyle='--', alpha=0.5)

    ax.set_xlabel(f'Return after {predict_minutes} minutes (%)')
    ax.set_title('Box Plot with Threshold Lines')
    ax.legend(loc='upper right', fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.suptitle(f'{symbol_filter} Return Analysis - {predict_minutes} Minutes Ahead',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()

    # Save plot
    plot_path = f'btc_return_analysis_{predict_minutes}min.png'
    plt.savefig(plot_path, dpi=100, bbox_inches='tight')
    logger.info(f"\nPlot saved to {plot_path}")
    plt.show()

    # Recommendation
    logger.info(f"\n{'='*60}")
    logger.info("THRESHOLD RECOMMENDATIONS:")
    logger.info(f"{'='*60}")

    # Find threshold that gives approximately 80% Hold, 10% Buy, 10% Sell
    best_threshold = None
    best_diff = float('inf')
    target_hold = 80

    for test_threshold in np.linspace(0.01, 2.0, 200):
        hold = np.sum((all_returns >= -test_threshold) & (all_returns <= test_threshold))
        hold_pct = hold / len(all_returns) * 100
        diff = abs(hold_pct - target_hold)

        if diff < best_diff:
            best_diff = diff
            best_threshold = test_threshold

    if best_threshold:
        hold = np.sum((all_returns >= -best_threshold) & (all_returns <= best_threshold))
        buy = np.sum(all_returns > best_threshold)
        sell = np.sum(all_returns < -best_threshold)
        total = len(all_returns)

        logger.info(f"\nFor ~80% Hold, 10% Buy, 10% Sell distribution:")
        logger.info(f"  Recommended threshold: ±{best_threshold:.4f}%")
        logger.info(f"  Actual distribution:")
        logger.info(f"    Hold: {hold/total*100:.1f}%")
        logger.info(f"    Buy:  {buy/total*100:.1f}%")
        logger.info(f"    Sell: {sell/total*100:.1f}%")

        # Convert to return value (not percentage)
        logger.info(f"\nFor config.json5:")
        logger.info(f"  return_threshold_buy:  {best_threshold/100:.6f}")
        logger.info(f"  return_threshold_sell: {-best_threshold/100:.6f}")


def find_crypto_data():
    """Try to find crypto data directories."""
    possible_paths = [
        '/workspace2/Metadata/crypto_train',
        '/workspace2/metadata/crypto_train',
        '/workspace2/data/crypto_train',
        '/workspace2/Data/crypto_train',
        '/workspace2/crypto_train',
        '/workspace2/Maeve/data/crypto_train',
    ]

    for path in possible_paths:
        if os.path.exists(path):
            logger.info(f"Found data path: {path}")
            return path

    # Try to find any directory with crypto data
    for root, dirs, files in os.walk('/workspace2'):
        if 'crypto' in root.lower():
            parquet_files = [f for f in files if f.endswith('.parquet')]
            if parquet_files:
                logger.info(f"Found crypto data in: {root}")
                return root

    return None


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Analyze BTC return distributions')
    parser.add_argument('--data-path', type=str,
                       default=None,
                       help='Path to data directory')
    parser.add_argument('--minutes', type=int, default=30,
                       help='Minutes to look ahead for returns')
    parser.add_argument('--symbol', type=str, default='BTC',
                       help='Symbol to analyze')
    parser.add_argument('--max-files', type=int, default=10,
                       help='Maximum number of files to process')

    args = parser.parse_args()

    # Auto-find data path if not provided
    data_path = args.data_path
    if data_path is None:
        logger.info("No data path provided, searching for crypto data...")
        data_path = find_crypto_data()
        if data_path is None:
            logger.error("Could not find crypto data directory. Please specify --data-path")
            logger.info("\nTrying to list available parquet files...")
            import subprocess
            result = subprocess.run(
                "find /workspace2 -name '*.parquet' 2>/dev/null | head -20",
                shell=True, capture_output=True, text=True
            )
            if result.stdout:
                logger.info("Found parquet files:")
                print(result.stdout)
                logger.info("\nPlease specify one of these paths with --data-path")
            else:
                logger.info("No parquet files found in /workspace2")
            exit(1)

    try:
        analyze_returns_distribution(
            data_path=data_path,
            predict_minutes=args.minutes,
            symbol_filter=args.symbol,
            max_files=args.max_files
        )
    except Exception as e:
        logger.error(f"Error: {e}")
        import traceback
        traceback.print_exc()
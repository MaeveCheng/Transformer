"""
Pre-normalization filtering for anomaly detection and removal.

This module provides row-based filtering that removes anomalous data BEFORE normalization.
If too many columns in a row are anomalous, the entire row is skipped.
If only a few columns are anomalous, those specific values are masked.
"""

import numpy as np
import pandas as pd
import torch
from typing import Dict, Optional, Tuple, List
import logging

logger = logging.getLogger(__name__)


class PreNormalizationFilter:
    """
    Filters raw data for anomalies before normalization.
    
    Features:
    - Category-specific valid ranges for different feature types
    - Row-based filtering when too many columns are anomalous
    - Column masking when few columns are anomalous
    - Detailed logging of filtering statistics
    """
    
    def __init__(
        self,
        # Valid ranges for each category
        price_range: Optional[Tuple[float, float]] = None,
        volume_range: Optional[Tuple[float, float]] = None,
        count_range: Optional[Tuple[float, float]] = None,
        derived_range: Optional[Tuple[float, float]] = None,
        time_range: Optional[Tuple[float, float]] = None,
        window_range: Optional[Tuple[float, float]] = None,
        meta_range: Optional[Tuple[float, float]] = None,
        other_range: Optional[Tuple[float, float]] = None,
        # Specific ranges for important computed features
        mid_price_range: Optional[Tuple[float, float]] = None,
        spread_range: Optional[Tuple[float, float]] = None,
        volume_imbalance_range: Optional[Tuple[float, float]] = None,
        # Placeholder values
        price_placeholder_values: Optional[List[float]] = None,
        # Thresholds
        row_threshold: float = 0.1,
        column_threshold: float = 0.1,
        # Column configuration for feature type detection
        column_config: Optional[Dict] = None
    ):
        """
        Initialize pre-normalization filter.
        
        Args:
            *_range: Valid ranges for each feature category (min, max)
            price_placeholder_values: List of exact values to filter in price columns
            row_threshold: If anomaly ratio > this, drop entire row
            column_threshold: If anomaly ratio < this, only mask columns
            column_config: Column configuration with feature type indices
        """
        self.valid_ranges = {
            'price': price_range,
            'volume': volume_range,
            'count': count_range,
            'derived': derived_range,
            'time': time_range,
            'window': window_range,
            'meta': meta_range,
            'other': other_range
        }
        
        # Store specific ranges for important features
        self.specific_ranges = {
            'mid_price': mid_price_range,
            'spread': spread_range,
            'volume_imbalance': volume_imbalance_range,
        }
        
        self.row_threshold = row_threshold
        self.column_threshold = column_threshold
        self.column_config = column_config
        self.price_placeholder_values = price_placeholder_values or []
        
        # Statistics tracking
        self.total_rows_processed = 0
        self.rows_filtered = 0
        self.values_masked = 0
        
        # Pre-compute pattern indices for fallback matching
        self._pattern_indices = None
        if column_config and 'column_names' in column_config:
            self._precompute_pattern_indices()
        
        logger.info(f"PreNormalizationFilter initialized with row_threshold={row_threshold}, "
                   f"column_threshold={column_threshold}")
        
        # Log active filters
        active_filters = [(cat, rng) for cat, rng in self.valid_ranges.items() if rng is not None]
        if active_filters:
            logger.info("Active pre-normalization filters:")
            for category, (min_val, max_val) in active_filters:
                logger.info(f"  {category}: [{min_val}, {max_val}]")
                if category == 'price' and self.price_placeholder_values:
                    logger.info(f"    + Special filter: exact values {self.price_placeholder_values} (placeholders)")
    
    def _precompute_pattern_indices(self):
        """Pre-compute pattern-based indices for each category to avoid repeated pattern matching."""
        column_names = self.column_config.get('column_names', [])
        
        # Define pattern matching functions
        patterns = {
            'price': lambda col: '_px' in col.lower() or col == 'mid_price',
            'volume': lambda col: '_qty' in col.lower() or 'volume' in col.lower(),
            'count': lambda col: '_ordcnt' in col.lower()
        }
        
        # Pre-compute indices for each pattern
        self._pattern_indices = {}
        for category, pattern_func in patterns.items():
            indices = [i for i, col in enumerate(column_names) if pattern_func(col)]
            self._pattern_indices[category] = indices
            logger.debug(f"Pre-computed {len(indices)} {category} indices from pattern matching")
    
    def filter_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Filter a DataFrame for anomalies.
        
        Args:
            df: Input DataFrame with raw data
            
        Returns:
            Filtered DataFrame with anomalous rows removed and values masked
        """
        if self.column_config is None:
            logger.warning("No column configuration provided, skipping pre-norm filtering")
            return df
        
        original_rows = len(df)
        
        # Create anomaly mask (True = anomalous)
        anomaly_mask = self._identify_anomalies(df)
        
        # Log how many columns we're checking
        logger.info(f"Checking {len(anomaly_mask.columns)} numeric columns out of {len(df.columns)} total columns")
        
        # Calculate anomaly ratio per row
        anomaly_counts = anomaly_mask.sum(axis=1)
        total_columns = anomaly_mask.shape[1]
        
        # Debug: Show distribution of anomaly counts
        if len(anomaly_mask) > 0 and len(anomaly_counts) > 0:
            unique_counts, count_freq = np.unique(anomaly_counts.values, return_counts=True)
            logger.info(f"Anomaly count distribution: {dict(zip(unique_counts, count_freq))}")
            
            # Show which columns have the most anomalies
            if total_columns > 0:
                col_anomaly_counts = anomaly_mask.sum(axis=0)
                worst_cols = col_anomaly_counts.nlargest(5)
                logger.info(f"Top 5 columns with most anomalies: {worst_cols.to_dict()}")
        
        # Only calculate ratios if we have columns to check
        if total_columns > 0:
            anomaly_ratios = anomaly_counts / total_columns
        else:
            # No columns to check, don't filter any rows
            anomaly_ratios = pd.Series(0, index=df.index)
        
        # Determine which rows to keep
        rows_to_drop = anomaly_ratios > self.row_threshold
        rows_to_mask = (anomaly_ratios > 0) & (anomaly_ratios <= self.column_threshold)
        
        # Log statistics
        n_rows_to_drop = rows_to_drop.sum()
        n_rows_to_mask = rows_to_mask.sum()
        
        if n_rows_to_drop > 0:
            logger.info(f"Dropping {n_rows_to_drop} rows ({n_rows_to_drop/original_rows*100:.2f}%) "
                       f"with >{self.row_threshold*100:.0f}% anomalous values")
            
            # Log detailed statistics about what's being dropped
            if logger.isEnabledFor(logging.DEBUG):
                # Find which categories are causing the most anomalies
                debug_mask = self._identify_anomalies(df[rows_to_drop])
                for col in debug_mask.columns[:5]:  # Show first 5 columns
                    anomaly_count = debug_mask[col].sum()
                    if anomaly_count > 0:
                        logger.debug(f"  Column '{col}': {anomaly_count} anomalies in dropped rows")
        
        # Apply row filtering
        df_filtered = df[~rows_to_drop].copy()
        
        # Apply column masking for remaining rows
        if n_rows_to_mask > 0 and len(anomaly_mask.columns) > 0:
            # Get the anomaly mask for remaining rows
            remaining_anomaly_mask = anomaly_mask[~rows_to_drop]
            rows_needing_masking = rows_to_mask[~rows_to_drop]
            
            # Count values to be masked
            values_to_mask = remaining_anomaly_mask[rows_needing_masking].sum().sum()
            if values_to_mask > 0:
                logger.info(f"Masking {values_to_mask} anomalous values in {rows_needing_masking.sum()} rows")
                
                # Get the rows that need masking
                mask_row_indices = rows_needing_masking[rows_needing_masking].index
                
                # Convert to numpy for faster operations
                mask_array = remaining_anomaly_mask.loc[rows_needing_masking].values
                
                # Apply masking using numpy where
                for i, col in enumerate(anomaly_mask.columns):
                    col_mask = mask_array[:, i]
                    if np.any(col_mask):
                        # Use numpy's where for faster masking
                        current_values = df_filtered.loc[mask_row_indices, col].values
                        df_filtered.loc[mask_row_indices, col] = np.where(col_mask, np.nan, current_values)
                
                self.values_masked += values_to_mask
        
        # Update statistics
        self.total_rows_processed += original_rows
        self.rows_filtered += n_rows_to_drop
        
        # Log filtering summary periodically
        if self.total_rows_processed % 100000 == 0:
            logger.info(f"Pre-norm filter summary - Total rows: {self.total_rows_processed}, "
                       f"Filtered: {self.rows_filtered} ({self.rows_filtered/self.total_rows_processed*100:.2f}%), "
                       f"Values masked: {self.values_masked}")
        
        return df_filtered
    
    def _identify_anomalies(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Identify anomalous values based on category-specific ranges.
        
        Returns:
            Boolean DataFrame where True indicates anomalous value
        """
        # Get column names from config
        column_names = self.column_config.get('column_names', [])
        
        # If column_names is empty, try to use DataFrame columns directly
        if not column_names:
            logger.warning("No column names in config, using DataFrame columns directly")
            column_names = df.columns.tolist()
        
        # Only create anomaly mask for numeric columns we're actually checking
        # This ensures we don't count non-numeric columns in our anomaly ratio
        numeric_columns = []
        numeric_indices = []
        for i, col in enumerate(column_names):
            if col in df.columns and df[col].dtype in [np.float32, np.float64, np.int32, np.int64]:
                numeric_columns.append(col)
                numeric_indices.append(i)
        
        logger.info(f"Column config has {len(column_names)} columns, DataFrame has {len(df.columns)} columns")
        logger.info(f"Found {len(numeric_columns)} matching numeric columns to check")
        
        # If no numeric columns found, there might be a mismatch
        if len(numeric_columns) == 0:
            logger.warning("No matching columns between config and DataFrame! Using all DataFrame columns.")
            numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
            logger.info(f"Using {len(numeric_columns)} numeric columns from DataFrame")
        
        # Convert to numpy for faster operations
        numeric_data = df[numeric_columns].values.astype(np.float32)
        anomaly_mask_array = np.zeros(numeric_data.shape, dtype=bool)
        
        # Check for NaN values first (vectorized)
        anomaly_mask_array |= np.isnan(numeric_data)
        
        # Pre-compute price placeholder values as numpy array
        if self.price_placeholder_values:
            price_placeholders_array = np.array(self.price_placeholder_values, dtype=np.float32)
        
        # First check specific ranges for important features
        for feature_name, specific_range in self.specific_ranges.items():
            if specific_range is None:
                continue
                
            min_val, max_val = specific_range
            
            # Find columns matching this specific feature
            feature_col_indices = []
            for i, col in enumerate(numeric_columns):
                if col == feature_name:
                    feature_col_indices.append(i)
            
            if feature_col_indices:
                # Apply specific range check
                feature_data = numeric_data[:, feature_col_indices]
                feature_anomalies = (feature_data < min_val) | (feature_data > max_val)
                
                # Update main anomaly mask
                anomaly_mask_array[:, feature_col_indices] |= feature_anomalies
                
                # Log statistics
                feature_anomaly_count = np.sum(feature_anomalies)
                if feature_anomaly_count > 0:
                    total_values = feature_data.size
                    anomaly_pct = feature_anomaly_count / total_values * 100
                    logger.info(f"{feature_name} specific anomalies: {feature_anomaly_count}/{total_values} ({anomaly_pct:.2f}%)")
        
        # Check each category
        for category, valid_range in self.valid_ranges.items():
            if valid_range is None:
                continue
                
            min_val, max_val = valid_range
            
            # Get indices for this category
            indices_key = f'{category}_indices'
            if indices_key not in self.column_config:
                continue
                
            indices = self.column_config[indices_key]
            if len(indices) == 0:
                continue
            
            # Get column indices for this category
            category_col_indices = []
            for idx in indices:
                if idx < len(column_names):
                    col_name = column_names[idx]
                    if col_name in numeric_columns:
                        # Find the index in numeric_columns
                        numeric_idx = numeric_columns.index(col_name)
                        category_col_indices.append(numeric_idx)
            
            if not category_col_indices:
                # Use pre-computed pattern indices as fallback
                if self._pattern_indices and category in self._pattern_indices:
                    # Map pre-computed indices to numeric_columns indices
                    pattern_indices = self._pattern_indices[category]
                    category_col_indices = []
                    for idx in pattern_indices:
                        if idx < len(column_names):
                            col_name = column_names[idx]
                            if col_name in numeric_columns:
                                numeric_idx = numeric_columns.index(col_name)
                                category_col_indices.append(numeric_idx)
                    
                    if category_col_indices:
                        logger.debug(f"Using pre-computed pattern indices for {category}, found {len(category_col_indices)} columns")
                
                if not category_col_indices:
                    continue
            
            # Skip columns that have specific ranges defined
            filtered_col_indices = []
            for idx in category_col_indices:
                col_name = numeric_columns[idx]
                # Skip if this column has a specific range
                if col_name not in self.specific_ranges or self.specific_ranges[col_name] is None:
                    filtered_col_indices.append(idx)
            
            if not filtered_col_indices:
                continue
            
            # Vectorized anomaly detection for all columns in this category
            category_data = numeric_data[:, filtered_col_indices]
            category_anomalies = (category_data < min_val) | (category_data > max_val)
            
            # Special handling for price placeholders
            if category == 'price' and self.price_placeholder_values:
                # Vectorized check for placeholder values
                for placeholder in price_placeholders_array:
                    category_anomalies |= (category_data == placeholder)
            
            # Update main anomaly mask
            anomaly_mask_array[:, filtered_col_indices] |= category_anomalies
            
            # Log anomaly statistics for this category
            category_anomaly_count = np.sum(category_anomalies)
            if category_anomaly_count > 0:
                total_values = category_data.size
                anomaly_pct = category_anomaly_count / total_values * 100
                logger.info(f"{category} anomalies: {category_anomaly_count}/{total_values} ({anomaly_pct:.2f}%)")
                
                # Show sample of problematic values for debugging
                if len(category_col_indices) > 0:
                    col_idx = category_col_indices[0]
                    col_name = numeric_columns[col_idx]
                    col_anomalies = anomaly_mask_array[:, col_idx]
                    if np.any(col_anomalies):
                        problematic_values = numeric_data[col_anomalies, col_idx][:10].tolist()
                        logger.info(f"  Sample anomalous {category} values from {col_name}: {problematic_values}")
        
        # Convert back to pandas DataFrame for compatibility
        anomaly_mask = pd.DataFrame(anomaly_mask_array, index=df.index, columns=numeric_columns)
        return anomaly_mask
    
    def reset_statistics(self):
        """Reset filtering statistics."""
        self.total_rows_processed = 0
        self.rows_filtered = 0
        self.values_masked = 0
        logger.info("Pre-normalization filter statistics reset")
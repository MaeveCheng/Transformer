"""Column processing utility functions for dataframe operations"""

import numpy as np
from typing import List, Dict, Optional, Union
import logging
import pandas as pd

logger = logging.getLogger(__name__)


def apply_column_mappings(df: pd.DataFrame, data_config=None) -> pd.DataFrame:
    """Apply symbol and data_type mappings to string columns in a DataFrame.
    
    Args:
        df: DataFrame to process
        data_config: DataConfig instance (will create if not provided)
    
    Returns:
        DataFrame with mappings applied and consistent column ordering
    """
    if data_config is None:
        from config import Config
        data_config = Config().data
    
    # Get the mapping dictionary
    mappings = data_config.train_data_columns_mapping
    
    # Create a copy to avoid modifying the original
    df = df.copy()
    
    # Convert string columns - apply mappings for symbol and data_type
    for col in df.columns:
        if df[col].dtype == 'object' or df[col].dtype.name == 'string':
            if col == 'symbol' and 'symbol_mapping' in mappings:
                # Apply symbol mapping
                df[col] = df[col].map(mappings['symbol_mapping']).fillna(0).astype(float)
            elif col == 'data_type' and 'data_type_mapping' in mappings:
                # Apply data_type mapping
                df[col] = df[col].map(mappings['data_type_mapping']).fillna(0).astype(float)
            elif col in ['date', 'seconds_of_day', 'milliseconds']:
                # Ensure time columns are numeric (handle both string and numeric cases)
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(int)
            else:
                # Default conversion for other string columns
                df[col] = 1.0
    
    # Ensure time columns are the correct type even if they were already numeric
    for time_col in ['date', 'seconds_of_day', 'milliseconds']:
        if time_col in df.columns:
            df[time_col] = pd.to_numeric(df[time_col], errors='coerce').fillna(0).astype(int)
    
    # If data_config has feature_columns defined, ensure consistent column ordering
    if hasattr(data_config, 'feature_columns') and data_config.feature_columns:
        # Reorder columns to match feature_columns order (only for columns that exist)
        existing_columns = [col for col in data_config.feature_columns if col in df.columns]
        remaining_columns = [col for col in df.columns if col not in existing_columns]
        
        # Preserve the expected order
        ordered_columns = existing_columns + remaining_columns
        df = df[ordered_columns]
    
    return df


def ensure_feature_columns(df: pd.DataFrame, 
                          feature_columns: List[str],
                          predict_start: Optional[float] = None,
                          predict_end: Optional[float] = None) -> pd.DataFrame:
    """Ensure DataFrame contains all required feature columns in correct order.
    
    Args:
        df: Input DataFrame
        feature_columns: List of required feature column names
        predict_start: Value for predict_start column if needed
        predict_end: Value for predict_end column if needed
    
    Returns:
        DataFrame with all feature columns in correct order
    """
    if not feature_columns:
        return df
    
    ordered_data = {}
    for col in feature_columns:
        if col in df.columns:
            ordered_data[col] = df[col]
        elif col == 'predict_start':
            ordered_data[col] = predict_start  # Don't check for None, keep original behavior
        elif col == 'predict_end':
            ordered_data[col] = predict_end    # Don't check for None, keep original behavior
        else:
            # Fill missing columns with 0.0
            ordered_data[col] = 0.0
    
    return pd.DataFrame(ordered_data)


def extract_price_data(df: pd.DataFrame) -> np.ndarray:
    """Extract price data from DataFrame.
    
    Looks for 'mid_price' column first, then any column containing 'price'.
    Returns zeros if no price column found.
    
    Args:
        df: Input DataFrame
    
    Returns:
        Numpy array of price values
    """
    if 'mid_price' in df.columns:
        return df['mid_price'].values
    
    # Look for any column containing 'price'
    price_cols = [col for col in df.columns if 'price' in col.lower()]
    if price_cols:
        return df[price_cols[0]].values
    
    if 'close' in df.columns:
        return df['close'].values

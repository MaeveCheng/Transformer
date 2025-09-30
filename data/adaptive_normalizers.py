"""
Adaptive Normalizers for Time-Series Data

This module implements adaptive normalization using rolling window statistics
from a pre-computed hourly statistics database. Supports DDP training with
process-local caching.
"""

import threading
from typing import List, Optional, Tuple, Union
import sqlite3
import torch
from dataclasses import dataclass
import logging
import time

@dataclass
class NormalizationStats:
    """Statistics for normalization"""
    mean: torch.Tensor
    std: torch.Tensor
    min_val: Optional[torch.Tensor] = None
    max_val: Optional[torch.Tensor] = None
    n_samples: int = 0

logger = logging.getLogger(__name__)


class AdaptiveNormalizerBase:
    """Base class for adaptive normalizers with thread safety."""
    
    def __init__(self, db_path: str):
        self.db_path = db_path
        self._local = threading.local()  # Thread-local storage
    
    @property
    def conn(self) -> sqlite3.Connection:
        """Get thread-local database connection with multi-process support."""
        if not hasattr(self._local, 'conn') or self._local.conn is None:
            self._local.conn = self._create_connection()
        return self._local.conn
    
    def _create_connection(self) -> sqlite3.Connection:
        """Create a new database connection with retry logic for multi-process access."""
        max_retries = 10
        retry_delay = 0.1
        
        for attempt in range(max_retries):
            try:
                # Create connection with longer timeout for multi-process access
                conn = sqlite3.connect(
                    self.db_path, 
                    timeout=30.0,  # 30 second timeout
                    check_same_thread=False,
                    isolation_level=None  # Autocommit mode
                )
                
                # Enable WAL mode for better concurrent access
                # This needs to be done on a writable connection first
                try:
                    conn.execute("PRAGMA journal_mode=WAL")
                except sqlite3.OperationalError:
                    # Database might already be in WAL mode or read-only
                    pass
                
                # Optimize for read-only access
                conn.execute("PRAGMA query_only = ON")
                conn.execute("PRAGMA temp_store = MEMORY")
                conn.execute("PRAGMA cache_size = -64000")  # 64MB cache
                
                # Additional optimizations for multi-process reads
                conn.execute("PRAGMA synchronous = OFF")  # We're read-only
                conn.execute("PRAGMA locking_mode = NORMAL")
                conn.execute("PRAGMA busy_timeout = 30000")  # 30 second busy timeout
                
                logger.debug(f"Database connection established (attempt {attempt + 1})")
                return conn
                
            except sqlite3.OperationalError as e:
                if "database is locked" in str(e) and attempt < max_retries - 1:
                    logger.warning(f"Database locked, retrying in {retry_delay}s (attempt {attempt + 1}/{max_retries})")
                    time.sleep(retry_delay)
                    retry_delay *= 2  # Exponential backoff
                else:
                    raise
        
        raise sqlite3.OperationalError(f"Failed to connect to database after {max_retries} attempts")
        
    def get_window_stats(self, timestamp: int, feature_names: List[str], 
                        file_path: Optional[str] = None, profile_name: Optional[str] = None) -> NormalizationStats:
        """Get daily cumulative statistics for the day before the given timestamp.
        
        IMPORTANT: This method should be overridden by subclasses to fetch
        the appropriate statistics (regular or log) based on the normalizer type.
        
        Args:
            timestamp: Unix timestamp
            feature_names: List of feature names
            file_path: Optional path to current file being processed
            profile_name: Optional name of current profile
        """
        # Validate inputs
        if timestamp <= 0:
            raise ValueError(f"Invalid timestamp: {timestamp}")
        if not feature_names:
            raise ValueError("No feature names provided")
        
        logger.debug(f"Querying daily cumulative database for timestamp {timestamp}")
        
        # Get the appropriate statistics based on normalizer type
        return self._query_daily_stats(timestamp, feature_names, file_path, profile_name)
    
    def _query_daily_stats(self, timestamp: int, feature_names: List[str], 
                           file_path: Optional[str] = None, profile_name: Optional[str] = None) -> NormalizationStats:
        """Query daily cumulative statistics for the day before the given timestamp."""
        import datetime
        
        # Convert timestamp to date and get previous day
        current_date = datetime.datetime.fromtimestamp(timestamp, tz=datetime.timezone.utc).date()
        previous_date = current_date - datetime.timedelta(days=1)
        date_str = previous_date.isoformat()
        
        logger.debug(f"Fetching daily cumulative stats for {date_str} (day before {current_date})")
        
        # Build query for daily cumulative stats
        feature_columns = []
        for feature in feature_names:
            # Handle column name mapping
            if '_' in feature:
                col_name = feature
            else:
                col_name = feature.replace(' ', '_').replace('-', '_')
            
            # Get columns based on whether we need log stats
            if hasattr(self, 'uses_log_stats') and self.uses_log_stats:
                feature_columns.extend([
                    f'"{col_name}_log_mean"',
                    f'"{col_name}_log_std"',
                    f'"{col_name}_min"',
                    f'"{col_name}_max"'
                ])
            else:
                feature_columns.extend([
                    f'"{col_name}_mean"',
                    f'"{col_name}_std"',
                    f'"{col_name}_min"',
                    f'"{col_name}_max"'
                ])
        
        query = f"""
        SELECT record_count, {', '.join(feature_columns)}
        FROM daily_cumulative_stats_wide
        WHERE day = ?
        """
        
        # Execute query with retry logic
        max_retries = 5
        retry_delay = 0.1
        
        for attempt in range(max_retries):
            try:
                cursor = self.conn.execute(query, [date_str])
                result = cursor.fetchone()
                
                if result is None:
                    # No data for this day, try earlier days
                    context = self._format_context(file_path, profile_name)
                    logger.warning(f"No data for {date_str}, trying earlier dates{context}")
                    return self._get_daily_fallback_stats(previous_date, feature_names, file_path, profile_name)
                
                # Parse the result
                record_count = result[0]
                mean_values = []
                std_values = []
                min_values = []
                max_values = []
                
                result_idx = 1
                for feature in feature_names:
                    if result_idx + 3 < len(result) + 1:
                        mean_val = result[result_idx]
                        std_val = result[result_idx + 1]
                        min_val = result[result_idx + 2]
                        max_val = result[result_idx + 3]
                        
                        if mean_val is None or std_val is None:
                            logger.warning(f"Missing stats for feature {feature}, using defaults")
                            mean_values.append(0.0)
                            std_values.append(1.0)
                            min_values.append(0.0 if min_val is None else float(min_val))
                            max_values.append(1.0 if max_val is None else float(max_val))
                        else:
                            mean_values.append(float(mean_val))
                            std_values.append(float(std_val))
                            min_values.append(float(min_val) if min_val is not None else float(mean_val))
                            max_values.append(float(max_val) if max_val is not None else float(mean_val))
                    else:
                        mean_values.append(0.0)
                        std_values.append(1.0)
                        min_values.append(0.0)
                        max_values.append(1.0)
                    result_idx += 4
                
                return NormalizationStats(
                    mean=torch.tensor(mean_values),
                    std=torch.tensor(std_values),
                    min_val=torch.tensor(min_values),
                    max_val=torch.tensor(max_values),
                    n_samples=record_count
                )
                
            except sqlite3.OperationalError as e:
                if "database is locked" in str(e).lower() and attempt < max_retries - 1:
                    logger.warning(f"Database locked, retrying in {retry_delay}s")
                    time.sleep(retry_delay)
                    retry_delay *= 2
                    # Reset connection
                    try:
                        self._local.conn.close()
                    except:
                        pass
                    self._local.conn = None
                else:
                    raise
        
        raise sqlite3.OperationalError(f"Query failed after {max_retries} attempts")
    
    def _get_daily_fallback_stats(self, start_date, feature_names: List[str], 
                                  file_path: Optional[str] = None, profile_name: Optional[str] = None) -> NormalizationStats:
        """Get fallback daily statistics by trying earlier dates."""
        import datetime
        
        # Try up to 30 days back
        for days_back in range(1, 31):
            fallback_date = start_date - datetime.timedelta(days=days_back)
            date_str = fallback_date.isoformat()
            
            logger.debug(f"Trying fallback date: {date_str}")
            
            # Build query
            feature_columns = []
            for feature in feature_names:
                if '_' in feature:
                    col_name = feature
                else:
                    col_name = feature.replace(' ', '_').replace('-', '_')
                
                if hasattr(self, 'uses_log_stats') and self.uses_log_stats:
                    feature_columns.extend([
                        f'"{col_name}_log_mean"',
                        f'"{col_name}_log_std"',
                        f'"{col_name}_min"',
                        f'"{col_name}_max"'
                    ])
                else:
                    feature_columns.extend([
                        f'"{col_name}_mean"',
                        f'"{col_name}_std"',
                        f'"{col_name}_min"',
                        f'"{col_name}_max"'
                    ])
            
            query = f"""
            SELECT record_count, {', '.join(feature_columns)}
            FROM daily_cumulative_stats_wide
            WHERE day = ?
            """
            
            try:
                cursor = self.conn.execute(query, [date_str])
                result = cursor.fetchone()
                
                if result is not None and result[0] is not None:
                    context = self._format_context(file_path, profile_name)
                    logger.info(f"Found fallback data for {date_str}{context}")
                    # Parse and return the result
                    record_count = result[0]
                    mean_values = []
                    std_values = []
                    min_values = []
                    max_values = []
                    
                    result_idx = 1
                    for feature in feature_names:
                        if result_idx + 3 < len(result) + 1:
                            mean_val = result[result_idx]
                            std_val = result[result_idx + 1]
                            min_val = result[result_idx + 2]
                            max_val = result[result_idx + 3]
                            
                            if mean_val is None or std_val is None:
                                mean_values.append(0.0)
                                std_values.append(1.0)
                                min_values.append(0.0)
                                max_values.append(1.0)
                            else:
                                mean_values.append(float(mean_val))
                                std_values.append(float(std_val))
                                min_values.append(float(min_val) if min_val is not None else float(mean_val))
                                max_values.append(float(max_val) if max_val is not None else float(mean_val))
                        else:
                            mean_values.append(0.0)
                            std_values.append(1.0)
                            min_values.append(0.0)
                            max_values.append(1.0)
                        result_idx += 4
                    
                    return NormalizationStats(
                        mean=torch.tensor(mean_values),
                        std=torch.tensor(std_values),
                        min_val=torch.tensor(min_values),
                        max_val=torch.tensor(max_values),
                        n_samples=record_count
                    )
            except:
                continue
        
        # Ultimate fallback: return default stats
        context = self._format_context(file_path, profile_name)
        logger.error(f"No daily cumulative data found, using default normalization{context}")
        return NormalizationStats(
            mean=torch.zeros(len(feature_names)),
            std=torch.ones(len(feature_names)),
            min_val=torch.zeros(len(feature_names)),
            max_val=torch.ones(len(feature_names)),
            n_samples=0
        )
    
    
    
    
    
    
    def _format_context(self, file_path: Optional[str] = None, profile_name: Optional[str] = None) -> str:
        """Format context information for logging."""
        context_parts = []
        if profile_name:
            context_parts.append(f"Profile: {profile_name}")
        if file_path:
            # Extract just the filename for cleaner logs
            import os
            filename = os.path.basename(file_path) if file_path else file_path
            context_parts.append(f"File: {filename}")
        
        if context_parts:
            return f" [{', '.join(context_parts)}]"
        return ""
    
    def close(self):
        """Close database connections."""
        if hasattr(self._local, 'conn'):
            self._local.conn.close()
    
    def __del__(self):
        """自动关闭数据库连接"""
        self.close()



class AdaptiveNormalizer(AdaptiveNormalizerBase):
    """Unified adaptive normalizer with strategy pattern."""
    
    def __init__(self, db_path: str, method: str = 'standard'):
        """
        Initialize adaptive normalizer with specified method.
        
        Args:
            db_path: Path to statistics database (daily cumulative)
            method: Normalization method ('standard', 'log_standard', 'minmax', 
                    'robust', 'context', 'cyclical', 'none')
        """
        super().__init__(db_path)
        self.method = method
        self.uses_log_stats = method in ['log_standard']
    
    
    def get_window_stats(self, timestamp: int, feature_names: List[str], 
                        file_path: Optional[str] = None, profile_name: Optional[str] = None) -> NormalizationStats:
        """Get statistics for the rolling window before the given timestamp."""
        # For methods that don't use database statistics, return dummy stats
        if self.method in ['context', 'cyclical', 'cyclical_sine', 'none']:
            # Return dummy stats that won't be used
            return NormalizationStats(
                mean=torch.zeros(len(feature_names)),
                std=torch.ones(len(feature_names)),
                n_samples=1
            )
        
        # For other methods, use the parent implementation
        return super().get_window_stats(timestamp, feature_names, file_path, profile_name)
    
    def normalize(self, data: torch.Tensor, stats: NormalizationStats, feature_names: Optional[List[str]] = None, scale_factor: float = 1.0) -> torch.Tensor:
        """Apply normalization based on configured method.
        
        Args:
            data: Input tensor to normalize
            stats: Normalization statistics 
            feature_names: Optional list of feature names for context-aware normalization
            scale_factor: Factor to divide normalized values by (default 1.0, no scaling)
        """
        if self.method == 'standard':
            # Normalize then optionally scale down
            normalized = (data - stats.mean) / (stats.std + 1e-8)
            if scale_factor != 1.0:
                normalized = normalized / scale_factor
            return normalized

        elif self.method == 'log_return':
            # Use log1p transformation
            log_data = torch.log(data + 1e-8)
            normalized = (log_data - stats.mean) / (stats.std + 1e-8)
            if scale_factor != 1.0:
                normalized = normalized / scale_factor
            return normalized
        
        elif self.method == 'log_standard':
            # Use log1p transformation
            log_data = torch.log1p(data)
            normalized = (log_data - stats.mean) / (stats.std + 1e-8)
            if scale_factor != 1.0:
                normalized = normalized / scale_factor
            return normalized
        
        elif self.method == 'minmax':
            if stats.min_val is None or stats.max_val is None:
                raise ValueError("Min/max values required for MinMaxNormalizer")
            
            # Add epsilon to prevent zero range
            eps = 1e-6
            safe_min = stats.min_val - eps
            safe_max = stats.max_val + eps
            range_val = safe_max - safe_min
            
            # Normalize and clamp to [0, 1]
            normalized = (data - safe_min) / range_val
            return torch.clamp(normalized, 0.0, 1.0)
        
        elif self.method == 'robust':
            # Robust normalization using median and IQR
            # Note: For adaptive normalization, we use mean/std from database
            # but interpret them as median/IQR for robust scaling
            # IQR = Q3 - Q1, robust_std ≈ IQR / 1.35
            normalized = (data - stats.mean) / ((stats.std * 1.35) + 1e-8)
            if scale_factor != 1.0:
                normalized = normalized / scale_factor
            return normalized
        
        
        elif self.method == 'context':
            # Fixed transform for window features: (x - 10000.0) / 10000.0
            return (data - 10000.0) / 10000.0
        
        elif self.method == 'cyclical':
            # Cyclical normalization for time features
            # Handle different time features based on their typical ranges
            if feature_names is not None:
                normalized = torch.zeros_like(data)
                for i, feature_name in enumerate(feature_names):
                    if i < data.shape[1]:
                        if feature_name == 'date':
                            # Convert YYMMDD or YYYYMMDD to cyclical encoding
                            dates = data[:, i]
                            
                            # Handle both YYMMDD and YYYYMMDD formats
                            years = torch.where(dates > 1000000, 
                                               torch.floor(dates / 10000),  # YYYYMMDD
                                               2000 + torch.floor(dates / 10000))  # YYMMDD -> YYYY
                            months = torch.floor((dates % 10000) / 100)
                            days = dates % 100
                            
                            # Validate dates
                            valid_months = (months >= 1) & (months <= 12)
                            valid_days = (days >= 1) & (days <= 31)
                            valid_dates = valid_months & valid_days
                            
                            if not torch.all(valid_dates):
                                invalid_count = torch.sum(~valid_dates).item()
                                logger.warning(f"Found {invalid_count} invalid dates, using default day of year")
                            
                            days_before_month = torch.tensor([0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334], 
                                                            dtype=torch.float32, device=data.device)
                            
                            day_of_year = torch.ones_like(dates)  # Default to day 1
                            
                            # Vectorized computation of day_of_year
                            for month_idx in range(1, 13):
                                month_mask = (months == month_idx) & valid_dates
                                if torch.any(month_mask):
                                    day_of_year[month_mask] = days_before_month[month_idx - 1] + days[month_mask]
                            
                            # Convert to cyclical feature (sine encoding)
                            import numpy as np
                            normalized[:, i] = torch.sin(2 * np.pi * day_of_year / 365.0)
                        elif feature_name == 'seconds_of_day':
                            # Normalize to [0, 1] range (86400 seconds in a day)
                            normalized[:, i] = torch.clamp(data[:, i], 0, 86400) / 86400.0
                        elif feature_name == 'milliseconds':
                            # Normalize to [0, 1] range (0-999 milliseconds)
                            normalized[:, i] = torch.clamp(data[:, i], 0, 999) / 1000.0
                        else:
                            # Default: assume already in reasonable range, just clamp to [0, 1]
                            normalized[:, i] = torch.clamp(data[:, i], 0, 1)
                return normalized
            else:
                # Without feature names, default to simple [0, 1] clamping
                return torch.clamp(data, 0, 1)
        
        
        elif self.method == 'none':
            # No normalization (for categorical features)
            return data
        
        else:
            raise ValueError(f"Unknown normalization method: {self.method}")


def create_adaptive_normalizer(method: str, db_path: str) -> AdaptiveNormalizerBase:
    """Factory function to create appropriate adaptive normalizer."""
    return AdaptiveNormalizer(db_path, method)
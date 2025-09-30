"""
CPU-based preprocessing with multicore support and vectorized operations.
Replaces GPU preprocessing while maintaining API compatibility.

Key features:
- Parallel normalization across CPU cores
- Vectorized NumPy operations
- Memory-efficient incremental statistics
- Support for adaptive normalization
- Zero-copy shared memory support
"""

import numpy as np
import torch
from typing import Dict, Optional, Union, Tuple, List, Any
import pandas as pd
import logging
from dataclasses import dataclass
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor
import time
from datetime import datetime
import sqlite3

# Import shared NormalizationStats for consistency
from .adaptive_normalizers import NormalizationStats
# Import FeatureMetadata for feature category tracking
from utils.feature_metadata import FeatureMetadata

logger = logging.getLogger(__name__)


@dataclass
class CPUNormalizationStats:
    """Statistics for normalization on CPU - wraps NormalizationStats with numpy arrays."""
    mean: np.ndarray
    std: np.ndarray
    min_val: Optional[np.ndarray] = None
    max_val: Optional[np.ndarray] = None
    n_samples: int = 0
    
    def to_normalization_stats(self) -> NormalizationStats:
        """Convert to unified NormalizationStats"""
        return NormalizationStats(
            mean=torch.from_numpy(self.mean),
            std=torch.from_numpy(self.std),
            min_val=torch.from_numpy(self.min_val) if self.min_val is not None else None,
            max_val=torch.from_numpy(self.max_val) if self.max_val is not None else None,
            n_samples=self.n_samples
        )




class CPUPreprocessor:
    """CPU-based data preprocessing using multicore processing and vectorized operations."""
    
    def __init__(
        self,
        # Required configuration
        feature_columns: List[str],  # Expected column names
        feature_categories: Dict[str, List[str]],  # Feature categories mapping
        # Main normalization methods (one per feature category)
        price_norm_method: str = 'standard',
        volume_norm_method: str = 'standard',
        count_norm_method: str = 'standard',
        time_norm_method: str = 'cyclical',
        spread_norm_method: str = 'standard',
        volume_imbalance_norm_method: str = 'standard',
        window_norm_method: str = 'context',
        meta_norm_method: str = 'none',
        # General settings
        normalize_all_features: bool = True,
        constant_feature_value: float = 0.0,
        categorical_features: Optional[List[str]] = None,
        adaptive_stats_db: Optional[str] = None,
        adaptive_window_hours: int = 6,
        num_workers: Optional[int] = None,
        chunk_size: int = 10000,
        # Category-specific clipping ranges
        price_clip_range: Optional[Tuple[float, float]] = None,
        volume_clip_range: Optional[Tuple[float, float]] = None,
        count_clip_range: Optional[Tuple[float, float]] = None,
        spread_clip_range: Optional[Tuple[float, float]] = None,
        volume_imbalance_clip_range: Optional[Tuple[float, float]] = None,
        time_clip_range: Optional[Tuple[float, float]] = None,
        window_clip_range: Optional[Tuple[float, float]] = None,
        meta_clip_range: Optional[Tuple[float, float]] = None,
        other_clip_range: Optional[Tuple[float, float]] = None,
        # Pre-normalization valid ranges (for logging suspicious values)
        price_valid_range: Optional[Tuple[float, float]] = None,
        price_placeholder_values: Optional[List[float]] = None,
        # Normalization scaling factors (divide normalized values by this factor)
        price_scale_factor: float = 1.0,
        volume_scale_factor: float = 1.0,
        count_scale_factor: float = 1.0,
        spread_scale_factor: float = 1.0,
        volume_imbalance_scale_factor: float = 1.0,
        time_scale_factor: float = 1.0,
        window_scale_factor: float = 1.0,
        meta_scale_factor: float = 1.0,
        other_scale_factor: float = 1.0,
        # Feature noise for regularization
        feature_noise_std: float = 0.0
    ):
        """
        Initialize CPU preprocessor.
        
        Args:
            price_norm_method: Normalization for all price features (bid/ask prices, mid_price)
            volume_norm_method: Normalization for all volume features (bid/ask volumes)
            count_norm_method: Normalization for order count features
            time_norm_method: Normalization for time features (date, seconds_of_day, milliseconds)
            spread_norm_method: Normalization for spread feature
            volume_imbalance_norm_method: Normalization for volume imbalance feature
            window_norm_method: Normalization for window features (predict_start, predict_end)
            meta_norm_method: Normalization for categorical features (symbol, data_type)
            normalize_all_features: Whether to normalize all features
            constant_feature_value: Value for constant features
            categorical_features: List of categorical feature names
            use_adaptive_normalization: Use adaptive normalization
            adaptive_stats_db: Path to adaptive stats database (daily cumulative)
            num_workers: Number of CPU workers (defaults to CPU count)
            chunk_size: Chunk size for parallel processing
        """
        # Store normalization methods (one per feature category)
        self.price_norm_method = price_norm_method
        self.volume_norm_method = volume_norm_method
        self.count_norm_method = count_norm_method
        self.time_norm_method = time_norm_method
        self.spread_norm_method = spread_norm_method
        self.volume_imbalance_norm_method = volume_imbalance_norm_method
        self.window_norm_method = window_norm_method
        self.meta_norm_method = meta_norm_method
        self.normalize_all_features = normalize_all_features
        self.constant_feature_value = constant_feature_value
        self.categorical_features = categorical_features or ['symbol', 'data_type', 'predict_start', 'predict_end']
        
        # Store clipping ranges
        self.clip_ranges = {
            'price': price_clip_range,
            'volume': volume_clip_range,
            'count': count_clip_range,
            'spread': spread_clip_range,
            'volume_imbalance': volume_imbalance_clip_range,
            'time': time_clip_range,
            'window': window_clip_range,
            'meta': meta_clip_range,
            'other': other_clip_range
        }
        
        # Store scaling factors
        self.scale_factors = {
            'price': price_scale_factor,
            'volume': volume_scale_factor,
            'count': count_scale_factor,
            'spread': spread_scale_factor,
            'volume_imbalance': volume_imbalance_scale_factor,
            'time': time_scale_factor,
            'window': window_scale_factor,
            'meta': meta_scale_factor,
            'other': other_scale_factor
        }
        
        # Store feature noise parameter
        self.feature_noise_std = feature_noise_std
        
        # Store pre-normalization valid ranges for suspicious value detection
        self.price_valid_range = price_valid_range
        self.price_placeholder_values = price_placeholder_values or []
        
        # Adaptive normalization
        self.adaptive_stats_db = adaptive_stats_db
        self.adaptive_window_hours = adaptive_window_hours
        
        # Multiprocessing
        self.num_workers = num_workers or mp.cpu_count()
        self.chunk_size = chunk_size
        self.executor = None
        
        # Use column-based configuration
        self.expected_columns = feature_columns
        self.feature_categories = feature_categories
        
        # Build internal index mapping from column names
        self._column_to_index = {col: i for i, col in enumerate(feature_columns)}
        
        # Build category indices for efficient numpy operations
        self._category_indices = {}
        for category, columns in feature_categories.items():
            indices = [self._column_to_index[col] for col in columns if col in self._column_to_index]
            self._category_indices[category] = np.array(indices, dtype=np.int64)
        
        # Build column config for compatibility
        self.column_config = {
            'column_names': feature_columns,
            **{f"{category}_indices": indices for category, indices in self._category_indices.items()}
        }
        
        # Pre-build feature type mapping using categories
        self._feature_type_map = [
            (category, self._category_indices.get(category, np.array([], dtype=np.int64)))
            for category in ['time', 'price', 'volume', 'count', 'spread', 'volume_imbalance', 'meta', 'window', 'other']
        ]
        
        # No longer need self.stats
        self._fitted = False
        self._columns = []
        self._normalizers_verified = False  # Track if normalizers have been verified
        
        # Initialize components
        self._init_components()
        
        logger.info(f"CPUPreprocessor initialized with {self.num_workers} workers")
    
    def _init_components(self):
        """Initialize preprocessing components"""
        # Initialize adaptive normalizers
        if self.adaptive_stats_db:
            from .adaptive_normalizers import create_adaptive_normalizer
            self.adaptive_normalizers = {
                'time': create_adaptive_normalizer(self.time_norm_method, self.adaptive_stats_db),
                'price': create_adaptive_normalizer(self.price_norm_method, self.adaptive_stats_db),
                'volume': create_adaptive_normalizer(self.volume_norm_method, self.adaptive_stats_db),
                'count': create_adaptive_normalizer(self.count_norm_method, self.adaptive_stats_db),
                'spread': create_adaptive_normalizer(self.spread_norm_method, self.adaptive_stats_db),
                'volume_imbalance': create_adaptive_normalizer(self.volume_imbalance_norm_method, self.adaptive_stats_db),
                'meta': create_adaptive_normalizer(self.meta_norm_method, self.adaptive_stats_db),
                'window': create_adaptive_normalizer(self.window_norm_method, self.adaptive_stats_db),
                'other': create_adaptive_normalizer('standard', self.adaptive_stats_db)
            }
        else:
            self.adaptive_normalizers = None
    
    
    def _extract_numeric_data(self, data: Union[np.ndarray, pd.DataFrame]) -> Tuple[np.ndarray, List[str]]:
        """Extract numeric data and column names."""
        if isinstance(data, pd.DataFrame):
            # Apply column mappings
            from data.pyarrow_utils import apply_column_mappings
            data = apply_column_mappings(data)
            
            # Log original columns before processing
            logger.debug(f"Original columns ({len(data.columns)}): {list(data.columns)[:10]}...")
            
            # Check for symbol and data_type columns
            if 'symbol' in data.columns:
                logger.debug(f"Symbol column found with unique values: {data['symbol'].unique()}")
            if 'data_type' in data.columns:
                logger.debug(f"Data_type column found with unique values: {data['data_type'].unique()}")
            
            # Get numeric columns
            numeric_data = data.select_dtypes(include=[np.number])
            
            columns = numeric_data.columns.tolist()
            array_data = numeric_data.values.astype(np.float32)
            
            # Log NaN statistics if any exist (from pre-norm filtering)
            nan_mask = np.isnan(array_data)
            if nan_mask.any():
                nan_count = nan_mask.sum()
                nan_pct = nan_count / array_data.size * 100
                logger.debug(f"Found {nan_count} NaN values ({nan_pct:.2f}%) in numeric data (likely from pre-norm filtering)")
            
            logger.debug(f"Numeric columns extracted ({len(columns)}): {columns[:10]}...")
            logger.debug(f"Array shape: {array_data.shape}")
            
            return array_data, columns
        else:
            array_data = data.astype(np.float32)
            
            
            return array_data, []
    
    def fit(self, data: Optional[Union[np.ndarray, pd.DataFrame]] = None) -> 'CPUPreprocessor':
        """Simplified fit - only initializes components.
        
        Args:
            data: Optional parameter, kept for backward compatibility (will be ignored)
        """
        # 1. Initialize executor if needed (lazy loading)
        if self.executor is None and self.num_workers > 1:
            self.executor = ThreadPoolExecutor(max_workers=self.num_workers)
            logger.info(f"CPUPreprocessor: Created ThreadPoolExecutor with {self.num_workers} workers")
        
        # 2. Mark as fitted (no verification here)
        self._columns = self.expected_columns  # Use expected columns directly
        self._fitted = True
        self._normalizers_verified = False  # Reset verification flag
        
        logger.info(f"Preprocessor fitted and ready")
        return self
    
    
    
    
    def transform(self, data: Union[np.ndarray, pd.DataFrame], timestamp: Optional[int] = None, 
                 file_path: Optional[str] = None, profile_name: Optional[str] = None) -> torch.Tensor:
        """Transform data using CPU preprocessing
        
        Args:
            data: Input data
            timestamp: Optional timestamp for adaptive normalization
            file_path: Optional file path for logging context
            profile_name: Optional profile name for logging context
        """
        if not self._fitted:
            raise ValueError("Preprocessor must be fitted before transform")
        
        # Store file path and profile for logging
        self._current_file = file_path
        self._current_profile = profile_name
        
        array_data, transform_columns = self._extract_numeric_data(data)
        
        # Validate columns match expected configuration
        if len(transform_columns) > 0:
            if transform_columns != self._columns:
                missing = set(self._columns) - set(transform_columns)
                extra = set(transform_columns) - set(self._columns)
                error_msg = f"Column mismatch! Expected {len(self._columns)} columns, got {len(transform_columns)} columns."
                if missing:
                    error_msg += f"\nMissing: {missing}"
                if extra:
                    error_msg += f"\nExtra: {extra}"
                    
                # Log detailed mismatch info for debugging
                logger.error(f"[CRITICAL] {error_msg}")
                logger.error(f"[CRITICAL] This will cause incorrect normalization and inf values!")
                
                # Find specific differences if same count
                if len(transform_columns) == len(self._columns):
                    for i, (tc, fc) in enumerate(zip(transform_columns, self._columns)):
                        if tc != fc:
                            logger.error(f"[CRITICAL] Column mismatch at index {i}: got '{tc}' vs expected '{fc}'")
                            break  # Just show first mismatch
                else:
                    logger.error(f"[CRITICAL] Transform columns: {transform_columns[:5]}...")
                    logger.error(f"[CRITICAL] Expected columns: {self._columns[:5]}...")
                
                # Raise an exception to stop processing with incorrect columns
                raise ValueError(error_msg)
        
        
        if timestamp is not None:
            return self._adaptive_transform(array_data, timestamp, file_path, profile_name)
        else:
            # Extract timestamp from data if available
            if isinstance(data, pd.DataFrame):
                timestamp = self._extract_data_timestamp(data)
            
            if timestamp is None:
                # Fallback to current time if no timestamp available
                timestamp = int(time.time())
                logger.warning("No timestamp provided or extracted, using current time for adaptive transform")
            
            return self._adaptive_transform(array_data, timestamp, file_path, profile_name)
    
    def _verify_adaptive_normalizers(self, timestamp: int):
        """Verify adaptive normalizers can work properly."""
        logger.debug("Verifying adaptive normalizers on first use...")
        
        for feature_type, indices in self._feature_type_map:
            if len(indices) == 0 or feature_type == 'time':
                continue
            
            feature_names = [self.expected_columns[idx] for idx in indices]
            
            # Handle 'other' categorical filtering
            if feature_type == 'other' and self.normalize_all_features:
                feature_names = [n for n in feature_names if n not in self.categorical_features]
                if not feature_names:
                    continue
            
            # Verify normalizer works
            if feature_type in self.adaptive_normalizers:
                try:
                    _ = self.adaptive_normalizers[feature_type].get_window_stats(
                        timestamp, feature_names, None, None
                    )
                    logger.debug(f"✓ Verified {feature_type} normalizer")
                except Exception as e:
                    logger.error(f"✗ Failed to verify {feature_type} normalizer: {e}")
                    raise
        
        logger.debug("All adaptive normalizers verified successfully")
    
    def _adaptive_transform(self, data: np.ndarray, timestamp: int, 
                           file_path: Optional[str] = None, profile_name: Optional[str] = None) -> torch.Tensor:
        """Transform using adaptive normalization with proper feature grouping."""
        # Verify normalizers on first use (lazy verification)
        if self.adaptive_normalizers and not self._normalizers_verified:
            self._verify_adaptive_normalizers(timestamp)
            self._normalizers_verified = True
        
        # Work with numpy array first, convert to torch at the end
        transformed = data.astype(np.float32, copy=True)
        
        # Log adaptive normalization usage only in debug mode
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"Applying adaptive normalization for timestamp {timestamp}")
        
        # Process each feature type using pre-computed mapping
        for feature_type, indices in self._feature_type_map:
            if len(indices) == 0 or not self.adaptive_normalizers or feature_type not in self.adaptive_normalizers:
                continue
            
            
            # Get feature names for this type
            feature_names = [self.column_config['column_names'][i] for i in indices]
            
            
            # Get normalizer
            normalizer = self.adaptive_normalizers[feature_type]
            
            # Get feature data (numpy slice, no copy)
            feature_data = transformed[:, indices]
            
            # Only log raw data statistics in specific cases to reduce overhead
            should_log_stats = logger.isEnabledFor(logging.DEBUG) or (feature_type == 'price' and self.price_valid_range)
            if should_log_stats and len(feature_data) > 0:
                # Create mask for non-NaN values
                valid_mask = ~np.isnan(feature_data)
                nan_count = (~valid_mask).sum()
                
                if valid_mask.any():
                    valid_data = feature_data[valid_mask]
                    raw_min = np.min(valid_data)
                    raw_max = np.max(valid_data)
                        
                    # Calculate NaN percentage
                    nan_pct = 100.0 * nan_count / feature_data.size
                        
                    if nan_pct > 0.01:  # Warn if more than 0.01% NaN
                        raw_mean = np.mean(valid_data)
                        raw_std = np.std(valid_data)
                        logger.warning(f"{feature_type} RAW data stats - Min: {raw_min:.4f}, Max: {raw_max:.4f}, Mean: {raw_mean:.4f}, Std: {raw_std:.4f} (NaN: {nan_count}, {nan_pct:.2f}%)")
                    else:
                        logger.debug(f"{feature_type} RAW data stats - Min: {raw_min:.4f}, Max: {raw_max:.4f}, Mean: {np.mean(valid_data):.4f}, Std: {np.std(valid_data):.4f} (NaN: {nan_count})")
                else:
                    logger.warning(f"{feature_type} RAW data contains only NaN values!")
                
                # Check for suspicious values in price data using config thresholds
                if feature_type == 'price' and 'valid_data' in locals() and len(valid_data) > 0:
                    # Get price valid range from config if available
                    price_range = self.price_valid_range
                    price_placeholders = self.price_placeholder_values
                    
                    if price_range and raw_min < price_range[0]:
                        file_info = f" [File: {getattr(self, '_current_file', 'unknown')}]" if hasattr(self, '_current_file') else ""
                        logger.warning(f"WARNING: {feature_type} has values below configured minimum (min={raw_min:.2f} < {price_range[0]:.2f}). Likely placeholder/missing data!{file_info}")
                        
                        # Count how many values are below threshold
                        suspicious_count = (valid_data < price_range[0]).sum()
                        suspicious_pct = 100.0 * suspicious_count / len(valid_data)
                        logger.warning(f"  {suspicious_count} values ({suspicious_pct:.2f}%) are below {price_range[0]}")
                    
                    # Check for exact placeholder values
                    if price_placeholders:
                        # Vectorized check for all placeholders at once
                        placeholder_mask = np.zeros_like(valid_data, dtype=bool)
                        for placeholder in price_placeholders:
                            placeholder_mask |= (valid_data == placeholder)
                        
                        placeholder_count = placeholder_mask.sum()
                        if placeholder_count > 0:
                            placeholder_pct = 100.0 * placeholder_count / len(valid_data)
                            logger.warning(f"  Found {placeholder_count} placeholder values {price_placeholders} ({placeholder_pct:.2f}%)")
                
                # Log sample values
                if logger.isEnabledFor(logging.DEBUG) and len(feature_data) > 0:
                    sample = feature_data[0, :min(3, feature_data.shape[1])]
                    logger.debug(f"{feature_type} sample raw values: {sample}")
            
            try:
                # Skip getting stats for methods that don't need them
                if normalizer.method not in ['context', 'cyclical', 'cyclical_sine', 'none']:
                    adaptive_stats = normalizer.get_window_stats(timestamp, feature_names, file_path, profile_name)
                    
                    # Log the normalization stats being used
                    logger.debug(f"{feature_type} normalization stats - Mean: {adaptive_stats.mean[:3]}, Std: {adaptive_stats.std[:3]}, n_samples: {adaptive_stats.n_samples}")
                else:
                    # Use dummy stats
                    adaptive_stats = NormalizationStats(
                        mean=torch.zeros(len(feature_names)),
                        std=torch.ones(len(feature_names)),
                        n_samples=1
                    )
                
                # Apply normalization and clipping in one pass
                # First convert feature_data to torch for normalizer (required by API)
                feature_data_torch = torch.from_numpy(feature_data).float()
                scale_factor = self.scale_factors.get(feature_type, 1.0)
                normalized_data_torch = normalizer.normalize(feature_data_torch, adaptive_stats, feature_names, scale_factor=scale_factor)
                
                # Convert back to numpy for efficient clipping
                normalized_data = normalized_data_torch.numpy()
                
                # Apply category-specific clipping in-place
                clip_range = self.clip_ranges.get(feature_type)
                
                if clip_range is not None:
                    clip_min, clip_max = clip_range
                    
                    # Only count values if logging is enabled to save computation
                    if logger.isEnabledFor(logging.DEBUG) or logger.isEnabledFor(logging.WARNING):
                        below_min = (normalized_data < clip_min).sum()
                        above_max = (normalized_data > clip_max).sum()
                    else:
                        below_min = above_max = 0
                    
                    # Apply clipping in-place for efficiency
                    np.clip(normalized_data, clip_min, clip_max, out=normalized_data)
                    
                    # Log if any values were clipped
                    if below_min > 0 or above_max > 0:
                        total_values = normalized_data.size
                        clip_pct = 100.0 * (below_min + above_max) / total_values
                        norm_method = getattr(self, f'{feature_type}_norm_method', 'unknown')
                        
                        if clip_pct > 0.2:  # Only warn if more than 0.2% clipped
                            logger.warning(
                                f"Clipped {feature_type} features ({norm_method} normalization): "
                                f"{below_min} below {clip_min}, {above_max} above {clip_max} "
                                f"({clip_pct:.2f}% total)"
                            )
                        else:  # Debug for any clipping below threshold
                            logger.debug(
                                f"Clipped {feature_type} features ({norm_method} normalization): "
                                f"{below_min} below {clip_min}, {above_max} above {clip_max} "
                                f"({clip_pct:.2f}% total)"
                            )
                
                # Assign normalized data back to transformed array
                transformed[:, indices] = normalized_data
                
                # Log sample after normalization only in debug mode
                if logger.isEnabledFor(logging.DEBUG) and len(normalized_data) > 0:
                    sample = normalized_data[0, :min(3, normalized_data.shape[1])]
                    logger.debug(f"{feature_type} after {normalizer.method} normalization: {sample}")
                
                
            except sqlite3.OperationalError as e:
                # Database query errors - use fallback
                logger.warning(f"Database error for {feature_type} features: {e}")
                logger.debug(f"Using fallback normalization for {feature_type}")
                # The adaptive normalizer already has fallback mechanisms
                # that will return default stats (mean=0, std=1)
                pass
            except KeyError as e:
                # Missing column in database
                logger.warning(f"Missing column statistics for {feature_type}: {e}")
                logger.debug(f"Using identity transformation for {feature_type}")
                # Keep original values if no adaptive stats available
            except Exception as e:
                # Other unexpected errors
                logger.error(f"Unexpected error normalizing {feature_type} features: {e}")
                logger.error(f"Feature names: {feature_names}")
                # Keep original values on error
                pass
        
        # Note: Category-specific clipping is now done within each feature type above
        # The global _clip_outliers method is no longer needed
        
        # Handle NaN values (including those from pre-normalization filtering)
        # Replace NaN with 0.0 (neutral value after normalization)
        nan_mask = np.isnan(transformed)
        nan_count = nan_mask.sum()
        if nan_count > 0:
            logger.debug(f"Replacing {nan_count} NaN values with 0.0 (likely from pre-norm filtering)")
            transformed[nan_mask] = 0.0
        
        # Also handle inf values
        inf_mask = np.isinf(transformed)
        inf_count = inf_mask.sum()
        if inf_count > 0:
            logger.debug(f"Replacing {inf_count} inf values with 0.0")
            transformed[inf_mask] = 0.0
        
        # Add feature noise if configured (for regularization)
        if self.feature_noise_std > 0:
            noise = np.random.normal(0, self.feature_noise_std, transformed.shape)
            transformed += noise
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(f"Added random noise with std={self.feature_noise_std} to features")
        
        # Convert to torch tensor at the very end
        transformed_torch = torch.from_numpy(transformed).float()
        
        # Log sample of transformed data
        if logger.isEnabledFor(logging.DEBUG) and transformed_torch.shape[0] > 0:
            logger.debug(f"Sample transformed data - mean: {transformed_torch[0].mean():.4f}, std: {transformed_torch[0].std():.4f}")
        
        return transformed_torch
    
    def fit_transform(self, data: Union[np.ndarray, pd.DataFrame]) -> torch.Tensor:
        """Fit and transform in one step"""
        return self.fit().transform(data)
    
    def cleanup(self):
        """Clean up resources"""
        if self.executor is not None:
            self.executor.shutdown(wait=True)
            self.executor = None
    
    def _get_normalization_method(self, feature_type: str) -> str:
        """Get normalization method for feature type."""
        
        method_map = {
            'time': self.time_norm_method,           # 'cyclical'
            'price': self.price_norm_method,         # 'log_return'
            'volume': self.volume_norm_method,       # 'log_standard'
            'count': self.count_norm_method,         # 'minmax'
            'spread': self.spread_norm_method,       # 'standard' for spread
            'volume_imbalance': self.volume_imbalance_norm_method,  # 'standard' for volume_imbalance
            'meta': 'none',                          # No normalization for categorical
            'window': self.window_norm_method        # 'context' - configured in DataConfig
        }
        
        return method_map.get(feature_type, 'standard')

    def _extract_data_timestamp(self, df: pd.DataFrame) -> Optional[int]:
        """Extract timestamp from dataframe - same logic as SimplifiedRandomDataset"""
        if 'timestamp' in df.columns:
            return int(df['timestamp'].iloc[0])
        elif 'datetime' in df.columns:
            datetime_val = df['datetime'].iloc[0]
            try:
                # 如果是字符串格式的 datetime
                if isinstance(datetime_val, str):
                    # 解析 UTC 时间字符串
                    dt = pd.to_datetime(datetime_val, utc=True)
                # 如果已经是 pandas Timestamp 或 datetime 对象
                elif hasattr(datetime_val, 'timestamp'):
                    dt = datetime_val
                    # 确保是 UTC 时区
                    if dt.tzinfo is None:
                        dt = pd.to_datetime(dt, utc=True)
                else:
                    logger.warning(f"Unexpected datetime type: {type(datetime_val)}")
                    return None

                # 转换为 Unix 时间戳（秒）
                return int(dt.timestamp())
            except Exception as e:
                logger.warning(f"Error parsing datetime={datetime_val}: {e}")
                return None
        elif 'date' in df.columns and 'seconds_of_day' in df.columns:
            date_val = df['date'].iloc[0]
            seconds_val = df['seconds_of_day'].iloc[0]
            
            try:
                # Handle potential string-to-int conversion for seconds_of_day
                if isinstance(seconds_val, str):
                    seconds = int(float(seconds_val))
                else:
                    seconds = int(seconds_val)
                
                # Handle both string and numeric date formats
                if isinstance(date_val, str):
                    date = int(float(date_val))  # Handle "1.0" -> 1
                else:
                    date = int(date_val)
                
                # Validate date is in reasonable range
                if date < 100101:  # Minimum valid date: 10-01-01 (YYMMDD) or 0001-01-01 (YYYYMMDD)
                    logger.warning(f"Date value {date} too small to be valid YYMMDD/YYYYMMDD format")
                    return None
                
                # Parse date - handle both YYMMDD and YYYYMMDD formats
                if date > 1000000:  # YYYYMMDD format
                    year = date // 10000
                    month = (date % 10000) // 100
                    day = date % 100
                else:  # YYMMDD format
                    year = 2000 + (date // 10000)
                    month = (date % 10000) // 100
                    day = date % 100
                
                # Validate parsed values
                if not (1 <= month <= 12):
                    logger.warning(f"Invalid month {month} from date {date_val} (parsed as {date})")
                    return None
                    
                if not (1 <= day <= 31):
                    logger.warning(f"Invalid day {day} from date {date_val} (parsed as {date})")
                    return None
                
                dt = datetime(year, month, day)
                base_timestamp = int(dt.timestamp())
                return base_timestamp + seconds
            except Exception as e:
                logger.warning(f"Error parsing timestamp from date={date_val}, seconds={seconds_val}: {e}")
                return None
        return None


class CPUPreprocessorCompat:
    """Compatibility wrapper for CPUPreprocessor to match GPU API"""
    
    def __init__(self, **kwargs):
        # Ensure normalize_all_features is True by default
        if 'normalize_all_features' not in kwargs:
            kwargs['normalize_all_features'] = True
        if 'constant_feature_value' not in kwargs:
            kwargs['constant_feature_value'] = 0.0
            
        self.cpu_preprocessor = CPUPreprocessor(**kwargs)
        self.expected_feature_count = None
    
    def fit(self, df: Optional[pd.DataFrame] = None) -> 'CPUPreprocessorCompat':
        """Fit preprocessor maintaining pandas compatibility
        
        Args:
            df: Optional DataFrame, kept for backward compatibility (will be ignored)
        """
        self.cpu_preprocessor.fit()  # No longer pass data
        self._columns = self.cpu_preprocessor._columns
        self.expected_feature_count = len(self._columns)
        return self
    
    def transform(self, df: pd.DataFrame, timestamp: Optional[int] = None, 
                 file_path: Optional[str] = None, profile_name: Optional[str] = None) -> torch.Tensor:
        """Transform data and return tensor"""
        return self.cpu_preprocessor.transform(df, timestamp, file_path, profile_name)
    
    def fit_transform(self, df: pd.DataFrame) -> torch.Tensor:
        """Fit and transform in one step"""
        return self.fit().transform(df)
    
    
    def to_tensor(self, df: pd.DataFrame) -> torch.Tensor:
        """Transform data and return tensor directly"""
        return self.cpu_preprocessor.transform(df)
    
    def cleanup(self):
        """Clean up resources"""
        self.cpu_preprocessor.cleanup()

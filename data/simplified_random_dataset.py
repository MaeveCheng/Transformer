"""
Simplified random dataset for efficient data loading.

Key features:
- Minimal initialization overhead
- True randomization at every level
- No complex state or thread locks
- Direct parquet reading with row slicing
- Worker-independent operation
"""

import torch
from torch.utils.data import IterableDataset
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from typing import Iterator, Dict, List, Optional, Any
import logging
import random
import os
import re
from datetime import datetime, timedelta
import multiprocessing as mp

# Config imports removed - using dict-like access
from .cpu_window_creator import CPUWindowCreator
from .cpu_preprocessor import CPUPreprocessorCompat
from .pyarrow_utils import apply_column_mappings, ensure_feature_columns, extract_price_data
from .pre_normalization_filter import PreNormalizationFilter

logger = logging.getLogger(__name__)


class SimplifiedRandomDataset(IterableDataset):
    """
    Stride-based dataset with systematic sampling.
    
    Features:
    - No initialization overhead (no file scanning)
    - Each worker independently selects random files and chunks
    - Direct parquet reading with efficient row slicing
    - Stride-based systematic sampling (no duplicates)
    - No complex state management
    
    Cross-file Reading:
    - Automatically reads from previous/next day files when needed
    - Uses date pattern in filenames (YYYY-MM-DD) to find neighbors
    - If previous file missing: shifts chunk position forward to ensure full lookback
    - If next file missing: adjusts position to stay within current file
    - Logs when positions are adjusted due to missing files
    """
    
    def __init__(
        self,
        file_list: List[str],
        data_config,
        profiles: List,  # Required: Profile configurations
        profile_files: Dict[str, List[str]],  # Required: Files per profile
        profile_configs: Optional[Dict[str, Any]] = None,
        rank: int = 0,
        world_size: int = 1,
        seed: Optional[int] = None,
        train_date_start: Optional[str] = None,
        train_date_end: Optional[str] = None,
        date_group_size: int = 10,
        date_group_lookback: int = 5,
        iter_every_group: int = 100,
        date_group_start_date: Optional[str] = None,
        binary_classification_config=None,
        model_config=None
    ):
        """
        Initialize simplified dataset.
        
        Args:
            file_list: List of parquet files
            data_config: Data configuration
            profiles: List of profile configurations (required)
            profile_files: Dictionary mapping profile names to file lists (required)
            profile_configs: Optional profile-specific configurations
            rank: Process rank for distributed training
            world_size: Total number of processes
            seed: Random seed (None for true randomness)
            train_date_start: Start date for training data range (YYYY-MM-DD)
            train_date_end: End date for training data range (YYYY-MM-DD)
            date_group_size: Size of date window in days
            date_group_lookback: Overlap between windows in days
            iter_every_group: Number of iterations before sliding window
            date_group_start_date: Starting date for the first window (YYYY-MM-DD)
        """
        # Validate required parameters
        if not profiles:
            raise ValueError("profiles parameter is required and cannot be empty")
        if not profile_files:
            raise ValueError("profile_files parameter is required and cannot be empty")
        
        # Store minimal state
        self.file_list = file_list
        self.data_config = data_config
        # chunking_config removed - each profile has its own chunk_size_lines
        self.rank = rank
        self.world_size = world_size
        self.binary_classification_config = binary_classification_config
        self.model_config = model_config
        
        # Store profile configurations
        self.profiles = profiles
        self.profile_files = profile_files
        self.profile_configs = profile_configs or {}
        
        # Initialize component storage (will be populated during iteration)
        self.profile_preprocessors = {}
        # No profile_window_creators needed - created dynamically per batch
        self.profile_pre_norm_filters = {}
        
        # Normalize weights
        total_weight = sum(p.weight for p in self.profiles)
        self.profile_weights = [p.weight / total_weight for p in self.profiles]
        
        # Extract key parameters
        # Note: chunk_size is now profile-specific, will be retrieved from profile
        self.feature_columns = data_config.feature_columns
        
        # Random seed
        if seed is None:
            seed = int.from_bytes(os.urandom(4), 'big')
        self.base_seed = seed
        
        # Store date range parameters
        self.train_date_start = train_date_start
        self.train_date_end = train_date_end
        self.date_group_size = date_group_size
        self.date_group_lookback = date_group_lookback
        self.iter_every_group = iter_every_group
        self.date_group_start_date = date_group_start_date
        
        # Parse dates if provided
        if self.train_date_start and self.train_date_end:
            self.train_date_start_dt = datetime.strptime(self.train_date_start, '%Y-%m-%d')
            self.train_date_end_dt = datetime.strptime(self.train_date_end, '%Y-%m-%d')
        else:
            self.train_date_start_dt = None
            self.train_date_end_dt = None
        
        total_files = sum(len(files) for files in self.profile_files.values())
        logger.info(f"SimplifiedRandomDataset initialized: {len(self.profiles)} profiles, {total_files} total files")
        if self.train_date_start and self.train_date_end:
            logger.info(f"Date range training enabled: {self.train_date_start} to {self.train_date_end}, "
                       f"window={self.date_group_size} days, lookback={self.date_group_lookback} days, "
                       f"slide every {self.iter_every_group} iterations")
            
            # Check and warn if any profiles have no files in the date range
            for profile in self.profiles:
                profile_files = self.profile_files.get(profile.name, [])
                files_in_range = []
                for file_path in profile_files:
                    file_date = self._extract_date_from_filename(file_path)
                    if file_date and self.train_date_start_dt <= file_date <= self.train_date_end_dt:
                        files_in_range.append(file_path)
                
                if not files_in_range:
                    logger.warning(f"Profile '{profile.name}' has NO files in date range [{self.train_date_start}, {self.train_date_end}]")
                else:
                    logger.info(f"Profile '{profile.name}' has {len(files_in_range)} files in date range")
    
    def _initialize_worker(self) -> tuple:
        """Initialize worker-specific settings and random state.
        
        Returns:
            tuple: (worker_id, num_workers, rng, np_rng, worker_seed, worker_info, 
                   current_start_date, current_end_date, global_batch_counter, current_window_files)
        """
        # Get worker info
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            worker_id = 0
            num_workers = 1
        else:
            worker_id = worker_info.id
            num_workers = worker_info.num_workers
        
        # CRITICAL FIX: Use truly random seed to avoid deterministic bias across ranks
        # Each worker gets a unique random seed that changes every time
        # This prevents systematic profile selection bias in distributed training
        worker_seed = int.from_bytes(os.urandom(4), 'big') + worker_id
        
        # Alternative: If you need some reproducibility, add epoch tracking:
        # worker_seed = self.base_seed + self.rank * 1000 + worker_id + (epoch * 100000)
        
        rng = random.Random(worker_seed)
        np_rng = np.random.RandomState(worker_seed)
        
        # Initialize date window if enabled
        current_start_date = None
        current_end_date = None
        current_window_files = self.profile_files  # Default to all files
        
        if self.train_date_start_dt and self.train_date_end_dt:
            # Use configured start date if provided, otherwise disable date windowing
            if self.date_group_start_date:
                # Parse the configured start date
                date_group_start_dt = datetime.strptime(self.date_group_start_date, '%Y-%m-%d')
                
                # Clamp to valid range
                if date_group_start_dt < self.train_date_start_dt:
                    logger.warning(f"date_group_start_date ({self.date_group_start_date}) is before train_date_start ({self.train_date_start}), clamping to train_date_start")
                    current_start_date = self.train_date_start_dt
                elif date_group_start_dt > self.train_date_end_dt:
                    logger.warning(f"date_group_start_date ({self.date_group_start_date}) is after train_date_end ({self.train_date_end}), clamping to train_date_end")
                    current_start_date = self.train_date_end_dt
                else:
                    current_start_date = date_group_start_dt
                
                current_end_date = min(current_start_date + timedelta(days=self.date_group_size - 1), self.train_date_end_dt)
            else:
                # No date_group_start_date specified, disable date windowing
                logger.info(f"No date_group_start_date specified, date windowing disabled for worker {worker_id}")
                current_start_date = None
                current_end_date = None
                current_window_files = self.profile_files
            
            # Only filter files if date windowing is enabled
            if current_start_date and current_end_date:
                # Filter files for initial window
                current_window_files = self._filter_files_by_date_window(current_start_date, current_end_date)
                
                # Log initial window stats
                total_files = sum(len(files) for files in current_window_files.values())
                #files_per_profile = ", ".join([f"{name}: {len(files)}" 
                #                             for name, files in current_window_files.items() if files])
                
                logger.info(f"Worker {worker_id}/{num_workers} (rank {self.rank}) initialized with date window "
                           f"[{current_start_date.strftime('%Y-%m-%d')}, {current_end_date.strftime('%Y-%m-%d')}]. "
                           #f"Total files: {total_files} ({files_per_profile})")
                           f"Total files: {total_files}")
        
        # Log once per worker with rank info for debugging
        logger.info(f"Worker {worker_id}/{num_workers} (rank {self.rank}) started with random seed {worker_seed}")
        
        return (worker_id, num_workers, rng, np_rng, current_start_date, current_end_date, current_window_files)
    
    def __iter__(self) -> Iterator[Dict[str, torch.Tensor]]:
        """
        Iterate over batches using stride-based sampling.
        
        Everything is initialized here to avoid pickling issues.
        """
        # Initialize worker
        (worker_id, num_workers, rng, np_rng, current_start_date, current_end_date, current_window_files) = self._initialize_worker()

        global_batch_counter = 0  # Total batches generated by this worker
        batch_counter = 0  # Batches generated in current date window
        
        # Simple file metadata cache (worker-local)
        file_metadata = {}
        
        # Stride usage tracking for logging
        stride_counts = {}
                
        # Infinite iteration
        while True:
            # Select profile and file
            profile_data = self._select_profile_and_file(
                rng, np_rng, num_workers, current_window_files
            )
            
            # Skip iteration if no valid profiles found
            if profile_data is None:
                continue
            
            # Extract values from profile data
            file_path = profile_data['file_path']
            # logger.info(f"Worker {worker_id}: Selected file: {os.path.basename(file_path)}")
            profile = profile_data['profile']
            profile_config = profile_data['profile_config']
            preprocessor = profile_data['preprocessor']
            pre_norm_filter = profile_data['pre_norm_filter']
            predict_start = profile_data['predict_start']
            predict_end = profile_data['predict_end']
            normalisation = profile_data['normalisation']
            min_stride = profile_data['min_stride']
            max_stride = profile_data['max_stride']
            exponential_lookback = profile_data['exponential_lookback']
            internal_workers = profile_data['internal_workers']
            chunk_size = profile_data['chunk_size']  # Get chunk_size from profile_data
            # Get file metadata (cached)
            metadata, success = self._get_or_cache_file_metadata(file_path, file_metadata)
            if not success:
                continue
            total_rows = metadata['num_rows']
            
            # Need enough rows for chunk + lookback + lookahead
            min_rows_needed = chunk_size + exponential_lookback + predict_end
            if total_rows < min_rows_needed:
                continue
            
            # Check for consecutive neighbor files (already checks date consecutiveness)
            prev_file = self._get_neighbor_file(file_path, -1)
            next_file = self._get_neighbor_file(file_path, 1)
            
            # Determine valid range based on consecutive neighbor availability
            if prev_file:
                min_valid_start = 0  # Can start anywhere, will read from prev day
            else:
                min_valid_start = exponential_lookback  # Must have enough lookback
            
            if next_file:
                max_valid_start = total_rows - 1  # Can end anywhere, will read from next day
            else:
                max_valid_start = total_rows - chunk_size - predict_end  # Must fit in current file
            
            # Check if valid range exists
            if max_valid_start < min_valid_start:
                continue  # File too small
            
            # Random starting position (only from valid range)
            start_pos_original = rng.randint(min_valid_start, max_valid_start)
            
            # Calculate read positions and handle file boundaries
            positions = self._calculate_read_positions(
                file_path, start_pos_original, total_rows,
                exponential_lookback, predict_end, chunk_size,
                prev_file, next_file
            )
            
            start_pos = positions['start_pos']
            read_start = positions['read_start']
            read_end = positions['read_end']
            prev_file = positions['prev_file']
            next_file = positions['next_file']
            need_lookback_rows = positions['need_lookback_rows']
            need_lookahead_rows = positions['need_lookahead_rows']
            
            try:
                # Read chunk data from parquet file
                df = self._read_chunk_data(file_path, read_start, read_end, metadata)
                
                # Add normalisation column directly to DataFrame
                df['normalisation'] = normalisation
                
                # Handle cross-file data if needed
                df = self._handle_cross_file_data(
                    df, file_path, prev_file, next_file,
                    need_lookback_rows, need_lookahead_rows, normalisation
                )
                
                # Preprocess chunk with normalization
                preprocessed, price_data, should_skip = self._preprocess_chunk(
                    df, pre_norm_filter, predict_start, predict_end, file_path, chunk_size, preprocessor, profile.name
                )
                
                if should_skip:
                    continue
                
                # Generate windows with dynamic sample offsets
                for batch in self._generate_windows_with_dynamic_offsets(
                    profile, profile_config, preprocessor, preprocessed, price_data,
                    df, file_path, start_pos, read_start, read_end, chunk_size,
                    prev_file, next_file, need_lookback_rows, need_lookahead_rows,
                    min_stride, max_stride, exponential_lookback, predict_end,
                    rng, np_rng, stride_counts, worker_id, internal_workers
                ):
                    yield batch
                    
                    # Update counters after yielding each batch
                    global_batch_counter += 1
                    batch_counter += 1
                    #if self.rank == 0:
                        #logger.error(f"******** rank {self.rank} - num: {num_workers}- global: {global_batch_counter}, current: {batch_counter}")
                    
                    # Check for window sliding after each batch
                    if current_start_date is not None:
                        # Check if we need to slide the window
                        if batch_counter * num_workers > self.iter_every_group:
                            # Calculate slide distance
                            slide_days = self.date_group_size - self.date_group_lookback
                        
                            # Slide window forward
                            current_start_date = current_start_date + timedelta(days=slide_days)
                            batch_counter = 0
                        
                            # Check if we've gone past the end date
                            if current_start_date > self.train_date_end_dt:
                                # Loop back to the beginning
                                current_start_date = self.train_date_start_dt
                                logger.info(f"Worker {worker_id}: Date window looped back to {current_start_date.strftime('%Y-%m-%d')}")
                        
                            # Update end date
                            current_end_date = min(current_start_date + timedelta(days=self.date_group_size - 1), 
                                                 self.train_date_end_dt)
                        
                            # Re-filter files for new window
                            current_window_files = self._filter_files_by_date_window(current_start_date, current_end_date)
                        
                            # Check if any files are available in the new window
                            if not self._has_any_files_in_window(current_window_files):
                                logger.warning(f"Worker {worker_id}: No files available in window "
                                            f"[{current_start_date.strftime('%Y-%m-%d')}, {current_end_date.strftime('%Y-%m-%d')}], "
                                            f"will continue sliding on next iteration")
                        
                            # Log window change with file counts
                            total_files = sum(len(files) for files in current_window_files.values())
                            files_per_profile = ", ".join([f"{name}: {len(files)}" for name, files in current_window_files.items() if files])
                            logger.info(f"Worker {worker_id}: Slide date window to "
                                      f"[{current_start_date.strftime('%Y-%m-%d')}, {current_end_date.strftime('%Y-%m-%d')}] "
                                      f"after {self.iter_every_group} batches. "
                                      f"Total files: {total_files} ({files_per_profile})")
                        
            except Exception as e:
                logger.error(
                    f"Failed to process chunk from {file_path}: {e}. "
                    f"Data integrity cannot be guaranteed. Please check the data file."
                )
                continue
    
    
    def _select_profile_and_file(self, rng, np_rng, num_workers, current_window_files=None) -> dict:
        """Select profile and random file based on profile weights.
        
        Args:
            rng: Random generator
            np_rng: Numpy random generator
            num_workers: Number of DataLoader workers
            current_window_files: Dict of files filtered by current date window (optional)
        
        Returns:
            dict: Contains all needed parameters
        """
        # Build list of profiles with available files
        valid_profiles = []
        valid_weights = []
        for i, profile in enumerate(self.profiles):
            profile_files = current_window_files.get(profile.name, [])
            if profile_files:
                valid_profiles.append((i, profile, profile_files))
                valid_weights.append(self.profile_weights[i])
        
        # Select profile from valid ones
        if valid_profiles:
            # Normalize weights for valid profiles
            total_weight = sum(valid_weights)
            normalized_weights = [w/total_weight for w in valid_weights]
            
            # Choose profile based on weights
            choice_idx = np_rng.choice(len(valid_profiles), p=normalized_weights)
            profile_idx, profile, profile_file_list = valid_profiles[choice_idx]
        else:
            # No files in current window - skip this iteration
            logger.warning("No profiles have files in current date window, skipping iteration")
            return None  # Signal to skip this iteration

        # Get profile-specific components
        profile_config = self.profile_configs.get(profile.name, self.data_config)

        # Calculate internal workers once here
        if num_workers > 1:
            cpu_count = mp.cpu_count()
            internal_workers = max(1, cpu_count // num_workers)
        else:
            internal_workers = mp.cpu_count()
        
        # Initialize profile-specific processors if not already initialized
        if profile.name not in self.profile_preprocessors:
            if num_workers > 1:
                logger.info(f"Profile '{profile.name}': Using {internal_workers} internal threads "
                          f"(CPU={cpu_count}, DataLoader workers={num_workers}, Total parallelization={num_workers * internal_workers})")
            else:
                logger.info(f"Profile '{profile.name}': Single process mode using all {internal_workers} CPU cores")
            
            # Initialize components with calculated internal_workers
            preprocessor = self._init_preprocessor_for_profile(profile, profile_config, internal_workers)
            # Skip window_creator initialization - we create dynamic ones per batch
            pre_norm_filter = self._init_pre_norm_filter_for_profile(profile, profile_config, preprocessor)

            # Cache components (no window_creator to cache anymore)
            self.profile_preprocessors[profile.name] = preprocessor
            self.profile_pre_norm_filters[profile.name] = pre_norm_filter
        
        # Use cached components
        preprocessor = self.profile_preprocessors[profile.name]
        # window_creator will be created dynamically per batch
        pre_norm_filter = self.profile_pre_norm_filters[profile.name]

        # Use profile-specific parameters
        seq_length = profile.seq_length
        batch_size = profile.batch_size
        predict_start = profile.predict_start
        predict_end = profile.predict_end
        normalisation = profile.get('normalisation', 0.1)  # Get normalisation with default
        min_stride = getattr(profile, 'min_stride', 50)  # Default to 50 if not specified
        max_stride = getattr(profile, 'max_stride', 200)  # Default to 200 if not specified
        exponential_lookback = profile.exponential_lookback
        
        # Get chunk_size from profile (required, will raise error if not present)
        if not hasattr(profile, 'chunk_size_lines'):
            raise ValueError(f"Profile '{profile.name}' must have 'chunk_size_lines' defined in config.json5")
        chunk_size = profile.chunk_size_lines

        # Select random file from this profile
        file_path = rng.choice(profile_file_list)

        return {
            'file_path': file_path,
            'profile': profile,  # Add the profile itself
            'profile_config': profile_config,  # Add the profile config
            'profile_file_list': profile_file_list,  # Added this for subsequent use
            'preprocessor': preprocessor,
            'pre_norm_filter': pre_norm_filter,
            'seq_length': seq_length,
            'batch_size': batch_size,
            'predict_start': predict_start,
            'predict_end': predict_end,
            'normalisation': normalisation,  # Add normalisation
            'min_stride': min_stride,
            'max_stride': max_stride,
            'exponential_lookback': exponential_lookback,
            'chunk_size': chunk_size,  # Add chunk_size to return dict
            'internal_workers': internal_workers  # Pass internal workers count
        }
    
    def _get_or_cache_file_metadata(self, file_path, file_metadata) -> tuple:
        """Get or cache file metadata.
        
        Returns:
            tuple: (metadata_dict, success_flag)
        """
        if file_path not in file_metadata:
            try:
                metadata = pq.read_metadata(file_path)
                file_metadata[file_path] = {
                    'num_rows': metadata.num_rows,
                    'num_row_groups': metadata.num_row_groups,
                    'row_group_sizes': [
                        metadata.row_group(i).num_rows 
                        for i in range(metadata.num_row_groups)
                    ]
                }
            except OSError as e:
                logger.error(f"WARNING: Too many open files error for {file_path}: {e}")
                return None, False
            except Exception as e:
                # Handle other exceptions
                raise IOError(
                    f"Failed to read metadata from {file_path}: {e}. "
                    f"File may be corrupted or inaccessible. Please verify file integrity."
                )
        
        return file_metadata[file_path], True
    
    def _calculate_read_positions(self, file_path, start_pos, total_rows, 
                                exponential_lookback, predict_end, chunk_size,
                                prev_file=None, next_file=None) -> dict:
        """Calculate read positions and handle file boundaries.
        
        Returns:
            dict: Position information
        """
        # Neighbor files are now passed as parameters to ensure consistency
        
        # Check if we need data from previous file for lookback
        need_lookback_rows = 0
        if start_pos < exponential_lookback:
            need_lookback_rows = exponential_lookback - start_pos
            
            # Verify consistency - this should never happen with our fix
            if not prev_file:
                raise ValueError(
                    f"Invalid state: start_pos={start_pos} requires lookback from previous file "
                    f"but no prev_file provided. This indicates a bug in position range calculation."
                )
        
        # Check if we need data from next file for prediction
        need_lookahead_rows = 0
        if start_pos + chunk_size + predict_end > total_rows:
            need_lookahead_rows = (start_pos + chunk_size + predict_end) - total_rows
            
            # Verify consistency - this should never happen with our fix
            if not next_file:
                raise ValueError(
                    f"Invalid state: start_pos={start_pos} requires lookahead from next file "
                    f"but no next_file provided. This indicates a bug in position range calculation."
                )
        
        # Read chunk with lookback/lookahead handling
        read_start = max(0, start_pos - exponential_lookback)
        read_end = min(total_rows, start_pos + chunk_size + predict_end)
        
        return {
            'start_pos': start_pos,
            'read_start': read_start,
            'read_end': read_end,
            'prev_file': prev_file,
            'next_file': next_file,
            'need_lookback_rows': need_lookback_rows,
            'need_lookahead_rows': need_lookahead_rows
        }
    
    def _read_chunk_data(self, file_path, read_start, read_end, metadata) -> pd.DataFrame:
        """Read chunk data from parquet file using row groups.
        
        Returns:
            pd.DataFrame: Read and sliced data with column mappings applied
        """
        # Efficient row group based reading
        with pq.ParquetFile(file_path) as pf:
            # Find which row groups we need
            row_group_starts = np.cumsum([0] + metadata['row_group_sizes'][:-1])
            start_group = np.searchsorted(row_group_starts, read_start, side='right') - 1
            end_group = np.searchsorted(row_group_starts, read_end, side='left')
            
            # Read only necessary row groups
            if end_group > start_group:
                table = pf.read_row_groups(list(range(start_group, end_group)))
            else:
                table = pf.read_row_groups([start_group])
            
            df = table.to_pandas()
        
        # Slice to exact range
        group_start = row_group_starts[start_group]
        df_start = read_start - group_start
        df_end = df_start + (read_end - read_start)
        df = df.iloc[df_start:df_end]
        
        # Apply column mappings to the main dataframe immediately
        df = apply_column_mappings(df, self.data_config)
        
        return df
    
    def _handle_cross_file_data(self, df, file_path, prev_file, next_file,
                               need_lookback_rows, need_lookahead_rows, normalisation=0.1) -> pd.DataFrame:
        """Handle cross-file reading for lookback and lookahead.
        
        Returns:
            pd.DataFrame: Combined dataframe with cross-file data
        """
        # Handle cross-file reading for lookback
        if need_lookback_rows > 0 and prev_file:
            try:
                # Read tail of previous file
                prev_df = self._read_file_tail(prev_file, need_lookback_rows)
                prev_df = apply_column_mappings(prev_df, self.data_config)
                
                # Add normalisation column to prev_df to match main df
                prev_df['normalisation'] = normalisation
                
                # Ensure column order consistency before concatenation
                if list(prev_df.columns) != list(df.columns):
                    raise IOError(f"Failed to read lookback data from {os.path.basename(prev_file)}: columns not match")

                df = pd.concat([prev_df, df], ignore_index=True)
                logger.debug(f"Added {len(prev_df)} lookback rows from {os.path.basename(prev_file)}")
            except Exception as e:
                raise IOError(
                    f"Failed to read lookback data from {os.path.basename(prev_file)}: {e}. "
                    f"Cannot guarantee temporal continuity. Please verify file accessibility."
                )
        
        # Handle cross-file reading for lookahead
        if need_lookahead_rows > 0 and next_file:
            try:
                # Read head of next file
                next_df = self._read_file_head(next_file, need_lookahead_rows)
                next_df = apply_column_mappings(next_df, self.data_config)
                
                # Add normalisation column to next_df to match main df
                next_df['normalisation'] = normalisation
                
                # Ensure column order consistency before concatenation
                if list(next_df.columns) != list(df.columns):
                    raise IOError(f"Failed to read lookahead data from {os.path.basename(next_file)}: columns not match")

                # Concatenate with current data
                df = pd.concat([df, next_df], ignore_index=True)
                logger.debug(f"Added {len(next_df)} lookahead rows from {os.path.basename(next_file)}")
            except Exception as e:
                raise IOError(
                    f"Failed to read lookahead data from {os.path.basename(next_file)}: {e}. "
                    f"Cannot guarantee temporal continuity. Please verify file accessibility."
                )
        
        return df
    
    def _preprocess_chunk(self, df, pre_norm_filter, predict_start, predict_end, file_path, chunk_size, preprocessor=None, profile_name=None) -> tuple:
        """Preprocess chunk data with chunk-level normalization.
        
        Args:
            df: DataFrame to preprocess
            pre_norm_filter: Pre-normalization filter
            predict_start: Prediction start offset
            predict_end: Prediction end offset
            file_path: Path to the file being processed
            chunk_size: Size of the chunk
            preprocessor: Preprocessor instance
            profile_name: Name of the current profile
        
        Returns:
            tuple: (preprocessed_array, price_data, should_skip)
        """
        # Apply pre-normalization filtering
        if pre_norm_filter is not None:
            original_len = len(df)
            df = pre_norm_filter.filter_dataframe(df)
            filtered_len = len(df)
            
            # Skip chunk if too many rows were filtered
            if filtered_len < chunk_size * 0.5:
                logger.warning(f"Pre-norm filter removed {original_len - filtered_len} rows "
                             f"({(original_len - filtered_len)/original_len*100:.1f}%), "
                             f"skipping chunk from {os.path.basename(file_path)}")
                return None, None, True
            elif filtered_len < original_len:
                logger.info(f"Pre-norm filter removed {original_len - filtered_len} rows "
                          f"({(original_len - filtered_len)/original_len*100:.1f}%) "
                          f"from {os.path.basename(file_path)}")
        
        # Extract timestamp BEFORE filtering columns (while time columns are still available)
        chunk_timestamp = None
        if preprocessor is not None:
            # Extract chunk timestamp using preprocessor's built-in method
            chunk_timestamp = preprocessor.cpu_preprocessor._extract_data_timestamp(df)
            logger.debug(f"Extracted timestamp {chunk_timestamp} before column filtering")
        
        # Extract price data BEFORE filtering columns (so price columns are still available)
        price_data = extract_price_data(df)

        # Ensure correct columns using utility function
        if self.feature_columns:
            df = ensure_feature_columns(df, self.feature_columns, predict_start, predict_end)
        
        # Apply chunk-level normalization
        if preprocessor is not None:
            # Normalize entire chunk using the pre-extracted timestamp
            chunk_array = df.values.astype(np.float32)
            normalized_tensor = preprocessor.transform(chunk_array, 
                                                      timestamp=chunk_timestamp, 
                                                      file_path=file_path,
                                                      profile_name=profile_name)
            preprocessed = normalized_tensor.numpy()
            
            logger.debug(f"Applied chunk-level normalization with timestamp {chunk_timestamp}")
        else:
            preprocessed = df.values.astype(np.float32)
            logger.debug("No preprocessor provided, using raw numpy data")
        
        return preprocessed, price_data, False
    
    def _generate_windows_with_dynamic_offsets(self, profile, profile_config,
                                              preprocessor, preprocessed, price_data,
                                              df, file_path, start_pos, read_start, read_end, chunk_size,
                                              prev_file, next_file, need_lookback_rows, need_lookahead_rows,
                                              min_stride, max_stride, exponential_lookback, predict_end,
                                              rng, np_rng, stride_counts, worker_id, internal_workers):
        """Generate windows using stride-based sampling.
        
        Creates windows with systematic stride through the chunk.
        """
        # Random stride selection for this chunk
        stride = rng.randint(min_stride, max_stride + 1)
        
        # Track stride usage for logging
        stride_counts[stride] = stride_counts.get(stride, 0) + 1
        
        logger.info(f"Worker {worker_id}: Selected stride={stride} for chunk from {os.path.basename(file_path)}")
        
        # Create window creator with exponential sample offsets
        from .cpu_window_creator import CPUWindowCreator
        
        # Create exponential sample offsets (with optional randomization)
        randomized_offsets = self._create_sample_offsets_for_profile(profile, rng)
        
        # Log stride selection
        logger.debug(f"Worker {worker_id}: Creating window creator with stride={stride}")
        
        # Get classification mode from model config
        classification_mode = 'binary'  # default
        if self.model_config and hasattr(self.model_config, 'classification_mode'):
            classification_mode = self.model_config.classification_mode

        # Get thresholds from config
        return_threshold_buy = 0.002
        return_threshold_sell = -0.002
        if self.binary_classification_config and hasattr(self.binary_classification_config, 'label_generation'):
            label_gen = self.binary_classification_config.label_generation
            return_threshold_buy = getattr(label_gen, 'return_threshold_buy', 0.002)
            return_threshold_sell = getattr(label_gen, 'return_threshold_sell', -0.002)

        # Create a new window creator with randomized offsets and RNG
        dynamic_window_creator = CPUWindowCreator(
            sample_offsets=randomized_offsets,
            feature_dim=len(profile_config.feature_columns),
            predict_start=profile.predict_start,
            predict_end=profile.predict_end,
            batch_size=profile.batch_size,
            num_workers=internal_workers,
            enable_shared_memory_mode=False,
            rng=np_rng,  # Pass the worker-specific RNG for future use
            return_threshold_buy=return_threshold_buy,
            return_threshold_sell=return_threshold_sell,
            classification_mode=classification_mode
        )
        
        # Calculate offsets accounting for cross-file data
        if need_lookback_rows > 0 and prev_file:
            actual_history_offset = exponential_lookback
        else:
            history_offset = start_pos - read_start
            actual_history_offset = history_offset
        
        # Future offset remains the same logic
        future_offset = read_end - (start_pos + chunk_size)
        
        if need_lookahead_rows > 0 and next_file:
            future_offset += need_lookahead_rows
        
        # Process windows with dynamic window creator
        chunk_data = preprocessed
        
        for batch in dynamic_window_creator.process_chunk(
            chunk_data,
            price_data,
            start_pos,
            stride,
            actual_history_offset,
            future_offset
        ):
            yield batch
    
    def _extract_date_from_filename(self, file_path: str) -> Optional[datetime]:
        """Extract date from filename pattern: YYYY-MM-DD"""
        match = re.search(r'(\d{4}-\d{2}-\d{2})', file_path)
        if match:
            return datetime.strptime(match.group(1), '%Y-%m-%d')
        return None
    
    def _get_neighbor_file(self, file_path: str, days_offset: int) -> Optional[str]:
        """Get neighbor file path by adding days_offset to current file's date"""
        current_date = self._extract_date_from_filename(file_path)
        if not current_date:
            return None
        
        neighbor_date = current_date + timedelta(days=days_offset)
        neighbor_date_str = neighbor_date.strftime('%Y-%m-%d')
        
        # Replace date in filename
        neighbor_path = re.sub(r'\d{4}-\d{2}-\d{2}', neighbor_date_str, file_path)
        
        # Check if neighbor is in the filtered list (respects date filtering)
        for profile_name, file_list in self.profile_files.items():
            if file_path in file_list:
                # Only return neighbor if it's also in the filtered list
                if neighbor_path in file_list:
                    return neighbor_path
                # Neighbor might exist on disk but is filtered out
                return None
        
        # File not found in any profile (shouldn't happen)
        return None
    
    def _read_file_tail(self, file_path: str, num_rows: int) -> pd.DataFrame:
        """Read last num_rows from a file efficiently"""
        with pq.ParquetFile(file_path) as pf:
            total_rows = pf.metadata.num_rows
            
            if num_rows >= total_rows:
                # Read entire file
                result = pf.read().to_pandas()
                return result
            
            # Read from the end
            start_row = total_rows - num_rows
            
            # Find which row groups we need
            row_group_starts = []
            cumsum = 0
            for i in range(pf.num_row_groups):
                row_group_starts.append(cumsum)
                cumsum += pf.metadata.row_group(i).num_rows
            
            # Find starting row group
            start_group = 0
            for i, rg_start in enumerate(row_group_starts):
                if rg_start <= start_row < (rg_start + pf.metadata.row_group(i).num_rows if i < len(row_group_starts)-1 else total_rows):
                    start_group = i
                    break
            
            # Read necessary row groups
            table = pf.read_row_groups(list(range(start_group, pf.num_row_groups)))
            df = table.to_pandas()
        
        # Slice to exact rows needed
        group_start = row_group_starts[start_group]
        df_start = start_row - group_start
        result = df.iloc[df_start:]
        return result
    
    def _read_file_head(self, file_path: str, num_rows: int) -> pd.DataFrame:
        """Read first num_rows from a file efficiently"""
        with pq.ParquetFile(file_path) as pf:
            # Find which row groups we need
            rows_read = 0
            row_groups_to_read = []
            
            for i in range(pf.num_row_groups):
                row_groups_to_read.append(i)
                rows_read += pf.metadata.row_group(i).num_rows
                if rows_read >= num_rows:
                    break
            
            # Read necessary row groups
            table = pf.read_row_groups(row_groups_to_read)
            df = table.to_pandas()
        
        # Slice to exact rows needed
        result = df.iloc[:num_rows]
        return result
    
    def _init_preprocessor_for_profile(self, profile, config, internal_workers):
        """Initialize preprocessor for a specific profile in worker process.
        
        Args:
            profile: Profile configuration
            config: Data configuration
            internal_workers: Number of internal workers (already calculated)
        """
        from .cpu_preprocessor import CPUPreprocessorCompat
        
        preprocessor = CPUPreprocessorCompat(
            # Required configuration
            feature_columns=config.feature_columns,
            feature_categories=config.feature_categories,
            price_norm_method=profile.price_norm_method,
            volume_norm_method=profile.volume_norm_method,
            count_norm_method=profile.count_norm_method,
            time_norm_method=getattr(profile, 'time_norm_method', 'cyclical'),
            spread_norm_method=getattr(profile, 'spread_norm_method', 'standard'),
            volume_imbalance_norm_method=getattr(profile, 'volume_imbalance_norm_method', 'standard'),
            window_norm_method=getattr(profile, 'window_norm_method', 'context'),
            meta_norm_method=getattr(profile, 'meta_norm_method', 'none'),
            normalize_all_features=True,
            adaptive_stats_db=profile.adaptive_stats_db,
            num_workers=internal_workers,  # Use calculated internal_workers
            price_clip_range=getattr(profile, 'price_clip_range', None),
            volume_clip_range=getattr(profile, 'volume_clip_range', None),
            count_clip_range=getattr(profile, 'count_clip_range', None),
            spread_clip_range=getattr(profile, 'spread_clip_range', None),
            volume_imbalance_clip_range=getattr(profile, 'volume_imbalance_clip_range', None),
            time_clip_range=getattr(profile, 'time_clip_range', None),
            window_clip_range=getattr(profile, 'window_clip_range', None),
            meta_clip_range=getattr(profile, 'meta_clip_range', None),
            other_clip_range=getattr(profile, 'other_clip_range', None),
            price_valid_range=getattr(profile, 'price_valid_range', None),
            price_placeholder_values=getattr(profile, 'price_placeholder_values', None),
            # Normalization scaling factors
            price_scale_factor=profile.price_scale_factor,
            volume_scale_factor=profile.volume_scale_factor,
            count_scale_factor=profile.count_scale_factor,
            spread_scale_factor=getattr(profile, 'spread_scale_factor', 1.0),
            volume_imbalance_scale_factor=getattr(profile, 'volume_imbalance_scale_factor', 1.0),
            time_scale_factor=profile.time_scale_factor,
            window_scale_factor=profile.window_scale_factor,
            meta_scale_factor=profile.meta_scale_factor,
            other_scale_factor=profile.other_scale_factor,
            # Feature noise for regularization
            feature_noise_std=getattr(config, 'feature_noise_std', 0.0)
        )
        
        # Fit preprocessor (no data needed anymore)
        preprocessor.fit()
        
        return preprocessor
    
    def _init_window_creator_for_profile(self, profile, config, internal_workers):
        """Initialize window creator for a specific profile in worker process.
        
        Args:
            profile: Profile configuration
            config: Data configuration
            internal_workers: Number of internal workers (already calculated)
        """
        from .cpu_window_creator import CPUWindowCreator
        
        # Create sample offsets for this profile
        sample_offsets = self._create_sample_offsets_for_profile(profile)
        
        window_creator = CPUWindowCreator(
            sample_offsets=sample_offsets,
            feature_dim=len(config.feature_columns),
            predict_start=profile.predict_start,
            predict_end=profile.predict_end,
            batch_size=profile.batch_size,
            num_workers=internal_workers,  # Use calculated internal_workers
            enable_shared_memory_mode=False
        )
        
        return window_creator
    
    def _init_pre_norm_filter_for_profile(self, profile, config, preprocessor):
        """Initialize pre-normalization filter for a specific profile."""
        from .pre_normalization_filter import PreNormalizationFilter
        
        if preprocessor.cpu_preprocessor.column_config is not None:
            pre_norm_filter = PreNormalizationFilter(
                price_range=getattr(profile, 'price_valid_range', None),
                volume_range=getattr(profile, 'volume_valid_range', None),
                count_range=getattr(profile, 'count_valid_range', None),
                derived_range=None,  # Deprecated, use specific ranges
                time_range=getattr(profile, 'time_valid_range', None),
                window_range=getattr(profile, 'window_valid_range', None),
                meta_range=getattr(profile, 'meta_valid_range', None),
                other_range=getattr(profile, 'other_valid_range', None),
                mid_price_range=getattr(profile, 'mid_price_valid_range', None),
                spread_range=getattr(profile, 'spread_valid_range', None),
                volume_imbalance_range=getattr(profile, 'volume_imbalance_valid_range', None),
                price_placeholder_values=getattr(profile, 'price_placeholder_values', None),
                row_threshold=getattr(profile, 'anomaly_row_threshold', 0.1),
                column_threshold=getattr(profile, 'anomaly_column_threshold', 0.1),
                column_config=preprocessor.cpu_preprocessor.column_config
            )
            return pre_norm_filter
        return None
    
    def _create_sample_offsets_for_profile(self, profile, rng=None) -> np.ndarray:
        """Create hybrid exponential sample offsets for a specific profile with randomness.
        
        Args:
            profile: Profile configuration
            rng: Optional random generator for adding variability
        
        Returns:
            np.ndarray: Sample offsets with optional randomness
        """
        # Get base config values from profile
        base_sequential_samples = getattr(profile, 'sequential_samples', 50)
        base_exponential_factor = getattr(profile, 'exponential_factor', 2.5)
        base_exponential_lookback = getattr(profile, 'exponential_lookback', 200000)
        
        # Add randomness if rng is provided (during actual data loading)
        if rng is not None:
            # Get randomization parameters from data config (with defaults)
            sequential_variation = getattr(self.data_config, 'sequential_randomness', 0.2)
            factor_variation = getattr(self.data_config, 'factor_randomness', 0.3)
            lookback_variation = getattr(self.data_config, 'lookback_randomness', 0.15)
            
            # Add randomness to sequential_samples
            sequential_samples = int(base_sequential_samples * (1 + rng.uniform(-sequential_variation, sequential_variation)))
            sequential_samples = max(10, min(sequential_samples, profile.seq_length // 2))  # Ensure reasonable bounds
            
            # Add randomness to exponential_factor
            exponential_factor = base_exponential_factor * (1 + rng.uniform(-factor_variation, factor_variation))
            exponential_factor = max(1.5, min(exponential_factor, 4.0))  # Keep factor in reasonable range
            
            # Add randomness to exponential_lookback
            exponential_lookback = int(base_exponential_lookback * (1 + rng.uniform(-lookback_variation, lookback_variation)))
            exponential_lookback = max(sequential_samples * 2, exponential_lookback)  # Ensure lookback > sequential
        else:
            # Use base values (for initialization or deterministic mode)
            sequential_samples = base_sequential_samples
            exponential_factor = base_exponential_factor
            exponential_lookback = base_exponential_lookback
        
        # First samples: sequential
        initial_offsets = np.arange(0, sequential_samples)
        
        # Remaining samples: exponential spacing
        remaining_samples = profile.seq_length - sequential_samples
        if remaining_samples > 0:
            positions = np.linspace(0, 1, remaining_samples)
            exp_factor = exponential_factor
            
            exp_positions = np.exp(exp_factor * positions) - 1
            exp_positions = exp_positions / exp_positions[-1]
            
            start_pos = sequential_samples
            offsets_exp = start_pos + exp_positions * (
                exponential_lookback - start_pos
            )
            offsets_exp = np.round(offsets_exp).astype(np.int64)
            
            # Ensure minimum spacing
            for i in range(1, len(offsets_exp)):
                if offsets_exp[i] <= offsets_exp[i-1]:
                    offsets_exp[i] = offsets_exp[i-1] + 1
            
            sample_offsets = np.concatenate([initial_offsets, offsets_exp])
        else:
            sample_offsets = initial_offsets[:profile.seq_length]
        
        # Ensure we don't exceed maximum lookback (use the potentially randomized value)
        sample_offsets = np.minimum(sample_offsets, exponential_lookback)
        
        return sample_offsets
    
    def _filter_files_by_date_window(self, start_date: datetime, end_date: datetime) -> Dict[str, List[str]]:
        """Filter files for all profiles within the date window.
        
        Args:
            start_date: Window start date
            end_date: Window end date
            
        Returns:
            Dictionary mapping profile names to filtered file lists
        """
        filtered_files = {}
        
        for profile_name, file_list in self.profile_files.items():
            profile_filtered = []
            
            for file_path in file_list:
                # Extract date from filename
                file_date = self._extract_date_from_filename(file_path)
                if file_date and start_date <= file_date <= end_date:
                    profile_filtered.append(file_path)
            
            filtered_files[profile_name] = profile_filtered
            
            if profile_filtered:
                logger.debug(f"Profile '{profile_name}': {len(profile_filtered)} files in window "
                           f"[{start_date.strftime('%Y-%m-%d')}, {end_date.strftime('%Y-%m-%d')}]")
        
        return filtered_files
    
    def _has_any_files_in_window(self, window_files: Dict[str, List[str]]) -> bool:
        """Check if any profile has files in the current window."""
        return any(len(files) > 0 for files in window_files.values())

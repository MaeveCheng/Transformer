"""
Sequential inference dataset inheriting from SimplifiedRandomDataset.

This dataset processes files sequentially rather than randomly sampling.
It inherits ~70% of its functionality from SimplifiedRandomDataset,
only overriding the iteration logic to be sequential.

Key features:
- Inherits all data processing from SimplifiedRandomDataset
- Sequential processing from file start to end
- Fixed stride for all windows (not random)
- Cross-file boundary handling (inherited)
- Compatible with existing preprocessing pipeline (inherited)
"""

from .simplified_random_dataset import SimplifiedRandomDataset
from .cpu_window_creator import CPUWindowCreator
import torch
import numpy as np
import logging
import os
import multiprocessing as mp
from typing import Iterator, Dict, List, Optional, Any

logger = logging.getLogger(__name__)


class SequentialInferenceDataset(SimplifiedRandomDataset):
    """
    Sequential inference dataset inheriting from SimplifiedRandomDataset.
    Reuses all data processing methods, only changes iteration strategy.
    
    Key differences from parent:
    - Files processed in sorted order (not random)
    - ALL chunks processed sequentially (not just one random chunk)
    - Fixed stride for windows (not random between min/max)
    """
    
    def __init__(
        self,
        file_list: List[str],
        data_config,
        profiles: List,
        profile_files: Dict[str, List[str]],
        inference_stride: int = None,  # FIXED stride for window sampling
        stride: int = None,  # Support both parameter names
        **kwargs
    ):
        """
        Initialize sequential inference dataset.
        
        Only difference from parent:
        - inference_stride: Fixed stride for ALL window sampling (not random)
        """
        # Call parent constructor with inference-specific defaults
        super().__init__(
            file_list=file_list,
            data_config=data_config,
            profiles=profiles,
            profile_files=profile_files,
            profile_configs=kwargs.get('profile_configs'),
            binary_classification_config=kwargs.get('binary_classification_config'),  # Pass this to parent!
            rank=kwargs.get('rank', 0),
            world_size=kwargs.get('world_size', 1),
            seed=kwargs.get('seed', 42),  # Fixed seed for reproducibility
            train_date_start=None,  # No date windowing for inference
            train_date_end=None,
            date_group_size=10,
            date_group_lookback=5,
            iter_every_group=100,
            date_group_start_date=None
        )
        
        # Store profile_configs if not provided (parent might not store it if None)
        if not hasattr(self, 'profile_configs'):
            self.profile_configs = kwargs.get('profile_configs', {})

        # Store model_config for classification_mode
        self.model_config = kwargs.get('model_config', None)

        # Store inference-specific parameter
        # Support both 'stride' and 'inference_stride' parameter names
        self.inference_stride = stride if stride is not None else (inference_stride if inference_stride is not None else 100)  # FIXED stride for windows
        
        # Sort files for sequential processing
        self.file_list = sorted(self.file_list)
        for profile_name in self.profile_files:
            self.profile_files[profile_name] = sorted(self.profile_files[profile_name])
        
        logger.info(f"Sequential inference: fixed stride={self.inference_stride} for all windows")
    
    def __iter__(self) -> Iterator[Dict[str, torch.Tensor]]:
        """
        Sequential iteration through files and chunks.
        
        Key changes from parent:
        1. Files processed in sorted order (not random)
        2. ALL chunks processed sequentially (not just one random chunk)
        3. Fixed stride for windows (not random)
        """
        # Initialize worker (reuse parent's method for consistency)
        # Parent returns: (worker_id, num_workers, rng, np_rng, current_start_date, current_end_date, current_window_files)
        worker_result = self._initialize_worker()
        worker_id = worker_result[0]
        num_workers = worker_result[1]
        rng = worker_result[2]
        np_rng = worker_result[3]
        # We don't need date windowing for inference, so ignore the rest
        
        # Select profile for inference
        if not self.profiles:
            raise ValueError("No profiles configured")
        
        # For inference, we typically use the first profile or could iterate through all
        # Make sure the profile has required attributes
        profile = self.profiles[0]
        if not hasattr(profile, 'chunk_size_lines'):
            raise ValueError(f"Profile '{profile.name}' must have 'chunk_size_lines' defined")
        
        # Get profile configuration
        profile_config = self.profile_configs.get(profile.name, self.data_config)
        
        # Calculate internal workers for window creation
        if num_workers > 1:
            internal_workers = max(1, mp.cpu_count() // num_workers)
        else:
            internal_workers = mp.cpu_count()
        
        # Initialize components using parent's methods
        if profile.name not in self.profile_preprocessors:
            # Use parent's initialization methods
            preprocessor = self._init_preprocessor_for_profile(profile, profile_config, internal_workers)
            pre_norm_filter = self._init_pre_norm_filter_for_profile(profile, profile_config, preprocessor)
            
            # Cache components
            self.profile_preprocessors[profile.name] = preprocessor
            self.profile_pre_norm_filters[profile.name] = pre_norm_filter
        
        preprocessor = self.profile_preprocessors[profile.name]
        pre_norm_filter = self.profile_pre_norm_filters[profile.name]
        
        # Get profile parameters with validation
        predict_start = getattr(profile, 'predict_start', 20)
        predict_end = getattr(profile, 'predict_end', 40)
        exponential_lookback = getattr(profile, 'exponential_lookback', 100000)
        chunk_size = profile.chunk_size_lines  # Required, will error if missing
        
        # For inference, we use the fixed inference_stride (not min/max from profile)
        # This is already set in __init__ as self.inference_stride
        
        # Simple file metadata cache
        file_metadata = {}
        
        # Stride usage tracking (for compatibility with parent)
        stride_counts = {}
        
        # SEQUENTIAL PROCESSING: Process files in order
        for file_idx, file_path in enumerate(self.file_list):
            logger.info(f"Processing file {file_idx + 1}/{len(self.file_list)}: {os.path.basename(file_path)}")
            
            # Get file metadata using parent's method
            metadata, success = self._get_or_cache_file_metadata(file_path, file_metadata)
            if not success:
                continue
            total_rows = metadata['num_rows']
            
            # Check for neighbor files using parent's method
            prev_file = self._get_neighbor_file(file_path, -1)
            next_file = self._get_neighbor_file(file_path, 1)
            
            # SEQUENTIAL PROCESSING: Determine starting position for first chunk
            # This is different from training's random selection!
            if prev_file:
                # Start from beginning, will read lookback from prev file
                first_chunk_start = 0
            else:
                # Start after lookback to ensure first window has enough history
                first_chunk_start = exponential_lookback
            
            # SEQUENTIAL CHUNKS: Process with NO gaps, NO overlap
            # Each chunk is exactly chunk_size (except possibly the last one)
            chunk_position = first_chunk_start
            chunk_count = 0
            
            # Process chunks sequentially until end of file
            while chunk_position < total_rows:
                # Define chunk boundaries (sequential, no gaps)
                chunk_start = chunk_position
                chunk_end = min(chunk_position + chunk_size, total_rows)
                actual_chunk_size = chunk_end - chunk_start
                
                # Skip if chunk is too small
                min_chunk_size = min(1000, chunk_size // 10)
                if actual_chunk_size < min_chunk_size:
                    break
                
                # Calculate what to read (with lookback/lookahead)
                # This is DIFFERENT from training which uses _calculate_read_positions
                # We calculate inline for clarity
                
                # Calculate lookback reading
                if chunk_start == 0 and prev_file:
                    read_start = 0
                    need_lookback_rows = exponential_lookback
                elif chunk_start >= exponential_lookback:
                    read_start = chunk_start - exponential_lookback
                    need_lookback_rows = 0
                else:
                    read_start = 0
                    need_lookback_rows = exponential_lookback - chunk_start if prev_file else 0
                
                # Calculate lookahead reading
                if chunk_end + predict_end > total_rows and next_file:
                    read_end = total_rows
                    need_lookahead_rows = (chunk_end + predict_end) - total_rows
                else:
                    read_end = min(chunk_end + predict_end, total_rows)
                    need_lookahead_rows = 0
                
                try:
                    # Read chunk using parent's method
                    df = self._read_chunk_data(file_path, read_start, read_end, metadata)
                    
                    # Add normalisation column (for compatibility with parent)
                    df['normalisation'] = getattr(profile, 'normalisation', 0.1)
                    
                    # Handle cross-file data using parent's method
                    df = self._handle_cross_file_data(
                        df, file_path, prev_file, next_file,
                        need_lookback_rows, need_lookahead_rows,
                        normalisation=df['normalisation'].iloc[0]
                    )
                    
                    # Preprocess using parent's method
                    preprocessed, price_data, should_skip = self._preprocess_chunk(
                        df, pre_norm_filter, predict_start, predict_end,
                        file_path, chunk_size, preprocessor, profile.name
                    )
                    
                    if should_skip:
                        chunk_position = chunk_end  # Move to next chunk (no gap!)
                        chunk_count += 1
                        continue
                    
                    # Call our sequential window generation method
                    # This replaces training's _generate_windows_with_dynamic_offsets
                    for batch in self._generate_windows_sequential(
                        profile, profile_config, preprocessor, preprocessed, price_data,
                        df, file_path, chunk_start, read_start, read_end, chunk_size,
                        prev_file, next_file, need_lookback_rows, need_lookahead_rows,
                        self.inference_stride, self.inference_stride,  # min/max stride both set to fixed value
                        exponential_lookback, predict_end,
                        rng, np_rng, stride_counts, worker_id, internal_workers
                    ):
                        yield batch
                    
                except Exception as e:
                    logger.error(f"Failed to process chunk at position {chunk_position}: {e}")
                
                # Move to next chunk position (NO GAP between chunks!)
                chunk_position = chunk_end
                chunk_count += 1
    
    def _generate_windows_sequential(self, profile, profile_config,
                                      preprocessor, preprocessed, price_data,
                                      df, file_path, start_pos, read_start, read_end, chunk_size,
                                      prev_file, next_file, need_lookback_rows, need_lookahead_rows,
                                      min_stride, max_stride, exponential_lookback, predict_end,
                                      rng, np_rng, stride_counts, worker_id, internal_workers):
        """
        Generate windows using FIXED stride for inference.
        
        This replaces parent's _generate_windows_with_dynamic_offsets.
        Only difference: uses fixed stride instead of random stride.
        
        Note: min_stride and max_stride are passed but ignored - we use self.inference_stride
        """
        # ONLY DIFFERENCE: Fixed stride for inference (ignore min/max_stride)
        stride = self.inference_stride  # Fixed for ALL chunks
        # Training would use: stride = rng.randint(min_stride, max_stride + 1)
        
        # Track stride usage (optional, for consistency with parent)
        stride_counts[stride] = stride_counts.get(stride, 0) + 1
        
        logger.info(f"Worker {worker_id}: Using FIXED stride={stride} for chunk from {os.path.basename(file_path)}")
        
        # Everything below is IDENTICAL to parent's _generate_windows_with_dynamic_offsets
        # Create window creator with exponential sample offsets
        from .cpu_window_creator import CPUWindowCreator
        
        # Create exponential sample offsets (with randomization, same as training)
        randomized_offsets = self._create_sample_offsets_for_profile(profile, rng)
        
        # Get classification mode from model config
        classification_mode = 'binary'  # default
        if self.model_config is not None and hasattr(self.model_config, 'classification_mode'):
            classification_mode = self.model_config.classification_mode

        # Get thresholds from config for ternary classification
        # Default values match those in config.json5
        return_threshold_buy = 0.000001
        return_threshold_sell = -0.000001
        if self.binary_classification_config and hasattr(self.binary_classification_config, 'label_generation'):
            label_gen = self.binary_classification_config.label_generation
            return_threshold_buy = getattr(label_gen, 'return_threshold_buy', 0.000001)
            return_threshold_sell = getattr(label_gen, 'return_threshold_sell', -0.000001)

        # Create a new window creator with randomized offsets and RNG
        dynamic_window_creator = CPUWindowCreator(
            sample_offsets=randomized_offsets,
            feature_dim=len(profile_config.feature_columns),
            predict_start=profile.predict_start,
            predict_end=profile.predict_end,
            batch_size=profile.batch_size,
            num_workers=internal_workers,
            enable_shared_memory_mode=False,
            rng=rng,
            return_threshold_buy=return_threshold_buy,
            return_threshold_sell=return_threshold_sell,
            classification_mode=classification_mode
        )
        
        # Calculate offsets (same as training)
        if need_lookback_rows > 0 and prev_file:
            actual_history_offset = exponential_lookback
        else:
            actual_history_offset = start_pos - read_start
        
        future_offset = read_end - (start_pos + chunk_size)
        if need_lookahead_rows > 0 and next_file:
            future_offset += need_lookahead_rows
        
        # Process windows with window creator
        chunk_data = preprocessed
        
        for batch in dynamic_window_creator.process_chunk(
            chunk_data,
            price_data,
            start_pos,  # chunk starting position in file
            stride,     # FIXED stride (not random)
            actual_history_offset,
            future_offset
        ):
            # Rename 'labels' to 'targets' for compatibility with Lightning module
            if 'labels' in batch:
                batch['targets'] = batch.pop('labels')
                # Debug: Log that targets are present
                if hasattr(self, '_first_batch_logged'):
                    pass
                else:
                    self._first_batch_logged = True
                    logger.debug(f"SequentialInferenceDataset: First batch contains targets with shape {batch['targets'].shape}")
            yield batch
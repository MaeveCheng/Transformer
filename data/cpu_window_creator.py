#!/usr/bin/env python3
"""
CPU-based memory-efficient window creation with multicore support.
Replaces GPU window creation with CPU-optimized implementation.

Key features:
- Multicore parallel processing using multiprocessing
- Vectorized NumPy operations for efficiency
- Memory pooling to reduce allocations
- Exponential lookback sampling support
- Zero-copy shared memory for inter-process communication
"""

import numpy as np
import torch
from typing import Dict, Optional, Generator, Tuple, List
import logging
import multiprocessing as mp
from multiprocessing import shared_memory
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import time
from functools import partial
import os

from .memory_utils import NumpyMemoryPool, SharedMemoryManager

logger = logging.getLogger(__name__)


class CPUWindowCreator:
    """
    CPU-based memory-efficient window creator using multicore processing.
    Maintains API compatibility with GPU version while using CPU resources.
    Supports shared memory mode for zero-copy data sharing in distributed training.
    """
    
    def __init__(
        self,
        sample_offsets: np.ndarray,
        feature_dim: int,
        predict_start: int,
        predict_end: int,
        batch_size: int = 256,
        num_workers: Optional[int] = None,
        use_shared_memory: bool = True,
        memory_pool_size: int = 10,
        enable_shared_memory_mode: bool = False,
        rng: Optional[np.random.RandomState] = None,
        **kwargs
    ):
        """
        Initialize CPU-based window creator.
        
        Args:
            sample_offsets: Array of sample offsets for exponential sampling
            feature_dim: Number of features in the data
            predict_start: Start of prediction window
            predict_end: End of prediction window
            batch_size: Number of windows to process at once
            num_workers: Number of CPU workers (defaults to CPU count)
            use_shared_memory: Use shared memory for inter-process communication
            memory_pool_size: Number of pre-allocated memory buffers
            enable_shared_memory_mode: Enable shared memory mode for distributed training
            rng: NumPy RandomState for stride selection
        """
        self.feature_dim = feature_dim
        self.predict_start = predict_start
        self.predict_end = predict_end
        self.batch_size = batch_size
        self.window_size = len(sample_offsets)
        self.use_shared_memory = use_shared_memory
        self.memory_pool_size = memory_pool_size
        self.enable_shared_memory_mode = enable_shared_memory_mode
        
        # Store RNG for stride selection (not used in process_chunk anymore)
        self.rng = rng if rng is not None else np.random.RandomState()
        
        # Store sample offsets
        self.sample_offsets = sample_offsets.astype(np.int64)
        
        # Shared memory manager for distributed training
        self.shared_memory_manager = None
        if self.enable_shared_memory_mode:
            self.shared_memory_manager = SharedMemoryManager(prefix="window_creator")
        
        # Setup multiprocessing
        self.num_workers = num_workers or mp.cpu_count()
        self.executor = None
        
        # Pre-calculate future offsets for target generation
        self.future_samples = min(10, self.predict_end - self.predict_start)
        self.future_offsets = np.linspace(
            self.predict_start, self.predict_end,
            self.future_samples, dtype=np.int64
        )
        
        # Memory pool for reusing allocations
        self.memory_pool = NumpyMemoryPool(max_size=self.memory_pool_size)
        
        # Statistics
        self._total_windows_created = 0
        
        # Get a test random value to verify RNG is unique
        test_random = self.rng.random()
        logger.info(f"CPUWindowCreator initialized: "
                   f"window_size={self.window_size}, batch_size={self.batch_size}, "
                   f"workers={self.num_workers}, shared_memory={self.use_shared_memory}, "
                   f"RNG test value={test_random:.6f}")
    
    def process_chunk(
        self,
        chunk_data: np.ndarray,
        price_data: np.ndarray,
        start_idx: int,
        stride: int,
        history_offset: int = 0,
        future_offset: int = 0
    ) -> Generator[Dict[str, torch.Tensor], None, None]:
        """
        Process windows from chunk data using stride-based systematic sampling.
        Each position is visited at most once, determined by the stride parameter.
        
        Args:
            chunk_data: Data chunk of shape (n_samples, feature_dim) - already normalized
            price_data: Price data of shape (n_samples,)
            start_idx: Starting index in the full dataset
            stride: Stride for systematic sampling through the chunk
            history_offset: Historical data offset
            future_offset: Future data offset
            
        Yields:
            Dict containing batch data:
                - 'features': Tensor of shape (batch_size, window_size, feature_dim)
                - 'labels': Tensor of shape (batch_size, 1)
                - 'groups': Tensor of shape (batch_size,)
                - 'positions': Tensor of shape (batch_size,)
                - 'batch_info': Dict with batch_idx and total_batches
        """
        chunk_size = chunk_data.shape[0]
        actual_chunk_size = chunk_size - history_offset - future_offset
        
        # Calculate valid range using same logic as old stride-based system
        # This ensures we don't waste data at boundaries
        
        # Find first valid position (checking lookback constraint)
        # From old system: has_lookback = (start_idx + history_offset + i) >= self.sample_offsets[-1]
        # Solving for i: i >= self.sample_offsets[-1] - start_idx - history_offset
        first_valid_i = max(0, self.sample_offsets[-1] - start_idx - history_offset)
        first_valid_local_pos = history_offset + first_valid_i
        
        # Find last valid position (checking lookahead constraint)
        # From old system: has_lookahead = (local_pos + self.future_offsets[-1]) < chunk_size
        # Where local_pos = history_offset + i
        # So: i < chunk_size - history_offset - self.future_offsets[-1]
        last_valid_i = min(actual_chunk_size, chunk_size - history_offset - self.future_offsets[-1])
        last_valid_local_pos = history_offset + last_valid_i
        
        # Check if we have any valid range
        if last_valid_local_pos <= first_valid_local_pos:
            logger.warning(f"[STRIDE SAMPLING] No valid positions in chunk! "
                         f"first_valid_pos={first_valid_local_pos}, last_valid_pos={last_valid_local_pos}, "
                         f"chunk_size={chunk_size}, history_offset={history_offset}, future_offset={future_offset}, "
                         f"start_idx={start_idx}, max_lookback={self.sample_offsets[-1]}, max_lookahead={self.future_offsets[-1]}")
            return
        
        # Stride-based systematic sampling
        valid_range_size = last_valid_local_pos - first_valid_local_pos
        
        # Generate positions using stride
        positions = np.arange(first_valid_local_pos, last_valid_local_pos, stride)
        
        # Log sampling info
        logger.info(f"[STRIDE SAMPLING] Using stride={stride} in valid range [{first_valid_local_pos}, {last_valid_local_pos}), "
                   f"generated {len(positions)} positions, "
                   f"first 5: {positions[:5] if len(positions) > 5 else positions}")
        
        n_positions = len(positions)
        positions_array = np.array(positions, dtype=np.int64)
        global_offset = start_idx
        
        logger.info(f"Processing chunk with {n_positions} positions, stride={stride}, chunk_size={chunk_size}")
        
        # Calculate actual number of batches based on positions from stride
        n_batches = (n_positions + self.batch_size - 1) // self.batch_size
        
        # Process in batches
        for batch_idx in range(n_batches):
            batch_start = batch_idx * self.batch_size
            batch_end = min(batch_start + self.batch_size, n_positions)
            batch_positions = positions_array[batch_start:batch_end]
            actual_batch_size = len(batch_positions)
            
            # Get buffer from pool or allocate new one
            windows = self.memory_pool.get_array(
                (actual_batch_size, self.window_size, self.feature_dim),
                dtype=np.float32
            )
            
            # Create windows using vectorized operations
            if self.num_workers > 1 and actual_batch_size > 16:
                # Parallel window creation for larger batches
                logger.debug(f"CPUWindowCreator: Using parallel processing for batch_size={actual_batch_size} with {self.num_workers} workers")
                windows = self._create_windows_parallel(
                    chunk_data, batch_positions, chunk_size
                )
            else:
                # Vectorized creation for smaller batches
                logger.debug(f"CPUWindowCreator: Using vectorized processing for batch_size={actual_batch_size}")
                windows = self._create_windows_vectorized(
                    chunk_data, batch_positions, chunk_size
                )
            
            # Calculate targets
            targets = self._calculate_targets_vectorized(
                price_data, batch_positions, chunk_size, global_offset
            )
            
            # Note: windows are already normalized at chunk level, no need for per-batch normalization
            
            # Create result tensors
            result = {
                'features': torch.from_numpy(windows),  # Changed to match lightning module
                'labels': torch.from_numpy(targets),    # Changed to match lightning module
                'groups': torch.full((actual_batch_size,), 0, dtype=torch.long),
                'positions': torch.from_numpy(batch_positions + global_offset),
                'batch_info': {
                    'batch_idx': batch_idx,
                    'total_batches': n_batches
                }
            }
            
            # Add worker_id for tracking
            worker_info = torch.utils.data.get_worker_info()
            if worker_info is not None:
                result['worker_id'] = worker_info.id
            else:
                result['worker_id'] = 0
            
            # Return buffer to pool
            self.memory_pool.return_array(windows)
            
            self._total_windows_created += actual_batch_size
            
            # Log batch info on first batch
            if batch_idx == 0 and self._total_windows_created <= actual_batch_size:
                logger.info(f"First batch created: batch_size={actual_batch_size}, window_shape={windows.shape}, target_shape={targets.shape}")
                logger.info(f"Window stats: mean={np.mean(windows):.4f}, std={np.std(windows):.4f}, min={np.min(windows):.4f}, max={np.max(windows):.4f}")
                logger.info(f"Target values: {targets.flatten()[:10]}... (first 10)")
                logger.info(f"Positions: {batch_positions[:5]}... (first 5)")
            
            yield result
    
    def _create_windows_vectorized(
        self,
        chunk_data: np.ndarray,
        positions: np.ndarray,
        chunk_size: int
    ) -> np.ndarray:
        """Create windows using vectorized NumPy operations"""
        batch_size = len(positions)
        windows = np.zeros((batch_size, self.window_size, self.feature_dim), dtype=np.float32)
        
        # Vectorized window creation
        for i, offset in enumerate(self.sample_offsets):
            # Calculate indices for all positions at once
            indices = positions - offset
            
            # Clip indices to valid range
            valid_mask = (indices >= 0) & (indices < chunk_size)
            valid_indices = np.where(valid_mask, indices, 0)
            
            # Extract data for all valid positions
            window_data = chunk_data[valid_indices]
            
            # Apply mask to set invalid positions to zero
            window_data[~valid_mask] = 0.0
            
            # Assign to windows
            windows[:, i, :] = window_data
        
        return windows
    
    def _create_windows_parallel(
        self,
        chunk_data: np.ndarray,
        positions: np.ndarray,
        chunk_size: int
    ) -> np.ndarray:
        """Create windows using parallel processing"""
        batch_size = len(positions)
        
        # Initialize executor if needed
        if self.executor is None:
            self.executor = ThreadPoolExecutor(max_workers=self.num_workers)
            logger.info(f"CPUWindowCreator: Created ThreadPoolExecutor with {self.num_workers} workers for parallel window creation")
        
        # Split positions for parallel processing
        chunk_positions = np.array_split(positions, self.num_workers)
        
        # Create partial function with fixed arguments
        process_func = partial(
            self._create_windows_worker,
            chunk_data=chunk_data,
            sample_offsets=self.sample_offsets,
            chunk_size=chunk_size,
            feature_dim=self.feature_dim
        )
        
        # Process in parallel
        futures = [self.executor.submit(process_func, pos_chunk) 
                  for pos_chunk in chunk_positions]
        
        # Collect results
        results = []
        for future in futures:
            results.append(future.result())
        
        # Concatenate results
        windows = np.vstack(results)
        
        return windows
    
    @staticmethod
    def _create_windows_worker(
        positions: np.ndarray,
        chunk_data: np.ndarray,
        sample_offsets: np.ndarray,
        chunk_size: int,
        feature_dim: int
    ) -> np.ndarray:
        """Worker function for parallel window creation"""
        batch_size = len(positions)
        window_size = len(sample_offsets)
        windows = np.zeros((batch_size, window_size, feature_dim), dtype=np.float32)
        
        for i, offset in enumerate(sample_offsets):
            indices = positions - offset
            valid_mask = (indices >= 0) & (indices < chunk_size)
            valid_indices = np.where(valid_mask, indices, 0)
            
            window_data = chunk_data[valid_indices]
            window_data[~valid_mask] = 0.0
            windows[:, i, :] = window_data
        
        return windows
    
    def _calculate_targets_vectorized(
        self,
        price_data: np.ndarray,
        positions: np.ndarray,
        chunk_size: int,
        global_offset: int
    ) -> np.ndarray:
        """Calculate targets using vectorized operations with proper bounds checking"""
        batch_size = len(positions)
        targets = np.zeros((batch_size, 1), dtype=np.float32)
        
        # CRITICAL FIX: Track data availability
        price_data_size = len(price_data)
        
        # Get current and future indices
        current_indices = positions
        future_indices = positions[:, np.newaxis] + self.future_offsets[np.newaxis, :]
        
        # PHASE 1 FIX: Don't clip future_indices - check validity instead
        # Current indices should be valid by construction
        current_indices = np.clip(current_indices, 0, price_data_size - 1)
        
        # Check which future indices are valid (within price_data bounds)
        valid_future_mask = (future_indices >= 0) & (future_indices < price_data_size)
        all_futures_valid = np.all(valid_future_mask, axis=1)
        
        # PHASE 3: Log data leakage detection
        max_future_idx = np.max(future_indices)
        if max_future_idx >= price_data_size:
            logger.warning(f"[DATA LEAKAGE PREVENTED] Future indices exceed bounds: "
                         f"max_future_idx={max_future_idx}, price_data_size={price_data_size}, "
                         f"positions range=[{np.min(positions)}, {np.max(positions)}], "
                         f"future_offsets={self.future_offsets.tolist()}")
        
        # Only process positions with ALL valid future data
        valid_positions_mask = all_futures_valid
        num_invalid = np.sum(~valid_positions_mask)
        
        if num_invalid > 0:
            logger.info(f"[LEAKAGE FIX] Skipping {num_invalid}/{batch_size} positions due to insufficient future data")
            # Mark invalid targets as -1 (will be filtered out in loss calculation)
            targets[~valid_positions_mask, 0] = -1.0
        
        if np.any(valid_positions_mask):
            # Get valid positions
            valid_pos_indices = np.where(valid_positions_mask)[0]
            valid_current_indices = current_indices[valid_positions_mask]
            valid_future_indices = future_indices[valid_positions_mask]
            
            # Safely clip future indices for valid positions only
            # This is safe because we've already verified these are within bounds
            valid_future_indices = np.clip(valid_future_indices, 0, price_data_size - 1)
            
            # Get prices for valid positions
            current_prices = price_data[valid_current_indices]
            future_prices = price_data[valid_future_indices]

            # Calculate average future prices
            avg_future_prices = np.mean(future_prices, axis=1)

            # Generate targets: 1 for buy, 0 for sell
            buy_mask = avg_future_prices > current_prices
            sell_mask = avg_future_prices < current_prices
            
            # Update targets for valid positions
            targets[valid_pos_indices[buy_mask], 0] = 1.0
            targets[valid_pos_indices[sell_mask], 0] = 0.0
            
            # Handle flat markets with deterministic random assignment
            flat_mask = ~buy_mask & ~sell_mask
            
            if np.any(flat_mask):
                flat_positions = positions[valid_positions_mask][flat_mask]
                global_positions = flat_positions + global_offset
                hash_vals = ((global_positions.astype(np.uint64) * 2654435761) % (2**32)).astype(np.float32) / (2**32)
                targets[valid_pos_indices[flat_mask], 0] = (hash_vals > 0.5).astype(np.float32)
            
            # PHASE 3: Enhanced logging for verification
            num_buys = np.sum(buy_mask)
            num_sells = np.sum(sell_mask)
            num_flats = np.sum(flat_mask)
            valid_total = len(valid_pos_indices)
            
            if valid_total > 0 and logger.isEnabledFor(logging.DEBUG):
                logger.debug(f"Target distribution - Buy: {num_buys}/{valid_total} ({100*num_buys/valid_total:.1f}%), "
                           f"Sell: {num_sells}/{valid_total} ({100*num_sells/valid_total:.1f}%), "
                           f"Flat: {num_flats}/{valid_total} ({100*num_flats/valid_total:.1f}%), "
                           f"Invalid: {num_invalid}/{batch_size}")
        
        return targets
    
    def create_shared_buffer(self, name: str, shape: Tuple[int, ...], dtype=np.float32) -> Tuple[str, np.ndarray]:
        """Create a shared memory buffer (only available in shared memory mode)"""
        if not self.enable_shared_memory_mode or self.shared_memory_manager is None:
            raise RuntimeError("Shared memory mode not enabled. Set enable_shared_memory_mode=True.")
        return self.shared_memory_manager.create_block(name, shape, dtype)
    
    def get_shared_array(self, name: str) -> Optional[np.ndarray]:
        """Get numpy array from shared memory (only available in shared memory mode)"""
        if not self.enable_shared_memory_mode or self.shared_memory_manager is None:
            raise RuntimeError("Shared memory mode not enabled. Set enable_shared_memory_mode=True.")
        return self.shared_memory_manager.get_array(name)
    
    def cleanup(self):
        """Clean up resources"""
        if self.executor is not None:
            self.executor.shutdown(wait=True)
            self.executor = None
        
        self.memory_pool.clear()
        
        if self.shared_memory_manager is not None:
            self.shared_memory_manager.cleanup()
        
        logger.info(f"CPUWindowCreator cleaned up. Total windows created: {self._total_windows_created}")
    
    def __del__(self):
        """Destructor to ensure cleanup"""
        self.cleanup()


# Backward compatibility alias
SharedMemoryWindowCreator = CPUWindowCreator

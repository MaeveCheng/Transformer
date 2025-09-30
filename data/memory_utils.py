"""
Memory management utilities for efficient data processing.

Provides shared memory pools and buffer management for CPU and GPU operations.
"""

import torch
import numpy as np
from typing import List, Tuple, Optional, Dict, Any
import threading
import logging
from multiprocessing import shared_memory

logger = logging.getLogger(__name__)


class MemoryPoolManager:
    """Manages memory pools for efficient buffer reuse."""
    
    def __init__(self, max_size: int = 10, use_pinned: bool = True):
        """
        Initialize memory pool manager.
        
        Args:
            max_size: Maximum number of buffers to keep in pool
            use_pinned: Use pinned memory for GPU transfers
        """
        self.max_size = max_size
        self.use_pinned = use_pinned and torch.cuda.is_available()
        self.buffers = []
        self.lock = threading.Lock()
        self.stats = {
            'hits': 0,
            'misses': 0,
            'allocations': 0
        }
    
    def get_buffer(self, shape: Tuple[int, ...], dtype: torch.dtype = torch.float32) -> Optional[torch.Tensor]:
        """
        Get a buffer from the pool or allocate a new one.
        
        Args:
            shape: Shape of the required buffer
            dtype: Data type of the buffer
            
        Returns:
            Buffer tensor or None if allocation fails
        """
        with self.lock:
            # Look for exact match first
            for i, buffer in enumerate(self.buffers):
                if buffer.shape == shape and buffer.dtype == dtype:
                    self.stats['hits'] += 1
                    return self.buffers.pop(i)
            
            # Look for buffer that can be reshaped
            total_elements = np.prod(shape)
            for i, buffer in enumerate(self.buffers):
                if buffer.numel() >= total_elements and buffer.dtype == dtype:
                    self.stats['hits'] += 1
                    buffer = self.buffers.pop(i)
                    return buffer.view(-1)[:total_elements].view(shape)
        
        # Allocate new buffer
        self.stats['misses'] += 1
        try:
            if self.use_pinned:
                buffer = torch.empty(shape, dtype=dtype, pin_memory=True)
            else:
                buffer = torch.empty(shape, dtype=dtype)
            self.stats['allocations'] += 1
            return buffer
        except Exception as e:
            logger.warning(f"Failed to allocate buffer {shape}: {e}")
            return None
    
    def return_buffer(self, buffer: torch.Tensor):
        """Return a buffer to the pool."""
        if buffer is None:
            return
        
        with self.lock:
            if len(self.buffers) < self.max_size:
                self.buffers.append(buffer)
    
    def get_stats(self) -> Dict[str, int]:
        """Get pool statistics."""
        return self.stats.copy()
    
    def clear(self):
        """Clear all buffers from the pool."""
        with self.lock:
            self.buffers.clear()


class SharedMemoryManager:
    """Manages shared memory blocks for inter-process data sharing."""
    
    def __init__(self, prefix: str = ""):
        """
        Initialize shared memory manager.
        
        Args:
            prefix: Prefix for shared memory block names
        """
        self.prefix = prefix
        self.blocks = {}
        self.arrays = {}
    
    def create_block(self, name: str, shape: Tuple[int, ...], dtype=np.float32) -> Tuple[str, np.ndarray]:
        """
        Create a shared memory block.
        
        Args:
            name: Name for the block
            shape: Shape of the array
            dtype: Data type
            
        Returns:
            Tuple of (block_name, numpy_array)
        """
        block_name = f"{self.prefix}_{name}" if self.prefix else name
        size = np.prod(shape) * np.dtype(dtype).itemsize
        
        try:
            shm = shared_memory.SharedMemory(create=True, size=size, name=block_name)
            self.blocks[name] = shm
            
            # Create numpy array view
            arr = np.ndarray(shape, dtype=dtype, buffer=shm.buf)
            self.arrays[name] = arr
            
            return block_name, arr
        except Exception as e:
            logger.error(f"Failed to create shared memory block {block_name}: {e}")
            raise
    
    def get_array(self, name: str) -> Optional[np.ndarray]:
        """Get numpy array for a shared memory block."""
        return self.arrays.get(name)
    
    def cleanup(self):
        """Clean up all shared memory blocks."""
        for name, shm in self.blocks.items():
            shm.close()
            try:
                shm.unlink()
            except FileNotFoundError:
                pass
        
        self.blocks.clear()
        self.arrays.clear()
        
        logger.info(f"Cleaned up {len(self.blocks)} shared memory blocks")


class NumpyMemoryPool:
    """Memory pool for numpy arrays."""
    
    def __init__(self, max_size: int = 10):
        """
        Initialize numpy memory pool.
        
        Args:
            max_size: Maximum number of arrays to keep in pool
        """
        self.max_size = max_size
        self.arrays = []
        self.lock = threading.Lock()
    
    def get_array(self, shape: Tuple[int, ...], dtype=np.float32) -> np.ndarray:
        """Get an array from the pool or allocate a new one."""
        with self.lock:
            # Look for matching array
            for i, arr in enumerate(self.arrays):
                if arr.shape == shape and arr.dtype == dtype:
                    return self.arrays.pop(i)
            
            # Look for array that can be reshaped
            total_elements = np.prod(shape)
            for i, arr in enumerate(self.arrays):
                if arr.size >= total_elements and arr.dtype == dtype:
                    arr = self.arrays.pop(i)
                    return arr.ravel()[:total_elements].reshape(shape)
        
        # Allocate new array
        return np.empty(shape, dtype=dtype)
    
    def return_array(self, arr: np.ndarray):
        """Return an array to the pool."""
        if arr is None:
            return
        
        with self.lock:
            if len(self.arrays) < self.max_size:
                self.arrays.append(arr)
    
    def clear(self):
        """Clear the pool."""
        with self.lock:
            self.arrays.clear()
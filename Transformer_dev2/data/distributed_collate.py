"""
Distributed collate function with CPU batching and efficient GPU transfer.

Key features:
- CPU-based batch assembly
- Pinned memory for fast GPU transfer
- Asynchronous prefetching to GPU
- Support for mixed precision
- Memory pooling for efficiency
"""

import torch
from typing import List, Tuple, Dict, Any, Optional, Union
import logging
from concurrent.futures import ThreadPoolExecutor

from .memory_utils import MemoryPoolManager

logger = logging.getLogger(__name__)


class DistributedCollateFunction:
    """
    Collate function optimized for distributed training with CPU preprocessing.
    
    Features:
    - Efficient CPU batch assembly
    - Pinned memory allocation for fast GPU transfer
    - Asynchronous GPU prefetching
    - Memory pool management
    """
    
    def __init__(
        self,
        use_pinned_memory: bool = True,
        prefetch_to_gpu: bool = True,
        device: Union[str, torch.device] = 'cuda',
        enable_mixed_precision: bool = True,
        memory_pool_size: int = 10,
        async_transfer: bool = True,
        num_transfer_streams: int = 2
    ):
        """
        Initialize distributed collate function.
        
        Args:
            use_pinned_memory: Use pinned memory for GPU transfer
            prefetch_to_gpu: Prefetch batches to GPU
            device: Target device for data
            enable_mixed_precision: Convert to float16 on GPU
            memory_pool_size: Size of pinned memory pool
            async_transfer: Use async GPU transfers
            num_transfer_streams: Number of CUDA streams for transfers
        """
        # Store configuration - delay initialization to avoid pickling issues
        self.config = {
            'use_pinned_memory': use_pinned_memory,
            'prefetch_to_gpu': prefetch_to_gpu,
            'device': device,
            'enable_mixed_precision': enable_mixed_precision,
            'memory_pool_size': memory_pool_size,
            'async_transfer': async_transfer,
            'num_transfer_streams': num_transfer_streams
        }
        
        # These will be initialized on first call
        self._initialized = False
        self.memory_pool = None
        self.transfer_streams = None
        self.stats = None
    
    def _lazy_init(self):
        """Lazily initialize components to avoid pickling issues."""
        if self._initialized:
            return
        
        # Check if we're in a worker process
        worker_info = torch.utils.data.get_worker_info()
        is_worker = worker_info is not None
        
        # Only use pinned memory and GPU in main process
        self.use_pinned_memory = self.config['use_pinned_memory'] and torch.cuda.is_available() and not is_worker
        self.prefetch_to_gpu = self.config['prefetch_to_gpu'] and torch.cuda.is_available() and not is_worker
        self.device = torch.device(self.config['device']) if not is_worker else torch.device('cpu')
        self.enable_mixed_precision = self.config['enable_mixed_precision']
        self.memory_pool_size = self.config['memory_pool_size']
        self.async_transfer = self.config['async_transfer']
        
        # Memory pool
        self.memory_pool = MemoryPoolManager(
            max_size=self.memory_pool_size,
            use_pinned=self.use_pinned_memory
        ) if self.use_pinned_memory else None
        
        # CUDA streams for async transfer
        self.transfer_streams = []
        if self.async_transfer and torch.cuda.is_available() and not is_worker:
            for _ in range(self.config['num_transfer_streams']):
                self.transfer_streams.append(torch.cuda.Stream(device=self.device))
        
        # Statistics
        self.stats = {
            'batches_processed': 0
        }
        
        self._initialized = True
        
        logger.info(f"Initialized DistributedCollateFunction: "
                   f"pinned_memory={self.use_pinned_memory}, "
                   f"prefetch_gpu={self.prefetch_to_gpu}, "
                   f"device={self.device}")
    
    def __call__(self, batch: List[Any]) -> Dict[str, torch.Tensor]:
        """
        Collate batch data efficiently.
        
        Args:
            batch: List of dictionaries with 'features' and 'labels' from dataset
            
        Returns:
            Dictionary with batched tensors on target device
        """
        # Initialize on first call
        self._lazy_init()
        
        if not batch:
            # Return empty dictionary to avoid CUDA init in worker
            empty_dict = {
                'features': torch.empty(0),
                'labels': torch.empty(0)
            }
            if self.prefetch_to_gpu and torch.cuda.is_available():
                # Only transfer to GPU in main process
                for key in empty_dict:
                    empty_dict[key] = empty_dict[key].to(self.device)
            return empty_dict
        
        self.stats['batches_processed'] += 1
        
        # Handle single dictionary item (common with IterableDataset)
        if isinstance(batch, dict) and 'features' in batch:
            return self._transfer_dict_to_device(batch)
        
        # Check if batch contains a wrapped list (from IterableDataset with batch_size=None)
        if batch and isinstance(batch, list):
            if len(batch) > 0:
                if len(batch) == 1 and isinstance(batch[0], dict) and 'features' in batch[0]:
                    # Single dictionary in a list
                    return self._transfer_dict_to_device(batch[0])
                elif len(batch) == 1 and isinstance(batch[0], list) and len(batch[0]) > 0:
                    # Nested list case
                    if len(batch[0]) == 1 and isinstance(batch[0][0], dict):
                        return self._transfer_dict_to_device(batch[0][0])
                elif isinstance(batch[0], str):
                    # Check if batch contains strings (dictionary keys passed incorrectly)
                    logger.warning(f"Received dictionary keys instead of data: {batch}")
                    # This happens when DataLoader incorrectly unpacks a dictionary
                    # Return empty batch to avoid crash
                    return {
                        'features': torch.empty(0, 1, 1),  # Empty 3D tensor
                        'labels': torch.empty(0, 1)
                    }
        
        # Check if data is already batched (from CPU dataset)
        if hasattr(batch, '__len__') and len(batch) == 1:
            item = batch[0]
            if isinstance(item, dict) and 'features' in item:
                # Already a dictionary, transfer to device if needed
                return self._transfer_dict_to_device(item)
            elif isinstance(item, tuple) and len(item) == 2:
                # Legacy tuple format - convert to dict
                data, targets = item
                if isinstance(data, torch.Tensor) and data.dim() >= 2:
                    return self._transfer_dict_to_device({
                        'features': data,
                        'labels': targets
                    })
            elif isinstance(item, torch.Tensor):
                # Single tensor without target - this shouldn't happen
                logger.error(f"Received single tensor without target: shape={item.shape}")
                # Create dummy targets
                targets = torch.zeros(item.shape[0], 1)
                return self._transfer_dict_to_device({
                    'features': item,
                    'labels': targets
                })
        
        # Collate individual samples
        return self._collate_samples(batch)
    
    def _collate_samples(self, samples: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """Collate individual samples into batch"""
        # Separate features and labels
        features_list = []
        labels_list = []
        
        for item in samples:
            if isinstance(item, dict) and 'features' in item:
                features_list.append(item['features'])
                labels_list.append(item.get('labels', torch.zeros(1)))
            elif isinstance(item, tuple) and len(item) == 2:
                # Handle legacy tuple format
                data, target = item
                features_list.append(data)
                labels_list.append(target)
            else:
                logger.warning(f"Unexpected item format in batch: {type(item)}")
        
        if not features_list:
            # Create empty dictionary on CPU first to avoid CUDA initialization in worker
            return {'features': torch.empty(0), 'labels': torch.empty(0)}
        
        # Determine batch shape
        batch_size = len(features_list)
        data_shape = features_list[0].shape
        batch_shape = (batch_size,) + data_shape
        
        # Try to use pinned memory buffer
        if self.memory_pool:
            data_buffer = self.memory_pool.get_buffer(batch_shape)
            target_shape = (batch_size,) + labels_list[0].shape
            target_buffer = self.memory_pool.get_buffer(target_shape)
        else:
            data_buffer = None
            target_buffer = None
        
        # Stack features
        if data_buffer is not None:
            # Use pre-allocated pinned buffer
            for i, data in enumerate(features_list):
                data_buffer[i] = data
            batched_features = data_buffer
        else:
            # Allocate new tensor
            if self.use_pinned_memory:
                batched_features = torch.stack(features_list).pin_memory()
            else:
                batched_features = torch.stack(features_list)
        
        # Stack labels
        if target_buffer is not None:
            for i, target in enumerate(labels_list):
                target_buffer[i] = target
            batched_labels = target_buffer
        else:
            if self.use_pinned_memory:
                batched_labels = torch.stack(labels_list).pin_memory()
            else:
                batched_labels = torch.stack(labels_list)
        
        # Create result dictionary
        result_dict = {
            'features': batched_features,
            'labels': batched_labels
        }
        
        # Transfer to device
        result = self._transfer_dict_to_device(result_dict)
        
        # Return buffers to pool if not transferred
        if not self.prefetch_to_gpu and self.memory_pool:
            self.memory_pool.return_buffer(data_buffer)
            self.memory_pool.return_buffer(target_buffer)
        
        return result
    
    def _transfer_to_device(self, data: torch.Tensor, targets: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Transfer tensors to target device"""
        if not self.prefetch_to_gpu:
            return data, targets
        
        # Check if already on target device
        if data.device == self.device and targets.device == self.device:
            if self.enable_mixed_precision and data.dtype != torch.float16:
                data = data.half()
            return data, targets
        
        # Async transfer using CUDA streams
        if self.async_transfer and self.transfer_streams:
            stream_idx = self.stats['batches_processed'] % len(self.transfer_streams)
            stream = self.transfer_streams[stream_idx]
            
            with torch.cuda.stream(stream):
                data_gpu = data.to(self.device, non_blocking=True)
                targets_gpu = targets.to(self.device, non_blocking=True)
                
                if self.enable_mixed_precision and data_gpu.dtype != torch.float16:
                    data_gpu = data_gpu.half()
        else:
            # Synchronous transfer
            data_gpu = data.to(self.device)
            targets_gpu = targets.to(self.device)
            
            if self.enable_mixed_precision and data_gpu.dtype != torch.float16:
                data_gpu = data_gpu.half()
        
        return data_gpu, targets_gpu
    
    def _transfer_dict_to_device(self, data_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Transfer dictionary of tensors to target device"""
        if not self.prefetch_to_gpu:
            return data_dict
        
        result = {}
        
        # Use async transfer if available
        if self.async_transfer and self.transfer_streams:
            stream_idx = self.stats['batches_processed'] % len(self.transfer_streams)
            stream = self.transfer_streams[stream_idx]
            
            with torch.cuda.stream(stream):
                for key, value in data_dict.items():
                    if isinstance(value, torch.Tensor):
                        if value.device != self.device:
                            tensor_gpu = value.to(self.device, non_blocking=True)
                            if self.enable_mixed_precision and key == 'features' and tensor_gpu.dtype != torch.float16:
                                tensor_gpu = tensor_gpu.half()
                            result[key] = tensor_gpu
                        else:
                            result[key] = value
                    else:
                        # Non-tensor values (like batch_info dict) pass through unchanged
                        result[key] = value
        else:
            # Synchronous transfer
            for key, value in data_dict.items():
                if isinstance(value, torch.Tensor):
                    if value.device != self.device:
                        tensor_gpu = value.to(self.device)
                        if self.enable_mixed_precision and key == 'features' and tensor_gpu.dtype != torch.float16:
                            tensor_gpu = tensor_gpu.half()
                        result[key] = tensor_gpu
                    else:
                        result[key] = value
                else:
                    # Non-tensor values (like batch_info dict) pass through unchanged
                    result[key] = value
        
        return result
    
    def get_stats(self) -> Dict[str, int]:
        """Get collation statistics"""
        if not self._initialized:
            return {}
        stats = self.stats.copy()
        if self.memory_pool:
            stats.update(self.memory_pool.get_stats())
        return stats
    
    def cleanup(self):
        """Clean up resources"""
        if not self._initialized:
            return
        
        # Clear memory pool
        if self.memory_pool:
            self.memory_pool.clear()
        
        # Synchronize streams
        if self.transfer_streams:
            for stream in self.transfer_streams:
                stream.synchronize()
        
        logger.info(f"DistributedCollateFunction stats: {self.get_stats()}")




def create_collate_fn(
    distributed: bool = False,
    **kwargs
) -> DistributedCollateFunction:
    """
    Factory function to create appropriate collate function.
    
    Args:
        distributed: Whether to use distributed collate
        **kwargs: Additional arguments
        
    Returns:
        Appropriate collate function instance
    """
    return DistributedCollateFunction(**kwargs)
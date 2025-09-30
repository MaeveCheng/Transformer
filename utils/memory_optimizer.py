import torch
import torch.nn as nn
import logging
import time
import math
import os
import pandas as pd
from typing import Optional, Dict, Any, Callable, Tuple
from contextlib import contextmanager
from pytorch_lightning.callbacks import Callback

logger = logging.getLogger(__name__)


class MemoryOptimizer:
    """GPU memory optimization utilities."""
    
    def __init__(self, device: torch.device = None, memory_fraction: float = 0.95):
        self.device = device or torch.device('cuda')
        self.initial_memory_fraction = memory_fraction
        self._configure_memory_pool()
        
    def _configure_memory_pool(self):
        """Configure PyTorch memory allocator for efficiency."""
        # Set memory fraction
        torch.cuda.set_per_process_memory_fraction(
            self.initial_memory_fraction, 
            self.device.index if self.device.index is not None else 0
        )
        
        # Configure allocator
        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512'


class AdaptiveBatchSizer:
    """Automatically adjust batch size based on GPU memory."""
    
    def __init__(
        self, 
        initial_batch_size: int,
        min_batch_size: int = 1,
        memory_fraction_target: float = 0.9
    ):
        self.current_batch_size = initial_batch_size
        self.min_batch_size = min_batch_size
        self.memory_fraction_target = memory_fraction_target
        self.memory_history = []
        
    def adjust_on_oom(self, model: nn.Module, optimizer: torch.optim.Optimizer) -> int:
        """Reduce batch size after OOM error."""
        # Clear cache
        torch.cuda.empty_cache()
        
        # Reduce batch size
        self.current_batch_size = max(
            self.min_batch_size,
            self.current_batch_size // 2
        )
        
        logger.warning(f"OOM detected, reducing batch size to {self.current_batch_size}")
        
        # Reset optimizer state to free memory
        optimizer.zero_grad(set_to_none=True)
        
        return self.current_batch_size
    
    def find_optimal_batch_size(
        self, 
        model: nn.Module, 
        sample_input_shape: Tuple[int, ...]
    ) -> int:
        """Find optimal batch size with proper error handling."""
        # Get device safely
        try:
            device = next(model.parameters()).device
        except StopIteration:
            device = torch.device('cuda')
        
        low = self.min_batch_size
        high = self.current_batch_size * 2
        optimal = self.min_batch_size
        
        # Reset memory stats
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        
        while low <= high:
            mid = (low + high) // 2
            
            try:
                test_input = torch.randn(mid, *sample_input_shape, device=device)
                
                with torch.cuda.amp.autocast(enabled=True):
                    _ = model(test_input)
                
                # Safe division
                max_mem = torch.cuda.max_memory_allocated()
                total_mem = torch.cuda.get_device_properties(device).total_memory
                memory_used = max_mem / total_mem if total_mem > 0 else 1.0
                
                if memory_used < self.memory_fraction_target:
                    optimal = mid
                    low = mid + 1
                else:
                    high = mid - 1
                
                del test_input
                    
            except RuntimeError as e:
                if "out of memory" in str(e):
                    high = mid - 1
                    torch.cuda.empty_cache()
                else:
                    raise
        
        logger.info(f"Optimal batch size found: {optimal}")
        return optimal


class MemoryManagementCallback(Callback):
    """PyTorch Lightning callback for memory optimization."""
    
    def __init__(
        self,
        enable_gradient_checkpointing: bool = True,
        profile_memory: bool = False,
        auto_batch_size: bool = True,
        min_batch_size: int = 1,
        memory_fraction: float = 0.95
    ):
        self.enable_gradient_checkpointing = enable_gradient_checkpointing
        self.profile_memory = profile_memory
        self.auto_batch_size = auto_batch_size
        self.min_batch_size = min_batch_size
        self.memory_fraction = memory_fraction
        self.batch_sizer = None
        self.step_count = 0
        self.memory_optimizer = None
        
    def on_fit_start(self, trainer, pl_module):
        """Configure memory optimizations at training start."""
        # Initialize memory optimizer
        self.memory_optimizer = MemoryOptimizer(memory_fraction=self.memory_fraction)
        
        # Enable gradient checkpointing if requested
        if self.enable_gradient_checkpointing:
            self._enable_gradient_checkpointing(pl_module)
            
        # Initialize batch size adjuster
        if self.auto_batch_size:
            self.batch_sizer = AdaptiveBatchSizer(
                initial_batch_size=trainer.datamodule.batch_size,
                min_batch_size=self.min_batch_size
            )
            
        # Configure memory allocator
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
            
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        """Periodic memory cleanup."""
        self.step_count += 1
            
        if self.profile_memory and self.step_count % 100 == 0:
            self._log_memory_stats(trainer)
            
    def on_exception(self, trainer, pl_module, exception):
        """Handle OOM errors gracefully."""
        if isinstance(exception, RuntimeError) and "out of memory" in str(exception):
            if self.batch_sizer:
                # Adjust batch size
                new_batch_size = self.batch_sizer.adjust_on_oom(
                    pl_module.model, 
                    trainer.optimizers[0]
                )
                
                # Update datamodule
                trainer.datamodule._batch_size = new_batch_size
                
                # Clear error and continue training
                return True  # Suppress exception
                
        return False  # Propagate other exceptions
        
    def _enable_gradient_checkpointing(self, pl_module):
        """Enable gradient checkpointing on the model."""
        model = pl_module.model
        
        # Check if model supports gradient checkpointing
        if hasattr(model, 'enable_memory_optimizations'):
            model.enable_memory_optimizations()
            logger.info("Memory optimizations enabled via model method")
        elif hasattr(model, 'gradient_checkpointing_enable'):
            model.gradient_checkpointing_enable()
            logger.info("Gradient checkpointing enabled")
        elif hasattr(model, 'enable_gradient_checkpointing'):
            model.enable_gradient_checkpointing()
            logger.info("Gradient checkpointing enabled")
        else:
            logger.warning("Model does not support gradient checkpointing")
            
    def _log_memory_stats(self, trainer):
        """Log detailed memory statistics."""
        stats = {
            'allocated_mb': torch.cuda.memory_allocated() / 1024**2,
            'reserved_mb': torch.cuda.memory_reserved() / 1024**2,
            'free_mb': (torch.cuda.memory_reserved() - torch.cuda.memory_allocated()) / 1024**2,
        }
        
        for key, value in stats.items():
            trainer.logger.log_metrics({f'memory/{key}': value}, step=trainer.global_step)


@contextmanager
def profile_memory(name: str = "operation"):
    """Context manager for memory profiling."""
    torch.cuda.synchronize()
    start_memory = torch.cuda.memory_allocated()
    
    yield
    
    torch.cuda.synchronize()
    end_memory = torch.cuda.memory_allocated()
    memory_used = (end_memory - start_memory) / 1024**2
    logger.info(f"{name} used {memory_used:.2f} MB")
        
        
class MemoryProfiler:
    """Detailed memory profiling for model operations."""
    
    def __init__(self):
        self.memory_snapshots = []
        
    def snapshot(self, label: str):
        """Take memory snapshot."""
        snapshot = {
            'label': label,
            'allocated': torch.cuda.memory_allocated(),
            'reserved': torch.cuda.memory_reserved(),
            'timestamp': time.time()
        }
        self.memory_snapshots.append(snapshot)
            
    def report(self) -> pd.DataFrame:
        """Generate memory usage report."""
        df = pd.DataFrame(self.memory_snapshots)
        df['allocated_mb'] = df['allocated'] / 1024**2
        df['reserved_mb'] = df['reserved'] / 1024**2
        return df


class SmartMemoryAllocator:
    """Intelligent memory allocation based on model size and available GPU memory."""
    
    @staticmethod
    def estimate_model_memory(model: nn.Module) -> int:
        """Estimate model memory requirements in bytes."""
        param_memory = sum(
            p.numel() * p.element_size() 
            for p in model.parameters()
        )
        
        # Add buffer memory
        buffer_memory = sum(
            b.numel() * b.element_size() 
            for b in model.buffers()
        )
        
        # Estimate gradient memory (same as parameters)
        gradient_memory = param_memory
        
        # Estimate optimizer state memory (2x parameters for Adam)
        optimizer_memory = param_memory * 2
        
        # Add overhead (20%)
        total_memory = int((param_memory + buffer_memory + gradient_memory + optimizer_memory) * 1.2)
        
        return total_memory
    
    @staticmethod
    def recommend_batch_size(
        model: nn.Module, 
        input_shape: Tuple[int, ...],
        available_memory: Optional[int] = None
    ) -> int:
        """Recommend optimal batch size based on available memory."""
        
        if available_memory is None:
            available_memory = torch.cuda.get_device_properties(0).total_memory
            
        # Estimate model memory
        model_memory = SmartMemoryAllocator.estimate_model_memory(model)
        
        # Estimate activation memory per sample
        sample_input = torch.randn(1, *input_shape, device=model.device)
        
        # Reset peak memory stats to get accurate measurement
        torch.cuda.reset_peak_memory_stats()
            
        with torch.no_grad():
            _ = model(sample_input)
            
        activation_memory_per_sample = torch.cuda.max_memory_allocated() - model_memory
            
        # Calculate maximum batch size
        # Reserve 10% for safety
        usable_memory = int(available_memory * 0.9) - model_memory
        max_batch_size = max(1, usable_memory // activation_memory_per_sample)
        
        # Round down to nearest power of 2 for efficiency
        if max_batch_size > 1:
            recommended_batch_size = 2 ** int(math.log2(max_batch_size))
        else:
            recommended_batch_size = 1
        
        return max(1, recommended_batch_size)